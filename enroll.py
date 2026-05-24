"""
声纹注册 + 验证 CLI

子命令:
  add      添加 / 覆盖说话人 (多样本平均, 注册时自动查重)
  list     列出所有已注册
  remove   删除
  match    给一段音频, 找库内最相似的 top N (debug 用)
  verify   验证音频是否匹配指定姓名 (含 vs 其他人 margin)
  audit    两两相似度矩阵, 找重复 / 命名错误

用法:
  python enroll.py add 张三 sample1.wav sample2.wav
  python enroll.py list
  python enroll.py remove 张三

  # 拿一段未知音频找库里最像谁
  python enroll.py match test.wav --top 3

  # 验证一段音频确实是张三 (含跟其他人的 margin)
  python enroll.py verify 张三 test.wav

  # 体检: 看库里有没有重复 / 不该相似的两个人却很像
  python enroll.py audit

  # 跨机器: 必须用同一个 embedder 注册和匹配
  python enroll.py --embedder-model iic/speech_eres2net_base_200k_sv_zh-cn_16k-common ...
"""

import os
import argparse

import numpy as np
import librosa

from speaker_db import (
    SpeakerDB,
    extract_embedding_from_wave,
    cosine_sim,
    set_embedder_model,
)


MIN_DURATION = 5.0
MATCH_HIGH = 0.55     # 高置信度匹配阈值
MATCH_LOW = 0.40      # 远场可接受的最低匹配阈值
DUPE_THRESHOLD = 0.70 # 库内两两相似度 > 此值视为可能重复


# ─────────── 工具 ───────────

def _load_emb(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    wav, _ = librosa.load(path, sr=16000, mono=True)
    dur = len(wav) / 16000.0
    return extract_embedding_from_wave(wav, sr=16000), dur


def _quality_tag(sim: float) -> str:
    if sim >= MATCH_HIGH:
        return "✓ 高置信"
    if sim >= MATCH_LOW:
        return "~ 弱匹配"
    return "✗ 不匹配"


# ─────────── 子命令 ───────────

def cmd_add(args):
    db = SpeakerDB(args.db)
    embs = []
    for path in args.samples:
        try:
            emb, dur = _load_emb(path)
        except FileNotFoundError:
            print(f"[skip] 文件不存在: {path}")
            continue
        if dur < MIN_DURATION:
            print(f"[warn] {path} 仅 {dur:.1f}s (建议 >= {MIN_DURATION:.0f}s)")
        embs.append(emb)
        print(f"[ok] {path} ({dur:.1f}s)")

    if not embs:
        print("没有有效样本, 中止")
        return

    avg = np.mean(np.stack(embs, axis=0), axis=0)

    # 查重: 跟库内已有名字以外的人比对
    if len(db.names) > 0:
        warned = False
        for i, name in enumerate(db.names):
            if name == args.name:
                continue
            sim = cosine_sim(avg, db.embeddings[i])
            if sim > DUPE_THRESHOLD:
                if not warned:
                    print()
                    warned = True
                print(f"  [!] 警告: 与已注册 '{name}' 相似度 {sim:.3f}, 可能是同一人")
        if warned:
            print("  → 如果确认是不同人, 用更干净/更长的样本重新注册")
            print("  → 如果是同一人, 用 'remove' 删冗余, 或直接覆盖")

    db.add(args.name, avg)
    db.save()


def cmd_list(args):
    db = SpeakerDB(args.db)
    if not db.names:
        print("(空)")
        return
    print(f"声纹库: {args.db}  (dim={db.embeddings.shape[1]})")
    for i, n in enumerate(db.names, 1):
        print(f"  {i}. {n}")


def cmd_remove(args):
    db = SpeakerDB(args.db)
    if db.remove(args.name):
        db.save()
        print(f"已删除: {args.name}")
    else:
        print(f"不存在: {args.name}")


def cmd_match(args):
    """给一段音频, 找库内最相似的 top N"""
    db = SpeakerDB(args.db)
    if len(db.names) == 0:
        print("声纹库为空")
        return
    emb, dur = _load_emb(args.audio)
    sims = sorted(
        [(name, cosine_sim(emb, db.embeddings[i]))
         for i, name in enumerate(db.names)],
        key=lambda x: -x[1],
    )

    print(f"\n[match] {args.audio} ({dur:.1f}s)  vs 库内 {len(db.names)} 人\n")
    print(f"  {'排名':<6}{'相似度':<10}{'判定':<10}{'姓名'}")
    for i, (name, sim) in enumerate(sims[:args.top], 1):
        tag = _quality_tag(sim) if i == 1 else ""
        print(f"  {i:<6}{sim:<10.3f}{tag:<10}{name}")

    best_name, best_sim = sims[0]
    print()
    if best_sim >= MATCH_HIGH:
        print(f"  推断: 这段是 {best_name} (相似度 {best_sim:.3f})")
    elif best_sim >= MATCH_LOW:
        print(f"  推断: 可能是 {best_name} (相似度 {best_sim:.3f}, 远场弱匹配)")
    else:
        print(f"  推断: 库里没有 (最高 {best_sim:.3f}, 阈值 {MATCH_LOW})")


def cmd_verify(args):
    """验证音频是否匹配指定名字"""
    db = SpeakerDB(args.db)
    if args.name not in db.names:
        print(f"声纹库中没有 '{args.name}'")
        return
    idx = db.names.index(args.name)
    emb, dur = _load_emb(args.audio)
    sim_self = cosine_sim(emb, db.embeddings[idx])

    others = sorted(
        [(n, cosine_sim(emb, db.embeddings[i]))
         for i, n in enumerate(db.names) if n != args.name],
        key=lambda x: -x[1],
    )

    print(f"\n[verify] {args.audio} ({dur:.1f}s)  vs '{args.name}'")
    print(f"  与 {args.name:<10}     相似度: {sim_self:.3f}  ({_quality_tag(sim_self)})")
    if others:
        top_other, top_other_sim = others[0]
        print(f"  与其他人最高:     {top_other_sim:.3f}  ({top_other})")
        margin = sim_self - top_other_sim
        print(f"  margin:           {margin:+.3f}")

    passed = sim_self >= MATCH_LOW and (not others or sim_self > others[0][1])
    print()
    if passed and sim_self >= MATCH_HIGH:
        print(f"  ✓ 验证通过: 这段是 {args.name}")
    elif passed:
        print(f"  ~ 弱通过: 倾向是 {args.name}, 但 margin 不大")
    else:
        print(f"  ✗ 验证失败")


def cmd_audit(args):
    """两两相似度矩阵 + 重复检测"""
    db = SpeakerDB(args.db)
    n = len(db.names)
    if n < 2:
        print(f"库内不足 2 人 ({n}), 无需 audit")
        return

    print(f"\n[audit] 两两相似度矩阵  ({n}×{n})")
    print(f"  ! 标记相似度 > {DUPE_THRESHOLD} 的可疑对\n")

    # 表头
    name_col = max(len(n) for n in db.names) + 6
    print(" " * name_col, end="")
    for j in range(n):
        print(f"  {j+1:>4}  ", end="")
    print()

    suspicious = []
    for i, name_i in enumerate(db.names):
        print(f"{i+1:>2}. {name_i:<{name_col-4}}", end="")
        for j in range(n):
            if i == j:
                print(f"  {'  -':>4}  ", end="")
                continue
            sim = cosine_sim(db.embeddings[i], db.embeddings[j])
            mark = "!" if sim > DUPE_THRESHOLD else " "
            print(f"  {sim:.2f}{mark} ", end="")
            if i < j and sim > DUPE_THRESHOLD:
                suspicious.append((name_i, db.names[j], sim))
        print()

    print()
    if suspicious:
        print(f"[!] 发现 {len(suspicious)} 对可疑重复:")
        for a, b, s in suspicious:
            print(f"    {a:<10} <=> {b:<10}  {s:.3f}")
        print()
        print("  → 如果是同一人录了多次, 删一个: python enroll.py remove <name>")
        print("  → 如果是不同人但声纹接近, 录更长更干净的样本重新注册")
    else:
        print("[✓] 没有可疑重复")


# ─────────── 入口 ───────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", default="speakers/db.npz", help="声纹库路径")
    ap.add_argument("--embedder-model", default="iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
                    help="覆盖默认声纹模型 (必须与注册时一致)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="注册 / 覆盖")
    p.add_argument("name")
    p.add_argument("samples", nargs="+")
    p.set_defaults(func=cmd_add)

    sub.add_parser("list", help="列出已注册").set_defaults(func=cmd_list)

    p = sub.add_parser("remove", help="删除")
    p.add_argument("name")
    p.set_defaults(func=cmd_remove)

    p = sub.add_parser("match", help="找库内最像的 top N")
    p.add_argument("audio")
    p.add_argument("--top", type=int, default=3)
    p.set_defaults(func=cmd_match)

    p = sub.add_parser("verify", help="验证音频是否匹配指定名字")
    p.add_argument("name")
    p.add_argument("audio")
    p.set_defaults(func=cmd_verify)

    sub.add_parser("audit", help="两两相似度矩阵, 找重复").set_defaults(func=cmd_audit)

    args = ap.parse_args()
    if args.embedder_model:
        set_embedder_model(args.embedder_model)
    args.func(args)


if __name__ == "__main__":
    main()
