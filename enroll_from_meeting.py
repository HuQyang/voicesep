"""
从一段会议音频自动建声纹库.

流程:
  1) (可选) FRCRN 降噪
  2) FSMN-VAD 切段 + 合并 + 滑窗 chunk
  3) ERes2NetV2 提 emb + dBFS 过滤
  4) AHC 聚类 → 发现 N 个 spk (或指定 --num-spk)
  5) 计算每个 spk 的 centroid embedding
  6) 给每个 spk 选 "最干净的代表 wav" (优先长、RMS 高、单 chunk 段)
  7) centroid 注册到 SpeakerDB (.npz), 代表 wav 存到 speakers/refs/

之后:
  - SpeakerDB 用于 main_pipeline.py 的 --enroll-db 替换真名
  - refs/*.wav 用于 TSE 推理 (Mossformer-TSE 等需要参考音频)

用法:
  python enroll_from_meeting.py --wav data/xxx.mp3 --num-spk 3
  python enroll_from_meeting.py --wav data/xxx.mp3 --num-spk 3 \
      --names "王总,毕姐,小雨"
  python enroll_from_meeting.py --wav data/xxx.mp3 --db speakers/db.npz \
      --refs-dir speakers/refs
"""
import argparse
import os
import tempfile
from collections import defaultdict, Counter

import numpy as np
import librosa
import soundfile as sf
import torch

from main_pipeline import (
    SR,
    _free_gpu, _release_model, _rms_dbfs,
    run_vad,
    extract_chunk_embs, cluster_embs, compute_centroids,
    segments_to_turns,
)
import main_bss
from main_bss import run_denoise
from main_diarization import merge_segments, chunk_long_segments
from speaker_db import extract_embedding_from_wave, SpeakerDB


def pick_best_ref_segment(wav, segments, kept_chunks, labels, spk_id,
                          min_dur_s=3.0, max_dur_s=10.0):
    """
    给 spk_id 选一段"最干净的代表"音频用于 TSE 参考.
    策略 (按优先级):
      1) 在该 spk 的所有 VAD segment 里, 选时长在 [min_dur, max_dur] 内、RMS 最高、
         且段内所有 chunk 都聚到这个 spk 的段 (纯净度高)
      2) 如果没有这样的段, 退一步: 只要段内多数 chunk 是这个 spk, 也算
      3) 实在不行, 取该 spk 任意单 chunk 长 emb (最少 2~3s)
    返回 (start_ms, end_ms, wav_slice) 或 None
    """
    # 把每段 segment 内的 chunk labels 收集起来
    seg_chunk_labels = defaultdict(list)  # seg_idx -> [labels]
    for (cs, ce), lab in zip(kept_chunks, labels):
        for i, (ss, se) in enumerate(segments):
            if cs >= ss and ce <= se + 1:
                seg_chunk_labels[i].append(int(lab))
                break

    # 候选: 段内多数 chunk 是 spk_id
    candidates = []
    for i, (ss, se) in enumerate(segments):
        labs = seg_chunk_labels.get(i, [])
        if not labs:
            continue
        top = Counter(labs).most_common(1)[0][0]
        if top != spk_id:
            continue
        pure_score = labs.count(spk_id) / len(labs)  # 1.0 = 完全纯, 0.5 = 一半
        dur_s = (se - ss) / 1000.0
        seg_wav = wav[int(ss/1000*SR):int(se/1000*SR)]
        rms = _rms_dbfs(seg_wav)
        # 评分: 纯度 + 时长合适性 + 能量
        dur_score = 1.0 if min_dur_s <= dur_s <= max_dur_s else (
            0.5 if dur_s > max_dur_s else dur_s / min_dur_s
        )
        score = pure_score * 2 + dur_score + (rms + 50) / 50  # rms ~ [-50, 0] → [0, 1]
        candidates.append((score, ss, se, dur_s, pure_score, rms))

    if not candidates:
        return None
    candidates.sort(reverse=True)  # 分数最高在前
    _, ss, se, dur_s, pure, rms = candidates[0]

    # 如果选中的段 > max_dur, 截到中间 max_dur 那块 (头尾常有静音 padding)
    if dur_s > max_dur_s:
        center = (ss + se) // 2
        half = int(max_dur_s * 1000 / 2)
        ss = max(ss, center - half)
        se = min(se, center + half)

    seg_wav = wav[int(ss/1000*SR):int(se/1000*SR)]
    return ss, se, seg_wav, pure, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="输入会议音频")
    ap.add_argument("--num-spk", type=int, default=None, help="已知人数 (最稳)")
    ap.add_argument("--threshold", type=float, default=0.7, help="AHC 距离阈值, 不传 num-spk 时用")
    ap.add_argument("--names", default=None, help="可选: 逗号分隔的 spk 名字 (按聚类顺序), 如 '王总,毕姐,小雨'")
    ap.add_argument("--db", default="speakers/db.npz", help="声纹库 .npz 路径")
    ap.add_argument("--refs-dir", default="speakers/refs", help="代表 wav 保存目录")
    ap.add_argument("--ref-min-dur", type=float, default=3.0)
    ap.add_argument("--ref-max-dur", type=float, default=10.0)
    ap.add_argument("--overwrite", action="store_true", help="同名 spk 直接覆盖")

    # 复用主管线参数
    ap.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--vad", choices=["fsmn", "silero"], default="fsmn")
    ap.add_argument("--min-dbfs", type=float, default=-50.0)
    ap.add_argument("--merge-gap", type=int, default=300)
    ap.add_argument("--min-dur", type=int, default=800)
    ap.add_argument("--chunk-max", type=int, default=3000)
    ap.add_argument("--chunk-hop", type=int, default=1500)
    ap.add_argument("--embedder-model", default=None,
                    help="声纹模型. 必须和 main_pipeline 用同一个, 否则维度/分布不兼容. "
                         "main_pipeline 默认 iic/speech_eres2net_large_200k_sv_zh-cn_16k-common (512-d)")
    args = ap.parse_args()

    if args.embedder_model:
        from speaker_db import set_embedder_model
        set_embedder_model(args.embedder_model)

    # 1. 加载
    print(f"\n=== [1/5] 加载 {args.wav} ===")
    wav, _ = librosa.load(args.wav, sr=SR, mono=True)
    print(f"  时长 {len(wav)/SR:.1f}s")

    # 2. 降噪
    if args.denoise:
        print(f"\n=== [2/5] FRCRN 降噪 ===")
        wav = run_denoise(wav, sr=SR)
        _release_model(main_bss, "_denoise_pipe")
        _free_gpu()
    else:
        print(f"\n=== [2/5] 降噪已跳过 ===")

    # 写降噪后 wav 到临时文件供 VAD
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, SR)
        clean_path = tmp.name

    try:
        # 3. VAD + diar
        print(f"\n=== [3/5] VAD + 聚类 ===")
        raw_segments = run_vad(clean_path, engine=args.vad)
        segments = merge_segments(raw_segments, args.merge_gap, args.min_dur)
        chunks = chunk_long_segments(segments, args.chunk_max, args.chunk_hop)
        print(f"  VAD={len(raw_segments)} → MERGE={len(segments)} → CHUNK={len(chunks)}")

        embs, kept_chunks = extract_chunk_embs(wav, chunks, min_dbfs=args.min_dbfs)
        print(f"  EMB {len(embs)} (192-d)")
        if len(embs) < 2:
            print("[!] embedding 不足 2 个, 无法聚类")
            return

        labels = cluster_embs(embs, num_spk=args.num_spk, threshold=args.threshold)
        centroids = compute_centroids(embs, labels)
        spk_ids = sorted(centroids.keys())
        print(f"  发现 {len(spk_ids)} 个 spk 簇: {spk_ids}")
        cluster_sizes = Counter(int(l) for l in labels)
        for sid in spk_ids:
            print(f"    spk_{sid}: {cluster_sizes[sid]} chunks")

        # 4. 给每个 spk 选 ref + 落盘
        print(f"\n=== [4/5] 选每个 spk 的代表 wav ===")
        os.makedirs(args.refs_dir, exist_ok=True)
        names = []
        if args.names:
            names = [n.strip() for n in args.names.split(",") if n.strip()]
            if len(names) != len(spk_ids):
                print(f"[!] --names 给了 {len(names)} 个, 但实际有 {len(spk_ids)} 个 spk, 多余的用默认 spk_X")

        chosen_refs = {}   # spk_id -> (name, ref_wav_path, ss, se, pure, rms)
        for idx, sid in enumerate(spk_ids):
            picked = pick_best_ref_segment(
                wav, segments, kept_chunks, labels, sid,
                min_dur_s=args.ref_min_dur, max_dur_s=args.ref_max_dur,
            )
            if picked is None:
                print(f"  spk_{sid}: 找不到合适的 ref 段, 跳过")
                continue
            ss, se, seg_wav, pure, rms = picked
            name = names[idx] if idx < len(names) else f"spk_{sid}"
            safe_name = name.replace("/", "_").replace(" ", "_")
            ref_path = os.path.join(args.refs_dir, f"{safe_name}.wav")
            sf.write(ref_path, seg_wav, SR)
            chosen_refs[sid] = (name, ref_path, ss, se, pure, rms)
            print(f"  {name}  ←  {ss/1000:.1f}-{se/1000:.1f}s "
                  f"(纯度 {pure:.2f}, RMS {rms:.1f} dBFS)  →  {ref_path}")

        # 5. 注册到 SpeakerDB
        print(f"\n=== [5/5] 注册到 SpeakerDB ===")
        db = SpeakerDB(args.db)
        for sid, (name, ref_path, ss, se, pure, rms) in chosen_refs.items():
            if (not args.overwrite) and name in db.names:
                print(f"  [skip] {name} 已存在, 用 --overwrite 强制覆盖")
                continue
            # 用 centroid (聚类平均) 而非单段 emb, 更稳
            db.add(name, centroids[sid])
        db.save()

        print(f"\n=== 完成 ===")
        print(f"  声纹库: {os.path.abspath(args.db)}")
        print(f"  ref wav: {os.path.abspath(args.refs_dir)}")
        print(f"\n下次跑 main_pipeline 加 --enroll-db {args.db} 就能用真名替换 spk_X")
        print(f"TSE 时把 {args.refs_dir} 下的 wav 作 reference")

    finally:
        try: os.unlink(clean_path)
        except OSError: pass


if __name__ == "__main__":
    main()
