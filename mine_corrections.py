"""
从 ASR 输出 + 参考文本中自动挖掘候选纠错对.

参考文本来源:
  - 讯飞 / 通义 / 飞书的导出 txt
  - 手工校对过的 PDF / Word
  - 同一会议的另一 ASR 输出 (相互校验)

挖掘逻辑:
  1. 两边都去掉时间戳 / 说话人标签 / 标点, 留纯中文文字
  2. difflib 字符级 diff, 找 replace 操作 (i1:i2 → j1:j2)
  3. 长度过滤: 错词长度 2-8 字, 长度差 ≤ 2 (太长不是错词, 太短噪音多)
  4. 按 (wrong, right) 聚合频次, 多次出现的可信度高
  5. 半交互式 review (y/n/e/q) 或自动接受高频候选

用法:
  # 单对文件: 交互式 review
  python mine_corrections.py --asr result/sim.txt --ref result/xunfei.txt

  # 多对批量: 跨会议出现 ≥ 3 次的自动收, 其余交互
  python mine_corrections.py --pairs pairs.json --auto-min-freq 3

  # 只看不写
  python mine_corrections.py --asr a.txt --ref b.pdf --dry-run

pairs.json 示例:
  [
    {"asr": "result/m1_simple.txt", "ref": "result/m1_xunfei.txt"},
    {"asr": "result/m2_simple.txt", "ref": "result/m2_xunfei.txt"}
  ]
"""

import os
import re
import json
import argparse
import difflib
from collections import Counter
from typing import List, Dict, Tuple

try:
    import pypdf
except ImportError:
    pypdf = None


# ─────────── 文件读取 ───────────

# 行级元信息正则: 时间戳 / 说话人 / markdown 标题
META_RE = re.compile(
    r"(\[\d+:\d+(?:-\d+:\d+)?\]|"
    r"说话人\s*\d+|发言人\s*\d+|spk[_\d]+|Speaker_\w+|"
    r"^#+\s|^---+\s*$)"
)
SPACE_BETWEEN_CJK = re.compile(r"(?<=[一-鿿])\s+(?=[一-鿿])")


def read_text(path: str) -> str:
    """读 txt / pdf / json, 返回原始文本"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.lower().endswith(".pdf"):
        if pypdf is None:
            raise ImportError("PDF 需要 pip install pypdf")
        reader = pypdf.PdfReader(path)
        return "".join((p.extract_text() or "") for p in reader.pages)
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return "\n".join(p.get("text", "") for p in data if p.get("text"))
        if isinstance(data, dict) and "paragraphs" in data:
            return "\n".join(p.get("text", "") for p in data["paragraphs"])
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def to_clean_chars(text: str) -> str:
    """只保留中文 + 字母数字, 标点/换行/空格/元信息全去 (做 diff 用)"""
    # 去元信息行
    text = META_RE.sub(" ", text)
    # 去 CJK 字符间的空格
    text = SPACE_BETWEEN_CJK.sub("", text)
    # 只留中文 + 字母数字
    text = re.sub(r"[^一-鿿\w]", "", text)
    return text


# ─────────── 候选挖掘 ───────────

def mine_diffs(asr_clean: str, ref_clean: str,
               min_len: int = 2, max_len: int = 8,
               max_len_diff: int = 2,
               context: int = 5) -> List[Dict]:
    """
    返回 [{"wrong", "right", "context_asr"}, ...]

    min_len/max_len: 错词长度
    max_len_diff:    错词与正确词的长度差 (超过认为不像同音字, 跳过)
    context:         展示用的上下文字符数
    """
    sm = difflib.SequenceMatcher(None, asr_clean, ref_clean, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace":
            continue
        wrong = asr_clean[i1:i2]
        right = ref_clean[j1:j2]
        if not (min_len <= len(wrong) <= max_len):
            continue
        if not (min_len <= len(right) <= max_len):
            continue
        if abs(len(wrong) - len(right)) > max_len_diff:
            continue
        left = asr_clean[max(0, i1 - context):i1]
        rgt = asr_clean[i2:min(len(asr_clean), i2 + context)]
        out.append({
            "wrong": wrong,
            "right": right,
            "context_asr": f"…{left}【{wrong}】{rgt}…",
        })
    return out


# ─────────── corrections.json 读写 ───────────

def load_corrections(path: str) -> Dict:
    if not os.path.exists(path):
        return {"rules": [], "hotwords": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_corrections(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")


def append_rules(data: Dict, new_rules: List[Dict]) -> Tuple[int, int]:
    """追加, 跳过已存在. 返回 (新加, 跳过)"""
    if "rules" not in data:
        data["rules"] = []
    existing = {(r["wrong"], r.get("right", "")) for r in data["rules"]}
    added = skipped = 0
    for r in new_rules:
        key = (r["wrong"], r["right"])
        if key in existing:
            skipped += 1
            continue
        data["rules"].append({"wrong": r["wrong"], "right": r["right"]})
        existing.add(key)
        added += 1
    return added, skipped


# ─────────── 交互式 review ───────────

def review_interactive(candidates: List[Dict], existing: set) -> List[Dict]:
    print(f"\n[review] {len(candidates)} 条候选")
    print("  y=收入, n=丢弃, e=编辑后收, s=跳过, q=结束 review\n")
    accepted = []
    for i, c in enumerate(candidates, 1):
        key = (c["wrong"], c["right"])
        flag = "  [库内已有]" if key in existing else ""
        freq = c.get("freq", 1)
        print(f"--- {i}/{len(candidates)} ---  出现 {freq} 次{flag}")
        print(f"  错: {c['wrong']!r}  →  对: {c['right']!r}")
        print(f"  {c.get('context_asr', '')}")
        try:
            ans = input("  [y/n/e/s/q] > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n[stop]")
            break
        if ans == "q":
            break
        if ans == "s" or key in existing:
            continue
        if ans == "y":
            accepted.append({"wrong": c["wrong"], "right": c["right"]})
            existing.add(key)
        elif ans == "e":
            new_w = input(f"    新 wrong (回车保留 {c['wrong']!r}): ").strip() or c["wrong"]
            new_r = input(f"    新 right (回车保留 {c['right']!r}): ").strip() or c["right"]
            accepted.append({"wrong": new_w, "right": new_r})
            existing.add((new_w, new_r))
    return accepted


# ─────────── 主流程 ───────────

def process_pair(asr_path: str, ref_path: str,
                 min_len: int, max_len: int) -> List[Dict]:
    asr_clean = to_clean_chars(read_text(asr_path))
    ref_clean = to_clean_chars(read_text(ref_path))
    if len(asr_clean) < 50 or len(ref_clean) < 50:
        print(f"  [warn] 文本太短, 跳过: asr={len(asr_clean)} ref={len(ref_clean)}")
        return []
    return mine_diffs(asr_clean, ref_clean, min_len=min_len, max_len=max_len)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--asr", help="ASR 输出 (.txt / .json / .pdf)")
    ap.add_argument("--ref", help="参考文本 (.txt / .json / .pdf)")
    ap.add_argument("--pairs", help="多对文件 JSON 配置")
    ap.add_argument("--corrections", default="corrections.json",
                    help="目标词典路径 (默认 corrections.json)")
    ap.add_argument("--min-len", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--auto-min-freq", type=int, default=0,
                    help="跨文件出现 ≥ N 次自动收入 (0=全部走交互)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只展示候选, 不写词典")
    ap.add_argument("--max-show", type=int, default=200,
                    help="最多 review 多少条 (按频次降序)")
    args = ap.parse_args()

    # ─── 收集所有候选 ───
    all_cands = []
    if args.pairs:
        with open(args.pairs, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        for p in pairs:
            try:
                c = process_pair(p["asr"], p["ref"], args.min_len, args.max_len)
                print(f"[mine] {p['asr']} vs {p['ref']}: {len(c)} 候选")
                all_cands.extend(c)
            except Exception as e:
                print(f"  [skip] {e}")
    elif args.asr and args.ref:
        all_cands = process_pair(args.asr, args.ref, args.min_len, args.max_len)
        print(f"[mine] {len(all_cands)} 候选")
    else:
        ap.error("必须给 --asr+--ref 或 --pairs")

    if not all_cands:
        print("[done] 没有候选")
        return

    # ─── 按 (wrong, right) 聚合频次 ───
    counter = Counter((c["wrong"], c["right"]) for c in all_cands)
    example_ctx = {}
    for c in all_cands:
        k = (c["wrong"], c["right"])
        if k not in example_ctx:
            example_ctx[k] = c["context_asr"]

    aggregated = [
        {"wrong": w, "right": r, "freq": f, "context_asr": example_ctx[(w, r)]}
        for (w, r), f in counter.most_common()
    ]
    print(f"[mine] 去重后 {len(aggregated)} 条独立候选")
    if len(aggregated) > args.max_show:
        print(f"  (按频次降序保留前 {args.max_show} 条)")
        aggregated = aggregated[:args.max_show]

    # ─── 加载已有词典 ───
    data = load_corrections(args.corrections)
    existing = {(r["wrong"], r.get("right", "")) for r in data.get("rules", [])}
    print(f"[load] 词典已有 {len(existing)} 条规则")

    # ─── 自动接受高频 ───
    auto_added = []
    if args.auto_min_freq > 0:
        auto_added = [
            {"wrong": c["wrong"], "right": c["right"]}
            for c in aggregated
            if c["freq"] >= args.auto_min_freq
            and (c["wrong"], c["right"]) not in existing
        ]
        if auto_added:
            print(f"\n[auto] 自动接受 {len(auto_added)} 条 (频次 ≥ {args.auto_min_freq})")
            for r in auto_added[:30]:
                f = next(c["freq"] for c in aggregated
                         if c["wrong"] == r["wrong"] and c["right"] == r["right"])
                print(f"    [{f}x] {r['wrong']} → {r['right']}")
            if len(auto_added) > 30:
                print(f"    ... 还有 {len(auto_added) - 30} 条")
            for r in auto_added:
                existing.add((r["wrong"], r["right"]))

    # ─── 剩余交互式 ───
    remaining = [
        c for c in aggregated
        if c["freq"] < max(1, args.auto_min_freq)
           and (c["wrong"], c["right"]) not in existing
    ]
    inter_added = []
    if remaining and not args.dry_run:
        inter_added = review_interactive(remaining, existing)

    # ─── 写回 ───
    total_new = auto_added + inter_added
    if args.dry_run:
        print(f"\n[dry-run] 会新增 {len(total_new)} 条 (未写入)")
        for r in total_new[:30]:
            print(f"  {r['wrong']} → {r['right']}")
        return

    if total_new:
        added, skipped = append_rules(data, total_new)
        print(f"\n[done] 新增 {added} 条, 跳过 {skipped} 条")
        save_corrections(args.corrections, data)
    else:
        print("\n[done] 没有新增")


if __name__ == "__main__":
    main()
