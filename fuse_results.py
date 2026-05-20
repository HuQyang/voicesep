"""
融合 Paraformer 和 FireRedASR 两路转写结果.

策略 (字符级 edit-distance + 领域词词典):
  - 用 difflib 对每对时间对齐的段落做字符级 align
  - 不一致片段: 含已知领域词的那一边胜出
  - 都没领域词时: 默认 FireRedASR (流畅度优先)
  - 段落时间对齐: 按 start 时间最近匹配

用法:
  python fuse_results.py \
      --paraformer result_paraformer.json \
      --firered    result_firered.json \
      --output     result_fused.json \
      --output-txt result_fused.txt
"""
import argparse
import difflib
import json
import os
from typing import List, Dict


def _fmt_time(ms):
    s = int(ms) // 1000
    return f"{s//60}:{s%60:02d}"


def load_paragraphs(path: str) -> List[Dict]:
    """读取 main_pipeline 输出的 JSON. 容错: 时间字段可能是 ms 或 s"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for p in data:
        start = p["start"]
        end = p["end"]
        # 老版本输出是秒, 新版本可能是 ms; 统一到 ms
        if start < 100000:  # 大概率是秒
            start = int(start * 1000)
            end = int(end * 1000)
        out.append({
            "start": int(start),
            "end": int(end),
            "speaker": p.get("speaker", ""),
            "text": p.get("text", ""),
            "overlap": p.get("overlap", False),
        })
    return out


def load_domain_words(corrections_path: str = "corrections.json") -> set:
    """从 corrections.json 读 hotwords + rules 的 right 端, 组成领域词集合"""
    if not os.path.exists(corrections_path):
        return set()
    with open(corrections_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = set()
    for w in data.get("hotwords", []):
        if isinstance(w, str) and w:
            words.add(w)
    for r in data.get("rules", []):
        if "right" in r and r["right"]:
            words.add(r["right"])
    return words


def _has_domain(text: str, domain: set) -> bool:
    return any(w in text for w in domain)


def fuse_pair(text_a: str, text_b: str, domain: set, default_side: str = "b") -> str:
    """
    字符级融合 text_a (Paraformer) 与 text_b (FireRedASR).
    diff 片段决策:
      - 一边含领域词另一边不含 → 含的胜
      - 都含或都不含 → default_side (默认 'b' = FireRedASR)
    """
    sm = difflib.SequenceMatcher(None, text_a, text_b)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            parts.append(text_a[i1:i2])
            continue
        a = text_a[i1:i2]
        b = text_b[j1:j2]
        a_has = _has_domain(a, domain)
        b_has = _has_domain(b, domain)
        if a_has and not b_has:
            parts.append(a)
        elif b_has and not a_has:
            parts.append(b)
        else:
            parts.append(a if default_side == "a" else b)
    return "".join(parts)


def time_align(pa_list, fr_list, max_gap_ms: int = 2000):
    """
    时间对齐: 对 FireRedASR 每段, 找 Paraformer 中 start 距离最近的段.
    返回 [(fr_para, pa_para_or_None), ...]
    """
    pairs = []
    used_pa = set()
    for fr in fr_list:
        best = None
        best_dist = float("inf")
        for i, pa in enumerate(pa_list):
            if i in used_pa:
                continue
            # 必须有时间重叠或紧邻
            d = max(abs(fr["start"] - pa["start"]), abs(fr["end"] - pa["end"]))
            # 必须时间窗有重叠
            overlap = min(fr["end"], pa["end"]) - max(fr["start"], pa["start"])
            if overlap < 0:
                continue
            if d < best_dist:
                best_dist = d
                best = i
        if best is not None and best_dist <= max_gap_ms + (fr["end"] - fr["start"]):
            pairs.append((fr, pa_list[best]))
            used_pa.add(best)
        else:
            pairs.append((fr, None))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paraformer", required=True, help="main_pipeline.py 的 JSON 输出")
    ap.add_argument("--firered", required=True, help="main_firered.py 的 JSON 输出")
    ap.add_argument("--output", default=None, help="融合后 JSON")
    ap.add_argument("--output-txt", default=None, help="融合后 TXT (可读格式)")
    ap.add_argument("--corrections", default="corrections.json",
                    help="领域词来源 (hotwords + rules.right)")
    ap.add_argument("--default-side", choices=["paraformer", "firered"], default="firered",
                    help="diff 片段双方都没领域词时, 默认采用哪一方")
    args = ap.parse_args()

    pa = load_paragraphs(args.paraformer)
    fr = load_paragraphs(args.firered)
    domain = load_domain_words(args.corrections)
    print(f"[domain] 加载 {len(domain)} 个领域词")
    print(f"[input] paraformer {len(pa)} 段, firered {len(fr)} 段")

    pairs = time_align(pa, fr)
    n_fused = n_fr_only = 0
    default_side_short = "a" if args.default_side == "paraformer" else "b"

    fused = []
    for fr_p, pa_p in pairs:
        if pa_p is None:
            fused.append(fr_p)  # 没匹配到 Paraformer, 直接用 FireRedASR
            n_fr_only += 1
            continue
        text = fuse_pair(pa_p["text"], fr_p["text"], domain,
                         default_side=default_side_short)
        # 用 FireRedASR 的时间戳和 spk (基础选 FireRedASR)
        fused.append({
            "start": fr_p["start"],
            "end": fr_p["end"],
            "speaker": fr_p["speaker"],
            "overlap": fr_p["overlap"] or pa_p["overlap"],
            "text": text,
        })
        n_fused += 1

    print(f"[fuse] 融合 {n_fused} 段, 单边(fr) {n_fr_only} 段")

    # 输出
    def _header(p):
        tag = " [可能重叠]" if p.get("overlap") else ""
        return f"{p['speaker']} - {_fmt_time(p['start'])}{tag}"

    print(f"\n=== 融合结果 ({len(fused)} 段) ===")
    for p in fused[:5]:
        print(f"\n{_header(p)}\n{p['text']}")
    if len(fused) > 5:
        print(f"\n... 还有 {len(fused) - 5} 段")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([
                {"start": round(p["start"]/1000, 2),
                 "end": round(p["end"]/1000, 2),
                 "speaker": p["speaker"],
                 "overlap": p.get("overlap", False),
                 "text": p["text"]}
                for p in fused
            ], f, ensure_ascii=False, indent=2)
        print(f"\n[saved json] {args.output}")

    if args.output_txt:
        with open(args.output_txt, "w", encoding="utf-8") as f:
            for p in fused:
                f.write(f"{_header(p)}\n{p['text']}\n\n")
        print(f"[saved txt] {args.output_txt}")


if __name__ == "__main__":
    main()
