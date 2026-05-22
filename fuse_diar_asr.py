"""
混合脚本: 用 A 的 diarization 时间线 + B 的 ASR 文本

适用场景: 两个 ASR 各对一半
  - main_pipeline.py (Paraformer + cam++) : diarization 准, 但远场 ASR 塌房
  - main_firered.py (FireRedASR-AED-L)    : ASR 文本准, 但 diarization 崩 (全 spk_0)
  → 拼起来: para 的 spk 标签 + firered 的文本

逻辑:
  1. 读两份 JSON (main_pipeline / main_firered 输出格式: list of {start, end, speaker, text, overlap})
  2. 对 ASR 来源的每个 paragraph:
     a. 找 diar 在同时段的所有段
     b. 相邻同 spk 合并成"spk 区间组"
     c. 单 spk → ASR 文本整段贴该 spk
     d. 多 spk → 按时间比例切 ASR 文本, 优先在标点处下刀
  3. 合并相邻同 spk paragraphs (带 3 分钟 / 1500 字硬上限)

用法:
  python fuse_diar_asr.py \\
      --diar result/m1_para.json \\
      --asr  result/m1_firered.json \\
      --output result/m1_fused.txt \\
      --output-json result/m1_fused.json
"""
import argparse
import json
import os
import sys
from typing import List, Dict


# ─────────── 读 ───────────

def load_paragraphs(path: str) -> List[Dict]:
    """读 main_pipeline/main_firered 输出的 JSON. 标准化为秒."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"{path} 顶层应为 list (main_pipeline/firered 输出格式)")
    out = []
    for p in data:
        if "start" not in p or "end" not in p:
            continue
        start = float(p["start"])
        end = float(p["end"])
        # 兼容: 如果看起来是毫秒, 转秒
        if start > 100000 or end > 100000:
            start /= 1000
            end /= 1000
        out.append({
            "start": start,
            "end": end,
            "speaker": p.get("speaker") or "spk_unknown",
            "text": (p.get("text") or "").strip(),
            "overlap": p.get("overlap", False),
        })
    return out


# ─────────── 切点工具 ───────────

_SENT_PUNC = "。！？!?"
_SOFT_PUNC = "，,；;、"


def _find_nearest_punc(text: str, target: int, window: int = 15) -> int:
    """在 target 附近 ±window 字内找最近标点, 优先句末标点. 返回切点位置."""
    n = len(text)
    for offset in range(window + 1):
        for p in (target + offset, target - offset):
            if 0 < p <= n and text[p - 1] in _SENT_PUNC:
                return p
    for offset in range(window + 1):
        for p in (target + offset, target - offset):
            if 0 < p <= n and text[p - 1] in _SOFT_PUNC:
                return p
    return min(max(target, 0), n)


# ─────────── 融合 ───────────

def fuse(diar_ps: List[Dict], asr_ps: List[Dict]) -> List[Dict]:
    """
    用 diar 的 spk 时间线 + asr 的文本, 输出新的 paragraphs 列表.
    """
    out = []
    for ap in asr_ps:
        a_start, a_end, a_text = ap["start"], ap["end"], ap["text"]
        if not a_text:
            continue

        # 1. 找 diar 跟 [a_start, a_end] 有交集的段
        overlaps = []
        for dp in diar_ps:
            ov_start = max(dp["start"], a_start)
            ov_end = min(dp["end"], a_end)
            if ov_end > ov_start:
                overlaps.append({
                    "start": ov_start,
                    "end": ov_end,
                    "speaker": dp["speaker"],
                    "overlap": dp.get("overlap", False),
                })

        if not overlaps:
            out.append({
                "start": a_start, "end": a_end,
                "speaker": ap.get("speaker") or "spk_unknown",
                "text": a_text,
                "overlap": ap.get("overlap", False),
            })
            continue

        # 2. 排序 + 合并相邻同 spk
        overlaps.sort(key=lambda x: x["start"])
        groups = [dict(overlaps[0])]
        for o in overlaps[1:]:
            if o["speaker"] == groups[-1]["speaker"]:
                groups[-1]["end"] = o["end"]
                groups[-1]["overlap"] = groups[-1]["overlap"] or o["overlap"]
            else:
                groups.append(dict(o))

        # 3. 单 spk → 整段归之
        if len(groups) == 1:
            out.append({
                "start": a_start, "end": a_end,
                "speaker": groups[0]["speaker"],
                "text": a_text,
                "overlap": groups[0]["overlap"] or ap.get("overlap", False),
            })
            continue

        # 4. 多 spk → 按时间比例切文本
        total_dur = sum(g["end"] - g["start"] for g in groups)
        text_len = len(a_text)
        cursor = 0
        for i, g in enumerate(groups):
            if i == len(groups) - 1:
                piece = a_text[cursor:].strip()
            else:
                ratio = (g["end"] - g["start"]) / total_dur if total_dur > 0 else 1.0
                target = cursor + int(text_len * ratio)
                cut_pos = _find_nearest_punc(a_text, target, window=15)
                cut_pos = max(cut_pos, cursor + 1)
                cut_pos = min(cut_pos, text_len)
                piece = a_text[cursor:cut_pos].strip()
                cursor = cut_pos
            if piece:
                out.append({
                    "start": g["start"], "end": g["end"],
                    "speaker": g["speaker"],
                    "text": piece,
                    "overlap": g["overlap"],
                })

    return out


# ─────────── 后处理: 合并相邻同 spk ───────────

def merge_consecutive(paragraphs: List[Dict],
                       max_gap_s: float = 5.0,
                       max_dur_s: float = 180.0,
                       max_chars: int = 1500) -> List[Dict]:
    """相邻同 spk 合并 (跟 main_pipeline 的 merge_consecutive_same_spk 一致)"""
    if not paragraphs:
        return []
    out = [dict(paragraphs[0])]
    for p in paragraphs[1:]:
        last = out[-1]
        same_spk = last["speaker"] == p["speaker"]
        neither_ov = not last.get("overlap", False) and not p.get("overlap", False)
        gap = p["start"] - last["end"]
        merged_dur = p["end"] - last["start"]
        merged_chars = len(last["text"]) + len(p["text"])
        if (same_spk and neither_ov
                and gap <= max_gap_s
                and merged_dur <= max_dur_s
                and merged_chars <= max_chars):
            last["end"] = p["end"]
            last["text"] += p["text"]
        else:
            out.append(dict(p))
    return out


# ─────────── 输出格式 ───────────

def _fmt_ts(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


def to_readable(paragraphs: List[Dict], source_basename: str = "") -> str:
    lines = []
    if source_basename:
        lines.append(source_basename)
        lines.append("")
    for p in paragraphs:
        tag = " [可能重叠]" if p.get("overlap") else ""
        lines.append(f"{p['speaker']} - {_fmt_ts(p['start'])}-{_fmt_ts(p['end'])}{tag}")
        lines.append(p["text"])
        lines.append("")
    return "\n".join(lines)


# ─────────── 主流程 ───────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    ap.add_argument("--diar", required=True,
                    help="diarization 时间线来源 JSON (通常 main_pipeline.py 输出)")
    ap.add_argument("--asr", required=True,
                    help="ASR 文本来源 JSON (通常 main_firered.py 输出)")
    ap.add_argument("--output", required=True, help="输出 .txt 路径")
    ap.add_argument("--output-json", default=None, help="可选: 输出结构化 JSON")
    ap.add_argument("--source-name", default="",
                    help="txt 顶部展示的源文件名 (默认从 --diar 推断)")
    ap.add_argument("--max-dur-s", type=float, default=180.0,
                    help="合并后段最大时长(s), 默认 180")
    ap.add_argument("--max-chars", type=int, default=1500,
                    help="合并后段最大字数, 默认 1500")
    ap.add_argument("--max-gap-s", type=float, default=5.0,
                    help="合并间隔上限(s), 默认 5")
    args = ap.parse_args()

    diar_ps = load_paragraphs(args.diar)
    asr_ps = load_paragraphs(args.asr)
    print(f"[load] diar: {len(diar_ps)} 段  ({args.diar})")
    print(f"[load] asr:  {len(asr_ps)} 段  ({args.asr})")

    # 简单一致性检查
    diar_dur = max((p["end"] for p in diar_ps), default=0)
    asr_dur = max((p["end"] for p in asr_ps), default=0)
    if abs(diar_dur - asr_dur) > 30:
        print(f"[!] 警告: 两份时长差距 {abs(diar_dur - asr_dur):.0f}s "
              f"(diar={diar_dur:.0f}s, asr={asr_dur:.0f}s), 可能不是同一段音频")

    fused = fuse(diar_ps, asr_ps)
    print(f"[fuse] 染色后 {len(fused)} 段")

    merged = merge_consecutive(
        fused,
        max_gap_s=args.max_gap_s,
        max_dur_s=args.max_dur_s,
        max_chars=args.max_chars,
    )
    print(f"[merge] 合并后 {len(merged)} 段")

    # 统计每个 spk 时长
    from collections import Counter
    spk_dur = Counter()
    for p in merged:
        spk_dur[p["speaker"]] += (p["end"] - p["start"])
    print(f"[stats] " + " | ".join(
        f"{spk}: {dur:.0f}s" for spk, dur in spk_dur.most_common()
    ))

    # 输出
    src_name = args.source_name or os.path.basename(args.diar).rsplit(".", 1)[0]
    readable = to_readable(merged, source_basename=src_name)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(readable)
    print(f"[save] {args.output}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump([
                {
                    "start": round(p["start"], 2),
                    "end": round(p["end"], 2),
                    "speaker": p["speaker"],
                    "overlap": p.get("overlap", False),
                    "text": p["text"],
                }
                for p in merged
            ], f, ensure_ascii=False, indent=2)
        print(f"[save] {args.output_json}")


if __name__ == "__main__":
    main()
