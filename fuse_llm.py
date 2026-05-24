"""
LLM 语义融合两份 ASR 输出.

跟 fuse_diar_asr.py 的区别:
  fuse_diar_asr.py:  规则融合, 按时间比例切文字, 完全信任 ASR 来源的内容
  fuse_llm.py:       语义融合, LLM 看 A 和 B 同一时间窗内的两段文字, 输出"信息最完整 + 最通顺"的版本

输入:
  --anchor: 时间结构来源 (通常 main_pipeline / para, 时间戳 + 说话人准)
  --alt:    替代文本来源 (通常 main_firered, 文字流畅但偶尔漏内容)

逻辑:
  1. 以 anchor 的每个段为锚点
  2. 找 alt 中和锚点时间重叠的所有段, 拼成对照文本
  3. 批量送 LLM: "对每段, 融合 A 和 B 得到最优文字, 保留 §N 时间/说话人"
  4. 解析 LLM 输出, 还原成 anchor 结构

环境变量同 llm_correction.py:
  LLM_BASE_URL / LLM_API_KEY / LLM_MODEL

用法:
  LLM_BASE_URL=http://localhost:11434/v1 \\
  LLM_API_KEY=ollama \\
  LLM_MODEL=qwen2.5:14b \\
  python fuse_llm.py \\
      --anchor result/钱部长_para.json \\
      --alt    result/钱部长_firered.json \\
      --output result/钱部长_fused_llm.txt \\
      --output-json result/钱部长_fused_llm.json
"""
import argparse
import json
import os
import re
import sys
import time
from typing import List, Dict, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# ─────────── Prompt ───────────

SYSTEM_PROMPT_FUSE = """你是会议转写融合助手. 接收同一段会议的两份 ASR 输出 (A 和 B), 目标是融合出"信息最完整 + 文字最通顺"的最终版.

输入特点:
- A (anchor): 时间戳和说话人分离更准, 但文字偶有噪声 / 错别字 / 重复
- B (alt):    文字相对流畅, 但偶尔漏内容 / 说话人分得不细

任务:
对输入的每一段 (按 §N 标记), 融合 A 与 B 同时间窗的两段文字, 输出最佳版本.

铁律 (违反等于失败):
1. 时间戳和说话人 (§N 行的 [MM:SS-MM:SS] spk_X) 来自 A, **绝不修改**
2. 文字融合策略:
   - A 和 B 内容一致 → 选更通顺的版本
   - A 有的内容 B 漏了 → **保留 A 的内容** (核心目标是不丢失信息)
   - B 有的合理内容 A 漏了 → 并入
   - 错别字 / 同音字 → 选明显更对的; 不能判断时倾向 A
3. 数字 / 人名 / 地名 / 业务术语 → 选更准的; 不确定保留 A 的版本
4. 不添加任何新事实; 不补充逻辑; 不展开缩写
5. 适度清掉口语重复 ("我我我"→"我", "就是就是"→"就是"), 但保留 2-3 次的口语强调
6. 保留 §N 标记和段顺序, 一段都不能漏

输出格式 (严格遵守):
§N [时间戳] 说话人:
<融合后的文字>

不要加任何前言 / 总结 / 解释. 段间空一行."""


# ─────────── 读 ───────────

def load_paragraphs(path: str) -> List[Dict]:
    if not os.path.exists(path):
        sys.exit(f"找不到: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        sys.exit(f"{path} 顶层应为 list")
    out = []
    for p in data:
        if "start" not in p or "end" not in p:
            continue
        s, e = float(p["start"]), float(p["end"])
        if s > 100000:   # 毫秒 → 秒
            s /= 1000
            e /= 1000
        out.append({
            "start": s,
            "end": e,
            "speaker": p.get("speaker") or "spk_unknown",
            "text": (p.get("text") or "").strip(),
            "overlap": p.get("overlap", False),
        })
    return out


# ─────────── 对齐 ───────────

def find_overlapping(anchor_seg: Dict, others: List[Dict],
                      min_overlap_s: float = 0.5) -> List[Dict]:
    """找 alt 中与 anchor 时间重叠 ≥ min_overlap_s 的段"""
    s, e = anchor_seg["start"], anchor_seg["end"]
    hits = []
    for o in others:
        os_, oe = o["start"], o["end"]
        ov = max(0, min(e, oe) - max(s, os_))
        if ov >= min_overlap_s:
            hits.append({**o, "_ov": ov})
    hits.sort(key=lambda x: x["start"])
    return hits


# ─────────── 批处理 ───────────

def _fmt_ts(start: float, end: float) -> str:
    s_m, s_s = divmod(int(start), 60)
    e_m, e_s = divmod(int(end), 60)
    return f"{s_m:02d}:{s_s:02d}-{e_m:02d}:{e_s:02d}"


def format_batch(pairs: List[Dict]) -> str:
    """
    pairs: [{"idx": int, "anchor": Dict, "alt_text": str}]
    输出 LLM 输入文本.
    """
    blocks = []
    for p in pairs:
        a = p["anchor"]
        ts = _fmt_ts(a["start"], a["end"])
        spk = a["speaker"]
        block = [
            f"§{p['idx']} [{ts}] {spk}:",
            f"  [A] {a['text']}",
            f"  [B] {p['alt_text'] or '<空>'}",
        ]
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


def parse_fused_output(output: str, n_pairs: int, anchors: List[Dict]) -> List[str]:
    """从 LLM 输出抽 §N 对应的融合文字"""
    parts = re.split(r"§\s*(\d+)\s*", output)
    # split 后: [前置, '1', §1, '2', §2, ...]
    sections = {}
    for i in range(1, len(parts), 2):
        idx = int(parts[i])
        body = parts[i + 1] if i + 1 < len(parts) else ""
        # 去掉首行的 [时间戳] 说话人: 前缀 (LLM 必然会保留, 我们要的是正文)
        body = body.strip()
        # 第一行通常是 [00:00-02:18] spk_2:, 跳过它
        lines = body.split("\n", 1)
        if len(lines) > 1 and re.match(r"\s*\[\d+:\d+", lines[0]):
            body = lines[1].strip()
        elif re.match(r"\s*\[\d+:\d+", lines[0]):
            # 整段就一行且是时间戳, 没正文
            body = ""
        sections[idx] = body

    out = []
    for i in range(1, n_pairs + 1):
        out.append(sections.get(i, anchors[i - 1]["text"]))   # 缺的回落 anchor
    return out


def batch_by_chars(pairs: List[Dict], batch_chars: int = 2500) -> List[List[Dict]]:
    """按字数攒批次. 一对的 A+B 算字数."""
    batches = []
    cur, cur_len = [], 0
    for p in pairs:
        size = len(p["anchor"]["text"]) + len(p["alt_text"])
        if cur and cur_len + size > batch_chars:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += size
    if cur:
        batches.append(cur)
    return batches


def llm_call(client, model: str, system: str, user: str,
              max_tokens: int = 6000, temperature: float = 0.1,
              retries: int = 2) -> Optional[str]:
    last_err = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(1 + attempt * 2)
                continue
    print(f"  [llm-fail] {last_err}", file=sys.stderr)
    return None


# ─────────── 主流程 ───────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    file_nm = "钱部长数据融合沟通"
    ap.add_argument("--anchor", default=f"result/{file_nm}_para2.json",
                    help="时间结构来源 JSON (通常 main_pipeline.py 输出)")
    ap.add_argument("--alt", default=f"result/{file_nm}_firered2.json",
                    help="替代文本来源 JSON (通常 main_firered.py 输出)")
    ap.add_argument("--output", default=f"result/{file_nm}_fused.txt", help="融合后 .txt 路径")
    ap.add_argument("--output-json", default=f"result/{file_nm}_fused.json", help="融合后 .json (可选)")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--batch-chars", type=int, default=2500)
    ap.add_argument("--skip-overlap", action="store_true",
                    help="跳过 anchor 中 overlap=True 的段 (不送 LLM, 直接保留 anchor 原文)")
    ap.add_argument("--min-anchor-chars", type=int, default=3,
                    help="anchor 文字少于此字数的段不送 LLM (太短没必要融合)")
    args = ap.parse_args()

    if OpenAI is None:
        sys.exit("需要 openai 库: pip install openai")

    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
    base_url = args.base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = args.model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        sys.exit("缺 API key. 设置环境变量 LLM_API_KEY 或传 --api-key")
    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"[llm] {model} @ {base_url}")

    anchors = load_paragraphs(args.anchor)
    alts = load_paragraphs(args.alt)
    print(f"[load] anchor: {len(anchors)} 段  alt: {len(alts)} 段")

    # 时长一致性
    a_dur = max((p["end"] for p in anchors), default=0)
    b_dur = max((p["end"] for p in alts), default=0)
    if abs(a_dur - b_dur) > 60:
        print(f"[!] 警告: 两份时长差 {abs(a_dur - b_dur):.0f}s, 可能不是同一段音频")

    # 构建 pairs: 每个 anchor 段 + 它时间窗内的 alt 拼起来
    pairs = []
    skipped = []
    for i, a in enumerate(anchors, 1):
        if args.skip_overlap and a.get("overlap"):
            skipped.append((i, a, "overlap"))
            continue
        if len(a["text"]) < args.min_anchor_chars:
            skipped.append((i, a, "short"))
            continue
        ovs = find_overlapping(a, alts, min_overlap_s=0.5)
        alt_text = " ".join(o["text"] for o in ovs).strip()
        pairs.append({
            "idx": i,
            "anchor": a,
            "alt_text": alt_text,
        })

    if not pairs:
        print("[!] 没有可融合的段, 直接输出 anchor")
        fused_texts = [a["text"] for a in anchors]
    else:
        print(f"[align] {len(pairs)} 段送 LLM, {len(skipped)} 段跳过 (短段/overlap)")

        batches = batch_by_chars(pairs, batch_chars=args.batch_chars)
        print(f"[fuse] 分 {len(batches)} 批")

        fused_lookup = {}   # idx → fused_text
        for bi, batch in enumerate(batches, 1):
            inp = format_batch(batch)
            n_chars = len(inp)
            print(f"  [batch {bi}/{len(batches)}] {len(batch)} 段 {n_chars} 字 ...", end=" ", flush=True)
            t0 = time.time()
            result = llm_call(client, model, SYSTEM_PROMPT_FUSE, inp)
            dt = time.time() - t0
            if result is None:
                print(f"FAIL ({dt:.1f}s), 保留 anchor")
                for p in batch:
                    fused_lookup[p["idx"]] = p["anchor"]["text"]
                continue
            batch_anchors = [p["anchor"] for p in batch]
            texts = parse_fused_output(result, len(batch), batch_anchors)
            ok = sum(1 for t in texts if t)
            print(f"OK {ok}/{len(batch)} ({dt:.1f}s)")
            for p, t in zip(batch, texts):
                fused_lookup[p["idx"]] = (t.strip() if t.strip() else p["anchor"]["text"])

        # 重建完整 anchors 顺序 (含 skipped)
        fused_texts = []
        for i, a in enumerate(anchors, 1):
            if i in fused_lookup:
                fused_texts.append(fused_lookup[i])
            else:
                fused_texts.append(a["text"])   # skipped 段保留 anchor 原文

    # ─── 写出 ───
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# {os.path.basename(args.anchor)} ⊕ {os.path.basename(args.alt)}\n")
        f.write(f"# LLM-fused via {model}\n\n")
        for a, t in zip(anchors, fused_texts):
            tag = " [可能重叠]" if a.get("overlap") else ""
            f.write(f"{a['speaker']} - {_fmt_ts(a['start'], a['end'])}{tag}\n")
            f.write(f"{t}\n\n")
    print(f"[save] {args.output}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump([
                {
                    "start": round(a["start"], 2),
                    "end": round(a["end"], 2),
                    "speaker": a["speaker"],
                    "overlap": a.get("overlap", False),
                    "text": t,
                }
                for a, t in zip(anchors, fused_texts)
            ], f, ensure_ascii=False, indent=2)
        print(f"[save] {args.output_json}")


if __name__ == "__main__":
    main()
