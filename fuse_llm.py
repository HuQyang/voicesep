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
  LLM_BASE_URL=http://localhost:11434/v1 \
  LLM_API_KEY=ollama \
  LLM_MODEL=qwen2.5:14b \
  python fuse_llm.py \
      --anchor result/钱部长_para.json \
      --alt    result/钱部长_firered.json \
      --output result/钱部长_fused_llm.txt \
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


# ─────────── Prompt (核心修改区) ───────────

SYSTEM_PROMPT_FUSE = """你是专业的会议转写文本融合专家。你将接收同一段会议的两份 ASR 输出 (A 和 B)，目标是融合出“信息最完整 + 语义最通顺 + 逻辑最严密”的最终版。

输入特点:
- A (anchor): 时间戳和说话人准，但文字可能有同音字错误、口语化或噪声。
- B (alt): 文字更流畅、上下文更准确，但偶尔会漏掉半句话或说话人切分不细。

你的任务是：对比每一段的 A 和 B，**结合整个批次的上下文语境**，输出该段的最佳版本。

【核心铁律】（违反会导致严重错误）：
1. **绝对禁止生硬拼接（缝合怪行为）**：当 A 和 B 对同一句话有不同的识别结果（如“多少种” vs “多久”），**必须结合前后文逻辑推理**选择最合理的一方！严禁将两者的差异词汇生硬拼凑在一起！
2. **全局上下文推理**：不要孤立地看一句话。如果后文提到了时间长度，那么前文的歧义就必须选择与时间相关的词汇；如果是专业术语，选择符合业务逻辑的一方。
3. **查漏补缺，不丢信息**：
   - A 和 B 意思一致 → 选表达更通顺的版本。
   - A 漏了 B 有的内容 → 并入 B 的内容。
   - B 漏了 A 有的内容 → 只要 A 的内容不是毫无意义的杂音，就必须保留。
   - 实在无法通过上下文判断对错的同音字，优先倾向 A 的版本。
4. **克制修改**：不添加原文不存在的新事实，不主动解释或展开缩写。适度清理严重的口语结巴（如“我我我”保留为“我”），但保留语气词。
5. **格式锚定**：时间戳和说话人（§N [MM:SS-MM:SS] spk_X）来自 A，**绝不允许有任何修改、遗漏或合并**。

输出格式 (严格遵守，一段都不能漏):
§N [时间戳] 说话人:
<融合后的文字>

不要加任何前言、总结或解释代码。段间空一行。"""


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
    """从 LLM 输出抽 §N 对应的融合文字 (增强了正则容错)"""
    # 匹配 § 符号，并允许后面跟着数字和可能存在的标点/换行
    parts = re.split(r"§\s*(\d+)[^\n]*\n", "\n" + output)
    sections = {}
    
    for i in range(1, len(parts), 2):
        if not parts[i].isdigit():
            continue
        idx = int(parts[i])
        body = parts[i + 1] if i + 1 < len(parts) else ""
        body = body.strip()
        
        # 很多时候大模型会保留 `[00:00-02:18] spk_2:` 作为正文第一行，需要清理
        lines = body.split("\n")
        clean_lines = []
        for line in lines:
            line = line.strip()
            # 如果这一行看起来像 "[12:34-12:56] spk_1:"，直接跳过
            if re.match(r"^\[\d{2}:\d{2}-\d{2}:\d{2}\]\s*spk_\w+:?", line):
                continue
            clean_lines.append(line)
            
        sections[idx] = " ".join(clean_lines).strip()

    out = []
    for i in range(1, n_pairs + 1):
        # 如果 LLM 漏掉了这一段或者解析为空，回退到 anchor
        result_text = sections.get(i, "")
        if not result_text:
            result_text = anchors[i - 1]["text"]
        out.append(result_text)
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
              max_tokens: int = 6000, temperature: float = 0.1,  # 保持低温，确保推理确定性
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
    ap.add_argument("--anchor", default=f"result/{file_nm}_seacopara_5.json",
                    help="时间结构来源 JSON (通常 main_pipeline.py 输出)")
    ap.add_argument("--alt", default=f"result/{file_nm}_fireredasr2.json",
                    help="替代文本来源 JSON (通常 main_firered.py 输出)")
    ap.add_argument("--output", default=f"result/{file_nm}_fused.txt", help="融合后 .txt 路径")
    ap.add_argument("--output-json", default=f"result/{file_nm}_fused.json", help="融合后 .json (可选)")
    ap.add_argument("--api-key", default="ollama")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen2.5:14b")
    # 增加一点 batch 长度，让它能看到更多的上下文来做推理
    ap.add_argument("--batch-chars", type=int, default=3000) 
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

    a_dur = max((p["end"] for p in anchors), default=0)
    b_dur = max((p["end"] for p in alts), default=0)
    if abs(a_dur - b_dur) > 60:
        print(f"[!] 警告: 两份时长差 {abs(a_dur - b_dur):.0f}s, 可能不是同一段音频")

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

        fused_lookup = {}   
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

        fused_texts = []
        for i, a in enumerate(anchors, 1):
            if i in fused_lookup:
                fused_texts.append(fused_lookup[i])
            else:
                fused_texts.append(a["text"])

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