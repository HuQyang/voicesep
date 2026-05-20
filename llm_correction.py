"""
LLM 后处理 (可选) - 接 ASR 输出, 做"可读性清洁" + "章节速览"

适合场景:
  1. 看 ASR 输出经 LLM 整理后能多干净 (评估上游 ASR 的下限)
  2. 临时生成可读纪要 (在最终下游 LLM 接入前)
  3. 不归 ASR 管线, 完全可选

输入支持:
  - main_simple.py / main_firered.py 的 JSON 输出 (含 start/end/text [/speaker])
  - 纯 TXT (按空行分段)

输出:
  - 纯净的 TXT (带时间戳, 可选章节速览)
  - 结构化 JSON (清洁后的段落 + 章节)

支持任何 OpenAI 兼容 API:
  本地: Ollama / vLLM / llama.cpp server
  云端: DeepSeek / 通义千问 / Moonshot / OpenAI

环境变量 (--xxx 参数会覆盖):
  LLM_API_KEY    API key (本地服务也要填, 随便给个非空字符串)
  LLM_BASE_URL   API endpoint
  LLM_MODEL      模型名

用法示例:
  # 1. 本地 Ollama
  LLM_BASE_URL=http://localhost:11434/v1 LLM_API_KEY=ollama LLM_MODEL=qwen2.5:7b \\
      python llm_correction.py --input result/simple.json --output result/cleaned.txt

  # 2. DeepSeek 云端
  LLM_BASE_URL=https://api.deepseek.com LLM_API_KEY=sk-xxx LLM_MODEL=deepseek-chat \\
      python llm_correction.py --input result/simple.json --output result/cleaned.txt --chapters

  # 3. 通义千问云端
  LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \\
      LLM_API_KEY=sk-xxx LLM_MODEL=qwen-plus \\
      python llm_correction.py --input result/simple.json --output result/cleaned.txt --chapters
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


# ─────────── Prompts ───────────

SYSTEM_PROMPT_CORRECT = """你是会议记录纠错助手. 输入是会议语音识别 (ASR) 转出的逐字稿, 可能有以下问题:
1. 同音字/近音字错误 (如"预值"应为"阈值", "于杭"应为"余杭")
2. 短串机械重复幻觉 (如"这个里面"连续 20 次)
3. 标点不准, 断句奇怪
4. 大量口头禅 ("嗯", "啊", "对对对", "首先首先这里呢")

请严格遵守:
- 【不改变任何事实信息】, 数字 / 人名 / 地名 / 业务术语原样保留
- 【不添加任何新内容】, 不补充逻辑, 不编造细节, 不展开缩写
- 修正明显错别字与同音字 (上下文清晰才改)
- 去除明显机械重复 (同一短串连续 5+ 次), 保留 2-3 次的口语强调
- 整理标点使句子完整可读, 但不强行书面化
- 保留输入的 §N 段标记, 段落数与顺序不变

输入示例:
§1 [00:00] 是是,我把那个标准的再给他。
§2 [00:15] 那个那个那个这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面这个里面。

输出:
§1 [00:00] 是的,我把那个标准的再给他。
§2 [00:15] 那个这个里面。

直接输出修正后的文本, 不要加任何解释/前言/总结."""


SYSTEM_PROMPT_CHAPTERS = """你是会议纪要助手. 输入是带时间戳的会议转写. 按主题切分章节, 每章 3-10 分钟.

严格输出格式 (每行一章, 不要任何前缀/解释):
HH:MM 章节标题

要求:
- 章节标题 6-12 字, 概括该时段的核心讨论点, 用名词短语
- 第一章必须从 00:00 开始
- 章节标题不要包含"讨论""探讨"等动词模板词

举例:
00:00 数据口径核对
05:30 电池续航分析
12:15 车辆画像维度设计
20:40 主数据规范化
"""


# ─────────── 输入读取 ───────────

def read_input(path: str) -> List[Dict]:
    """
    读 JSON 或 TXT. 返回 paragraphs list, 每项 dict 至少有 'text', 可选 'start'/'end'/'speaker'.
    """
    if not os.path.exists(path):
        sys.exit(f"输入文件不存在: {path}")

    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            sys.exit(f"JSON 顶层应为 list (simple/firered 输出格式)")
        return data

    # TXT: 按空行分段
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    paragraphs = []
    cur_lines = []
    cur_meta = {}
    ts_pat = re.compile(r"^\[(\d+):(\d+)(?:-(\d+):(\d+))?\]\s*(?:(.+?)\s*:\s*)?$")
    for line in text.split("\n"):
        line = line.rstrip()
        if not line:
            if cur_lines:
                paragraphs.append({**cur_meta, "text": " ".join(cur_lines)})
                cur_lines, cur_meta = [], {}
            continue
        m = ts_pat.match(line)
        if m:
            if cur_lines:
                paragraphs.append({**cur_meta, "text": " ".join(cur_lines)})
                cur_lines = []
            sm, ss = int(m.group(1)), int(m.group(2))
            cur_meta = {"start": sm * 60 + ss}
            if m.group(3):
                em, es = int(m.group(3)), int(m.group(4))
                cur_meta["end"] = em * 60 + es
            if m.group(5):
                cur_meta["speaker"] = m.group(5).strip()
        else:
            cur_lines.append(line)
    if cur_lines:
        paragraphs.append({**cur_meta, "text": " ".join(cur_lines)})
    return [p for p in paragraphs if p.get("text", "").strip()]


# ─────────── 段落 → LLM 输入格式 ───────────

def _fmt_ts(start) -> str:
    """秒或毫秒 → MM:SS"""
    if start is None:
        return ""
    s = float(start)
    if s > 10000:   # 看着像 ms
        s = s / 1000
    m, sec = divmod(int(s), 60)
    return f"{m:02d}:{sec:02d}"


def format_batch_input(paragraphs: List[Dict]) -> str:
    """带 §N 标记, 方便 LLM 保段"""
    lines = []
    for i, p in enumerate(paragraphs):
        prefix = ""
        if p.get("start") is not None:
            prefix += f"[{_fmt_ts(p['start'])}] "
        if p.get("speaker"):
            prefix += f"{p['speaker']}: "
        lines.append(f"§{i + 1} {prefix}{p['text']}")
    return "\n\n".join(lines)


def parse_batch_output(output: str, n_paragraphs: int) -> List[str]:
    """从 LLM 输出还原段落 (按 §N 切)"""
    parts_raw = re.split(r"§\s*(\d+)\s*", output)
    # split 后: [前置文本, '1', §1正文, '2', §2正文, ...]
    sections = {}
    for i in range(1, len(parts_raw), 2):
        idx = int(parts_raw[i])
        body = parts_raw[i + 1] if i + 1 < len(parts_raw) else ""
        # 去掉行首的 [MM:SS] / 说话人前缀
        body = re.sub(r"^\s*\[\d+:\d+\]\s*", "", body.strip())
        body = re.sub(r"^[^\[]+?:\s*", "", body) if ":" in body[:30] else body
        sections[idx] = body.strip()

    out = []
    for i in range(1, n_paragraphs + 1):
        out.append(sections.get(i, ""))
    return out


# ─────────── 批处理 ───────────

def batch_by_chars(paragraphs: List[Dict], batch_chars: int = 2500) -> List[List[Dict]]:
    """按字数攒批次. 单段超长则单独成批."""
    batches = []
    cur, cur_len = [], 0
    for p in paragraphs:
        t_len = len(p.get("text", ""))
        if cur and cur_len + t_len > batch_chars:
            batches.append(cur)
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += t_len
    if cur:
        batches.append(cur)
    return batches


def llm_call(client, model: str, system: str, user: str,
             max_tokens: int = 4000, temperature: float = 0.1,
             retries: int = 2) -> Optional[str]:
    """带重试的 LLM 调用"""
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
        description="LLM 后处理 ASR 输出 (纠错 + 可选章节速览)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input", required=True,
                    help="ASR 输出: .json (simple/firered) 或 .txt")
    ap.add_argument("--output", required=True, help="清洁后的 .txt")
    ap.add_argument("--output-json", default=None,
                    help="可选: 同时输出结构化 JSON (含章节)")
    ap.add_argument("--chapters", action="store_true",
                    help="同时生成章节速览 (会多调一次 LLM)")
    ap.add_argument("--api-key", default=None, help="覆盖 LLM_API_KEY")
    ap.add_argument("--base-url", default=None, help="覆盖 LLM_BASE_URL")
    ap.add_argument("--model", default=None, help="覆盖 LLM_MODEL")
    ap.add_argument("--batch-chars", type=int, default=2500,
                    help="一批送多少字 (越大越快, 但可能超 context)")
    ap.add_argument("--no-correct", action="store_true",
                    help="跳过纠错, 只生成章节 (要求同时给 --chapters)")
    args = ap.parse_args()

    if OpenAI is None:
        sys.exit("缺 openai 库: pip install openai")

    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
    base_url = args.base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model = args.model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        sys.exit("缺 API key. 设置环境变量 LLM_API_KEY 或传 --api-key")

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"[llm] {model} @ {base_url}")

    paragraphs = read_input(args.input)
    total_chars = sum(len(p.get("text", "")) for p in paragraphs)
    print(f"[load] {len(paragraphs)} 段, 共 {total_chars} 字")

    # ───── 1. 纠错 ─────
    cleaned = list(paragraphs)
    if not args.no_correct:
        batches = batch_by_chars(paragraphs, batch_chars=args.batch_chars)
        print(f"[correct] 分 {len(batches)} 批 (每批 ≤ {args.batch_chars} 字)")
        cleaned = []
        for i, batch in enumerate(batches, 1):
            inp = format_batch_input(batch)
            print(f"  [batch {i}/{len(batches)}] {len(batch)} 段, {len(inp)} 字 ...", end=" ", flush=True)
            t0 = time.time()
            result = llm_call(client, model, SYSTEM_PROMPT_CORRECT, inp)
            dt = time.time() - t0
            if result is None:
                print(f"FAIL ({dt:.1f}s), 保留原文")
                cleaned.extend({**p} for p in batch)
                continue
            texts = parse_batch_output(result, len(batch))
            ok = sum(1 for t in texts if t)
            print(f"OK {ok}/{len(batch)} ({dt:.1f}s)")
            for p, t in zip(batch, texts):
                cleaned.append({**p, "text": t.strip() if t.strip() else p["text"]})
        print(f"[correct] 完成, 共 {sum(len(p['text']) for p in cleaned)} 字 (原 {total_chars})")

    # ───── 2. 章节 (可选) ─────
    chapters_text = ""
    if args.chapters:
        print("[chapters] 生成章节速览 ...")
        # 用清洁后的段落, 只取时间戳 + 头 80 字, 控制输入长度
        snippets = []
        for p in cleaned:
            ts = _fmt_ts(p.get("start"))
            snippet = (p["text"] or "")[:80]
            snippets.append(f"[{ts}] {snippet}")
        full = "\n".join(snippets)
        t0 = time.time()
        chapters_text = llm_call(
            client, model, SYSTEM_PROMPT_CHAPTERS, full,
            max_tokens=600, temperature=0.2,
        )
        dt = time.time() - t0
        if chapters_text:
            print(f"[chapters] 完成 ({dt:.1f}s)")
        else:
            print(f"[chapters] FAIL ({dt:.1f}s)")
            chapters_text = ""

    # ───── 3. 输出 TXT ─────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(f"# {os.path.basename(args.input)}\n\n")
        if chapters_text:
            f.write("## 章节速览\n\n")
            f.write(chapters_text.strip() + "\n\n")
            f.write("---\n\n## 全文转写\n\n")
        for p in cleaned:
            prefix = ""
            if p.get("start") is not None:
                prefix += f"[{_fmt_ts(p['start'])}] "
            if p.get("speaker"):
                prefix += f"{p['speaker']}: "
            f.write(f"{prefix}{p['text']}\n\n")
    print(f"[save] {args.output}")

    # ───── 4. 输出 JSON (可选) ─────
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({
                "source": args.input,
                "model": model,
                "chapters_raw": chapters_text,
                "paragraphs": cleaned,
            }, f, ensure_ascii=False, indent=2)
        print(f"[save] {args.output_json}")


if __name__ == "__main__":
    main()
