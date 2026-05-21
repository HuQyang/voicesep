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

_BASE_RULES = """你是会议记录纠错助手. 输入是会议语音识别 (ASR) 转出的逐字稿, 典型问题:
- 同音字/近音字 ("预值"→"阈值", "于杭"→"余杭", "驾驶语言"→"驾驶员")
- 短串机械重复 ("这个里面"连续 20 次)
- 大量重复词 / 结巴 / 填充词 ("我我我", "就是就是", "嗯嗯嗯", "对对对")
- ITN 错乱 ("5010000人"应为"501 万人", "11沟通"应为"一对一沟通")
- 标点不准 / 断句奇怪

铁律 (违反等于失败):
- 不改变事实: 数字 / 人名 / 地名 / 业务术语 / 时间, 全部原样保留
- 不添加内容: 不补充逻辑, 不编造细节, 不展开缩写
- 保留 §N 段标记, 段落数与顺序不变 (一段都不能漏)
- 段内的 [MM:SS] 时间戳和 "说话人:" 前缀也要保留

直接输出修正后的文本, 不要加任何解释/前言/总结. 段间空一行."""


_POLISH_LIGHT = """
清洁强度: 轻度 (保留口语风格, 仅修明显错误)
- 同音字 / 错别字: 修
- 连续重复 ≥ 4 次的短串: 削减到 1 次
- 单字/双字重复 ≥ 3 次 (我我我我, 就是就是就是): 削减到 1 次
- 保留 "嗯" "啊" 等语气词
- 保留 "对对对" "好好" 这种 2-3 次的口语强调
"""

_POLISH_MEDIUM = """
清洁强度: 中度 (默认推荐, 狠去口语噪声 + 保留原句结构)

【要做的】
1. 同音字 / 错别字 / ITN 错乱: 修
2. 连续重复, 一律削到 1 次
   - 单字重复: "我我我" → "我", "就就就" → "就"
   - 词组重复: "就是就是" → "就是", "这个这个" → "这个", "那那那那" → "那"
   - 长串重复: "这个里面"×20 → "这个里面"
3. 语气词全删: 嗯、啊、呢、吧、哦、唉、诶 (句末"是吧"保留)
4. 无意义口头禅删: "就是说", "怎么说呢", "你比如说", "反正就是说"
5. 标点修正使句子完整

【不要做的】
- 不要合并句子. 原来是几句还是几句, 哪怕一句话只有 5 个字
- 不要改写句式. 原话怎么说就怎么说, 别替换成更"书面"的版本
- 不要删代词. "那个"/"这个"/"什么"这种, 不重复就保留 (它们承载语义)
- 不要丢任何事实信息. 数字/人名/地名/术语/数据原样
- 不要补全省略的逻辑. 原话跳跃就跳跃

【对比示范】

输入:
§1 我我我我觉得这个就是就是非常非常对的, 你这个呢, 嗯, 就是说很好, 然后然后那个那个, 我们这个 5010000 人, 出现了, 出现了, 嗯, 社会舆情。

输出 (正确 - 保留结构, 清重复):
§1 我觉得这个非常对, 你这个很好, 然后那个, 我们这 501 万人, 出现了, 社会舆情。

错误示范 1 (过度合并):
§1 我觉得这个非常对, 你这个很好。我们这 501 万人出现了社会舆情。  ← 错: 合并了句子

错误示范 2 (过度删代词):
§1 我觉得非常对, 你很好, 然后, 我们 501 万人, 出现了社会舆情。  ← 错: 删了"这个/那个"

错误示范 3 (改写):
§1 我认为这非常正确, 你这点也很好。然而我们这 501 万人引发了社会舆情。  ← 错: 改写句式
"""

_POLISH_HEAVY = """
清洁强度: 重度 (面向阅读, 接近书面)
- 同音字 / 错别字: 修
- 所有口语重复 / 结巴 / 填充词: 删干净
- 把零碎短句合并成完整长句
- 把口语化的"那个" "这个" "什么的" 删掉或替换
- 保持原意但允许小幅改写让句子通顺
- 仍然不可: 添加事实 / 改变数字 / 改变名词

输入示例:
§1 你看你看你看, 就是那个, 我们这个 5010000 人, 然后然后呢就是, 出现了, 出现了, 你懂的, 嗯, 就是社会舆情。
输出:
§1 我们 501 万人, 出现了社会舆情。
"""


def build_correct_prompt(polish_level: str = "medium") -> str:
    rules = {
        "light": _POLISH_LIGHT,
        "medium": _POLISH_MEDIUM,
        "heavy": _POLISH_HEAVY,
    }
    if polish_level not in rules:
        polish_level = "medium"
    return _BASE_RULES + "\n" + rules[polish_level]


# 保留旧名字以兼容
SYSTEM_PROMPT_CORRECT = build_correct_prompt("medium")


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
    # file_nm = "04.21公交数据要素比赛决赛培训"
    file_nm = "钱部长数据融合沟通"
    ap.add_argument("--input",default=f"result/{file_nm}_firered.json", 
                    help="ASR 输出: .json (simple/firered) 或 .txt")
    ap.add_argument("--output", default=f"result/{file_nm}_simple_medium_32.txt", help="清洁后的 .txt")
    ap.add_argument("--output-json", default=f"result/{file_nm}_simple_medium_32.json",
                    help="可选: 同时输出结构化 JSON (含章节)")
    ap.add_argument("--chapters", action="store_true",
                    help="同时生成章节速览 (会多调一次 LLM)")
    ap.add_argument("--api-key", default="ollama", help="覆盖 LLM_API_KEY")
    ap.add_argument("--base-url", default="http://localhost:11434/v1", help="覆盖 LLM_BASE_URL")
    ap.add_argument("--model", default="qwen2.5:32b", help="覆盖 LLM_MODEL")
    ap.add_argument("--batch-chars", type=int, default=2500,
                    help="一批送多少字 (越大越快, 但可能超 context)")
    ap.add_argument("--no-correct", action="store_true",
                    help="跳过纠错, 只生成章节 (要求同时给 --chapters)")
    ap.add_argument("--polish-level", choices=["light", "medium", "heavy"],
                    default="medium",
                    help="清洁强度: light=保留口语风格 / medium=平衡(默认) / heavy=面向阅读, 接近书面")
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
        prompt_correct = build_correct_prompt(args.polish_level)
        print(f"[correct] polish-level={args.polish_level}")
        batches = batch_by_chars(paragraphs, batch_chars=args.batch_chars)
        print(f"[correct] 分 {len(batches)} 批 (每批 ≤ {args.batch_chars} 字)")
        cleaned = []
        for i, batch in enumerate(batches, 1):
            inp = format_batch_input(batch)
            print(f"  [batch {i}/{len(batches)}] {len(batch)} 段, {len(inp)} 字 ...", end=" ", flush=True)
            t0 = time.time()
            result = llm_call(client, model, prompt_correct, inp)
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
