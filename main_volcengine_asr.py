"""
火山引擎「录音文件识别 - 大模型版」会议转写

文档: https://www.volcengine.com/docs/6561/1354869

特性 (火山官方提供, 不用本地 GPU):
  - 中文会议 ASR (大模型版, 准确率 > Whisper-large-v3 中文场景)
  - 内置 VAD / 标点 / ITN / 顺滑 (去口语化)
  - 内置 speaker diarization (enable_speaker_info=True)
  - 异步任务: submit → poll → result

凭证 (3 个, 控制台拿):
  export VOLC_APP_KEY=xxx          # App ID
  export VOLC_ACCESS_KEY=xxx       # Access Token
  # 音频要 HTTPS 公网 URL, 可用 TOS / OSS / 七牛 / 自建 nginx

用法:
  # 1. 把 mp3 传到任意能给 HTTPS URL 的地方 (TOS/OSS/...) 拿到 url
  # 2. 跑这个脚本
  python main_volcengine_asr.py \\
      --audio-url "https://your-bucket.tos-cn-beijing.volces.com/xxx.mp3" \\
      --output result/钱部长_volc.json \\
      --output-txt result/钱部长_volc.txt \\
      --num-spk 5

  # 已经知道任务 ID, 直接拉结果 (任务最长保留 24h)
  python main_volcengine_asr.py --query-only --task-id xxx-xxx-xxx

输出格式: 与 result/*_seacopara_5.json 完全兼容, 可直接喂 fuse_llm.py 当 anchor.
"""
import argparse
import json
import os
import sys
import time
import uuid
from typing import Dict, List, Optional

import requests


SUBMIT_URL = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
QUERY_URL  = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
RESOURCE_ID = "volc.bigasr.auc"   # 大模型版固定值


# ─────────── 鉴权 header ───────────

def make_headers(app_key: str, access_key: str, request_id: str = None) -> Dict[str, str]:
    return {
        "X-Api-App-Key":     app_key,
        "X-Api-Access-Key":  access_key,
        "X-Api-Resource-Id": RESOURCE_ID,
        "X-Api-Request-Id":  request_id or str(uuid.uuid4()),
        "X-Api-Sequence":    "-1",          # 非流式
        "Content-Type":      "application/json",
    }


# ─────────── 提交任务 ───────────

def submit_task(
    audio_url: str,
    app_key: str,
    access_key: str,
    audio_format: str = "mp3",
    num_spk: Optional[int] = None,
    enable_itn: bool = True,
    enable_punc: bool = True,
    enable_ddc: bool = True,         # 顺滑 / 去口语化
    hotwords: List[str] = None,
    language: str = "zh-CN",
) -> str:
    """提交录音文件识别任务, 返回 task_id (后续轮询用)"""
    request_id = str(uuid.uuid4())
    payload = {
        "user": {"uid": "voice_sep_user"},
        "audio": {
            "url": audio_url,
            "format": audio_format,
        },
        "request": {
            "model_name": "bigmodel",
            "show_utterances": True,        # 输出句子级时间戳 + speaker
            "enable_speaker_info": True,    # ← diarization 开关
            "enable_itn": enable_itn,
            "enable_punc": enable_punc,
            "enable_ddc": enable_ddc,
            "language": language,
        },
    }
    if num_spk and num_spk > 0:
        payload["request"]["speaker_number"] = int(num_spk)
    if hotwords:
        payload["request"]["context"] = {"hotwords": hotwords}

    headers = make_headers(app_key, access_key, request_id)
    print(f"[submit] POST {SUBMIT_URL}")
    print(f"[submit] X-Api-Request-Id = {request_id}")
    r = requests.post(SUBMIT_URL, headers=headers, json=payload, timeout=30)

    # 火山 API 的状态在 header 里
    status = r.headers.get("X-Api-Status-Code")
    msg    = r.headers.get("X-Api-Message")
    log_id = r.headers.get("X-Tt-Logid")
    print(f"[submit] status={status} msg={msg} log_id={log_id}")
    if status != "20000000":
        print(f"[submit] body = {r.text}")
        sys.exit(f"提交失败: {status} {msg}")

    # task_id 就是我们传的 X-Api-Request-Id, 后续轮询用同一个
    return request_id


# ─────────── 轮询结果 ───────────

def query_task(
    task_id: str,
    app_key: str,
    access_key: str,
    interval_s: float = 5.0,
    timeout_s: int = 1800,
) -> Dict:
    """轮询任务结果, 拿到完整 JSON 后返回"""
    headers = make_headers(app_key, access_key, request_id=task_id)
    t0 = time.time()
    print(f"[query] 开始轮询 task_id={task_id} (每 {interval_s}s 一次, 最长等 {timeout_s}s)")

    while True:
        r = requests.post(QUERY_URL, headers=headers, json={}, timeout=30)
        status = r.headers.get("X-Api-Status-Code", "")
        msg    = r.headers.get("X-Api-Message", "")
        elapsed = time.time() - t0

        # 20000000 = 完成
        # 20000001 = 排队中
        # 20000002 = 处理中
        if status == "20000000":
            print(f"[query] 完成 ({elapsed:.0f}s)")
            return r.json()
        if status in ("20000001", "20000002"):
            print(f"  [{elapsed:5.0f}s] {msg or status}, 继续等...")
        else:
            print(f"[query] body = {r.text}")
            sys.exit(f"查询失败: status={status} msg={msg}")

        if elapsed > timeout_s:
            sys.exit(f"超时 (>{timeout_s}s), 任务可能卡了, 之后可用 --query-only --task-id {task_id} 拉")
        time.sleep(interval_s)


# ─────────── 结果解析: 火山 JSON → 你的格式 ───────────

def parse_volc_result(result: Dict) -> List[Dict]:
    """
    火山返回结构 (大致):
      result.utterances = [
        {
          "text": "...",
          "start_time": 0,         # ms
          "end_time": 2300,
          "additions": {"speaker": "0"} 或 "speaker_id": 0,
          "words": [{"text":..,"start_time":..,"end_time":..}]
        },
        ...
      ]

    输出: [{"start": s, "end": s, "speaker": "spk_N", "overlap": False, "text": str}, ...]
    与 result/*_seacopara_5.json 完全兼容.
    """
    # result 顶层 = {"result": {...}} 或直接 {...}, 兼容两种
    root = result.get("result") if isinstance(result, dict) and "result" in result else result
    utterances = root.get("utterances") or root.get("Utterances") or []
    if not utterances:
        # 退化: 只有整段 text
        return [{
            "start": 0.0,
            "end":   round(root.get("audio_duration", 0) / 1000, 2),
            "speaker": "spk_unknown",
            "overlap": False,
            "text":  root.get("text", ""),
        }]

    out = []
    for u in utterances:
        # speaker 可能在多个字段, 都试一遍
        spk = (
            u.get("speaker") if u.get("speaker") is not None else
            (u.get("additions") or {}).get("speaker") if isinstance(u.get("additions"), dict) else
            u.get("speaker_id")
        )
        if spk is None:
            spk_label = "spk_unknown"
        else:
            spk_label = f"spk_{spk}"
        out.append({
            "start": round(float(u.get("start_time", 0)) / 1000, 2),
            "end":   round(float(u.get("end_time",   0)) / 1000, 2),
            "speaker": spk_label,
            "overlap": False,        # 火山 diar 不输出重叠
            "text":  (u.get("text") or "").strip(),
        })
    return out


# ─────────── 段落合并 (同一说话人 + 小间隔) ───────────

def merge_consecutive_same_spk(
    sentences: List[Dict],
    max_gap_s: float = 0.8,
    max_dur_s: float = 30.0,
    max_chars: int = 200,
) -> List[Dict]:
    out = []
    for s in sentences:
        if not out:
            out.append(dict(s)); continue
        last = out[-1]
        same   = last["speaker"] == s["speaker"]
        gap    = s["start"] - last["end"]
        new_d  = s["end"]   - last["start"]
        new_c  = len(last["text"]) + len(s["text"])
        if same and gap <= max_gap_s and new_d <= max_dur_s and new_c <= max_chars:
            last["end"] = s["end"]
            last["text"] += s["text"]
        else:
            out.append(dict(s))
    return out


# ─────────── 输出 ───────────

def _fmt_time(s: float) -> str:
    s = int(s)
    return f"{s//60}:{s%60:02d}"


def save_outputs(paragraphs: List[Dict], json_path: str, txt_path: str, wav_name: str = ""):
    if json_path:
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(paragraphs, f, ensure_ascii=False, indent=2)
        print(f"[save] {json_path}")
    if txt_path:
        with open(txt_path, "w", encoding="utf-8") as f:
            if wav_name:
                f.write(f"{wav_name}\n\n")
            for p in paragraphs:
                f.write(f"{p['speaker']} - {_fmt_time(p['start'])}-{_fmt_time(p['end'])}\n")
                f.write(f"{p['text']}\n\n")
        print(f"[save] {txt_path}")


# ─────────── 主流程 ───────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    file_nm = "钱部长数据融合沟通"

    # 输入: 二选一
    ap.add_argument("--audio-url", default=None,
                    help="音频公网 HTTPS URL (TOS/OSS/七牛/自建)")
    ap.add_argument("--audio-format", default="mp3", choices=["mp3", "wav", "m4a", "ogg", "flac", "aac"])

    # 凭证
    ap.add_argument("--app-key", default=os.environ.get("VOLC_APP_KEY", ""))
    ap.add_argument("--access-key", default=os.environ.get("VOLC_ACCESS_KEY", ""))

    # 识别参数
    ap.add_argument("--num-spk", type=int, default=None, help="已知人数 (建议传, 准确率高)")
    ap.add_argument("--language", default="zh-CN")
    ap.add_argument("--hotwords", default=None,
                    help="热词列表 (逗号分隔), e.g. '钱部长,毕姐,公交云,OD分析'")
    ap.add_argument("--no-itn",  action="store_true", help="关闭 ITN (中文数字→阿拉伯)")
    ap.add_argument("--no-punc", action="store_true", help="关闭标点")
    ap.add_argument("--no-ddc",  action="store_true", help="关闭顺滑 (去口语化 嗯/啊/重复)")

    # 段落合并
    ap.add_argument("--merge-gap", type=float, default=0.8)
    ap.add_argument("--merge-max-dur", type=float, default=30.0)
    ap.add_argument("--merge-max-chars", type=int, default=200)

    # 输出
    ap.add_argument("--output",     default=f"result/{file_nm}_volc.json")
    ap.add_argument("--output-txt", default=f"result/{file_nm}_volc.txt")
    ap.add_argument("--dump-raw",   default=None, help="保存火山原始 JSON (调试用)")

    # 只查询模式
    ap.add_argument("--query-only", action="store_true",
                    help="跳过 submit, 直接用已有 task_id 拉结果")
    ap.add_argument("--task-id", default=None, help="--query-only 时必填")

    # 轮询
    ap.add_argument("--poll-interval", type=float, default=5.0)
    ap.add_argument("--poll-timeout",  type=int,   default=1800)

    args = ap.parse_args()

    if not args.app_key or not args.access_key:
        sys.exit("缺凭证: export VOLC_APP_KEY=xxx VOLC_ACCESS_KEY=xxx (或用 --app-key/--access-key)")

    # 1) 提交 (或跳过)
    if args.query_only:
        if not args.task_id:
            sys.exit("--query-only 需要 --task-id xxx")
        task_id = args.task_id
        print(f"[main] query-only 模式, task_id={task_id}")
    else:
        if not args.audio_url:
            sys.exit("缺 --audio-url (火山只收公网 HTTPS URL)")
        hotwords = [h.strip() for h in args.hotwords.split(",") if h.strip()] if args.hotwords else None
        task_id = submit_task(
            audio_url=args.audio_url,
            app_key=args.app_key,
            access_key=args.access_key,
            audio_format=args.audio_format,
            num_spk=args.num_spk,
            enable_itn=not args.no_itn,
            enable_punc=not args.no_punc,
            enable_ddc=not args.no_ddc,
            hotwords=hotwords,
            language=args.language,
        )
        print(f"[main] task_id = {task_id}  (失败可用 --query-only --task-id 重试)")

    # 2) 轮询拿结果
    raw = query_task(
        task_id,
        app_key=args.app_key,
        access_key=args.access_key,
        interval_s=args.poll_interval,
        timeout_s=args.poll_timeout,
    )

    if args.dump_raw:
        with open(args.dump_raw, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        print(f"[dump] {args.dump_raw}")

    # 3) 解析 + 合段
    sentences = parse_volc_result(raw)
    print(f"[parse] {len(sentences)} 句")
    paragraphs = merge_consecutive_same_spk(
        sentences,
        max_gap_s=args.merge_gap,
        max_dur_s=args.merge_max_dur,
        max_chars=args.merge_max_chars,
    )
    print(f"[merge] {len(sentences)} → {len(paragraphs)} 段")

    # speaker 统计
    from collections import Counter
    spk_counts = Counter(p["speaker"] for p in paragraphs)
    print(f"[stats] speakers: {dict(spk_counts)}")

    # 4) 写出
    save_outputs(paragraphs, args.output, args.output_txt,
                 wav_name=os.path.basename(args.audio_url or "task_" + task_id))

    print("\n=== 完成 ===")
    print(f"  task_id (24h 内可重复 query): {task_id}")


if __name__ == "__main__":
    main()
