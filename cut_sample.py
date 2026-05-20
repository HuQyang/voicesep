"""
裁剪声纹注册样本工具

用法:
  # 单段裁剪 (时间点 + 时长)
  python cut_sample.py --wav meeting.mp3 --time 2:50 --dur 8 --out speakers/raw/王总.wav

  # 单段裁剪 (起止时间)
  python cut_sample.py --wav meeting.mp3 --start 2:50 --end 2:58 --out speakers/raw/王总.wav

  # 批量裁剪 (一个 JSON 配置, 一次出多个声纹样本)
  python cut_sample.py --wav meeting.mp3 --batch enroll_plan.json --out-dir speakers/raw/

时间格式支持:
  2:50       → 2 分 50 秒
  0:02:50.5  → 0 时 2 分 50.5 秒
  170        → 170 秒
  170.5      → 170.5 秒

批量配置 (enroll_plan.json):
  [
    {"name": "王总",    "start": "0:55", "end": "1:05"},
    {"name": "小雨",    "start": "5:30", "end": "5:40"},
    {"name": "君姐",    "time": "12:08", "dur": 7}
  ]

输出:
  16kHz mono float32 wav (cam++ / ERes2Net 都标准吃这个)
  控制台打印 RMS dBFS + 时长, 用来确认样本能量是否够强 (建议 RMS > -30 dBFS)
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import librosa
import soundfile as sf


SR_OUT = 16000


def parse_time(t) -> float:
    """
    解析时间字符串为秒.
    支持: float / int / "MM:SS" / "MM:SS.S" / "HH:MM:SS" / "HH:MM:SS.S"
    """
    if t is None:
        return None
    if isinstance(t, (int, float)):
        return float(t)
    s = str(t).strip()
    if not s:
        return None
    # 纯数字 (秒)
    if re.fullmatch(r"[0-9]+(\.[0-9]+)?", s):
        return float(s)
    # MM:SS / HH:MM:SS
    parts = s.split(":")
    if len(parts) == 2:
        m, sec = parts
        return int(m) * 60 + float(sec)
    if len(parts) == 3:
        h, m, sec = parts
        return int(h) * 3600 + int(m) * 60 + float(sec)
    raise ValueError(f"无法解析时间: {s!r}")


def rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)) + 1e-10)
    return 20.0 * np.log10(rms)


def cut_one(wav: np.ndarray, sr: int, start_s: float, end_s: float, out_path: str,
            label: str = ""):
    """裁剪一段, 保存为 16k mono wav, 打印质量指标"""
    if start_s is None or end_s is None:
        raise ValueError("start/end 必须给定")
    if end_s <= start_s:
        raise ValueError(f"end ({end_s}) <= start ({start_s})")

    total_s = len(wav) / sr
    start_s = max(0.0, start_s)
    end_s = min(total_s, end_s)
    s_idx = int(start_s * sr)
    e_idx = int(end_s * sr)
    seg = wav[s_idx:e_idx]
    dur = len(seg) / sr

    if dur < 2.0:
        print(f"  ⚠️  片段仅 {dur:.1f}s, 建议 >=5s 注册才稳")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    sf.write(out_path, seg.astype(np.float32), sr)

    db = rms_dbfs(seg)
    quality = "✅" if db > -30 else ("⚠️" if db > -45 else "❌")
    tag = f"[{label}] " if label else ""
    print(f"  {quality} {tag}{start_s:7.2f}-{end_s:7.2f}s ({dur:.1f}s) "
          f"RMS {db:.1f} dBFS  →  {out_path}")
    if db <= -45:
        print(f"     能量太低, 多半是噪声段, 换一段")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="输入音频 (mp3/wav)")

    # 单段模式
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--time", default=None, help="单段中心时间点, 配合 --dur")
    g.add_argument("--start", default=None, help="单段起始时间, 配合 --end")

    ap.add_argument("--dur", type=float, default=8.0, help="单段时长(s, --time 模式用), 默认 8")
    ap.add_argument("--end", default=None, help="单段结束时间, 配合 --start")
    ap.add_argument("--out", default=None, help="单段输出路径 (--time/--start 模式)")
    ap.add_argument("--label", default="", help="打印日志时的标签 (一般填人名)")

    # 批量模式
    ap.add_argument("--batch", default=None, help="批量配置 JSON 路径")
    ap.add_argument("--out-dir", default="speakers/raw", help="批量模式的输出目录")

    args = ap.parse_args()

    if not os.path.exists(args.wav):
        sys.exit(f"输入音频不存在: {args.wav}")

    print(f"[load] 加载 {args.wav} (16kHz mono)...")
    wav, _ = librosa.load(args.wav, sr=SR_OUT, mono=True)
    total = len(wav) / SR_OUT
    print(f"[load] 总时长 {total:.1f}s ({total/60:.1f} min)")

    if args.batch:
        with open(args.batch, "r", encoding="utf-8") as f:
            plans = json.load(f)
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[batch] {len(plans)} 个样本待裁剪 → {args.out_dir}/")
        for i, p in enumerate(plans, 1):
            name = p.get("name") or f"spk_{i:02d}"
            safe = re.sub(r"[/\\\s]", "_", name)
            out_path = p.get("out") or os.path.join(args.out_dir, f"{safe}.wav")
            # time + dur 优先, 没给就用 start + end
            if p.get("time") is not None:
                center = parse_time(p["time"])
                dur = float(p.get("dur", args.dur))
                start_s = center - dur / 2
                end_s = center + dur / 2
            else:
                start_s = parse_time(p.get("start"))
                end_s = parse_time(p.get("end"))
                if end_s is None and p.get("dur") is not None:
                    end_s = start_s + float(p["dur"])
            try:
                cut_one(wav, SR_OUT, start_s, end_s, out_path, label=name)
            except Exception as e:
                print(f"  ❌ [{name}] 裁剪失败: {e}")
        print("\n[done] 批量裁剪完成. 接下来:")
        print(f"  for w in {args.out_dir}/*.wav; do")
        print(f"      python enroll.py add \"$(basename ${{w%.wav}})\" \"$w\"")
        print(f"  done")
        return

    # 单段模式
    if args.time is not None:
        center = parse_time(args.time)
        start_s = center - args.dur / 2
        end_s = center + args.dur / 2
    elif args.start is not None:
        start_s = parse_time(args.start)
        end_s = parse_time(args.end) if args.end else (start_s + args.dur)
    else:
        sys.exit("单段模式: 必须给 --time 或 --start, 或者改用 --batch")

    if not args.out:
        sys.exit("单段模式需要 --out 指定输出路径")

    cut_one(wav, SR_OUT, start_s, end_s, args.out, label=args.label)


if __name__ == "__main__":
    main()
