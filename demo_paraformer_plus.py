"""
Paraformer + 热词 + 纠错词典 增强版会议转写 Demo

思路:
  - 用回 paraformer-zh (中文离线 ASR 天花板之一)
  - 加热词: 解码时让模型倾向正确词 (如 "阈值")
  - 加纠错词典: 解码后正则替换错词 (如 "预值" -> "阈值")
  - 词典/热词都从 corrections.json 加载, 持续积累

用法:
  python demo_paraformer_plus.py <audio_file>
  python demo_paraformer_plus.py <audio_file> --corrections corrections.json

输出:
  <audio>.pp.txt   人类可读
  <audio>.pp.json  喂给下游 LLM 的结构化数据
"""

import os
import re
import json
import argparse
from funasr import AutoModel


def load_corrections(path: str) -> dict:
    if not os.path.exists(path):
        return {"rules": [], "hotwords": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 按 wrong 长度倒序: 长串先替换, 防止 "预值阈值" 这种被先替换 "预值" 干扰
    data["rules"] = sorted(data.get("rules", []), key=lambda r: -len(r["wrong"]))
    return data


def apply_corrections(text: str, rules: list) -> str:
    for r in rules:
        text = text.replace(r["wrong"], r["right"])
    return text


class ParaformerPlusPipeline:
    def __init__(self):
        print("[init] 加载 Paraformer + VAD + 标点 + cam++ ...")
        self.model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            disable_update=True,
        )
        print("[init] 模型加载完成")

    def process(self, audio_path: str, hotwords: list, rules: list) -> list:
        hotword_str = " ".join(hotwords) if hotwords else ""
        if hotword_str:
            print(f"[asr] 热词: {hotword_str}")

        res = self.model.generate(
            input=audio_path,
            batch_size_s=300,
            hotword=hotword_str,
        )
        if not res:
            return []

        segments = []
        for item in res:
            for s in item.get("sentence_info", []):
                raw_text = s.get("text", "")
                fixed_text = apply_corrections(raw_text, rules)
                segments.append({
                    "speaker": f"Speaker_{s.get('spk', 'X')}",
                    "start": round(s.get("start", 0) / 1000.0, 2),
                    "end": round(s.get("end", 0) / 1000.0, 2),
                    "text": fixed_text,
                    "text_raw": raw_text if raw_text != fixed_text else None,
                })

        n_fixed = sum(1 for s in segments if s["text_raw"])
        if n_fixed:
            print(f"[fix] 纠错命中 {n_fixed} 段")
        return segments

    @staticmethod
    def merge_consecutive(segments: list) -> list:
        if not segments:
            return []
        merged = [dict(segments[0])]
        for seg in segments[1:]:
            if seg["speaker"] == merged[-1]["speaker"]:
                merged[-1]["end"] = seg["end"]
                merged[-1]["text"] += seg["text"]
            else:
                merged.append(dict(seg))
        return merged

    @staticmethod
    def to_readable(segments: list) -> str:
        out = ["========== 会议记录 ==========\n"]
        for seg in segments:
            s, e = seg["start"], seg["end"]
            ts = f"[{int(s//60):02d}:{int(s%60):02d}-{int(e//60):02d}:{int(e%60):02d}]"
            out.append(f"{ts} {seg['speaker']}:\n{seg['text']}\n")
        return "\n".join(out)

    @staticmethod
    def to_llm_json(segments: list, audio_path: str) -> dict:
        speakers = sorted({s["speaker"] for s in segments})
        duration = max((s["end"] for s in segments), default=0.0)
        return {
            "source": os.path.basename(audio_path),
            "duration_sec": duration,
            "speakers": [{"id": spk, "name": None} for spk in speakers],
            "segments": [
                {k: v for k, v in s.items() if k != "text_raw"}
                for s in segments
            ],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--corrections", default="corrections.json")
    parser.add_argument("--extra-hotword", default="",
                        help="本场会议追加的热词, 空格分隔 (会和 json 里的合并)")
    args = parser.parse_args()

    conf = load_corrections(args.corrections)
    hotwords = list(conf.get("hotwords", []))
    if args.extra_hotword:
        hotwords += args.extra_hotword.split()

    if not os.path.exists(args.audio):
        print(f"错误: 找不到音频文件 '{args.audio}'。请检查路径是否正确。")
        return

    pipeline = ParaformerPlusPipeline()
    segments = pipeline.process(args.audio, hotwords, conf["rules"])
    if not segments:
        print("未识别到内容")
        return

    merged = pipeline.merge_consecutive(segments)
    readable = pipeline.to_readable(merged)
    print(readable)

    base = os.path.splitext(args.audio)[0]
    with open(f"{base}.pp.txt", "w", encoding="utf-8") as f:
        f.write(readable)
    with open(f"{base}.pp.json", "w", encoding="utf-8") as f:
        json.dump(pipeline.to_llm_json(segments, args.audio),
                  f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {base}.pp.txt / {base}.pp.json")


if __name__ == "__main__":
    main()
