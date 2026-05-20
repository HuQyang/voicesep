"""
路线 B: Paraformer-zh + 热词 + 纠错词典

思路:
  - 用回 paraformer-zh (中文离线 ASR 天花板之一, 你原项目就用的它)
  - 热词: 解码时让模型倾向正确词 (如 "阈值")
  - 纠错词典: 解码后正则替换错词 (如 "预值" -> "阈值")
  - 词典/热词从 corrections.json 加载, 持续积累领域知识

用法:
  python demo_route_b.py <audio> [--corrections corrections.json]
                                  [--extra-hotword "钱部长 数据要素"]

输出: <audio>.B.txt / <audio>.B.json
"""

import os
import json
import argparse
from funasr import AutoModel


def load_corrections(path: str) -> dict:
    if not os.path.exists(path):
        return {"rules": [], "hotwords": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["rules"] = sorted(data.get("rules", []), key=lambda r: -len(r["wrong"]))
    return data


def apply_corrections(text: str, rules: list) -> str:
    for r in rules:
        text = text.replace(r["wrong"], r["right"])
    return text


class RouteBPipeline:
    def __init__(self):
        print("[init] Paraformer-zh + cam++ ...")
        self.model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            disable_update=True,
        )
        print("[init] 完成")

    def process(self, audio: str, hotwords: list, rules: list) -> list:
        if not os.path.exists(audio):
            print(f"错误: 找不到音频文件 '{audio}'。请检查路径是否正确。")
            return []
            
        hot_str = " ".join(hotwords) if hotwords else ""
        if hot_str:
            print(f"[asr] 热词: {hot_str}")

        res = self.model.generate(input=audio, batch_size_s=300, hotword=hot_str)
        segs = []
        for it in res or []:
            for s in it.get("sentence_info", []):
                raw = s.get("text", "")
                fixed = apply_corrections(raw, rules)
                segs.append({
                    "speaker": f"Speaker_{s.get('spk', 'X')}",
                    "start": round(s.get("start", 0) / 1000.0, 2),
                    "end": round(s.get("end", 0) / 1000.0, 2),
                    "text": fixed,
                    "_fixed": raw != fixed,
                })
        n_fix = sum(1 for s in segs if s["_fixed"])
        if n_fix:
            print(f"[fix] 纠错命中 {n_fix} 段")
        return segs

    @staticmethod
    def merge(segments: list) -> list:
        if not segments:
            return []
        m = [dict(segments[0])]
        for s in segments[1:]:
            if s["speaker"] == m[-1]["speaker"]:
                m[-1]["end"] = s["end"]
                m[-1]["text"] += s["text"]
            else:
                m.append(dict(s))
        return m

    @staticmethod
    def to_readable(segments: list) -> str:
        out = ["========== 会议记录 (路线B) ==========\n"]
        for seg in segments:
            s, e = seg["start"], seg["end"]
            ts = f"[{int(s//60):02d}:{int(s%60):02d}-{int(e//60):02d}:{int(e%60):02d}]"
            out.append(f"{ts} {seg['speaker']}:\n{seg['text']}\n")
        return "\n".join(out)

    @staticmethod
    def to_json(segments: list, audio: str) -> dict:
        speakers = sorted({s["speaker"] for s in segments})
        dur = max((s["end"] for s in segments), default=0.0)
        return {
            "source": os.path.basename(audio),
            "route": "B_paraformer_hotword_dict",
            "duration_sec": dur,
            "speakers": [{"id": s, "name": None} for s in speakers],
            "segments": [{k: v for k, v in s.items() if not k.startswith("_")}
                         for s in segments],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--corrections", default="corrections.json")
    ap.add_argument("--extra-hotword", default="",
                    help="本场临时热词, 空格分隔 (与 json 中 hotwords 合并)")
    args = ap.parse_args()

    conf = load_corrections(args.corrections)
    hotwords = list(conf.get("hotwords", []))
    if args.extra_hotword:
        hotwords += args.extra_hotword.split()

    pipe = RouteBPipeline()
    segs = pipe.process(args.audio, hotwords, conf["rules"])
    if not segs:
        print("无输出")
        return
    merged = pipe.merge(segs)
    readable = pipe.to_readable(merged)
    print(readable)

    base = os.path.splitext(args.audio)[0]
    with open(f"{base}.B.txt", "w", encoding="utf-8") as f:
        f.write(readable)
    with open(f"{base}.B.json", "w", encoding="utf-8") as f:
        json.dump(pipe.to_json(segs, args.audio), f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {base}.B.txt / {base}.B.json")


if __name__ == "__main__":
    main()
