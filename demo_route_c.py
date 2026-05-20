"""
路线 C: faster-whisper (large-v3) + Paraformer 声纹时间戳对齐

思路:
  Pass 1 (Paraformer+cam++): 拿到说话人时间线 [(speaker, start, end)]
  Pass 2 (faster-whisper):    word-level 时间戳 + initial_prompt (=隐形热词)
  Pass 3 (时间戳匹配):        每个词找重叠最多的说话人窗口 -> 拼回带说话人的句子

依赖:
  pip install faster-whisper

首次运行会自动下载 Whisper large-v3 (~3GB), 走 HuggingFace 或 OpenAI CDN.

输出: <audio>.C.txt / <audio>.C.json
"""

import os
import json
import argparse
from funasr import AutoModel


def load_initial_prompt(corrections_path: str, extra: str = "") -> str:
    """
    把 corrections.json 中的 hotwords + right 词都拼成 initial_prompt
    Whisper 看到这个前缀, 解码时会更倾向出现这些词
    """
    words = []
    if os.path.exists(corrections_path):
        with open(corrections_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        words += list(data.get("hotwords", []))
        words += [r["right"] for r in data.get("rules", [])]
    if extra:
        words += extra.split()
    seen, uniq = set(), []
    for w in words:
        if w and w not in seen:
            seen.add(w)
            uniq.append(w)
    # Whisper initial_prompt 不要太长 (官方建议 <224 tokens)
    return "本段会议涉及以下专有名词与术语: " + "、".join(uniq[:60]) if uniq else ""


class RouteCPipeline:
    def __init__(self, whisper_model: str = "large-v3",
                 compute_type: str = "default"):
        print("[init] Pass1: Paraformer + cam++ (仅取说话人时间线) ...")
        self.diar = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            disable_update=True,
        )

        print(f"[init] Pass2: faster-whisper {whisper_model} ...")
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise SystemExit(
                "需要先安装: pip install faster-whisper"
            )
        # device 让 ctranslate2 自己选; compute_type=default 让其选最快
        self.whisper = WhisperModel(
            whisper_model,
            device="auto",
            compute_type=compute_type,
        )
        print("[init] 完成")

    def _speaker_timeline(self, audio: str, hotword: str) -> list:
        res = self.diar.generate(input=audio, batch_size_s=300, hotword=hotword)
        timeline = []
        for it in res or []:
            for s in it.get("sentence_info", []):
                timeline.append({
                    "speaker": f"Speaker_{s.get('spk', 'X')}",
                    "start": s.get("start", 0) / 1000.0,
                    "end": s.get("end", 0) / 1000.0,
                })
        timeline.sort(key=lambda x: x["start"])
        return timeline

    @staticmethod
    def _assign_speaker(word_start: float, word_end: float, timeline: list) -> str:
        """挑与该词时间区间重叠最多的说话人窗口"""
        best, best_ov = None, 0.0
        for win in timeline:
            ov = max(0.0, min(word_end, win["end"]) - max(word_start, win["start"]))
            if ov > best_ov:
                best_ov, best = ov, win["speaker"]
        return best or "Speaker_X"

    def process(self, audio: str, initial_prompt: str = "",
                hotword_for_diar: str = "") -> list:
        if not os.path.exists(audio):
            print(f"错误: 找不到音频文件 '{audio}'。请检查路径是否正确。")
            return []
            
        print(f"[pass1] 说话人时间线: {audio}")
        timeline = self._speaker_timeline(audio, hotword_for_diar)
        print(f"[pass1] {len(timeline)} 个说话人片段, "
              f"{len({t['speaker'] for t in timeline})} 个说话人")

        print(f"[pass2] Whisper 转写 + word_timestamps ...")
        if initial_prompt:
            print(f"[pass2] initial_prompt: {initial_prompt[:120]}...")
        segments_iter, info = self.whisper.transcribe(
            audio,
            language="zh",
            initial_prompt=initial_prompt or None,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
            condition_on_previous_text=True,
        )
        # 收集所有 word
        words = []
        for seg in segments_iter:
            for w in (seg.words or []):
                if w.word is None:
                    continue
                words.append({
                    "start": float(w.start),
                    "end": float(w.end),
                    "text": w.word,
                })
        print(f"[pass2] 共 {len(words)} 个词")

        print("[pass3] 词级说话人分配 + 同说话人合并 ...")
        out = []
        cur = None
        for w in words:
            spk = self._assign_speaker(w["start"], w["end"], timeline)
            if cur is None or spk != cur["speaker"]:
                if cur is not None:
                    out.append(cur)
                cur = {
                    "speaker": spk,
                    "start": round(w["start"], 2),
                    "end": round(w["end"], 2),
                    "text": w["text"].strip(),
                }
            else:
                cur["end"] = round(w["end"], 2)
                # Whisper 中文 word 自带空格, 这里粘合
                cur["text"] += w["text"]
        if cur is not None:
            out.append(cur)
        # 清理空白和首尾
        for s in out:
            s["text"] = s["text"].replace(" ", "").strip()
        return out

    @staticmethod
    def to_readable(segments: list) -> str:
        lines = ["========== 会议记录 (路线C) ==========\n"]
        for seg in segments:
            s, e = seg["start"], seg["end"]
            ts = f"[{int(s//60):02d}:{int(s%60):02d}-{int(e//60):02d}:{int(e%60):02d}]"
            lines.append(f"{ts} {seg['speaker']}:\n{seg['text']}\n")
        return "\n".join(lines)

    @staticmethod
    def to_json(segments: list, audio: str) -> dict:
        speakers = sorted({s["speaker"] for s in segments})
        dur = max((s["end"] for s in segments), default=0.0)
        return {
            "source": os.path.basename(audio),
            "route": "C_whisper_large_v3_align",
            "duration_sec": dur,
            "speakers": [{"id": s, "name": None} for s in speakers],
            "segments": segments,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--corrections", default="corrections.json",
                    help="复用 B 路线的词典/热词, 生成 Whisper initial_prompt")
    ap.add_argument("--extra-hotword", default="")
    ap.add_argument("--whisper-model", default="large-v3",
                    help="large-v3 / medium / small")
    args = ap.parse_args()

    initial_prompt = load_initial_prompt(args.corrections, args.extra_hotword)
    pipe = RouteCPipeline(whisper_model=args.whisper_model)
    segs = pipe.process(
        args.audio,
        initial_prompt=initial_prompt,
        hotword_for_diar=args.extra_hotword,
    )
    if not segs:
        print("无输出")
        return
    readable = pipe.to_readable(segs)
    print(readable)

    base = os.path.splitext(args.audio)[0]
    with open(f"{base}.C.txt", "w", encoding="utf-8") as f:
        f.write(readable)
    with open(f"{base}.C.json", "w", encoding="utf-8") as f:
        json.dump(pipe.to_json(segs, args.audio), f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {base}.C.txt / {base}.C.json")


if __name__ == "__main__":
    main()
