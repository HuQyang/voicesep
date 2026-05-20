"""
SenseVoice + cam++ 会议转写 Demo (两遍 pass 方案)

由于 SenseVoice 不输出时间戳，cam++ 声纹聚类依赖时间戳，二者不能直接组合。
本 demo 采用两遍 pass:
  Pass 1: Paraformer + VAD + 标点 + cam++ → 拿到带说话人的分段 (含 start/end)
  Pass 2: 用 SenseVoice 对每段音频重新转写 → 替换文字

输入: 一段会议音频 (wav/mp3)
输出:
  1) 控制台打印带说话人的会议记录
  2) <audio>.sv.json    —— 结构化转写，喂给下游 LLM
  3) <audio>.sv.txt     —— 人类可读版本

用法:
  python demo_sensevoice.py <audio_file> [--hotword "张三 李四 K8s"]
"""

import os
import re
import sys
import json
import argparse
import tempfile

import librosa
import soundfile as sf
from funasr import AutoModel


SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]*\|>")


def clean_sv_text(text: str) -> str:
    """剥掉 SenseVoice 的 <|zh|><|NEUTRAL|><|Speech|> 等标签"""
    return SENSEVOICE_TAG_RE.sub("", text).strip()


class SenseVoiceMeetingPipeline:
    def __init__(self):
        print("[init] 加载 Pass1: Paraformer + VAD + 标点 + cam++ ...")
        self.diar_model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            disable_update=True,
        )

        print("[init] 加载 Pass2: SenseVoiceSmall ...")
        self.asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            disable_update=True,
        )
        print("[init] 模型加载完成")

    def process(self, audio_path: str, hotword: str = "") -> list:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)

        # ---------- Pass 1: 拿分段 + 说话人 ----------
        print(f"[pass1] 说话人分离: {audio_path}")
        res = self.diar_model.generate(
            input=audio_path,
            batch_size_s=300,
            hotword=hotword,
        )
        if not res:
            return []

        diar_segments = []
        for item in res:
            for s in item.get("sentence_info", []):
                diar_segments.append({
                    "speaker": f"Speaker_{s.get('spk', 'X')}",
                    "start": s.get("start", 0) / 1000.0,
                    "end": s.get("end", 0) / 1000.0,
                    "text_paraformer": s.get("text", ""),  # 兜底用
                })

        if not diar_segments:
            return []

        print(f"[pass1] 共 {len(diar_segments)} 段，"
              f"说话人 {len(set(s['speaker'] for s in diar_segments))} 人")

        # ---------- Pass 2: SenseVoice 重新转写每段 ----------
        print("[pass2] SenseVoice 重新转写...")
        wav, sr = librosa.load(audio_path, sr=16000, mono=True)

        segments = []
        for i, seg in enumerate(diar_segments, 1):
            s_idx = max(0, int(seg["start"] * sr))
            e_idx = min(len(wav), int(seg["end"] * sr))
            if e_idx - s_idx < int(0.2 * sr):
                # 太短的段（<0.2s）跳过 SenseVoice，直接用 Paraformer 结果
                text = seg["text_paraformer"]
            else:
                chunk = wav[s_idx:e_idx]
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    sf.write(tmp.name, chunk, sr)
                    tmp_path = tmp.name
                try:
                    sv_res = self.asr_model.generate(
                        input=tmp_path,
                        cache={},
                        language="zh",
                        use_itn=True,
                        hotword=hotword,
                    )
                    text = clean_sv_text(sv_res[0].get("text", "")) if sv_res else ""
                    if not text:
                        text = seg["text_paraformer"]
                finally:
                    os.unlink(tmp_path)

            segments.append({
                "speaker": seg["speaker"],
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": text,
            })

            if i % 10 == 0 or i == len(diar_segments):
                print(f"[pass2] 进度 {i}/{len(diar_segments)}")

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
            "segments": segments,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="音频文件路径 (wav/mp3)")
    parser.add_argument("--hotword", default="", help="热词，空格分隔")
    args = parser.parse_args()

    pipeline = SenseVoiceMeetingPipeline()
    raw_segments = pipeline.process(args.audio, hotword=args.hotword)
    if not raw_segments:
        print("未识别到内容")
        return

    merged = pipeline.merge_consecutive(raw_segments)
    readable = pipeline.to_readable(merged)
    print(readable)

    base = os.path.splitext(args.audio)[0]
    txt_path = f"{base}.sv.txt"
    json_path = f"{base}.sv.json"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(readable)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            pipeline.to_llm_json(raw_segments, args.audio),
            f, ensure_ascii=False, indent=2,
        )

    print(f"\n[saved] 人类可读版: {txt_path}")
    print(f"[saved] LLM 结构化版: {json_path}")


if __name__ == "__main__":
    main()
