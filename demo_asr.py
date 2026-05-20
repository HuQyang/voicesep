"""
ASR demo
- 路线 A: Paraformer-large + VAD + Punc + 热词 (中文主线)
- 路线 B: SenseVoice-Small (多语种 + 情感/事件)

用法:
    python demo_asr.py test_audio.wav
    python demo_asr.py test_audio.wav --engine sensevoice
    python demo_asr.py test_audio.wav --hotword "韦小宝 鳌拜 康熙"
"""
import argparse
import json
import os
from funasr import AutoModel


def load_corrections(path: str = "corrections.json"):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def apply_corrections(text: str, table: dict) -> str:
    for wrong, right in table.items():
        text = text.replace(wrong, right)
    return text


def asr_paraformer(wav_path: str, hotword: str = ""):
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        # spk_model="cam++",   # 想要带说话人就打开, 等价于 route_b 路线
        disable_update=True,
    )
    res = model.generate(
        input=wav_path,
        batch_size_s=300,
        hotword=hotword,
    )
    return res


def asr_sensevoice(wav_path: str):
    model = AutoModel(
        model="iic/SenseVoiceSmall",
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        disable_update=True,
    )
    res = model.generate(
        input=wav_path,
        cache={},
        language="auto",       # auto / zh / en / ja / ko / yue
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
        merge_length_s=15,
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--engine", choices=["paraformer", "sensevoice"], default="paraformer")
    ap.add_argument("--hotword", default="")
    ap.add_argument("--no-correct", action="store_true")
    args = ap.parse_args()

    if args.engine == "paraformer":
        res = asr_paraformer(args.wav, hotword=args.hotword)
    else:
        res = asr_sensevoice(args.wav)

    corrections = {} if args.no_correct else load_corrections()
    print("\n=== ASR ===")
    for r in res:
        text = r.get("text", "")
        if corrections:
            text = apply_corrections(text, corrections)
        print(text)
        # Paraformer 走 VAD 时段会带 timestamp; 想看就打开:
        # if "timestamp" in r: print(r["timestamp"])


if __name__ == "__main__":
    main()
