"""
VAD demo
- 路线 A: FunASR FSMN-VAD (离线, 输出 segment 列表)
- 路线 B: Silero-VAD     (流式, 帧级 30ms)

用法:
    python demo_vad.py test_audio.wav                 # FSMN
    python demo_vad.py test_audio.wav --engine silero
    python demo_vad.py test_audio.wav --stream        # 模拟流式 (Silero)
"""
import argparse
import numpy as np
import librosa
import soundfile as sf

SR = 16000


def vad_fsmn(wav_path: str):
    from funasr import AutoModel
    vad = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
    res = vad.generate(input=wav_path)
    return [(s / 1000, e / 1000) for s, e in res[0]["value"]]


def vad_silero(wav_path: str):
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    (get_speech_timestamps, _, read_audio, *_) = utils
    wav = read_audio(wav_path, sampling_rate=SR)
    ts = get_speech_timestamps(wav, model, sampling_rate=SR)
    return [(t["start"] / SR, t["end"] / SR) for t in ts]


def vad_silero_stream(wav_path: str, frame_ms: int = 32):
    """模拟流式: 一帧一帧灌入, 用 VADIterator"""
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    VADIterator = utils[3]
    it = VADIterator(model, sampling_rate=SR)

    wav, _ = librosa.load(wav_path, sr=SR, mono=True)
    win = int(SR * frame_ms / 1000)
    # silero v4/v5 要求 512 sample (16k) 一帧
    win = 512
    import torch as _t
    events = []
    for i in range(0, len(wav) - win, win):
        chunk = _t.from_numpy(wav[i : i + win])
        ev = it(chunk, return_seconds=True)
        if ev:
            events.append(ev)
    it.reset_states()
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--engine", choices=["fsmn", "silero"], default="fsmn")
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()

    if args.stream:
        print("[stream/silero] events =")
        for ev in vad_silero_stream(args.wav):
            print(" ", ev)
        return

    segs = vad_fsmn(args.wav) if args.engine == "fsmn" else vad_silero(args.wav)
    total = sum(e - s for s, e in segs)
    print(f"[{args.engine}] {len(segs)} segments, 语音总时长 {total:.2f}s")
    for s, e in segs:
        print(f"  {s:7.2f}s -> {e:7.2f}s  ({e - s:.2f}s)")


if __name__ == "__main__":
    main()
