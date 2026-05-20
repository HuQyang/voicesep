"""
降噪 demo: FRCRN (16kHz)

保存 原始 / 降噪后 两份, 方便 A/B 听感对比 + 打印能量指标.

用法:
  python demo_denoise.py --wav test_audio.wav
  python demo_denoise.py --wav "data/xxx.mp3" --start 130 --end 145
  python demo_denoise.py --wav "..." --model zipenhancer       # 换降噪模型

输出 (默认 denoise_out/):
  <name>_orig.wav        原始 (切片后)
  <name>_denoised.wav    FRCRN 降噪后
"""
import argparse
import os
import tempfile

import numpy as np
import librosa
import soundfile as sf

SR = 16000

MODELS = {
    "frcrn":       "damo/speech_frcrn_ans_cirm_16k",
    "zipenhancer": "damo/speech_zipenhancer_ans_multiloss_16k_base",
}


_pipe = None


def get_model(name: str = "frcrn"):
    global _pipe
    if _pipe is None:
        from modelscope.pipelines import pipeline as ms_pipeline
        from modelscope.utils.constant import Tasks
        model_id = MODELS[name]
        print(f"[denoise] 加载模型: {model_id}")
        try:
            _pipe = ms_pipeline(
                task=Tasks.acoustic_noise_suppression, model=model_id,
            )
        except Exception:
            # iic/ 新前缀兜底
            _pipe = ms_pipeline(
                task=Tasks.acoustic_noise_suppression,
                model=model_id.replace("damo/", "iic/"),
            )
    return _pipe


def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)) + 1e-10)
    return 20.0 * np.log10(rms)


def denoise(wav: np.ndarray, sr: int = SR, model_name: str = "frcrn") -> np.ndarray:
    """输入 1D mono 波形, 输出降噪后波形 (16k float32)."""
    if sr != SR:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, SR)
        tmp_path = tmp.name
    out_path = tmp_path.replace(".wav", "_dn.wav")
    try:
        model = get_model(model_name)
        result = model(tmp_path, output_path=out_path)
        if os.path.exists(out_path):
            out, _ = sf.read(out_path, dtype="float32")
        else:
            pcm = (
                result.get("output_pcm")
                or result.get("output_wav")
                or result.get("output")
            )
            if isinstance(pcm, bytes):
                out = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                out = np.asarray(pcm, dtype=np.float32).reshape(-1)
                if np.max(np.abs(out)) > 2.0:
                    out = out / 32768.0
    finally:
        for p in (tmp_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass
    return out


def measure(wav: np.ndarray):
    """简单分析: RMS dBFS + 估算"噪声底" (最安静 10% 帧的 RMS)"""
    rms = _rms_dbfs(wav)
    # 切 50ms 帧算 frame-level RMS
    hop = int(0.05 * SR)
    n = len(wav) // hop
    if n < 4:
        return {"rms_dbfs": rms, "noise_floor_dbfs": rms, "peak_dbfs": rms}
    frame_rms = np.array([
        np.sqrt(np.mean(wav[i*hop:(i+1)*hop] ** 2) + 1e-10) for i in range(n)
    ])
    frame_db = 20 * np.log10(frame_rms + 1e-10)
    noise_floor = float(np.percentile(frame_db, 10))
    peak = float(np.percentile(frame_db, 95))
    return {
        "rms_dbfs": rms,
        "noise_floor_dbfs": noise_floor,
        "peak_dbfs": peak,
        "dynamic_range": peak - noise_floor,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="输入音频")
    ap.add_argument("--start", type=float, default=None, help="切片起点(s)")
    ap.add_argument("--end", type=float, default=None, help="切片终点(s)")
    ap.add_argument("--out-dir", default="denoise_out")
    ap.add_argument("--model", choices=list(MODELS.keys()), default="frcrn")
    args = ap.parse_args()

    wav, _ = librosa.load(args.wav, sr=SR, mono=True)
    if args.start is not None and args.end is not None:
        s, e = int(args.start * SR), int(args.end * SR)
        wav = wav[s:e]
        print(f"[denoise] 切出 {args.start:.2f}s -> {args.end:.2f}s, 长度 {len(wav)/SR:.2f}s")
    else:
        print(f"[denoise] 处理整条音频, 长度 {len(wav)/SR:.2f}s")

    out = denoise(wav, sr=SR, model_name=args.model)
    # 长度对齐 (FRCRN 偶尔会有几样本偏差, 截到一致)
    n = min(len(wav), len(out))
    wav, out = wav[:n], out[:n]

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.wav))[0]
    if args.start is not None:
        base = f"{base}_{args.start:.1f}-{args.end:.1f}"

    orig_path = os.path.join(args.out_dir, f"{base}_orig.wav")
    dn_path = os.path.join(args.out_dir, f"{base}_denoised_{args.model}.wav")
    sf.write(orig_path, wav, SR)
    sf.write(dn_path, out, SR)

    m_in = measure(wav)
    m_out = measure(out)

    print("\n=== A/B 对比 ===")
    fmt = lambda d: (
        f"RMS={d['rms_dbfs']:6.1f} dBFS  "
        f"noise_floor(p10)={d['noise_floor_dbfs']:6.1f}  "
        f"peak(p95)={d['peak_dbfs']:6.1f}  "
        f"dyn_range={d['dynamic_range']:5.1f} dB"
    )
    print(f"  原始:   {fmt(m_in)}")
    print(f"  降噪后: {fmt(m_out)}")
    nf_drop = m_in["noise_floor_dbfs"] - m_out["noise_floor_dbfs"]
    dr_gain = m_out["dynamic_range"] - m_in["dynamic_range"]
    print(f"\n  噪声底下降: {nf_drop:+.1f} dB   动态范围变化: {dr_gain:+.1f} dB")
    if nf_drop < 3:
        print("  ⚠  噪声底没显著下降, 可能是 (a) 原音频已较干净, 或 (b) FRCRN 对此场景不灵敏")
    if dr_gain < 0:
        print("  ⚠  动态范围反而缩小, 可能语音也被一起压了, 试试 --model zipenhancer")

    print(f"\n  原始:   {orig_path}")
    print(f"  降噪后: {dn_path}")


if __name__ == "__main__":
    main()
