"""
音频质量估算 (近场 vs 远场 / 干净 vs 混响噪声).

用法:
    from audio_quality import estimate_quality
    score, detail = estimate_quality(wav, sr=16000)
    # score: [0, 1], 越高越像近场清晰
    # detail: 各分量原始值, 用于调参

阈值经验:
    > 0.6   近场清晰 (放心喂 FireRedASR)
    0.4-0.6 中等 (Paraformer 更稳)
    < 0.4   远场/噪声大 (考虑降噪后再 ASR, 或直接 Paraformer)

可选: 接 DNSMOS (Microsoft 开源, ONNX 模型, < 1MB)
    pip install onnxruntime requests
    然后 estimate_quality(..., use_dnsmos=True)
"""
import os
import numpy as np
import librosa


_dnsmos_model = None


def _rms_dbfs(wav: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(wav.astype(np.float32) ** 2)) + 1e-10)
    return 20.0 * np.log10(rms)


def _spectral_features(wav: np.ndarray, sr: int):
    """谱质心 / 谱平坦度 / 谱滚降, 越高频信息丰富越像近场"""
    # 用 short-time 谱分析
    S = np.abs(librosa.stft(wav, n_fft=512, hop_length=160))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    flatness = librosa.feature.spectral_flatness(S=S).mean()
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85).mean()
    return float(centroid), float(flatness), float(rolloff)


def _crest_factor_db(wav: np.ndarray) -> float:
    """峰均比 (dB). 近场语音 6~15 dB, 远场混响 3~6 dB"""
    rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-10)
    peak = float(np.max(np.abs(wav)) + 1e-10)
    return 20.0 * np.log10(peak / rms)


def _heuristic_score(wav: np.ndarray, sr: int) -> tuple:
    """
    用经验特征算质量分 [0, 1].
    返回 (score, detail_dict)
    """
    if len(wav) < sr * 0.3:
        return 0.5, {"reason": "too_short"}

    dbfs = _rms_dbfs(wav)
    centroid, flatness, rolloff = _spectral_features(wav, sr)
    crest_db = _crest_factor_db(wav)

    # 把每个量映射到 [0, 1]
    # dBFS: -50 → 0, -25 → 1
    dbfs_s = np.clip((dbfs + 50) / 25, 0, 1)
    # 谱质心: 800 Hz → 0, 2300 Hz → 1
    cent_s = np.clip((centroid - 800) / 1500, 0, 1)
    # 谱平坦度: 0.4 → 0, 0.05 → 1 (越低越好)
    flat_s = np.clip(1 - (flatness - 0.05) / 0.35, 0, 1)
    # 峰均比: 3 dB → 0, 12 dB → 1
    crest_s = np.clip((crest_db - 3) / 9, 0, 1)

    # 综合加权 (谱质心 + 平坦度比 dBFS 更可靠, 因为 dBFS 受录音电平影响)
    score = 0.20 * dbfs_s + 0.30 * cent_s + 0.30 * flat_s + 0.20 * crest_s

    return float(score), {
        "dbfs": round(dbfs, 1),
        "centroid_hz": round(centroid, 0),
        "flatness": round(flatness, 3),
        "rolloff_hz": round(rolloff, 0),
        "crest_db": round(crest_db, 1),
        "components": {
            "dbfs_s": round(dbfs_s, 2),
            "cent_s": round(cent_s, 2),
            "flat_s": round(flat_s, 2),
            "crest_s": round(crest_s, 2),
        },
    }


def _get_dnsmos():
    """懒加载 DNSMOS ONNX 模型 (Microsoft 开源, 单 wav < 1 GFLOPs)"""
    global _dnsmos_model
    if _dnsmos_model is None:
        try:
            import onnxruntime as ort
            # 假设你预先下载了 sig_bak_ovr.onnx 到 ./pretrained/dnsmos/
            # 模型 URL: https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS
            model_path = os.environ.get(
                "DNSMOS_MODEL",
                "./pretrained/dnsmos/sig_bak_ovr.onnx",
            )
            if not os.path.exists(model_path):
                print(f"[dnsmos] 模型未找到: {model_path}, 跳过 (用启发式打分)")
                _dnsmos_model = False
                return None
            _dnsmos_model = ort.InferenceSession(model_path)
            print("[dnsmos] 加载成功")
        except Exception as e:
            print(f"[dnsmos] 加载失败 ({e}), 跳过")
            _dnsmos_model = False
    return _dnsmos_model if _dnsmos_model is not False else None


def _dnsmos_score(wav: np.ndarray, sr: int) -> dict:
    """跑 DNSMOS, 返回 {sig, bak, ovr, p808_mos}"""
    model = _get_dnsmos()
    if model is None:
        return None
    # DNSMOS 要 16k mono float; 输入要切到 9s 窗 (官方推荐)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    # 切 9s 窗求平均
    win = sr * 9
    if len(wav) < sr * 2:   # < 2s 不可靠
        return None
    scores = []
    for i in range(0, max(1, len(wav) - win + 1), win):
        clip = wav[i:i+win]
        if len(clip) < sr * 2:
            break
        inp = clip[np.newaxis, :].astype(np.float32)
        out = model.run(None, {model.get_inputs()[0].name: inp})
        # 输出: [SIG, BAK, OVR] 或 [SIG, BAK, OVR, P808_MOS]
        scores.append(out[0][0] if out[0].ndim > 1 else out[0])
    if not scores:
        return None
    avg = np.mean(scores, axis=0)
    return {
        "sig": float(avg[0]),
        "bak": float(avg[1]),
        "ovr": float(avg[2]),
        "p808": float(avg[3]) if len(avg) > 3 else None,
    }


def estimate_quality(wav: np.ndarray, sr: int = 16000, use_dnsmos: bool = False) -> tuple:
    """
    估算音频质量 [0, 1]. 越高越像近场清晰录音.

    返回 (score, detail).
    detail 包含各原始声学特征, 用于调试.

    use_dnsmos=True 时如果 DNSMOS 可用, 用它的 OVR 映射 + 启发式做加权.
    """
    score, detail = _heuristic_score(wav, sr)

    if use_dnsmos:
        d = _dnsmos_score(wav, sr)
        if d is not None:
            # OVR 1~5, 映射到 [0, 1]
            dnsmos_s = np.clip((d["ovr"] - 1.5) / 3.0, 0, 1)
            # 半启发式半 DNSMOS
            score = 0.5 * score + 0.5 * dnsmos_s
            detail["dnsmos"] = d

    return score, detail


def classify(score: float) -> str:
    """方便阅读的分级"""
    if score >= 0.6:
        return "near"
    if score >= 0.4:
        return "medium"
    return "far"


if __name__ == "__main__":
    import argparse
    import soundfile as sf
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--dnsmos", action="store_true")
    ap.add_argument("--window", type=float, default=5.0, help="按 N 秒窗扫描整段音频, 看分布")
    args = ap.parse_args()

    wav, sr = sf.read(args.wav, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000

    print(f"=== 整段质量 ===")
    score, det = estimate_quality(wav, sr, use_dnsmos=args.dnsmos)
    print(f"  score = {score:.3f} ({classify(score)})")
    for k, v in det.items():
        print(f"  {k}: {v}")

    print(f"\n=== 按 {args.window}s 窗扫描 ===")
    win = int(args.window * sr)
    for i in range(0, len(wav) - win, win):
        clip = wav[i:i+win]
        s, _ = estimate_quality(clip, sr, use_dnsmos=args.dnsmos)
        bar = "█" * int(s * 30)
        print(f"  {i/sr:7.1f}s  score={s:.2f}  {classify(s):6s}  {bar}")
