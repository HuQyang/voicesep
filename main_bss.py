"""
BSS demo: Mossformer2 2-spk 盲源分离

用法:
  # 对整条音频做 BSS
  python demo_bss.py --wav test_audio.wav

  # 只对怀疑是重叠的时间段做 BSS
  python demo_bss.py --wav test_audio.wav --start 130.0 --end 135.0

  # 输出后顺手匹配回 SpeakerDB, 给两路赋 spk_id
  python demo_bss.py --wav test_audio.wav --start 130 --end 135 --enroll-db speakers/db.npz

输出:
  bss_out/<name>_spk0.wav  (16kHz)
  bss_out/<name>_spk1.wav  (16kHz)
"""
import argparse
import os
import tempfile

import numpy as np
import librosa
import soundfile as sf

SR_BSS = 8000      # Mossformer2 模型采样率
SR_OUT = 16000     # 输出 / 下游 ASR & 声纹模型采样率


_denoise_pipe = None
_bss_pipe = None


def get_denoise_model():
    """懒加载 FRCRN 降噪 (16kHz)"""
    global _denoise_pipe
    if _denoise_pipe is None:
        from modelscope.pipelines import pipeline as ms_pipeline
        from modelscope.utils.constant import Tasks
        print("[bss] 加载 FRCRN 降噪模型...")
        try:
            _denoise_pipe = ms_pipeline(
                task=Tasks.acoustic_noise_suppression,
                model="damo/speech_frcrn_ans_cirm_16k",
            )
        except Exception:
            _denoise_pipe = ms_pipeline(
                task=Tasks.acoustic_noise_suppression,
                model="iic/speech_frcrn_ans_cirm_16k",
            )
    return _denoise_pipe


def run_denoise(wav: np.ndarray, sr: int = SR_OUT) -> np.ndarray:
    """FRCRN 降噪. 输入/输出都是 16kHz float32 1D mono."""
    if sr != SR_OUT:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=SR_OUT)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, SR_OUT)
        tmp_path = tmp.name
    out_path = tmp_path.replace(".wav", "_denoised.wav")
    try:
        model = get_denoise_model()
        # FRCRN pipeline 支持 output_path 直接写文件; 也可以走内存
        result = model(tmp_path, output_path=out_path)
        if os.path.exists(out_path):
            out, _sr = sf.read(out_path, dtype="float32")
        else:
            # 兜底: 走 result 里的 pcm 字段
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


def get_bss_model():
    """懒加载 Mossformer2 2-spk BSS"""
    global _bss_pipe
    if _bss_pipe is None:
        from modelscope.pipelines import pipeline as ms_pipeline
        from modelscope.utils.constant import Tasks
        print("[bss] 加载 Mossformer2 2-spk 分离模型...")
        try:
            _bss_pipe = ms_pipeline(
                task=Tasks.speech_separation,
                model="damo/speech_mossformer2_separation_temporal_8k",
            )
        except Exception:
            # 新仓库前缀
            _bss_pipe = ms_pipeline(
                task=Tasks.speech_separation,
                model="iic/speech_mossformer2_separation_temporal_8k",
            )
    return _bss_pipe


def _to_float32_mono(w):
    """ModelScope 输出兼容: 可能是 bytes / int16 ndarray / float ndarray"""
    if isinstance(w, bytes):
        w = np.frombuffer(w, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        w = np.asarray(w)
        if w.ndim > 1:
            w = w.reshape(-1)
        if w.dtype.kind in ("i", "u"):
            w = w.astype(np.float32) / 32768.0
        else:
            w = w.astype(np.float32)
            if np.max(np.abs(w)) > 2.0:   # 兜底: float 但值域是 int 范围
                w = w / 32768.0
    return w


def run_bss(wav: np.ndarray, sr_in: int = 16000):
    """
    输入 1D mono 波形 (任意 sr), 输出 2 路独立波形 (16kHz float32)
    """
    if sr_in != SR_BSS:
        wav_8k = librosa.resample(wav.astype(np.float32), orig_sr=sr_in, target_sr=SR_BSS)
    else:
        wav_8k = wav.astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav_8k, SR_BSS)
        tmp_path = tmp.name

    try:
        model = get_bss_model()
        result = model(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # 兼容不同版本字段
    outs = (
        result.get("output_pcm_list")
        or result.get("output_pcm")
        or result.get("output_wav")
        or result.get("output")
    )
    if outs is None:
        raise RuntimeError(f"未能从 BSS 输出中找到波形, keys={list(result.keys())}")

    wavs_8k = [_to_float32_mono(w) for w in outs]
    wavs_16k = [librosa.resample(w, orig_sr=SR_BSS, target_sr=SR_OUT) for w in wavs_8k]
    return wavs_16k


def identify(wav: np.ndarray, db_path: str, threshold: float = 0.55):
    """用 ERes2NetV2 (speaker_db) 提 emb, 在 SpeakerDB 中匹配"""
    from speaker_db import extract_embedding_from_wave, SpeakerDB
    emb = extract_embedding_from_wave(wav, sr=SR_OUT)
    db = SpeakerDB(db_path)
    name, score = db.match(emb, threshold=threshold)
    return (name or "unknown"), float(score)


def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)) + 1e-10)
    return 20.0 * np.log10(rms)


_vad_model = None


def _get_vad():
    global _vad_model
    if _vad_model is None:
        from funasr import AutoModel
        print("[bss] 加载 FSMN-VAD (post-check 用)...")
        _vad_model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
    return _vad_model


def vad_speech_ratio(wav: np.ndarray, sr: int = SR_OUT) -> float:
    """跑 FSMN-VAD, 返回语音时长占比 [0, 1]. 纯噪声接近 0, 纯语音接近 1."""
    vad = _get_vad()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, sr)
        tmp_path = tmp.name
    try:
        res = vad.generate(input=tmp_path)
        segments = res[0]["value"]
        speech_ms = sum(e - s for s, e in segments)
        total_ms = len(wav) / sr * 1000.0
        return speech_ms / total_ms if total_ms > 0 else 0.0
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def check_bss_output(
    wavs,
    energy_ratio_db: float = 12.0,
    emb_sim_threshold: float = 0.80,
    min_speech_ratio_abs: float = 0.05,
    speech_ratio_gap: float = 0.10,
):
    """
    判定 BSS 两路的"含义". 远场友好版: 用相对比较而非绝对阈值.
      mode: "overlap" | "single" | "noise_one" | "all_noise"
      keep_indices: 要保留的输出索引
      detail: 各路 metric
    """
    from speaker_db import extract_embedding_from_wave

    rms = [_rms_dbfs(w) for w in wavs]
    speech_ratios = [vad_speech_ratio(w) for w in wavs]

    detail = {
        "rms_dbfs": rms,
        "speech_ratio": speech_ratios,
        "emb_sim": None,
    }

    high_sr = max(speech_ratios)
    low_sr = min(speech_ratios)
    high_idx = int(np.argmax(speech_ratios))
    low_idx = 1 - high_idx

    # 全噪声: 连最高的一路也几乎没有语音
    if high_sr < min_speech_ratio_abs:
        return {
            "mode": "all_noise",
            "keep_indices": [],
            "reason": (
                f"两路 speech_ratio 都 < {min_speech_ratio_abs} "
                f"({speech_ratios[0]:.2f}, {speech_ratios[1]:.2f})"
            ),
            **detail,
        }

    # 一路明显比另一路更像语音 → 另一路视为噪声 (远场场景常见)
    if (high_sr - low_sr) >= speech_ratio_gap:
        return {
            "mode": "noise_one",
            "keep_indices": [high_idx],
            "reason": (
                f"spk{high_idx} 语音占比 {high_sr:.2f} 显著高于 spk{low_idx} ({low_sr:.2f}), "
                f"差 {high_sr - low_sr:.2f} ≥ {speech_ratio_gap}"
            ),
            **detail,
        }

    # 两路 speech_ratio 接近, 判是否同一人
    strong_idx = int(np.argmax(rms))
    weak_idx = 1 - strong_idx
    energy_gap = rms[strong_idx] - rms[weak_idx]

    if energy_gap >= energy_ratio_db:
        return {
            "mode": "single",
            "keep_indices": [strong_idx],
            "reason": f"能量悬殊 (差 {energy_gap:.1f} dB) → 视为独白",
            **detail,
        }

    try:
        emb_a = extract_embedding_from_wave(wavs[0], sr=SR_OUT)
        emb_b = extract_embedding_from_wave(wavs[1], sr=SR_OUT)
        sim = float(
            np.dot(emb_a, emb_b)
            / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
        )
        detail["emb_sim"] = sim
    except Exception as e:
        print(f"[post-check] 声纹相似度计算失败: {e}")
        sim = None

    if sim is not None and sim >= emb_sim_threshold:
        return {
            "mode": "single",
            "keep_indices": [strong_idx],
            "reason": f"两路声纹高度相似 (cos_sim={sim:.3f}) → 视为独白",
            **detail,
        }

    return {
        "mode": "overlap",
        "keep_indices": [0, 1],
        "reason": (
            f"真重叠 (能量差 {energy_gap:.1f} dB, "
            f"emb_sim={sim if sim is None else f'{sim:.3f}'})"
        ),
        **detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="输入音频")
    ap.add_argument("--start", type=float, default=None, help="重叠段起点(s)")
    ap.add_argument("--end", type=float, default=None, help="重叠段终点(s)")
    ap.add_argument("--out-dir", default="bss_out")
    ap.add_argument("--enroll-db", default=None, help="可选: 声纹库, 给两路输出赋 spk_id")
    ap.add_argument("--threshold", type=float, default=0.55, help="SpeakerDB 匹配阈值")
    ap.add_argument("--energy-gap-db", type=float, default=12.0,
                    help="post-check: 两路能量差 >= 此值判为独白")
    ap.add_argument("--emb-sim", type=float, default=0.85,
                    help="post-check: 两路声纹相似度 >= 此值判为独白")
    ap.add_argument("--min-speech-ratio-abs", type=float, default=0.05,
                    help="post-check: 两路 speech_ratio 都 < 此值才算 all_noise")
    ap.add_argument("--speech-ratio-gap", type=float, default=0.10,
                    help="post-check: 两路 speech_ratio 差 >= 此值, 低的一路视为噪声")
    ap.add_argument("--no-postcheck", action="store_true", help="跳过 post-check")
    ap.add_argument("--denoise", action="store_true", default=True,
                    help="BSS 前用 FRCRN 降噪 (默认开)")
    ap.add_argument("--no-denoise", dest="denoise", action="store_false",
                    help="禁用降噪")
    ap.add_argument("--save-denoised", action="store_true",
                    help="把降噪后的输入也存一份, 方便对比")
    args = ap.parse_args()

    wav, _ = librosa.load(args.wav, sr=SR_OUT, mono=True)
    if args.start is not None and args.end is not None:
        s = int(args.start * SR_OUT)
        e = int(args.end * SR_OUT)
        wav = wav[s:e]
        print(f"[bss] 切出 {args.start:.2f}s -> {args.end:.2f}s, 长度 {len(wav)/SR_OUT:.2f}s")
    else:
        print(f"[bss] 处理整条音频, 长度 {len(wav)/SR_OUT:.2f}s")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.wav))[0]
    if args.start is not None:
        base = f"{base}_{args.start:.1f}-{args.end:.1f}"

    if args.denoise:
        rms_before = _rms_dbfs(wav)
        wav = run_denoise(wav, sr=SR_OUT)
        rms_after = _rms_dbfs(wav)
        print(f"[denoise] 完成, RMS dBFS: {rms_before:.1f} → {rms_after:.1f}")
        if args.save_denoised:
            dn_path = os.path.join(args.out_dir, f"{base}_denoised.wav")
            sf.write(dn_path, wav, SR_OUT)
            print(f"[denoise] 降噪后波形已保存: {dn_path}")

    wavs = run_bss(wav, sr_in=SR_OUT)
    print(f"[bss] 输出 {len(wavs)} 路")

    # post-check: 判断 BSS 输出含义
    keep_indices = list(range(len(wavs)))
    mode = "raw"
    if not args.no_postcheck and len(wavs) == 2:
        chk = check_bss_output(
            wavs,
            energy_ratio_db=args.energy_gap_db,
            emb_sim_threshold=args.emb_sim,
            min_speech_ratio_abs=args.min_speech_ratio_abs,
            speech_ratio_gap=args.speech_ratio_gap,
        )
        mode = chk["mode"]
        keep_indices = chk["keep_indices"]
        print(f"[post-check] {mode.upper()} - {chk['reason']}")
        print(f"             RMS dBFS:    spk0={chk['rms_dbfs'][0]:.1f}, spk1={chk['rms_dbfs'][1]:.1f}")
        print(f"             speech_rate: spk0={chk['speech_ratio'][0]:.2f}, spk1={chk['speech_ratio'][1]:.2f}")
        if chk["emb_sim"] is not None:
            print(f"             emb_sim:     {chk['emb_sim']:.3f}")

    if not keep_indices:
        print("  (无可保留输出)")
        return

    if mode in ("single", "noise_one") and len(keep_indices) == 1:
        i = keep_indices[0]
        out_path = os.path.join(args.out_dir, f"{base}_mono.wav")
        sf.write(out_path, wavs[i], SR_OUT)
        info = ""
        if args.enroll_db:
            name, score = identify(wavs[i], args.enroll_db, args.threshold)
            info = f"  → {name} (sim={score:.3f})"
        print(f"  mono (spk{i} of BSS): {out_path}{info}")
    else:
        for i in keep_indices:
            out_path = os.path.join(args.out_dir, f"{base}_spk{i}.wav")
            sf.write(out_path, wavs[i], SR_OUT)
            info = ""
            if args.enroll_db:
                name, score = identify(wavs[i], args.enroll_db, args.threshold)
                info = f"  → {name} (sim={score:.3f})"
            print(f"  spk{i}: {out_path}{info}")


if __name__ == "__main__":
    main()
