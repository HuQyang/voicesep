"""
端到端转写 demo - ASR 换成 Dolphin (DataoceanAI) - 其他阶段复用 main_pipeline 的模块

管线:
  音频
   → [1] FRCRN 降噪              (默认关, --denoise 开)
   → [2] FSMN/Silero VAD 切段
   → [3] ERes2NetV2 声纹聚类 → spk turn
   → [4] BSS 重叠检测            (可关 --no-bss)
   → [5] Dolphin ASR (Whisper-like 架构, 强化亚洲语言)
         + 反幻觉黑词重转 (复用 main_firered 的黑词表)
         + CT-Punc + ITN + corrections.json 纠错
   → [6] 按时间戳染 spk + 段落化

前置准备:
  方式 1 (官方仓库):
    git clone https://github.com/DataoceanAI/Dolphin.git
    cd Dolphin && pip install -e .

  方式 2 (pip, 如果发布了):
    pip install dataoceanai-dolphin

  模型权重 (首次自动下载, 也可以预下到 --cache-dir):
    base   ~140 M (最小, 快)
    small  ~372 M
    medium ~743 M
    large  ~1.5 B (最强, 显存高)

  Dolphin 支持 40+ 亚洲语言, 这里固定 zh-CN.
  Cantonese 改 --region HK; 其他参考 https://github.com/DataoceanAI/Dolphin#supported-languages

用法:
  python main_dolphin.py --wav data/xxx.mp3
  python main_dolphin.py --wav xxx.mp3 --model-name small --num-spk 5
  python main_dolphin.py --wav xxx.mp3 --output result/dolphin.json \
      --output-dir result --debug-dir result/debug_dolphin
"""
import argparse
import gc
import json
import os
import sys
import tempfile

import numpy as np
import librosa
import soundfile as sf
import torch
from funasr import AutoModel

import main_pipeline
from main_pipeline import (
    SR,
    _free_gpu, _release_model, _dump_debug,
    run_vad, _clean_text,
    apply_itn,
    asr_wav as paraformer_asr_wav,   # 反幻觉重转用
    assign_speaker, group_paragraphs, merge_consecutive_same_spk,
    _fmt_time, _fmt_time_range,
    load_corrections, load_hotwords, apply_corrections,
    extract_chunk_embs, cluster_embs, compute_centroids, match_centroid,
    segments_to_turns, chunks_to_fine_turns,
)
import main_bss
from main_bss import run_denoise, run_bss, check_bss_output
from main_diarization import merge_segments, chunk_long_segments
from speaker_db import extract_embedding_from_wave, SpeakerDB
# 复用 FireRedASR 的反幻觉黑词表 (Dolphin 同样可能在低 SNR 段产生日常口语幻觉)
from main_firered import HALLU_BLACKLIST, has_hallucination, apply_punc


# ─────────── Dolphin 后端 ───────────
_dolphin = None


def _ensure_dolphin_importable():
    """
    Dolphin 不一定 pip install 成功. 如果当前目录或上级目录有 Dolphin/ 源码,
    自动加到 sys.path. 也支持环境变量 DOLPHIN_HOME 指定路径.
    """
    try:
        import dolphin  # noqa
        return
    except ImportError:
        pass

    candidates = []
    if os.environ.get("DOLPHIN_HOME"):
        candidates.append(os.environ["DOLPHIN_HOME"])
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(here, "Dolphin"),
        os.path.join(here, "..", "Dolphin"),
        os.path.join(os.getcwd(), "Dolphin"),
    ]
    for p in candidates:
        if p and os.path.isdir(p) and os.path.isfile(os.path.join(p, "dolphin", "__init__.py")):
            sys.path.insert(0, p)
            print(f"[asr] 自动添加 Dolphin 源码路径: {p}")
            return
    # 找不到, 让上层抛 ImportError
    return


def get_dolphin(model_name: str, cache_dir: str, device: str = None):
    """懒加载 Dolphin"""
    global _dolphin
    if _dolphin is None:
        _ensure_dolphin_importable()
        try:
            import dolphin
        except ImportError:
            print("[!] 找不到 dolphin 包. 三种方式之一:")
            print("    1) pip 装: cd Dolphin && pip install -e . --no-build-isolation")
            print("    2) 把 Dolphin/ 源码放到当前项目目录下 (本脚本会自动加 sys.path)")
            print("    3) export DOLPHIN_HOME=/path/to/Dolphin")
            raise
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[asr] 加载 Dolphin-{model_name} → {dev} (cache_dir={cache_dir})")
        os.makedirs(cache_dir, exist_ok=True)
        _dolphin = dolphin.load_model(model_name, cache_dir, dev)
    return _dolphin


def dolphin_transcribe_one(model, wav_np: np.ndarray, sr: int, lang: str, region: str,
                           predict_time: bool = False) -> str:
    """
    用 Dolphin 转写一段 numpy 波形 (1D float32 mono), 返回纯文本.
    Dolphin 自带轻度标点, 但建议后面再过一次 CT-Punc 统一风格.

    Dolphin API 在不同版本有差异, 这里按优先级尝试多种调用方式:
      1. dolphin.transcribe(model, waveform, ...)        ← 推荐 (公开 API)
      2. model.transcribe(waveform, ...)
      3. model.predict(waveform, ...)
      4. model(waveform, lang_sym=..., region_sym=...)   ← 老版本
    """
    import dolphin

    if sr != 16000:
        wav_np = librosa.resample(wav_np.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    waveform = torch.from_numpy(wav_np.astype(np.float32))
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, T)

    kwargs = {"lang_sym": lang, "region_sym": region, "predict_time": predict_time}

    last_err = None
    candidates = []
    if hasattr(dolphin, "transcribe"):
        candidates.append(("dolphin.transcribe", lambda: dolphin.transcribe(model, waveform, **kwargs)))
    if hasattr(model, "transcribe"):
        candidates.append(("model.transcribe", lambda: model.transcribe(waveform, **kwargs)))
    if hasattr(model, "predict"):
        candidates.append(("model.predict", lambda: model.predict(waveform, **kwargs)))
    candidates.append(("model.__call__", lambda: model(waveform, **kwargs)))

    for name, fn in candidates:
        try:
            with torch.no_grad():
                result = fn()
            # Dolphin 返回对象通常有 .text 属性
            text = getattr(result, "text", None)
            if text is None and isinstance(result, dict):
                text = result.get("text", "")
            if text is None:
                text = str(result)
            return _clean_text(text)
        except TypeError as e:
            # 接口签名不匹配, 试下一个
            last_err = (name, e)
            continue
        except Exception as e:
            print(f"  [dolphin-fail] {name}: {e}")
            return ""

    print(f"  [dolphin-fail] 没找到兼容的 API. 最后错误: {last_err}")
    return ""


def dolphin_transcribe_segments(wav: np.ndarray, segments, args) -> list:
    """
    对所有 segment 跑 Dolphin, 返回 [(start_ms, end_ms, text), ...].
    Dolphin 单段输入限制 ~30s, 超过强切.
    """
    model = get_dolphin(args.model_name, args.cache_dir, args.device or None)
    sr = SR
    max_samples = int(args.dolphin_max_seg * sr)
    out = []
    total = len(segments)
    for idx, (s_ms, e_ms) in enumerate(segments):
        s_idx = int(s_ms / 1000 * sr)
        e_idx = int(e_ms / 1000 * sr)
        cur = s_idx
        while cur < e_idx:
            nxt = min(cur + max_samples, e_idx)
            seg = wav[cur:nxt]
            if len(seg) < sr * 0.3:
                cur = nxt
                continue
            text = dolphin_transcribe_one(
                model, seg, sr=sr,
                lang=args.lang, region=args.region,
                predict_time=False,
            )
            cs_ms = int(cur / sr * 1000)
            ce_ms = int(nxt / sr * 1000)
            out.append((cs_ms, ce_ms, text))
            cur = nxt
        if (idx + 1) % 20 == 0 or idx + 1 == total:
            print(f"  [dolphin] 进度 {idx+1}/{total}")
        _free_gpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="data/04.21公交数据要素比赛决赛培训.mp3", help="输入音频")

    # Dolphin 相关
    ap.add_argument("--model-name", default="small",
                    choices=["base", "small", "medium", "large"],
                    help="Dolphin 模型规格 (建议 small 或 medium 起步)")
    ap.add_argument("--cache-dir", default="pretrained/dolphin",
                    help="Dolphin 权重缓存目录")
    ap.add_argument("--device", default=None, help="cuda / cpu, 默认自动")
    ap.add_argument("--lang", default="zh", help="ISO 语言码, 中文 zh")
    ap.add_argument("--region", default="CN", help="区域: CN/HK/TW (zh) 等")
    ap.add_argument("--dolphin-max-seg", type=float, default=30.0,
                    help="单次喂 Dolphin 的最大时长(s), 超过强切. Dolphin 训练 max=30s")

    # 反幻觉 + 输出控制
    ap.add_argument("--anti-hallu", action=argparse.BooleanOptionalAction, default=True,
                    help="Dolphin 命中黑词 (宝宝/字幕由社区提供等) → 用 Paraformer 重转该段")
    ap.add_argument("--overlap-engine", choices=["dolphin", "paraformer"], default="paraformer",
                    help="重叠区分离后两路用哪个 ASR (paraformer 在 BSS 伪影上更稳)")
    ap.add_argument("--overlap-min-chars", type=int, default=3)

    # 与 main_pipeline 完全对齐的参数 (确保可对比)
    ap.add_argument("--num-spk", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--enroll-db", default=None)
    ap.add_argument("--match-threshold", type=float, default=0.55)
    ap.add_argument("--hotword", default="", help="附加热词 (Dolphin 不支持解码热词, 此处仅与 corrections.json hotwords 合并供反幻觉重转的 Paraformer 用)")
    ap.add_argument("--itn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wetext-itn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vad", choices=["fsmn", "silero"], default="fsmn")
    ap.add_argument("--no-bss", action="store_true")
    ap.add_argument("--bss-min-dur", type=float, default=1.0)
    ap.add_argument("--bss-max-dur", type=float, default=30.0)
    ap.add_argument("--bss-dump-dir", default=None)
    ap.add_argument("--vad-show-n", type=int, default=20)
    ap.add_argument("--diar-mode", choices=["segment", "chunk"], default="segment")
    ap.add_argument("--diar-smooth", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-dbfs", type=float, default=-50.0)
    ap.add_argument("--merge-gap", type=int, default=300)
    ap.add_argument("--min-dur", type=int, default=800)
    ap.add_argument("--chunk-max", type=int, default=3000)
    ap.add_argument("--chunk-hop", type=int, default=1500)
    ap.add_argument("--output", default=None)
    ap.add_argument("--output-dir", default="result")
    ap.add_argument("--para-gap", type=int, default=800)
    ap.add_argument("--para-max-dur", type=int, default=60_000)
    ap.add_argument("--para-max-chars", type=int, default=600)
    ap.add_argument("--post-merge-gap", type=int, default=5000)
    ap.add_argument("--post-merge-max-dur", type=int, default=180_000,
                    help="post-merge 硬上限: 合并后段最大时长(ms)")
    ap.add_argument("--post-merge-max-chars", type=int, default=1500,
                    help="post-merge 硬上限: 合并后段最大字符数")
    ap.add_argument("--debug-dir", default=None,
                    help="若指定, 每个阶段 dump 一份 JSON (vad/turns/bss/asr/final)")
    args = ap.parse_args()

    # ─── 1. Load ───
    print(f"\n=== [1/6] 加载音频 ===")
    wav, _ = librosa.load(args.wav, sr=SR, mono=True)
    dur_s = len(wav) / SR
    print(f"  {args.wav}  时长 {dur_s:.1f}s ({dur_s/60:.1f} min)")

    # ─── 2. Denoise ───
    if not args.denoise:
        print(f"\n=== [2/6] 降噪已跳过 ===")
    else:
        print(f"\n=== [2/6] FRCRN 降噪 ===")
        wav = run_denoise(wav, sr=SR)
        _release_model(main_bss, "_denoise_pipe")
        _free_gpu("after denoise")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, SR)
        clean_path = tmp.name

    try:
        # ─── 3. VAD + 声纹聚类 ───
        print(f"\n=== [3/6] VAD ({args.vad}) + 声纹聚类 ===")
        raw_segments = run_vad(clean_path, engine=args.vad)
        print(f"  [VAD]   {len(raw_segments)} segments")
        show_n = args.vad_show_n if args.vad_show_n > 0 else len(raw_segments)
        for i, (s_ms, e_ms) in enumerate(raw_segments[:show_n]):
            print(f"    seg{i:03d}: {_fmt_time_range(s_ms, e_ms)}  ({(e_ms-s_ms)/1000:.2f}s)")
        if args.vad_show_n > 0 and len(raw_segments) > args.vad_show_n:
            print(f"    ... 还有 {len(raw_segments) - args.vad_show_n} 段未列出")

        _dump_debug(args.debug_dir, "01_vad_raw.json", {
            "engine": args.vad,
            "count": len(raw_segments),
            "segments": [{"start_s": round(s/1000, 2), "end_s": round(e/1000, 2),
                          "dur_s": round((e-s)/1000, 2)}
                         for s, e in raw_segments],
        })

        segments = merge_segments(raw_segments, args.merge_gap, args.min_dur)
        print(f"  [MERGE] {len(segments)} segments")
        chunks = chunk_long_segments(segments, args.chunk_max, args.chunk_hop)
        print(f"  [CHUNK] {len(chunks)} chunks")

        _dump_debug(args.debug_dir, "02_vad_merged.json", {
            "merged_count": len(segments),
            "chunk_count": len(chunks),
            "merge_gap_ms": args.merge_gap,
            "min_dur_ms": args.min_dur,
            "merged_segments": [{"start_s": round(s/1000, 2), "end_s": round(e/1000, 2),
                                 "dur_s": round((e-s)/1000, 2)}
                                for s, e in segments],
            "chunks": [{"start_s": round(s/1000, 2), "end_s": round(e/1000, 2)}
                       for s, e in chunks],
        })

        embs, kept_chunks = extract_chunk_embs(wav, chunks, min_dbfs=args.min_dbfs)
        print(f"  [EMB]   {len(embs)} embeddings")
        if len(embs) < 2:
            print("  embedding 不足 2 个, 退出")
            return

        labels = cluster_embs(embs, num_spk=args.num_spk, threshold=args.threshold)
        centroids = compute_centroids(embs, labels)
        spk_ids = sorted(centroids.keys())
        print(f"  [CLUS]  {len(centroids)} 个说话人簇: {spk_ids}")

        if args.diar_mode == "chunk":
            turns = chunks_to_fine_turns(kept_chunks, labels, smooth=args.diar_smooth)
            print(f"  [TURNS] {len(turns)} turns (chunk-level, smooth={args.diar_smooth})")
        else:
            turns = segments_to_turns(segments, chunks, kept_chunks, labels)
            print(f"  [TURNS] {len(turns)} turns (segment-vote)")

        from collections import Counter
        spk_dur_ms = Counter()
        for _ts, _te, _spk in turns:
            spk_dur_ms[int(_spk)] += (_te - _ts)
        print(f"  [SPK-DUR] " + " | ".join(
            f"spk_{spk}: {dur/1000:.1f}s" for spk, dur in spk_dur_ms.most_common()
        ))
        _dump_debug(args.debug_dir, "03_turns.json", {
            "diar_mode": args.diar_mode,
            "num_spk_requested": args.num_spk,
            "threshold": args.threshold,
            "spk_count": len(spk_ids),
            "spk_total_duration_s": {
                f"spk_{spk}": round(dur/1000, 1)
                for spk, dur in spk_dur_ms.most_common()
            },
            "turn_count": len(turns),
            "turns": [{"start_s": round(s/1000, 2), "end_s": round(e/1000, 2),
                       "dur_s": round((e-s)/1000, 2), "speaker": f"spk_{spk}"}
                      for s, e, spk in turns],
        })

        # ─── 4. BSS 重叠检测 ───
        overlap_ranges = []
        if args.no_bss:
            print(f"\n=== [4/6] BSS 已跳过 ===")
        else:
            print(f"\n=== [4/6] BSS 重叠检测 ===")
            n_overlap = 0
            for s_ms, e_ms, spk in turns:
                dur = (e_ms - s_ms) / 1000.0
                if not (args.bss_min_dur <= dur <= args.bss_max_dur):
                    continue
                seg = wav[int(s_ms/1000*SR):int(e_ms/1000*SR)]
                try:
                    with torch.no_grad():
                        wavs = run_bss(seg, sr_in=SR)
                        chk = check_bss_output(wavs)
                except torch.cuda.OutOfMemoryError:
                    print(f"  [BSS-OOM] {s_ms/1000:.1f}-{e_ms/1000:.1f}s")
                    _free_gpu()
                    continue
                except Exception as e:
                    print(f"  [BSS-FAIL] {s_ms/1000:.1f}-{e_ms/1000:.1f}s: {e}")
                    _free_gpu()
                    continue
                if chk["mode"] == "overlap":
                    n_overlap += 1
                    with torch.no_grad():
                        spk_pair = []
                        for w in wavs:
                            emb = extract_embedding_from_wave(w, sr=SR)
                            mspk, _ = match_centroid(emb, centroids)
                            spk_pair.append(mspk)
                    overlap_ranges.append((s_ms, e_ms, spk_pair, [w.copy() for w in wavs]))
                    print(f"  [OVERLAP] {_fmt_time_range(s_ms, e_ms)} → spk_{spk_pair}")

                    if args.bss_dump_dir:
                        os.makedirs(args.bss_dump_dir, exist_ok=True)
                        stem = os.path.splitext(os.path.basename(args.wav))[0]
                        tag = f"{int(s_ms):06d}-{int(e_ms):06d}"
                        orig_path = os.path.join(args.bss_dump_dir, f"{stem}_{tag}_orig.wav")
                        sf.write(orig_path, seg, SR)
                        for i, w in enumerate(wavs):
                            sp = spk_pair[i] if spk_pair[i] is not None else "?"
                            sep_path = os.path.join(args.bss_dump_dir, f"{stem}_{tag}_spk{sp}_track{i}.wav")
                            sf.write(sep_path, w, SR)
                        print(f"    [DUMP] {orig_path}  (+ 2 tracks)")
                _free_gpu()
            print(f"  发现重叠段 {n_overlap}")
            _release_model(main_bss, "_bss_pipe")
            _free_gpu("after bss")

            _dump_debug(args.debug_dir, "04_bss.json", {
                "overlap_count": len(overlap_ranges),
                "overlap_ranges": [
                    {"start_s": round(s/1000, 2), "end_s": round(e/1000, 2),
                     "spk_pair": [(f"spk_{x}" if x is not None else None) for x in spk_pair]}
                    for s, e, spk_pair, _wavs in overlap_ranges
                ],
            })

        # ─── 5. Dolphin ASR ───
        print(f"\n=== [5/6] Dolphin-{args.model_name} (lang={args.lang}/{args.region}) ===")
        seg_outputs = dolphin_transcribe_segments(wav, segments, args)
        print(f"  完成 {len(seg_outputs)} 段")

        # 反幻觉重转 (复用 FireRedASR 路线那套黑词)
        effective_hotword = " ".join(filter(None, [args.hotword, load_hotwords()])).strip()
        if args.anti_hallu:
            n_retry = 0
            for i, (s_ms, e_ms, text) in enumerate(seg_outputs):
                hit = has_hallucination(text)
                if not hit:
                    continue
                seg = wav[int(s_ms/1000*SR):int(e_ms/1000*SR)]
                try:
                    with torch.no_grad():
                        para_text = paraformer_asr_wav(seg, sr=SR, hotword=effective_hotword)
                except Exception as e:
                    print(f"  [anti-hallu-fail] {s_ms/1000:.1f}-{e_ms/1000:.1f}s: {e}")
                    para_text = ""
                if para_text and not has_hallucination(para_text):
                    print(f"  [ANTI-HALLU] {s_ms/1000:7.2f}-{e_ms/1000:7.2f}s '{hit}' → Paraformer 重转")
                    seg_outputs[i] = (s_ms, e_ms, para_text)
                    n_retry += 1
            if n_retry:
                print(f"  [ANTI-HALLU] 共重转 {n_retry} 段")
            _free_gpu()

        # CT-Punc + ITN + 纠错
        print(f"  [PUNC] 应用 CT-Punc + ITN + 纠错...")
        corrections = load_corrections()
        sentences = []
        for s_ms, e_ms, text in seg_outputs:
            if not text.strip():
                continue
            text = apply_punc(text)
            if args.itn:
                text = apply_itn(text, use_wetext=args.wetext_itn)
            if corrections:
                text = apply_corrections(text, corrections)
            sentences.append({"start": s_ms, "end": e_ms, "text": text})
        print(f"  [SENT] {len(sentences)} 个句子")

        # 释放 Dolphin (大模型不要常驻显存)
        _release_model(sys.modules[__name__], "_dolphin")
        _free_gpu("after dolphin")

        _dump_debug(args.debug_dir, "05_asr_raw.json", {
            "model_name": args.model_name,
            "lang_region": f"{args.lang}/{args.region}",
            "hotword": effective_hotword,
            "itn_enabled": args.itn,
            "correction_count": len(corrections) if corrections else 0,
            "sentence_count": len(sentences),
            "sentences": [
                {"start_s": round(s["start"]/1000, 2),
                 "end_s": round(s["end"]/1000, 2),
                 "text": s["text"]}
                for s in sentences
            ],
        })

        # ─── 6. 重叠区 ASR + 染色 + 段落化 ───
        print(f"\n=== [6/6] 重叠区 ASR + 染色 + 段落化 ===")
        sentences = assign_speaker(sentences, turns)
        for s in sentences:
            s["overlap"] = False

        if overlap_ranges:
            engine = args.overlap_engine
            print(f"  [OVERLAP-ASR] 重叠区用 {engine.upper()}")
            n_dropped = 0
            for os_ms, oe_ms, spk_pair, sep_wavs in overlap_ranges:
                if spk_pair[0] == spk_pair[1] or None in spk_pair:
                    continue
                sentences = [
                    s for s in sentences
                    if not (os_ms <= (s["start"] + s["end"]) / 2.0 <= oe_ms)
                ]
                texts2 = []
                if engine == "paraformer":
                    for w in sep_wavs:
                        try:
                            with torch.no_grad():
                                t = paraformer_asr_wav(w, sr=SR, hotword=effective_hotword)
                        except Exception as e:
                            print(f"  [overlap-paraformer-fail] {e}")
                            t = ""
                        texts2.append(t)
                else:  # dolphin
                    model = get_dolphin(args.model_name, args.cache_dir, args.device)
                    for w in sep_wavs:
                        t = dolphin_transcribe_one(
                            model, w, sr=SR,
                            lang=args.lang, region=args.region,
                        )
                        texts2.append(t)

                for idx, t in enumerate(texts2):
                    if not t.strip():
                        continue
                    if len(t.strip()) < args.overlap_min_chars:
                        n_dropped += 1
                        continue
                    if engine == "dolphin":
                        t = apply_punc(t)
                    if args.itn:
                        t = apply_itn(t, use_wetext=args.wetext_itn)
                    if corrections:
                        t = apply_corrections(t, corrections)
                    sentences.append({
                        "start": os_ms, "end": oe_ms,
                        "text": t,
                        "speaker": spk_pair[idx],
                        "overlap": True,
                    })
                _free_gpu()
            if n_dropped:
                print(f"  [OVERLAP-DROP] 丢弃 {n_dropped} 条疑似伪影 (< {args.overlap_min_chars} 字)")
            sentences.sort(key=lambda r: (r["start"], r.get("speaker") if r.get("speaker") is not None else -1))
            _release_model(sys.modules[__name__], "_dolphin")
            _release_model(main_pipeline, "_asr")
            _free_gpu("after overlap asr")

        # 真名映射
        if args.enroll_db and os.path.exists(args.enroll_db):
            db = SpeakerDB(args.enroll_db)
            spk_names = {}
            for spk, c in centroids.items():
                name, _ = db.match(c, threshold=args.match_threshold)
                spk_names[spk] = name if name else f"spk_{spk}"
            print(f"  [ENROLL] spk → 真名: {spk_names}")
        else:
            spk_names = {spk: f"spk_{spk}" for spk in centroids}

        for s in sentences:
            s["speaker"] = spk_names.get(
                s["speaker"],
                f"spk_{s['speaker']}" if s["speaker"] is not None else "spk_unknown",
            )

        paragraphs = group_paragraphs(
            sentences,
            max_gap_ms=args.para_gap,
            max_paragraph_dur_ms=args.para_max_dur,
            max_paragraph_chars=args.para_max_chars,
        )
        n_before = len(paragraphs)
        paragraphs = merge_consecutive_same_spk(
            paragraphs,
            max_gap_ms=args.post_merge_gap,
            max_dur_ms=args.post_merge_max_dur,
            max_chars=args.post_merge_max_chars,
        )
        if len(paragraphs) < n_before:
            print(f"  [POST-MERGE] {n_before} → {len(paragraphs)} 段")

        _dump_debug(args.debug_dir, "06_final.json", {
            "paragraph_count": len(paragraphs),
            "spk_names": list(spk_names.values()),
            "paragraphs": [
                {"start_s": round(p["start"]/1000, 2),
                 "end_s": round(p["end"]/1000, 2),
                 "dur_s": round((p["end"]-p["start"])/1000, 2),
                 "speaker": p["speaker"],
                 "overlap": p.get("overlap", False),
                 "char_count": len(p["text"]),
                 "text": p["text"]}
                for p in paragraphs
            ],
        })

        # ─── 输出 ───
        def _header(p):
            tag = " [可能重叠]" if p.get("overlap") else ""
            return f"{p['speaker']} - {_fmt_time_range(p['start'], p['end'])}{tag}"

        print(f"\n=== 转写结果 ({len(paragraphs)} 段) ===")
        for p in paragraphs:
            print(f"\n{_header(p)}")
            print(p["text"])

        if args.output:
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump([
                    {"start": round(p["start"]/1000, 2),
                     "end": round(p["end"]/1000, 2),
                     "speaker": p["speaker"],
                     "overlap": p.get("overlap", False),
                     "text": p["text"]}
                    for p in paragraphs
                ], f, ensure_ascii=False, indent=2)
            print(f"\n[saved json] {os.path.abspath(args.output)}")

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            filename = os.path.splitext(os.path.basename(args.wav))[0]
            txt_file = os.path.join(args.output_dir, f"{filename}_dolphin.txt")
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(f"{os.path.basename(args.wav)}\n\n")
                for p in paragraphs:
                    f.write(f"{_header(p)}\n")
                    f.write(f"{p['text']}\n\n")
            print(f"[saved txt ] {os.path.abspath(txt_file)}")

    finally:
        try: os.unlink(clean_path)
        except OSError: pass


if __name__ == "__main__":
    main()
