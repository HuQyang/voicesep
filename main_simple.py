"""
极简版 ASR 转写 (不做说话人识别)

适合: 只关心"说了什么"而不关心"谁说的"; 单一发言人为主的录音;
      或者多人会议想要快速出可读纪要交给下游 LLM 自己分人.

管线:
  音频
   → [1] FSMN-VAD 切段 + 合并
   → [2] 段合并到 ~25s (给 FireRedASR 充足上下文)
   → [3] FireRedASR (anti-hallu) + CT-Punc + ITN + corrections.json 纠错
   → [4] 按静音长度分段落 (没有 speaker, 只按时间空隙)

用法:
  python main_simple.py --wav meeting.mp3
  python main_simple.py --wav meeting.mp3 --para-gap-ms 1500 --para-max-chars 800
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

# 复用 main_firered 的 ASR + 反幻觉 + 段块合并
from main_firered import (
    firered_transcribe_files,
    apply_punc,
    merge_segments_for_asr,
    has_hallucination,
)
# 复用 main_pipeline 的 VAD/ITN/纠错/工具
from main_pipeline import (
    SR,
    _free_gpu, _release_model,
    run_vad, _clean_text,
    apply_itn,
    asr_wav as paraformer_asr_wav,
    load_corrections, load_hotwords, apply_corrections,
    _fmt_time, _fmt_time_range,
)
from main_diarization import merge_segments
import main_pipeline
import main_bss


def group_paragraphs_by_gap(sentences, max_gap_ms: int = 2000, max_chars: int = 1500):
    """
    按相邻段时间间隙 + 字数上限分段落, 不考虑说话人.
    - 静音 > max_gap_ms: 新起一段 (会议里的自然停顿)
    - 当前段已 >= max_chars: 强制新起一段, 避免过长
    """
    if not sentences:
        return []
    paragraphs = [{"start": sentences[0]["start"],
                   "end": sentences[0]["end"],
                   "text": sentences[0]["text"]}]
    for s in sentences[1:]:
        last = paragraphs[-1]
        gap = s["start"] - last["end"]
        too_long = len(last["text"]) >= max_chars
        if gap > max_gap_ms or too_long:
            paragraphs.append({
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
            })
        else:
            last["end"] = s["end"]
            last["text"] += s["text"]
    return paragraphs


def main():
    ap = argparse.ArgumentParser()
    file_nm = "车辆管理业务研讨"
    ap.add_argument("--wav",default=f"data/{file_nm}.mp3", help="输入音频")

    # FireRedASR
    ap.add_argument("--model-dir", default="pretrained/FireRedASR-AED-L")
    ap.add_argument("--variant", choices=["aed", "llm"], default="aed")
    ap.add_argument("--firered-batch", type=int, default=1)
    ap.add_argument("--beam-size", type=int, default=3)
    ap.add_argument("--firered-max-seg", type=float, default=30.0)
    ap.add_argument("--asr-merge-target-s", type=float, default=25.0,
                    help="ASR 输入合并目标长度(s), 接近 FireRedASR 训练 30s 上限")

    # 后处理
    ap.add_argument("--hotword", default="", help="附加热词 (与 corrections.json hotwords 合并)")
    ap.add_argument("--itn", action=argparse.BooleanOptionalAction, default=True,
                    help="中文数字 → 阿拉伯数字")
    ap.add_argument("--wetext-itn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--anti-hallu", action=argparse.BooleanOptionalAction, default=True,
                    help="FireRed 输出命中黑词 → 用 Paraformer 重转该段")

    # VAD
    ap.add_argument("--vad", choices=["fsmn", "silero"], default="fsmn")
    ap.add_argument("--vad-fsmn-max-seg-ms", type=int, default=60000)
    ap.add_argument("--vad-fsmn-end-sil-ms", type=int, default=800)
    ap.add_argument("--merge-gap", type=int, default=300,
                    help="VAD 后相邻段间隙 < 此值(ms) 合并")
    ap.add_argument("--min-dur", type=int, default=800,
                    help="合并后短于此值(ms) 的段丢弃")

    # 段落
    ap.add_argument("--para-gap-ms", type=int, default=2000,
                    help="静音 > 此值(ms) 视为新段落起点")
    ap.add_argument("--para-max-chars", type=int, default=800,
                    help="段落字数硬上限")

    # 输出
    ap.add_argument("--output", default=None, help="JSON 路径")
    ap.add_argument("--output-dir", default="result", help="TXT 输出目录")
    ap.add_argument("--output-txt", default=f"result/{file_nm}_simple.txt")
    args = ap.parse_args()

    if not os.path.exists(args.wav):
        sys.exit(f"输入音频不存在: {args.wav}")

    # ─── 1. 加载 ───
    print(f"\n=== [1/3] 加载 {args.wav} ===")
    wav, _ = librosa.load(args.wav, sr=SR, mono=True)
    dur_s = len(wav) / SR
    print(f"  时长 {dur_s:.1f}s ({dur_s/60:.1f} min)")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, SR)
        clean_path = tmp.name

    try:
        # ─── 2. VAD + 合并 ───
        print(f"\n=== [2/3] VAD + ASR 块合并 ===")
        # 兼容老版 run_vad (没有 fsmn_* kwargs), 只在新版才传
        import inspect
        _vad_kwargs = {}
        _vad_sig = inspect.signature(run_vad).parameters
        if "fsmn_max_seg_ms" in _vad_sig:
            _vad_kwargs["fsmn_max_seg_ms"] = args.vad_fsmn_max_seg_ms
        if "fsmn_end_sil_ms" in _vad_sig:
            _vad_kwargs["fsmn_end_sil_ms"] = args.vad_fsmn_end_sil_ms
        raw_segments = run_vad(clean_path, engine=args.vad, **_vad_kwargs)
        print(f"  [VAD]   {len(raw_segments)} 段")
        segments = merge_segments(raw_segments, args.merge_gap, args.min_dur)
        print(f"  [MERGE] {len(segments)} 段")
        blocks = merge_segments_for_asr(
            segments,
            target_dur_s=args.asr_merge_target_s,
            max_dur_s=args.firered_max_seg,
        )
        print(f"  [BLOCK] {len(blocks)} 个 ASR 块 "
              f"(target {args.asr_merge_target_s}s)")

        # ─── 3. ASR + 后处理 + 段落 ───
        print(f"\n=== [3/3] FireRedASR-{args.variant.upper()} + 后处理 ===")
        block_records = []
        tmp_paths = []
        for bs, be, _subs in blocks:
            s_idx = int(bs / 1000 * SR)
            e_idx = int(be / 1000 * SR)
            seg = wav[s_idx:e_idx]
            if len(seg) < int(SR * 0.3):
                continue
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(f.name, seg, SR)
            f.close()
            block_records.append((bs, be, f.name))
            tmp_paths.append(f.name)

        print(f"  落盘 {len(block_records)} 块, ASR 中...")
        try:
            texts = firered_transcribe_files([r[2] for r in block_records], args)
        finally:
            for p in tmp_paths:
                try: os.unlink(p)
                except OSError: pass

        # 反幻觉
        effective_hotword = " ".join(filter(None, [args.hotword, load_hotwords()])).strip()
        if args.anti_hallu:
            n_retry = 0
            for i, ((bs, be, _), text) in enumerate(zip(block_records, texts)):
                hit = has_hallucination(text)
                if not hit:
                    continue
                seg = wav[int(bs/1000*SR):int(be/1000*SR)]
                try:
                    with torch.no_grad():
                        para_text = paraformer_asr_wav(seg, sr=SR, hotword=effective_hotword)
                except Exception as e:
                    print(f"  [anti-hallu-fail] {bs/1000:.1f}-{be/1000:.1f}s: {e}")
                    para_text = ""
                if para_text and not has_hallucination(para_text):
                    print(f"  [ANTI-HALLU] {bs/1000:7.2f}-{be/1000:7.2f}s '{hit}' → Paraformer 重转")
                    texts[i] = para_text
                    n_retry += 1
            if n_retry:
                print(f"  [ANTI-HALLU] 共重转 {n_retry} 块")
            _free_gpu()

        # Punc + ITN + 纠错
        print(f"  [PUNC] 应用 CT-Punc + ITN + 纠错...")
        corrections = load_corrections()
        sentences = []
        for (bs, be, _), text in zip(block_records, texts):
            if not text.strip():
                continue
            text = apply_punc(text)
            if args.itn:
                text = apply_itn(text, use_wetext=args.wetext_itn)
            if corrections:
                text = apply_corrections(text, corrections)
            sentences.append({"start": bs, "end": be, "text": text})

        # 释放显存
        _release_model(sys.modules.get("main_firered"), "_firered")
        _release_model(main_pipeline, "_asr")
        _free_gpu("after asr")

        # 段落划分
        paragraphs = group_paragraphs_by_gap(
            sentences,
            max_gap_ms=args.para_gap_ms,
            max_chars=args.para_max_chars,
        )
        print(f"  [PARA] {len(sentences)} 句 → {len(paragraphs)} 段")

        # ─── 输出 ───
        print(f"\n=== 转写结果 ({len(paragraphs)} 段) ===")
        for p in paragraphs:
            print(f"\n[{_fmt_time(p['start'])}-{_fmt_time(p['end'])}]")
            print(p["text"])

        if args.output:
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump([
                    {"start": round(p["start"]/1000, 2),
                     "end": round(p["end"]/1000, 2),
                     "text": p["text"]}
                    for p in paragraphs
                ], f, ensure_ascii=False, indent=2)
            print(f"\n[saved json] {os.path.abspath(args.output)}")

        if args.output_txt:
            os.makedirs(args.output_dir, exist_ok=True)
            
            txt_file = args.output_txt
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(f"{os.path.basename(args.wav)}\n\n")
                for p in paragraphs:
                    f.write(f"[{_fmt_time(p['start'])}-{_fmt_time(p['end'])}]\n")
                    f.write(f"{p['text']}\n\n")
            print(f"[saved txt ] {os.path.abspath(txt_file)}")


    finally:
        try: os.unlink(clean_path)
        except OSError: pass


if __name__ == "__main__":
    main()
