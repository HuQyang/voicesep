"""
端到端转写 demo - ASR 换成 FireRedASR (其他阶段复用 main_pipeline 的模块)

管线:
  音频
   → [1] FRCRN 降噪              (可关 --no-denoise)
   → [2] FSMN-VAD 切段
   → [3] 声纹聚类 → spk turn
   → [4] BSS 重叠检测            (可关 --no-bss)
   → [5] FireRedASR (AED-L 或 LLM-L) + CT-Punc + ITN
   → [6] 按时间戳染 spk + 段落化

前置准备:
  1) 克隆 FireRedASR 仓库, 让 Python 找到 fireredasr 包:
       git clone https://github.com/FireRedTeam/FireRedASR.git
       cd FireRedASR && pip install -e .
       # 或者 export PYTHONPATH=/path/to/FireRedASR:$PYTHONPATH
  2) 下载模型权重 (3090 推荐 AED-L, 1.1B):
       huggingface-cli download FireRedTeam/FireRedASR-AED-L \
           --local-dir ./pretrained/FireRedASR-AED-L
     (LLM-L 8.3B 也可以, 但显存吃紧)

用法:
  python main_firered.py --wav data/xxx.mp3 \
      --model-dir ./pretrained/FireRedASR-AED-L \
      --output-txt result_firered.txt

  python main_firered.py --wav xxx --variant llm \
      --model-dir ./pretrained/FireRedASR-LLM-L
"""
import argparse
import gc
import json
import os
import re
import sys
import tempfile
from collections import Counter

import numpy as np
import librosa
import soundfile as sf
import torch
from funasr import AutoModel

# 复用 main_pipeline.py 里所有非 ASR 的工具
import main_pipeline   # 用于按需访问 _asr / asr_wav (Paraformer)
from main_pipeline import (
    SR,
    _free_gpu, _release_model,
    run_vad, _clean_text,
    apply_itn,
    asr_wav as paraformer_asr_wav,   # 重叠区用 Paraformer (BSS 输出鲁棒性更好)
    assign_speaker, group_paragraphs, merge_consecutive_same_spk, _fmt_time, _fmt_time_range,
    load_corrections, load_hotwords, apply_corrections,
    extract_chunk_embs, cluster_embs, compute_centroids, match_centroid,
    segments_to_turns, chunks_to_fine_turns,
)
import main_bss
from main_bss import run_denoise, run_bss, check_bss_output
from main_diarization import merge_segments, chunk_long_segments
from speaker_db import extract_embedding_from_wave, SpeakerDB
from scd import split_segments_by_scd
from main_pipeline import _dump_debug, merge_consecutive_same_spk as _merge_consecutive_same_spk_v2


# ─────────── FireRedASR 后端 ───────────
_firered = None
_punc = None


def get_firered(model_dir: str, variant: str = "aed"):
    """懒加载 FireRedASR-AED-L 或 LLM-L"""
    global _firered
    if _firered is None:
        try:
            from fireredasr.models.fireredasr import FireRedAsr
        except ImportError as e:
            print("[!] 找不到 fireredasr 包. 先按文件头说明克隆并 pip install -e .")
            raise
        print(f"[asr] 加载 FireRedASR-{variant.upper()}: {model_dir}")
        _firered = FireRedAsr.from_pretrained(variant, model_dir)
    return _firered


def _firered_args(variant: str, beam_size: int = 3) -> dict:
    """变体专属推理参数 (与 FireRedASR README 一致)"""
    if variant == "aed":
        return {
            "use_gpu": 1 if torch.cuda.is_available() else 0,
            "beam_size": beam_size,
            "nbest": 1,
            "decode_max_len": 0,
            "softmax_smoothing": 1.25,
            "aed_length_penalty": 0.6,
            "eos_penalty": 1.0,
        }
    # LLM 变体
    return {
        "use_gpu": 1 if torch.cuda.is_available() else 0,
        "beam_size": beam_size,
        "decode_max_len": 0,
        "decode_min_len": 0,
        "repetition_penalty": 3.0,
        "llm_length_penalty": 1.0,
        "temperature": 1.0,
    }


def firered_transcribe_files(wav_paths, args) -> list:
    """
    批量转写一组 wav 文件路径, 返回每条的纯文本 (清理空格, 未加标点).
    """
    if not wav_paths:
        return []
    model = get_firered(args.model_dir, args.variant)
    cfg = _firered_args(args.variant, args.beam_size)
    out = [""] * len(wav_paths)
    BATCH = args.firered_batch
    for i in range(0, len(wav_paths), BATCH):
        batch = wav_paths[i:i+BATCH]
        uttids = [f"utt_{i+j}" for j in range(len(batch))]
        try:
            with torch.no_grad():
                # FireRedASR README 用法: (batch_uttid, batch_wav_path, args_dict)
                results = model.transcribe(uttids, batch, cfg)
            for j, r in enumerate(results):
                text = r.get("text", "") if isinstance(r, dict) else str(r)
                out[i+j] = _clean_text(text)
        except Exception as e:
            print(f"  [firered-fail] batch {i//BATCH}: {e}")
        _free_gpu()
    return out


# ─────────── CT-Punc (FireRedASR 自己不带标点) ───────────
def get_punc():
    global _punc
    if _punc is None:
        print("[punc] 加载 CT-Punc...")
        _punc = AutoModel(model="ct-punc", disable_update=True)
    return _punc


def apply_punc(text: str) -> str:
    if not text.strip():
        return text
    try:
        res = get_punc().generate(input=text)
        if res and res[0].get("text"):
            return res[0]["text"]
    except Exception as e:
        print(f"  [punc-fail] {e}")
    return text


# ─────────── 反幻觉黑词表 ───────────
# FireRedASR 在低 SNR 或 OOD 音频上倾向用训练语料 (小红书系) 高频日常词凑句子.
# 命中黑词的句子 → 视为幻觉, 用 Paraformer 重转该段.
# 这里只放"业务会议几乎不可能出现"的高风险词, 避免误杀.
HALLU_BLACKLIST = {
    # 称谓/情感 (业务场景不会出现)
    "宝宝", "老婆", "老公", "媳妇", "亲爱的", "想你了", "爱你",
    "好可爱", "哈哈哈哈", "嘻嘻", "么么哒",
    # 娱乐/游戏/平台
    "火鸡面", "王者荣耀", "和平精英", "原神", "抖音", "小红书",
    "微博", "B站", "网易云",
    # 日常生活 (会议场景出现概率低)
    "睡觉", "睡饱", "想睡觉", "起床", "刷牙",
    "吃饭了吗", "今天吃啥", "好吃", "好饿",
    # 常见幻觉短语
    "字幕由社区提供", "感谢观看", "请订阅",
}


def has_hallucination(text: str, extra: set = None) -> str:
    """命中返回触发词, 否则返回空串."""
    pool = HALLU_BLACKLIST | (extra or set())
    for w in pool:
        if w in text:
            return w
    return ""


def has_repetition(text: str,
                   min_pattern_len: int = 3,
                   max_pattern_len: int = 10,
                   min_repeats: int = 4) -> str:
    """
    检测短串机械重复 (FireRedASR-AED 在弱信号长段的典型幻觉模式).
    例: "这个里面这个里面...这个里面" × 21 次.

    规则:
      - pattern 长度 [3, 10] 字 (太短易误伤口语强调, 太长不像幻觉)
      - 连续出现 >= 4 次才报警
      - 单字符 pattern (如 "对对对") 不算 (正常口语)

    返回: 命中描述字符串, 未命中空串.
    """
    for n in range(min_pattern_len, max_pattern_len + 1):
        for m in re.finditer(rf"(.{{{n}}})\1{{{min_repeats - 1},}}", text):
            pattern = m.group(1)
            # 单字符重复 (如 "对对对", "提提提") 是正常口语
            if len(set(pattern)) < 2:
                continue
            n_reps = len(m.group(0)) // n
            return f"重复 {pattern!r} × {n_reps}"
    return ""


def has_anomaly(text: str, extra_blacklist: set = None) -> str:
    """统一异常检测: 黑词幻觉 OR 短串重复幻觉, 任一命中即返回触发描述."""
    return has_hallucination(text, extra_blacklist) or has_repetition(text)


# ─────────── ASR 输入合并 + 文本时间戳切回 ───────────
def merge_segments_for_asr(segments, target_dur_s: float = 25.0, max_dur_s: float = 30.0):
    """
    把相邻 VAD/SCD 段合并到接近 target_dur_s 的"ASR 块", 提高 FireRedASR 上下文长度.
    单段已超 max_dur_s 的不动.
    返回: [(block_start_ms, block_end_ms, [(sub_start, sub_end), ...]), ...]
    """
    if not segments:
        return []
    target_ms = int(target_dur_s * 1000)
    max_ms = int(max_dur_s * 1000)
    blocks = []
    cur = None  # [start_ms, end_ms, [subs]]

    for s, e in segments:
        seg_dur = e - s
        if seg_dur > max_ms:
            if cur:
                blocks.append((cur[0], cur[1], cur[2]))
                cur = None
            blocks.append((s, e, [(s, e)]))
            continue

        if cur is None:
            cur = [s, e, [(s, e)]]
            continue

        if (e - cur[0]) <= max_ms:
            cur[1] = e
            cur[2].append((s, e))
            if (cur[1] - cur[0]) >= target_ms:
                blocks.append((cur[0], cur[1], cur[2]))
                cur = None
        else:
            blocks.append((cur[0], cur[1], cur[2]))
            cur = [s, e, [(s, e)]]

    if cur:
        blocks.append((cur[0], cur[1], cur[2]))
    return blocks


def _dominant_speaker_for_range(ss_ms: int, se_ms: int, turns) -> int:
    """[ss, se] 范围内重叠时长最长的 spk"""
    best_spk, best_ov = None, 0
    for ts, te, spk in turns:
        ov = max(0, min(se_ms, te) - max(ss_ms, ts))
        if ov > best_ov:
            best_ov, best_spk = ov, spk
    return best_spk


_SENT_PUNC = "。！？!?"
_SOFT_PUNC = "，,；;、"


def _find_nearest_punc(text: str, target_pos: int, window: int = 15) -> int:
    """在 target_pos 附近找标点, 优先句末 (。！？), 退而求其次 (，；、). 切点是标点 *后面* 的位置."""
    n = len(text)
    for offset in range(window + 1):
        for p in (target_pos + offset, target_pos - offset):
            if 0 < p <= n and text[p - 1] in _SENT_PUNC:
                return p
    for offset in range(window + 1):
        for p in (target_pos + offset, target_pos - offset):
            if 0 < p <= n and text[p - 1] in _SOFT_PUNC:
                return p
    return min(max(target_pos, 0), n)


def redistribute_text_by_speaker(text: str, sub_segments, turns):
    """
    给一个"ASR 块"的输出文本, 按 turns 时间戳和 sub_segments 的速率, 切回每个 speaker 的子段.
    返回: [{"start", "end", "text", "speaker"}, ...]

    规则:
      - 每个 sub_segment 先按主导 spk 染色
      - 相邻同 spk 的 sub 合并成一个 group
      - groups 数 ≤ 1 → 整块文字归一个 spk
      - groups 数 ≥ 2 → 按时间比例切文字, 优先在标点处下刀
    """
    if not sub_segments or not text.strip():
        return []

    # 1. 每个 sub 主导 spk
    sub_spks = [_dominant_speaker_for_range(ss, se, turns) for ss, se in sub_segments]

    # 2. 合并相邻同 spk
    groups = []
    g_start, g_end, g_spk = sub_segments[0][0], sub_segments[0][1], sub_spks[0]
    for (ss, se), spk in zip(sub_segments[1:], sub_spks[1:]):
        if spk == g_spk:
            g_end = se
        else:
            groups.append((g_start, g_end, g_spk))
            g_start, g_end, g_spk = ss, se, spk
    groups.append((g_start, g_end, g_spk))

    # 3. 单一 spk → 不切
    if len(groups) == 1:
        return [{
            "start": groups[0][0], "end": groups[0][1],
            "text": text.strip(), "speaker": groups[0][2],
        }]

    # 4. 多 spk → 按时间比例切, 标点优先
    total_dur = sum(ge - gs for gs, ge, _ in groups)
    text_len = len(text)
    pieces = []
    char_cursor = 0
    for i, (gs, ge, gspk) in enumerate(groups):
        if i == len(groups) - 1:
            piece_text = text[char_cursor:].strip()
        else:
            ratio = (ge - gs) / total_dur
            target = char_cursor + int(text_len * ratio)
            best_pos = _find_nearest_punc(text, target, window=15)
            best_pos = max(best_pos, char_cursor + 1)
            best_pos = min(best_pos, text_len)
            piece_text = text[char_cursor:best_pos].strip()
            char_cursor = best_pos
        if piece_text:
            pieces.append({
                "start": gs, "end": ge,
                "text": piece_text, "speaker": gspk,
            })
    return pieces


# ─────────── 工具 ───────────
def _dump_segments_to_tmp(wav, segments, sr=SR, min_dur_s=0.3, max_dur_s=30.0):
    """
    把每个 segment 切片存临时 wav. 超过 max_dur_s 的段按 max_dur_s 强切.
    FireRedASR-AED 训练时 input_length_max=60s, 大于这个直接 OOM.
    """
    out, tmps = [], []
    max_samples = int(max_dur_s * sr)
    for s_ms, e_ms in segments:
        s_idx = int(s_ms / 1000 * sr)
        e_idx = int(e_ms / 1000 * sr)
        # 大段强切
        cur = s_idx
        while cur < e_idx:
            nxt = min(cur + max_samples, e_idx)
            seg = wav[cur:nxt]
            if len(seg) >= sr * min_dur_s:
                f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(f.name, seg, sr)
                f.close()
                cs_ms = int(cur / sr * 1000)
                ce_ms = int(nxt / sr * 1000)
                out.append((cs_ms, ce_ms, f.name))
                tmps.append(f.name)
            cur = nxt
    return out, tmps


def main():
    ap = argparse.ArgumentParser()
    # file_nm = "2026-03-18 14_28 记录"
    # file_nm = "车辆管理业务研讨"
    file_nm = "04.21公交数据要素比赛决赛培训"
    # file_nm = "2025-09-30 15_56 记录"
    # file_nm = "钱部长数据融合沟通"
    ap.add_argument("--wav",default=f"data/{file_nm}.mp3", help="输入音频")
    ap.add_argument("--model-dir", default="pretrained/FireRedASR-AED-L", help="FireRedASR 权重目录")
    ap.add_argument("--variant", choices=["aed", "llm"], default="aed")
    ap.add_argument("--firered-batch", type=int, default=1,
                    help="一次喂 FireRedASR 的段数 (3090 24G 建议 1, 大于 1 容易 OOM)")
    ap.add_argument("--beam-size", type=int, default=3, help="解码 beam, 1=贪心更省显存")
    ap.add_argument("--firered-max-seg", type=float, default=30.0,
                    help="ASR 输入 segme dnt 长度上限(s), 超过强切. FireRedASR 训练 max=60s")
    ap.add_argument("--asr-merge", action=argparse.BooleanOptionalAction, default=True,
                    help="合并相邻短 VAD 段到 ~target_s, 喂 FireRedASR 更长上下文, "
                         "ASR 后用 turns 时间戳切回多 speaker 子段. 默认开启.")
    ap.add_argument("--asr-merge-target-s", type=float, default=25.0,
                    help="ASR 合并目标长度(s). 默认 25, 接近 FireRedASR 训练 30s 上限")
    ap.add_argument("--overlap-engine", choices=["firered", "paraformer"], default="paraformer",
                    help="重叠区分离后的两路用哪个 ASR. paraformer 在 BSS 伪影上幻觉少, 推荐")
    ap.add_argument("--overlap-min-chars", type=int, default=3,
                    help="重叠区 ASR 出来 < 此字数 视为伪影丢弃")
    ap.add_argument("--anti-hallu", action=argparse.BooleanOptionalAction, default=True,
                    help="FireRedASR 输出命中黑词 (宝宝/睡觉/王者荣耀...) → 用 Paraformer 重转该段")
    # 复用 main_pipeline 的参数
    ap.add_argument("--num-spk", type=int, default=3)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--enroll-db", default="speaker/db.npz")
    ap.add_argument("--match-threshold", type=float, default=0.55)
    ap.add_argument("--itn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wetext-itn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vad", choices=["fsmn", "silero"], default="silero")
    ap.add_argument("--no-bss", action="store_true")
    ap.add_argument("--bss-min-dur", type=float, default=2.0)
    ap.add_argument("--bss-max-dur", type=float, default=30.0)
    ap.add_argument("--bss-dump-dir", default=None,
                    help="BSS 检测到重叠时落盘 (原始混合 + 分离两路 wav), 展示用")
    ap.add_argument("--vad-show-n", type=int, default=20,
                    help="VAD 阶段打印前 N 个段 (0=全打印, -1=不打印)")
    ap.add_argument("--diar-mode", choices=["segment", "chunk"], default="chunk",
                    help="diar 模式: segment=段内投票(传统稳); chunk=每个 chunk 投票(能 catch 快速轮替)")
    ap.add_argument("--diar-smooth", action=argparse.BooleanOptionalAction, default=True,
                    help="chunk 模式时是否平滑孤立点 (X Y X → X X X)")
    ap.add_argument("--min-dbfs", type=float, default=-60.0)
    ap.add_argument("--merge-gap", type=int, default=300)
    ap.add_argument("--min-dur", type=int, default=800)
    ap.add_argument("--chunk-max", type=int, default=2000)
    ap.add_argument("--chunk-hop", type=int, default=1000)
    ap.add_argument("--output", default=f"result/{file_nm}_firered.json")
    ap.add_argument("--output-dir", default="result")
    ap.add_argument("--output-txt", default=f"result/{file_nm}_firered.txt")
    ap.add_argument("--para-gap", type=int, default=800)
    ap.add_argument("--para-max-dur", type=int, default=60000)
    ap.add_argument("--para-max-chars", type=int, default=600)
    ap.add_argument("--post-merge-gap", type=int, default=5000,
                    help="后合并: 同 spk + 都非 overlap, 间隔(ms)<=此值则合并")
    ap.add_argument("--post-merge-max-dur", type=int, default=120000,
                    help="后合并硬上限: 合并后段最大时长(ms), 默认 3 分钟")
    ap.add_argument("--post-merge-max-chars", type=int, default=1500,
                    help="后合并硬上限: 合并后段最大字符数, 默认 1500")
    # SCD (Speaker Change Detection)
    ap.add_argument("--scd", default=True, action="store_true",
                    help="VAD 后追加 SCD: 滑窗 cam++ 距离检测说话人切换点")
    ap.add_argument("--scd-min-seg-s", type=float, default=5.0)
    ap.add_argument("--scd-window-s", type=float, default=0.75)
    ap.add_argument("--scd-hop-s", type=float, default=0.1)
    ap.add_argument("--scd-threshold", type=float, default=0.5)
    ap.add_argument("--scd-min-spk-dur-s", type=float, default=0.8)
    ap.add_argument("--whiten", default=False, action="store_true",
                    help="聚类前对所有 embedding 减全局均值, 移除房间/通道共同分量. "
                         "远场/多男声场景必开.")
    ap.add_argument("--embedder-model", default="iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
                    help="声纹模型 (覆盖默认 ERes2NetV2). 推荐: "
                         "iic/speech_eres2net_sv_zh-cn_3dspeaker_16k (远场强); "
                         "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common (最强, 512-d)")
    ap.add_argument("--debug-dir", default=None,
                    help="若指定, 每个阶段 dump JSON 到此目录")
    args = ap.parse_args()

    # 在任何 embedding 调用之前切换模型
    if args.embedder_model:
        from speaker_db import set_embedder_model
        set_embedder_model(args.embedder_model)

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
        # ─── 3. VAD + Diar ───
        print(f"\n=== [3/6] VAD ({args.vad}) + 声纹聚类 ===")
        raw_segments = run_vad(clean_path, engine=args.vad)
        print(f"  [VAD]   {len(raw_segments)} segments")
        show_n = args.vad_show_n if args.vad_show_n > 0 else len(raw_segments)
        for i, (s_ms, e_ms) in enumerate(raw_segments[:show_n]):
            print(f"    seg{i:03d}: {_fmt_time_range(s_ms, e_ms)}  ({(e_ms-s_ms)/1000:.2f}s)")
        if args.vad_show_n > 0 and len(raw_segments) > args.vad_show_n:
            print(f"    ... 还有 {len(raw_segments) - args.vad_show_n} 段未列出")
        segments = merge_segments(raw_segments, args.merge_gap, args.min_dur)
        print(f"  [MERGE] {len(segments)} segments (gap<{args.merge_gap}ms 合并, <{args.min_dur}ms 丢弃)")

        # SCD: 长段按 embedding 距离切说话人切换点
        if args.scd:
            print(f"  [SCD]   开始检测 (window={args.scd_window_s}s, hop={args.scd_hop_s}s, "
                  f"threshold={args.scd_threshold}, 只处理 > {args.scd_min_seg_s}s 的段)")
            segments, scd_stats = split_segments_by_scd(
                wav, segments,
                min_segment_for_scd_s=args.scd_min_seg_s,
                window_s=args.scd_window_s,
                hop_s=args.scd_hop_s,
                distance_threshold=args.scd_threshold,
                min_speaker_dur_s=args.scd_min_spk_dur_s,
                sr=SR,
                verbose=True,
            )
            print(f"  [SCD]   {scd_stats['n_segments_in']} → {scd_stats['n_segments_out']} 段 "
                  f"(共 {scd_stats['n_scd_run']} 段被分析, 找到 {scd_stats['n_total_change_points']} 个切点)")
            if scd_stats["per_segment"]:
                d_maxs = [s["distance_max"] for s in scd_stats["per_segment"] if s.get("distance_max") is not None]
                d_means = [s["distance_mean"] for s in scd_stats["per_segment"] if s.get("distance_mean") is not None]
                if d_maxs:
                    print(f"  [SCD]   距离分布: max 中位 {sorted(d_maxs)[len(d_maxs)//2]:.3f} / "
                          f"mean 中位 {sorted(d_means)[len(d_means)//2]:.3f} "
                          f"(当前阈值 {args.scd_threshold:.2f}; 找到 0 切点请把阈值调到约 max 中位的 80%)")
            _dump_debug(args.debug_dir, "02b_scd.json", scd_stats)

        chunks = chunk_long_segments(segments, args.chunk_max, args.chunk_hop)
        print(f"  [CHUNK] {len(chunks)} chunks (>{args.chunk_max}ms 按 {args.chunk_hop}ms hop 切)")
        embs, kept_chunks = extract_chunk_embs(wav, chunks, min_dbfs=args.min_dbfs)
        print(f"  [EMB]   {len(embs)} embeddings")
        if len(embs) < 2:
            print("  embedding 不足 2 个, 退出")
            return
        labels = cluster_embs(embs, num_spk=args.num_spk, threshold=args.threshold,
                              whiten=args.whiten)
        centroids = compute_centroids(embs, labels)
        print(f"  [CLUS]  {len(centroids)} 个说话人簇: {sorted(centroids.keys())}")
        if args.diar_mode == "chunk":
            turns = chunks_to_fine_turns(kept_chunks, labels, smooth=args.diar_smooth)
            print(f"  [TURNS] {len(turns)} turns (chunk-level, smooth={args.diar_smooth})")
        else:
            turns = segments_to_turns(segments, chunks, kept_chunks, labels)
            print(f"  [TURNS] {len(turns)} turns (segment-vote)")

        # 关键诊断: 每个 spk 的总时长 (一眼看出是否坍缩)
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
            "spk_count": len(centroids),
            "spk_total_duration_s": {f"spk_{spk}": round(dur/1000, 1)
                                     for spk, dur in spk_dur_ms.most_common()},
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

                    # 落盘: 原始混合 + 分离两路 (展示用)
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

        # ─── 5. FireRedASR ───
        print(f"\n=== [5/6] FireRedASR-{args.variant.upper()} ===")

        # 决定 ASR 输入单元: 合并模式拼到 ~target_s, 否则一个 VAD 段一块
        if args.asr_merge:
            asr_blocks = merge_segments_for_asr(
                segments,
                target_dur_s=args.asr_merge_target_s,
                max_dur_s=args.firered_max_seg,
            )
            print(f"  [ASR-MERGE] {len(segments)} VAD/SCD 段 → {len(asr_blocks)} ASR 块 "
                  f"(target {args.asr_merge_target_s}s, max {args.firered_max_seg}s)")
        else:
            asr_blocks = [(s, e, [(s, e)]) for s, e in segments]
            print(f"  [ASR-MERGE] off, {len(asr_blocks)} 块 = 段数")

        # 落盘每个块到临时 wav
        block_records = []  # [(bs, be, subs, tmp_path)]
        tmp_paths = []
        for bs, be, subs in asr_blocks:
            s_idx = int(bs / 1000 * SR)
            e_idx = int(be / 1000 * SR)
            seg = wav[s_idx:e_idx]
            if len(seg) < int(SR * 0.3):
                continue
            f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(f.name, seg, SR)
            f.close()
            block_records.append((bs, be, subs, f.name))
            tmp_paths.append(f.name)

        print(f"  落盘 {len(block_records)} 个 ASR 块, 喂 FireRedASR...")
        try:
            texts = firered_transcribe_files([r[3] for r in block_records], args)
        finally:
            for p in tmp_paths:
                try: os.unlink(p)
                except OSError: pass

        # 反幻觉重转 (块级)
        effective_hotword = " ".join(filter(None, [args.__dict__.get('hotword', ''), load_hotwords()])).strip()
        if args.anti_hallu:
            n_retry = 0
            for i, ((bs, be, _subs, _), text) in enumerate(zip(block_records, texts)):
                hit = has_anomaly(text)
                if not hit:
                    continue
                seg = wav[int(bs/1000*SR):int(be/1000*SR)]
                try:
                    with torch.no_grad():
                        para_text = paraformer_asr_wav(seg, sr=SR, hotword=effective_hotword)
                except Exception as e:
                    print(f"  [anti-hallu-fail] {bs/1000:.1f}-{be/1000:.1f}s: {e}")
                    para_text = ""
                if para_text and not has_anomaly(para_text):
                    print(f"  [ANTI-HALLU] {bs/1000:7.2f}-{be/1000:7.2f}s '{hit}' → Paraformer 重转")
                    texts[i] = para_text
                    n_retry += 1
            if n_retry:
                print(f"  [ANTI-HALLU] 共重转 {n_retry} 块")
            _free_gpu()

        print(f"  [PUNC] 应用 CT-Punc + ITN + 按 turns 切回 speaker ...")
        corrections = load_corrections()
        sentences = []
        n_pieces_per_block = []
        for (bs, be, subs, _), text in zip(block_records, texts):
            if not text.strip():
                continue
            text = apply_punc(text)
            if args.itn:
                text = apply_itn(text, use_wetext=args.wetext_itn)
            if corrections:
                text = apply_corrections(text, corrections)

            # 按 turns 把整块文本切回各 speaker
            pieces = redistribute_text_by_speaker(text, subs, turns)
            n_pieces_per_block.append(len(pieces))
            sentences.extend(pieces)

        n_multi = sum(1 for n in n_pieces_per_block if n > 1)
        print(f"  [SENT] {len(sentences)} 个 sentence "
              f"(其中 {n_multi} 块被按 spk 切成 ≥2 片)")

        # FireRedASR 跑完, 释放显存 (大模型必须释放)
        _release_model(sys.modules[__name__], "_firered")
        _free_gpu("after firered")

        # ─── 6. 重叠区分别 ASR + 染色 + 段落化 ───
        print(f"\n=== [6/6] 重叠区 ASR + 段落化 ===")

        # sentences 在 redistribute 时已经有 speaker, 跳过 assign_speaker
        # (传统路径走到这里仍需要 assign, 通过 sub_segments 长度判断)
        if not any("speaker" in s for s in sentences):
            sentences = assign_speaker(sentences, turns)

        for s in sentences:
            s["overlap"] = False
        if overlap_ranges:
            engine = args.overlap_engine
            print(f"  [OVERLAP-ASR] 重叠区用 {engine.upper()} ({'更稳' if engine=='paraformer' else '保持流畅'})")
            effective_hotword = " ".join(filter(None, [args.__dict__.get('hotword', ''), load_hotwords()])).strip()
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
                    # 用 Paraformer 处理 BSS 输出 (鲁棒性更好, 不容易幻觉)
                    for w in sep_wavs:
                        try:
                            with torch.no_grad():
                                t = paraformer_asr_wav(w, sr=SR, hotword=effective_hotword)
                        except Exception as e:
                            print(f"  [overlap-paraformer-fail] {e}")
                            t = ""
                        texts2.append(t)
                else:
                    # 继续用 FireRedASR
                    paths, tmp_paths = [], []
                    for w in sep_wavs:
                        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        sf.write(f.name, w, SR)
                        f.close()
                        paths.append(f.name)
                        tmp_paths.append(f.name)
                    try:
                        texts2 = firered_transcribe_files(paths, args)
                    finally:
                        for p in tmp_paths:
                            try: os.unlink(p)
                            except OSError: pass

                for idx, t in enumerate(texts2):
                    if not t.strip():
                        continue
                    # 反幻觉: 太短的过滤
                    if len(t.strip()) < args.overlap_min_chars:
                        n_dropped += 1
                        continue
                    # Paraformer 自带 punc, FireRedASR 没有, 统一过一次 CT-Punc
                    if engine == "firered":
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
            _release_model(sys.modules[__name__], "_firered")
            # 释放 Paraformer (如果加载过)
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
        paragraphs = _merge_consecutive_same_spk_v2(
            paragraphs,
            max_gap_ms=args.post_merge_gap,
            max_dur_ms=args.post_merge_max_dur,
            max_chars=args.post_merge_max_chars,
        )
        if len(paragraphs) < n_before:
            print(f"  [POST-MERGE] {n_before} → {len(paragraphs)} 段 "
                  f"(同 spk + 无 overlap + gap≤{args.post_merge_gap}ms + "
                  f"<{args.post_merge_max_dur//1000}s + <{args.post_merge_max_chars} 字)")

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

        if args.output_txt:
            os.makedirs(args.output_dir, exist_ok=True)
            # filename = os.path.splitext(os.path.basename(args.wav))[0]
            # txt_file = os.path.join(args.output_dir, f"{filename}_fire3_denoise.txt")
            txt_file = args.output_txt
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
