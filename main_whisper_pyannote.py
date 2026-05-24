"""
Whisper-large-v3 + pyannote.audio 端到端会议转写

管线:
  音频
   → [1] (可选) FRCRN 降噪
   → [2] pyannote speaker-diarization-3.1 → turns [(start_ms, end_ms, spk_id), ...]
   → [3] faster-whisper large-v3 全片转写 (word-level 时间戳)
   → [4] 按 word 中点对齐到 turn → 每句话归属一个 speaker
   → [5] (可选) SpeakerDB 把 spk_X 替换成真名 (复用 main_pipeline 的库)
   → [6] 输出 JSON + TXT (格式同 result/*_seacopara_5.json)

为什么这么搭:
  - Whisper 在长上下文下识别质量最高, 切碎了喂会丢上下文 → 选全片 + 词时间戳对齐
  - pyannote 的 diar 精度好, turn 边界自然不重叠, 比"VAD+聚类"的两阶段方案省心
  - 词中点法 (不是词的 start 也不是 end) 对快速换说话人最稳

安装:
  pip install faster-whisper "pyannote.audio>=3.1" librosa soundfile
  conda install -c conda-forge ffmpeg
  # FRCRN 降噪 (可选, 已在 main_bss.py)

HuggingFace:
  export HF_TOKEN=hf_xxx
  浏览器同意条款: pyannote/segmentation-3.0  +  pyannote/speaker-diarization-3.1

用法:
  python main_whisper_pyannote.py \\
      --wav data/钱部长数据融合沟通.mp3 \\
      --num-spk 5 \\
      --output result/钱部长_whisper.json \\
      --output-txt result/钱部长_whisper.txt

  # 用已注册的声纹库换真名
  python main_whisper_pyannote.py --wav ... --enroll-db speakers/db.npz
"""
import argparse
import gc
import json
import os
import sys
import tempfile
from typing import List, Tuple, Dict, Optional

import numpy as np
import librosa
import soundfile as sf
import torch


SR = 16000


# ─────────── 公共工具 (和 demo_pipeline_pyannote 对齐) ───────────

def _free_gpu(tag: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if tag:
            free, total = torch.cuda.mem_get_info()
            print(f"  [gpu] {tag}: 空闲 {free/1e9:.2f} / {total/1e9:.2f} GB")


def _fmt_time(ms: float) -> str:
    s = int(ms // 1000)
    return f"{s//60}:{s%60:02d}"


def _from_pretrained_compat(cls, model_id: str, token: str):
    """兼容 pyannote/hf_hub 不同版本 token 参数名"""
    try:
        return cls.from_pretrained(model_id, use_auth_token=token)
    except TypeError:
        return cls.from_pretrained(model_id, token=token)


def _get_hf_token(arg_token: str = "") -> str:
    token = arg_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not token:
        raise RuntimeError(
            "缺少 HuggingFace token. export HF_TOKEN=hf_xxx, "
            "并在 HuggingFace 网页接受 pyannote/speaker-diarization-3.1 条款"
        )
    return token


# ─────────── pyannote diarization ───────────

_diar_pipe = None


def run_diarization(
    wav_path: str,
    num_spk: Optional[int] = None,
    min_spk: Optional[int] = None,
    max_spk: Optional[int] = None,
    model_id: str = "pyannote/speaker-diarization-3.1",
    hf_token: str = "",
    device: str = None,
    merge_gap_ms: int = 250,
    min_turn_ms: int = 250,
) -> Tuple[List[Tuple[int, int, int]], Dict[str, int]]:
    """
    返回:
      turns      = [(start_ms, end_ms, spk_int_id), ...]  按时间升序, 同 spk 小间隔已合并
      label_map  = {pyannote_label_str: spk_int_id}
    """
    global _diar_pipe
    if _diar_pipe is None:
        from pyannote.audio import Pipeline
        token = _get_hf_token(hf_token)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[diar] 加载 {model_id} → {device}")
        _diar_pipe = _from_pretrained_compat(Pipeline, model_id, token)
        _diar_pipe.to(torch.device(device))

    kwargs = {}
    if num_spk and int(num_spk) > 0:
        kwargs["num_speakers"] = int(num_spk)
    else:
        if min_spk: kwargs["min_speakers"] = int(min_spk)
        if max_spk: kwargs["max_speakers"] = int(max_spk)
    print(f"[diar] kwargs = {kwargs}")

    with torch.no_grad():
        annotation = _diar_pipe(wav_path, **kwargs)

    # 把 pyannote 的 SPEAKER_00/01/... 映射成 0/1/...
    label_map: Dict[str, int] = {}
    raw_turns: List[Tuple[int, int, int]] = []
    for seg, _track, label in annotation.itertracks(yield_label=True):
        if label not in label_map:
            label_map[label] = len(label_map)
        s_ms = int(round(seg.start * 1000))
        e_ms = int(round(seg.end * 1000))
        if e_ms - s_ms < min_turn_ms:
            continue
        raw_turns.append((s_ms, e_ms, label_map[label]))
    raw_turns.sort(key=lambda x: (x[0], x[1]))

    # 同 spk 小间隔合并 (pyannote 偶尔切碎)
    turns: List[Tuple[int, int, int]] = []
    for s_ms, e_ms, spk in raw_turns:
        if turns and turns[-1][2] == spk and s_ms - turns[-1][1] <= merge_gap_ms:
            turns[-1] = (turns[-1][0], max(turns[-1][1], e_ms), spk)
        else:
            turns.append((s_ms, e_ms, spk))

    print(f"[diar] turns: raw={len(raw_turns)} → merged={len(turns)}, "
          f"speakers={len(label_map)} (label_map={label_map})")
    return turns, label_map


# ─────────── Whisper-large-v3 (faster-whisper 后端) ───────────

_whisper = None


def get_whisper(
    model_size: str = "large-v3",
    device: str = None,
    compute_type: str = None,
    cpu_threads: int = 4,
    download_root: str = None,
):
    """
    懒加载 faster-whisper. 默认 large-v3.
    - GPU: compute_type=float16 (24G+) / int8_float16 (8~12G)
    - CPU: compute_type=int8
    """
    global _whisper
    if _whisper is not None:
        return _whisper

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("缺 faster-whisper: pip install faster-whisper")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"[whisper] 加载 {model_size} ({device}, {compute_type})")
    _whisper = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        download_root=download_root,
    )
    return _whisper


def run_whisper(
    wav_path: str,
    language: str = "zh",
    beam_size: int = 5,
    initial_prompt: str = "以下是普通话会议录音。",
    vad_filter: bool = True,
    word_timestamps: bool = True,
    condition_on_previous_text: bool = True,
    **load_kwargs,
) -> List[Dict]:
    """
    全片 ASR. 返回 segments = [{
        "start": s, "end": s, "text": str,
        "words": [{"start": s, "end": s, "word": str, "prob": float}, ...]
    }, ...]
    """
    model = get_whisper(**load_kwargs)
    print(f"[whisper] 转写 {wav_path}  lang={language} beam={beam_size} vad_filter={vad_filter}")

    segments_iter, info = model.transcribe(
        wav_path,
        language=language,
        beam_size=beam_size,
        initial_prompt=initial_prompt,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        condition_on_previous_text=condition_on_previous_text,
    )
    print(f"[whisper] 检测语言 = {info.language} (p={info.language_probability:.2f}), "
          f"音频时长 = {info.duration:.1f}s")

    out = []
    n_words = 0
    for seg in segments_iter:
        words = []
        if seg.words:
            for w in seg.words:
                words.append({
                    "start": float(w.start),
                    "end":   float(w.end),
                    "word":  w.word,
                    "prob":  float(w.probability) if w.probability is not None else 0.0,
                })
                n_words += 1
        out.append({
            "start": float(seg.start),
            "end":   float(seg.end),
            "text":  seg.text.strip(),
            "words": words,
        })
    print(f"[whisper] {len(out)} segments, {n_words} words")
    return out


# ─────────── 词 → turn 对齐 ───────────

def assign_words_to_turns(
    whisper_segments: List[Dict],
    turns: List[Tuple[int, int, int]],
) -> List[Dict]:
    """
    词中点 → 落到哪个 turn 区间, 就归该 spk.
    无词时间戳的 segment 退化用 segment 中点.
    返回 sentences = [{"start": ms, "end": ms, "speaker": int, "text": str}, ...]
    每个 Whisper segment 内若 spk 变更, 自动拆成多句.
    """
    if not turns:
        # 没 diar 结果, 全部 spk_unknown
        return [{"start": int(s["start"]*1000), "end": int(s["end"]*1000),
                 "speaker": None, "text": s["text"]} for s in whisper_segments]

    # 构建 turn 查找: 把 turn 按起点排序, 二分
    turn_starts = np.array([t[0] for t in turns], dtype=np.int64)
    turn_ends   = np.array([t[1] for t in turns], dtype=np.int64)
    turn_spks   = np.array([t[2] for t in turns], dtype=np.int32)

    def spk_at(ms: float) -> int:
        """二分找 ms 落在哪个 turn"""
        idx = int(np.searchsorted(turn_starts, ms, side="right") - 1)
        if 0 <= idx < len(turn_starts) and turn_starts[idx] <= ms <= turn_ends[idx]:
            return int(turn_spks[idx])
        # 落在 turn 之间的静音: 取最近的 turn
        candidates = []
        if 0 <= idx < len(turn_starts):
            candidates.append((abs(ms - turn_ends[idx]), int(turn_spks[idx])))
        if 0 <= idx + 1 < len(turn_starts):
            candidates.append((abs(turn_starts[idx + 1] - ms), int(turn_spks[idx + 1])))
        if not candidates:
            return -1
        candidates.sort()
        return candidates[0][1]

    sentences: List[Dict] = []
    for seg in whisper_segments:
        words = seg.get("words") or []
        if not words:
            # 没词时间戳: 整 segment 当一句, 用中点判 spk
            mid_ms = (seg["start"] + seg["end"]) * 500
            sp = spk_at(mid_ms)
            sentences.append({
                "start": int(seg["start"] * 1000),
                "end":   int(seg["end"]   * 1000),
                "speaker": sp if sp >= 0 else None,
                "text": seg["text"],
            })
            continue

        # 有词时间戳: 逐词分配 spk, 同 spk 连续词拼成一句, spk 切换则开新句
        cur_words = []
        cur_spk = None
        cur_start_s = None
        for w in words:
            mid = (w["start"] + w["end"]) * 500  # ms
            sp = spk_at(mid)
            if sp < 0:
                sp = cur_spk if cur_spk is not None else 0
            if cur_spk is None:
                cur_spk = sp
                cur_start_s = w["start"]
            if sp != cur_spk:
                # 收一句
                text = "".join(x["word"] for x in cur_words).strip()
                if text:
                    sentences.append({
                        "start": int(cur_start_s * 1000),
                        "end":   int(cur_words[-1]["end"] * 1000),
                        "speaker": cur_spk,
                        "text": text,
                    })
                cur_words = []
                cur_spk = sp
                cur_start_s = w["start"]
            cur_words.append(w)
        if cur_words:
            text = "".join(x["word"] for x in cur_words).strip()
            if text:
                sentences.append({
                    "start": int(cur_start_s * 1000),
                    "end":   int(cur_words[-1]["end"] * 1000),
                    "speaker": cur_spk,
                    "text": text,
                })

    return sentences


# ─────────── 段落合并 (同 spk + 小间隔) ───────────

def merge_consecutive_same_spk(
    sentences: List[Dict],
    max_gap_ms: int = 800,
    max_dur_ms: int = 30000,
    max_chars: int = 200,
) -> List[Dict]:
    """同说话人 + 间隔小 + 不超长 → 合段, 出来即"段落"."""
    out: List[Dict] = []
    for s in sentences:
        if not out:
            out.append(dict(s)); continue
        last = out[-1]
        same_spk = last["speaker"] == s["speaker"]
        gap = s["start"] - last["end"]
        new_dur = s["end"] - last["start"]
        new_chars = len(last["text"]) + len(s["text"])
        if same_spk and gap <= max_gap_ms and new_dur <= max_dur_ms and new_chars <= max_chars:
            last["end"] = s["end"]
            # 中文无空格直接拼; 已经有标点就保留
            sep = "" if (last["text"] and last["text"][-1] in "。！？，、,.!?") else ""
            last["text"] = last["text"] + sep + s["text"]
        else:
            out.append(dict(s))
    return out


# ─────────── SpeakerDB (复用 main_pipeline 的库, 可选) ───────────

def apply_enroll_db(
    sentences: List[Dict],
    db_path: str,
    wav_path: str,
    turns: List[Tuple[int, int, int]],
    match_threshold: float = 0.45,
    embedder_model: str = None,
) -> List[Dict]:
    """
    给每个 pyannote spk_id 算 centroid (拼接该 spk 所有 turn 的音频 → emb), 然后查库换真名.
    """
    from speaker_db import SpeakerDB, extract_embedding_from_wave, set_embedder_model
    if embedder_model:
        set_embedder_model(embedder_model)
    db = SpeakerDB(db_path)
    if len(db.names) == 0:
        print(f"[enroll] 库 {db_path} 是空的, 跳过换名")
        return sentences

    print(f"[enroll] 库里 {len(db.names)} 人: {db.names}")
    wav, _ = librosa.load(wav_path, sr=SR, mono=True)
    spk_emb: Dict[int, np.ndarray] = {}
    # 按 spk 收集 turn, 拼成一条音频提 emb
    by_spk: Dict[int, List[Tuple[int, int]]] = {}
    for s, e, sp in turns:
        by_spk.setdefault(sp, []).append((s, e))

    for sp, segs in by_spk.items():
        # 取累计时长前 30s, 避免太长
        segs.sort(key=lambda x: -(x[1] - x[0]))   # 长在前
        picked, total_ms = [], 0
        for s, e in segs:
            picked.append((s, e))
            total_ms += e - s
            if total_ms >= 30_000:
                break
        clip = np.concatenate([wav[int(s/1000*SR):int(e/1000*SR)] for s, e in picked])
        if len(clip) < SR * 1.0:
            print(f"  spk_{sp}: 累计 <1s, 跳过")
            continue
        emb = extract_embedding_from_wave(clip, sr=SR)
        spk_emb[sp] = emb

    spk_names: Dict[int, str] = {}
    for sp, emb in spk_emb.items():
        name, sim = db.match(emb, threshold=match_threshold)
        spk_names[sp] = name if name else f"spk_{sp}"
        print(f"  spk_{sp} → {spk_names[sp]} (sim={sim:.3f}, thr={match_threshold})")

    for s in sentences:
        sp = s["speaker"]
        s["speaker"] = spk_names.get(sp, f"spk_{sp}" if sp is not None else "spk_unknown")
    return sentences


def get_next_filepath(filepath: str) -> str:
    """
    自动递增文件名避免覆盖。
    传入 'result/xxx.json'，若已存在，则返回 'result/xxx_1.json'，以此类推。
    """
    # 如果最原始的文件名还不存在，直接用它
    if not os.path.exists(filepath):
        return filepath
        
    # 拆分路径、文件名和后缀
    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    
    # 开始递增寻找可用序号
    counter = 1
    while True:
        new_name = f"{name}_{counter}{ext}"
        new_path = os.path.join(base_dir, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

# ─────────── 主流程 ───────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    file_nm = "钱部长数据融合沟通"
    ap.add_argument("--wav", default=f"data/{file_nm}.mp3", help="输入音频")
    ap.add_argument("--output", default=f"result/{file_nm}_whisper.json", help="输出 JSON")
    ap.add_argument("--output-txt", default=f"result/{file_nm}_whisper.txt", help="输出 TXT (人读)")

    # 降噪
    ap.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=False,
                    help="是否 FRCRN 降噪 (默认关; 远场/低 SNR 建议开)")

    # diar
    ap.add_argument("--num-spk", type=int, default=None, help="已知人数 (最稳)")
    ap.add_argument("--min-spk", type=int, default=None)
    ap.add_argument("--max-spk", type=int, default=None)
    ap.add_argument("--diar-model", default="pyannote/speaker-diarization-3.1")
    ap.add_argument("--hf-token", default="", help="HF token, 默认读 $HF_TOKEN")
    ap.add_argument("--diar-merge-gap", type=int, default=250,
                    help="同 spk turn 间隔 ≤ 此值就合并 (ms)")
    ap.add_argument("--diar-min-turn", type=int, default=250,
                    help="丢弃 < 此时长的 turn (ms)")

    # whisper
    ap.add_argument("--whisper-model", default="large-v3",
                    help="large-v3 / large-v3-turbo / medium / small")
    ap.add_argument("--whisper-device", default=None, help="cuda / cpu (默认自动)")
    ap.add_argument("--whisper-compute-type", default=None,
                    help="float16 (24G+) / int8_float16 (8~12G) / int8 (CPU)")
    ap.add_argument("--whisper-download-root", default=None, help="模型下载目录")
    ap.add_argument("--language", default="zh", help="zh / en / auto")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--initial-prompt", default="以下是普通话会议录音。",
                    help="给 Whisper 的引导词 (影响繁简/口语化)")
    ap.add_argument("--no-vad-filter", action="store_true",
                    help="关闭 Whisper 内置 Silero-VAD (默认开, 减少幻觉)")
    ap.add_argument("--no-condition-prev", action="store_true",
                    help="关闭 condition_on_previous_text (长音频幻觉太多时关)")

    # 段落合并
    ap.add_argument("--merge-gap", type=int, default=800,
                    help="同 spk 句间 ≤ 此值就合段 (ms)")
    ap.add_argument("--merge-max-dur", type=int, default=30000,
                    help="合段后最长 (ms)")
    ap.add_argument("--merge-max-chars", type=int, default=200)

    # 真名替换
    ap.add_argument("--enroll-db", default=None, help="声纹库 .npz, 把 spk_X 换真名")
    ap.add_argument("--match-threshold", type=float, default=0.45)
    ap.add_argument("--embedder-model", default=None,
                    help="必须和注册时一致, 留空用 speaker_db.py 默认")

    args = ap.parse_args()

    # ── [1] 加载 (统一转 16k mono wav, 绕开 pyannote 的 torchcodec 解码路径) ──
    print(f"\n=== [1] 加载 / 预处理 {args.wav} ===")
    wav, _ = librosa.load(args.wav, sr=SR, mono=True)
    print(f"  时长 {len(wav)/SR:.1f}s, sr={SR}")
    if args.denoise:
        print(f"  → FRCRN 降噪")
        import main_bss
        from main_bss import run_denoise
        wav = run_denoise(wav, sr=SR)
        main_bss._denoise_pipe = None
        _free_gpu("after-denoise")
    # 一律写临时 wav, pyannote/whisper 都吃文件
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, wav, SR)
    wav_path = tmp.name
    print(f"  → {wav_path}")

    # ── [2] pyannote diarization ──
    print(f"\n=== [2] pyannote diarization ===")
    turns, label_map = run_diarization(
        wav_path,
        num_spk=args.num_spk, min_spk=args.min_spk, max_spk=args.max_spk,
        model_id=args.diar_model, hf_token=args.hf_token,
        merge_gap_ms=args.diar_merge_gap,
        min_turn_ms=args.diar_min_turn,
    )
    _free_gpu("after-diar")

    # ── [3] Whisper 全片转写 ──
    print(f"\n=== [3] Whisper {args.whisper_model} ===")
    whisper_segs = run_whisper(
        wav_path,
        language=None if args.language == "auto" else args.language,
        beam_size=args.beam_size,
        initial_prompt=args.initial_prompt,
        vad_filter=not args.no_vad_filter,
        word_timestamps=True,
        condition_on_previous_text=not args.no_condition_prev,
        model_size=args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
        download_root=args.whisper_download_root,
    )
    _free_gpu("after-whisper")

    # ── [4] 词 → turn 对齐 ──
    print(f"\n=== [4] 词级对齐 ===")
    sentences = assign_words_to_turns(whisper_segs, turns)
    print(f"  对齐后 {len(sentences)} 句")

    # ── [5] 同 spk 合段 ──
    print(f"\n=== [5] 段落合并 ===")
    paragraphs = merge_consecutive_same_spk(
        sentences,
        max_gap_ms=args.merge_gap,
        max_dur_ms=args.merge_max_dur,
        max_chars=args.merge_max_chars,
    )
    print(f"  {len(sentences)} → {len(paragraphs)} 段")

    # ── [6] (可选) 真名替换 ──
    if args.enroll_db and os.path.exists(args.enroll_db):
        print(f"\n=== [6] SpeakerDB 真名替换 ===")
        paragraphs = apply_enroll_db(
            paragraphs, args.enroll_db, wav_path, turns,
            match_threshold=args.match_threshold,
            embedder_model=args.embedder_model,
        )
    else:
        # 没库, 用 spk_X 占位
        for p in paragraphs:
            sp = p["speaker"]
            p["speaker"] = f"spk_{sp}" if sp is not None else "spk_unknown"

    # ── 写出 ──

    def _header(p):
            tag = " [可能重叠]" if p.get("overlap") else ""
            return f"{p['speaker']} - {_fmt_time_range(p['start'], p['end'])}{tag}"

    # print(f"\n=== 转写结果 ({len(paragraphs)} 段) ===")
    # for p in paragraphs:
    #     print(f"\n{_header(p)}")
    #     print(p["text"])
        
    print(f"\n=== 写出 ===")
    out_json = [
        {
            "start": round(p["start"] / 1000, 2),
            "end":   round(p["end"]   / 1000, 2),
            "speaker": p["speaker"],
            "overlap": False,   # pyannote 完整 diar 不重叠
            "text": p["text"],
        }
        for p in paragraphs
    ]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"  [save] {args.output}")

    if args.output:
            # 【改动点】获取递增后的 json 文件名
            final_json_path = get_next_filepath(args.output)
            
            results_json = [
                {"start": round(p["start"]/1000, 2),
                 "end": round(p["end"]/1000, 2),
                 "speaker": p["speaker"],
                 "overlap": p.get("overlap", False),
                 "text": p["text"]}
                for p in paragraphs
            ]
            # 【改动点】写入新文件名
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(results_json, f, ensure_ascii=False, indent=2)
            print(f"\n[saved json] {final_json_path}")

    if args.output_txt:
        # 【改动点】获取递增后的 txt 文件名
        final_txt_path = get_next_filepath(args.output_txt)
        
        with open(final_txt_path, "w", encoding="utf-8") as f:
            f.write(f"{os.path.basename(args.wav)}\n\n")
            for p in paragraphs:
                f.write(f"{_header(p)}\n")
                f.write(f"{p['text']}\n\n")
        print(f"[saved txt ] {os.path.abspath(final_txt_path)}")

    # 清理临时 wav
    try: os.unlink(wav_path)
    except OSError: pass

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()
