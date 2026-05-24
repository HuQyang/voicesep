"""
端到端语音转文字 demo：pyannote 版整合文件

管线:
  音频
   → [1] FRCRN 降噪                         (可关 --no-denoise / 默认关)
   → [2] 说话人分离 / VAD
        A. --diar-backend local
           FSMN/Silero VAD + ERes2Net/CAM++ embedding + AHC 聚类
        B. --diar-backend pyannote
           pyannote speaker diarization 直接输出 turns
        C. --diar-backend pyannote-seg-vad
           pyannote segmentation 只做 VAD，后面继续用本地 embedding + AHC
   → [3] 可选 Mossformer2 BSS overlap 检测/分离
        注意：pyannote 完整 diarization 模式默认没有本地 centroids，BSS 会自动跳过。
   → [4] Paraformer ASR + ITN + corrections.json 纠错
   → [5] 输出 [(start, end, speaker, text), ...]

安装 pyannote:
  pip install "pyannote.audio>=3.1"
  conda install -c conda-forge ffmpeg

HuggingFace:
  export HF_TOKEN=hf_xxx
  同时需要在 HuggingFace 网页接受模型条款：
    - pyannote/segmentation-3.0
    - pyannote/speaker-diarization-3.1

推荐用法:
  python demo_pipeline_pyannote_full.py \
    --wav data/钱部长数据融合沟通.mp3 \
    --diar-backend pyannote \
    --num-spk 5 \
    --output result/钱部长_pyannote.json \
    --output-txt result/钱部长_pyannote.txt

只用 pyannote segmentation 替换 VAD，保留你自己的 embedding 聚类:
  python demo_pipeline_pyannote_full.py \
    --wav data/钱部长数据融合沟通.mp3 \
    --diar-backend pyannote-seg-vad \
    --num-spk 5 --diar-mode chunk --whiten
"""

import argparse
import gc
import json
import os
import re
import tempfile
from collections import Counter

import numpy as np
import librosa
import soundfile as sf
import torch
from funasr import AutoModel

import main_bss  # 需要操作其内部 _denoise_pipe / _bss_pipe 引用以释放显存
from speaker_db import extract_embedding_from_wave, SpeakerDB
from main_diarization import merge_segments, chunk_long_segments
from main_bss import run_denoise, run_bss, check_bss_output
from scd import split_segments_by_scd

SR = 16000


# ============================================================
# 基础工具
# ============================================================

def _free_gpu(tag: str = ""):
    """GC + empty_cache. 跨阶段调一次, 避免显存碎片堆积."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if tag:
            free, total = torch.cuda.mem_get_info()
            print(f"  [gpu] {tag}: 空闲 {free/1e9:.2f} / {total/1e9:.2f} GB")


def _release_model(module, attr: str):
    """把 module.attr 设为 None 并释放显存"""
    if getattr(module, attr, None) is not None:
        setattr(module, attr, None)
        _free_gpu()


def get_next_filepath(filepath: str) -> str:
    """自动递增文件名避免覆盖。"""
    if not os.path.exists(filepath):
        return filepath

    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)

    counter = 1
    while True:
        new_name = f"{name}_{counter}{ext}"
        new_path = os.path.join(base_dir, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def _fmt_time(ms: int) -> str:
    s = ms // 1000
    return f"{s//60}:{s%60:02d}"


def _fmt_time_range(start_ms: int, end_ms: int) -> str:
    return f"{_fmt_time(start_ms)}-{_fmt_time(end_ms)}"


def _dump_debug(debug_dir, name: str, data):
    """把任意可序列化的 dict/list 写到 debug_dir/<name>."""
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [debug] {path}")


# ============================================================
# 懒加载 VAD / ASR
# ============================================================
_vad_fsmn = None
_vad_silero = None
_asr = None


def run_vad_fsmn(wav_path: str, max_segment_ms: int = 60000, max_end_silence_ms: int = 800):
    """FSMN-VAD, 返回 [[start_ms, end_ms], ...]"""
    global _vad_fsmn
    if _vad_fsmn is None:
        print(f"[vad] 加载 FSMN-VAD (max_seg={max_segment_ms}ms, end_sil={max_end_silence_ms}ms)...")
        _vad_fsmn = AutoModel(
            model="fsmn-vad",
            model_revision="v2.0.4",
            disable_update=True,
            max_single_segment_time=max_segment_ms,
            max_end_silence_time=max_end_silence_ms,
        )
    return _vad_fsmn.generate(input=wav_path)[0]["value"]


def run_vad_silero(wav_path: str):
    """Silero-VAD, 返回 [[start_ms, end_ms], ...] 与 FSMN 对齐"""
    global _vad_silero
    if _vad_silero is None:
        print("[vad] 加载 Silero-VAD...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        _vad_silero = (model, utils)
    model, utils = _vad_silero
    get_speech_timestamps, _save, read_audio, *_ = utils
    wav = read_audio(wav_path, sampling_rate=SR)
    ts = get_speech_timestamps(wav, model, sampling_rate=SR)
    return [[int(t["start"] / SR * 1000), int(t["end"] / SR * 1000)] for t in ts]


def run_vad(wav_path: str, engine: str = "fsmn", fsmn_max_seg_ms: int = 60000, fsmn_end_sil_ms: int = 800):
    if engine == "silero":
        return run_vad_silero(wav_path)
    return run_vad_fsmn(
        wav_path,
        max_segment_ms=fsmn_max_seg_ms,
        max_end_silence_ms=fsmn_end_sil_ms,
    )


def get_asr():
    """整段 ASR: Paraformer-large + 内置 VAD + CT-Punc + 时间戳"""
    global _asr
    if _asr is None:
        print("[asr] 加载 Paraformer + FSMN-VAD + CT-Punc...")
        _asr = AutoModel(
            model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            disable_update=True,
        )
    return _asr


# ============================================================
# pyannote diarization / segmentation VAD
# ============================================================
_pyannote_diar_pipeline = None
_pyannote_seg_inference = None


def _from_pretrained_compat(cls_or_obj, model_id: str, token: str):
    """兼容 pyannote/hf_hub 不同版本的 token 参数名。"""
    try:
        return cls_or_obj.from_pretrained(model_id, use_auth_token=token)
    except TypeError:
        return cls_or_obj.from_pretrained(model_id, token=token)


def get_pyannote_token(hf_token: str = "") -> str:
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or ""
    if not token:
        raise RuntimeError(
            "缺少 HuggingFace token。请先 export HF_TOKEN=hf_xxx，"
            "并确认已经在 HuggingFace 接受 pyannote 模型条款。"
        )
    return token


def get_pyannote_diarization_pipeline(
    model_id: str = "pyannote/speaker-diarization-3.1",
    hf_token: str = "",
    device: str = None,
):
    """加载 pyannote 完整说话人分离 pipeline。"""
    global _pyannote_diar_pipeline
    if _pyannote_diar_pipeline is not None:
        return _pyannote_diar_pipeline

    from pyannote.audio import Pipeline

    token = get_pyannote_token(hf_token)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[pyannote] 加载 diarization pipeline: {model_id}")
    pipe = _from_pretrained_compat(Pipeline, model_id, token)
    pipe.to(torch.device(device))
    print(f"[pyannote] device = {device}")

    _pyannote_diar_pipeline = pipe
    return pipe


def run_pyannote_diarization(
    wav_path: str,
    num_spk: int = None,
    min_spk: int = None,
    max_spk: int = None,
    model_id: str = "pyannote/speaker-diarization-3.1",
    hf_token: str = "",
    device: str = None,
    merge_gap_ms: int = 250,
    min_turn_ms: int = 250,
):
    """
    完整替换：FSMN-VAD + ERes2Net embedding + AHC。
    返回 turns = [(start_ms, end_ms, spk_int), ...]
    """
    pipe = get_pyannote_diarization_pipeline(
        model_id=model_id,
        hf_token=hf_token,
        device=device,
    )

    kwargs = {}
    if num_spk is not None and int(num_spk) > 0:
        kwargs["num_speakers"] = int(num_spk)
    else:
        if min_spk is not None:
            kwargs["min_speakers"] = int(min_spk)
        if max_spk is not None:
            kwargs["max_speakers"] = int(max_spk)

    print(f"[pyannote] diarization kwargs = {kwargs}")
    with torch.no_grad():
        # annotation = pipe(wav_path, **kwargs)
        # 避开 pyannote 内部 AudioDecoder / torchcodec，直接传 waveform
        wav_np, _ = librosa.load(wav_path, sr=SR, mono=True)
        waveform = torch.from_numpy(wav_np).float().unsqueeze(0)  # [1, num_samples]

        file_for_pyannote = {
            "waveform": waveform,
            "sample_rate": SR,
        }

        annotation = pipe(file_for_pyannote, **kwargs)
        

    label_to_id = {}
    raw_turns = []
    # for segment, _track, label in annotation.itertracks(yield_label=True):
    #     if label not in label_to_id:
    #         label_to_id[label] = len(label_to_id)
    #     s_ms = int(round(segment.start * 1000))
    #     e_ms = int(round(segment.end * 1000))
    #     if e_ms - s_ms < min_turn_ms:
    #         continue
    #     raw_turns.append((s_ms, e_ms, label_to_id[label]))


    # 兼容不同 pyannote 输出：
    # - 老版 Pipeline: 直接返回 Annotation，有 itertracks
    # - 新版/community Pipeline: 返回 DiarizeOutput，结果在 .speaker_diarization
    if hasattr(annotation, "itertracks"):
        diarization = annotation
    elif hasattr(annotation, "speaker_diarization"):
        diarization = annotation.speaker_diarization
    else:
        raise TypeError(
            f"未知的 pyannote 输出类型: {type(annotation)}，"
            f"可用属性: {dir(annotation)[:50]}"
        )

    label_to_id = {}
    raw_turns = []

    for segment, _track, label in diarization.itertracks(yield_label=True):
        if label not in label_to_id:
            label_to_id[label] = len(label_to_id)

        s_ms = int(round(segment.start * 1000))
        e_ms = int(round(segment.end * 1000))

        if e_ms - s_ms < min_turn_ms:
            continue

        raw_turns.append((s_ms, e_ms, label_to_id[label]))



    raw_turns.sort(key=lambda x: (x[0], x[1], x[2]))

    turns = []
    for s_ms, e_ms, spk in raw_turns:
        if turns and turns[-1][2] == spk and s_ms - turns[-1][1] <= merge_gap_ms:
            turns[-1] = (turns[-1][0], max(turns[-1][1], e_ms), spk)
        else:
            turns.append((s_ms, e_ms, spk))

    print(f"[pyannote] raw_turns={len(raw_turns)}, merged_turns={len(turns)}, speakers={len(label_to_id)}")
    print(f"[pyannote] label_to_id={label_to_id}")
    return turns, label_to_id


def get_pyannote_segmentation_inference(
    model_id: str = "pyannote/segmentation-3.0",
    hf_token: str = "",
    device: str = None,
):
    """加载 pyannote segmentation 模型，只用来做 VAD。"""
    global _pyannote_seg_inference
    if _pyannote_seg_inference is not None:
        return _pyannote_seg_inference

    from pyannote.audio import Model, Inference

    token = get_pyannote_token(hf_token)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[pyannote] 加载 segmentation model: {model_id}")
    model = _from_pretrained_compat(Model, model_id, token)
    model.to(torch.device(device))

    inference = Inference(
        model,
        window="sliding",
        duration=5.0,
        step=0.5,
        device=torch.device(device),
    )
    _pyannote_seg_inference = inference
    return inference


def _activity_to_vad_segments(
    prob: np.ndarray,
    frame_starts_s: np.ndarray,
    frame_step_s: float,
    onset: float = 0.50,
    offset: float = 0.35,
    min_duration_on_s: float = 0.25,
    min_duration_off_s: float = 0.20,
    pad_onset_s: float = 0.05,
    pad_offset_s: float = 0.05,
    audio_dur_s: float = 0.0,
):
    """把 frame-level speech probability 转成 [[start_ms, end_ms], ...]。"""
    segments = []
    active = False
    start_s = None

    for i, p in enumerate(prob):
        t = float(frame_starts_s[i])
        if (not active) and p >= onset:
            active = True
            start_s = t
        elif active and p < offset:
            end_s = t + frame_step_s
            segments.append([start_s, end_s])
            active = False
            start_s = None

    if active and start_s is not None:
        segments.append([start_s, audio_dur_s])

    padded = []
    for s, e in segments:
        s = max(0.0, s - pad_onset_s)
        e = min(audio_dur_s, e + pad_offset_s)
        if e > s:
            padded.append([s, e])

    merged = []
    for s, e in padded:
        if merged and s - merged[-1][1] <= min_duration_off_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    merged = [[s, e] for s, e in merged if e - s >= min_duration_on_s]
    return [[int(round(s * 1000)), int(round(e * 1000))] for s, e in merged]


def run_vad_pyannote_segmentation(
    wav_path: str,
    model_id: str = "pyannote/segmentation-3.0",
    hf_token: str = "",
    device: str = None,
    onset: float = 0.50,
    offset: float = 0.35,
    min_duration_on: float = 0.25,
    min_duration_off: float = 0.20,
    pad_onset: float = 0.05,
    pad_offset: float = 0.05,
):
    """只用 pyannote segmentation 做 VAD。"""
    inference = get_pyannote_segmentation_inference(
        model_id=model_id,
        hf_token=hf_token,
        device=device,
    )

    with torch.no_grad():
        output = inference(wav_path)

    data = np.asarray(output.data)
    data = np.squeeze(data)

    if data.ndim == 1:
        speech_prob = data
    else:
        speech_prob = data.max(axis=-1)

    sw = output.sliding_window
    n = len(speech_prob)
    frame_starts = np.array([sw[i].start for i in range(n)], dtype=np.float32)
    if n >= 2:
        frame_step = float(frame_starts[1] - frame_starts[0])
    else:
        frame_step = float(getattr(sw, "step", 0.016))

    info = sf.info(wav_path)
    audio_dur_s = float(info.frames / info.samplerate)

    segments = _activity_to_vad_segments(
        speech_prob,
        frame_starts,
        frame_step,
        onset=onset,
        offset=offset,
        min_duration_on_s=min_duration_on,
        min_duration_off_s=min_duration_off,
        pad_onset_s=pad_onset,
        pad_offset_s=pad_offset,
        audio_dur_s=audio_dur_s,
    )

    print(
        f"[pyannote-seg-vad] {len(segments)} segments "
        f"onset={onset}, offset={offset}, min_on={min_duration_on}, min_off={min_duration_off}"
    )
    return segments


def print_turn_stats(turns, title="TURNS", show_n: int = 20):
    spk_dur_ms = Counter()
    for s, e, spk in turns:
        spk_dur_ms[int(spk)] += max(0, e - s)

    print(f"  [{title}] {len(turns)} turns, {len(spk_dur_ms)} speakers")
    if spk_dur_ms:
        print("  [SPK-DUR] " + " | ".join(
            f"spk_{spk}: {dur/1000:.1f}s" for spk, dur in spk_dur_ms.most_common()
        ))

    for i, (s, e, spk) in enumerate(turns[:show_n]):
        print(f"    turn{i:03d}: {_fmt_time_range(s, e)} spk_{spk} ({(e-s)/1000:.2f}s)")
    if len(turns) > show_n:
        print(f"    ... 还有 {len(turns)-show_n} 个 turns")


# ============================================================
# ASR 文本清理 / ITN
# ============================================================
_CJK_RE = re.compile(r"[一-鿿]")
_SPACE_BETWEEN_CJK = re.compile(r"(?<=[一-鿿])\s+(?=[一-鿿])")
_SENT_SPLIT = re.compile(r"([。！？!?；;]+)")


def _clean_text(text: str) -> str:
    """Paraformer 默认 token 间带空格, 中文要去掉; 英文/数字间空格保留"""
    return _SPACE_BETWEEN_CJK.sub("", text).strip()


_DIGIT_CHARS = "零一二三四五六七八九幺两"
_DIGIT_MAP = {"零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
              "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
              "幺": "1", "两": "2"}
_DIGIT_RUN = re.compile(f"[{_DIGIT_CHARS}]{{2,}}")
_UNIT_VAL = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100_000_000}
_FORMAL_NUM = re.compile(
    r"[零一二三四五六七八九两幺]?"
    r"(?:[十百千万亿][零一二三四五六七八九两幺]?)+"
)
_wetext_itn = None


def _parse_formal_cn_num(s: str):
    if not s:
        return None
    total = 0
    section = 0
    digit = 0
    has_digit = False
    for ch in s:
        if ch in _DIGIT_MAP:
            digit = int(_DIGIT_MAP[ch])
            has_digit = True
        elif ch in _UNIT_VAL:
            unit = _UNIT_VAL[ch]
            if unit >= 10_000 and not has_digit and section == 0:
                return None
            if unit >= 10_000:
                section = (section + (digit if digit > 0 else 1)) * unit
                total += section
                section = 0
            else:
                if digit == 0:
                    digit = 1
                section += digit * unit
            digit = 0
        else:
            return None
    return (total + section + digit) if has_digit else None


def get_wetext_itn():
    global _wetext_itn
    if _wetext_itn is None:
        try:
            from itn.chinese.inverse_normalizer import InverseNormalizer
            _wetext_itn = InverseNormalizer()
            print("[itn] 加载 WeTextProcessing.InverseNormalizer (itn.chinese)")
            return _wetext_itn
        except Exception:
            pass
        try:
            from wetext import Normalizer
            try:
                _wetext_itn = Normalizer(lang="zh", operator="itn")
            except TypeError:
                _wetext_itn = Normalizer(remove_interjections=False)
            print("[itn] 加载 wetext.Normalizer")
            return _wetext_itn
        except Exception as e:
            print(f"[itn] WeTextProcessing / wetext 都不可用 ({e}), 走正则 quick_itn")
            _wetext_itn = False
    return _wetext_itn if _wetext_itn is not False else None


_APPROX_PREFIX_2 = ("好几",)
_APPROX_PREFIX_1 = ("几", "数", "上")


def _has_approx_prefix(text: str, pos: int) -> bool:
    if pos >= 2 and text[pos - 2:pos] in _APPROX_PREFIX_2:
        return True
    if pos >= 1 and text[pos - 1] in _APPROX_PREFIX_1:
        return True
    return False


def quick_itn(text: str) -> str:
    def repl_formal(m):
        if _has_approx_prefix(m.string, m.start()):
            return m.group()
        n = _parse_formal_cn_num(m.group())
        return str(n) if n is not None else m.group()

    text = _FORMAL_NUM.sub(repl_formal, text)

    def repl_run(m):
        if _has_approx_prefix(m.string, m.start()):
            return m.group()
        return "".join(_DIGIT_MAP.get(c, c) for c in m.group())

    text = _DIGIT_RUN.sub(repl_run, text)
    return text


def smart_large_number_format(text: str) -> str:
    KEEP_DIGIT_UNITS = "年号期章节楼层室届任季度版页页码秒分时第"

    def repl(m):
        s = m.group()
        n = int(s)
        start, end = m.start(), m.end()
        full = m.string
        if start > 0 and (full[start - 1].isalnum() or full[start - 1] == "."):
            return s
        if end < len(full) and (full[end].isalnum() or full[end] == "."):
            return s
        if end < len(full) and full[end] in KEEP_DIGIT_UNITS:
            return s
        if start > 0 and full[start - 1] in "几数上多":
            return s
        if start > 1 and full[start - 2:start] == "好几":
            return s
        if n < 10000:
            return s
        if n >= 100000000:
            yi = n / 100000000
            return f"{int(yi)}亿" if yi == int(yi) else f"{yi:.1f}亿"
        wan = n / 10000
        return f"{int(wan)}万" if wan == int(wan) else f"{wan:.1f}万"

    return re.sub(r"\d{5,}", repl, text)


_POST_ITN_UNIT_FIXES = [
    (re.compile(r"(\d+(?:\.\d+)?)\s*多\s*10000(?!\d)"), r"\1多万"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*多\s*100000000(?!\d)"), r"\1多亿"),
    (re.compile(r"(?<![\d.])10000(?![\d.])"), r"万"),
    (re.compile(r"(?<![\d.])100000000(?![\d.])"), r"亿"),
]


def _post_fix_itn_units(text: str) -> str:
    for pat, repl in _POST_ITN_UNIT_FIXES:
        text = pat.sub(repl, text)
    return text


def apply_itn(text: str, use_wetext: bool = True) -> str:
    if use_wetext:
        n = get_wetext_itn()
        if n is not None:
            try:
                text = n.normalize(text)
            except Exception:
                pass
    text = quick_itn(text)
    text = smart_large_number_format(text)
    text = _post_fix_itn_units(text)
    return text


# ============================================================
# ASR
# ============================================================

def asr_wav(wav: np.ndarray, sr: int = SR, hotword: str = "") -> str:
    """对一段 numpy 波形跑 ASR, 返回清理后的纯文本"""
    asr = get_asr()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, sr)
        tmp_path = tmp.name
    try:
        res = asr.generate(input=tmp_path, hotword=hotword)
        if not res:
            return ""
        return _clean_text(res[0].get("text", ""))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def asr_full(wav_path: str, hotword: str = ""):
    """
    对整段音频跑 ASR. 返回句子列表:
      [{"start": ms, "end": ms, "text": str}, ...]
    """
    asr = get_asr()
    res = asr.generate(input=wav_path, batch_size_s=300, hotword=hotword)
    if not res:
        return []
    r = res[0]

    if isinstance(r.get("sentence_info"), list) and r["sentence_info"]:
        out = []
        for s in r["sentence_info"]:
            txt = _clean_text(s.get("text", ""))
            if not txt:
                continue
            out.append({
                "start": int(s.get("start", 0)),
                "end": int(s.get("end", 0)),
                "text": txt,
            })
        return out

    text = r.get("text", "")
    timestamps = r.get("timestamp", [])
    if not text or not timestamps:
        return [{"start": 0, "end": 0, "text": _clean_text(text)}] if text else []

    text_clean = _clean_text(text)
    sentences = []
    char_idx = 0
    cur_chars = []
    cur_start = None
    n_ts = len(timestamps)

    for ch in text_clean:
        if _CJK_RE.match(ch):
            if cur_start is None and char_idx < n_ts:
                cur_start = int(timestamps[char_idx][0])
            cur_chars.append(ch)
            char_idx += 1
            continue
        cur_chars.append(ch)
        if ch in "。！？.!?；;":
            end_ms = int(timestamps[min(char_idx - 1, n_ts - 1)][1]) if char_idx > 0 else (cur_start or 0)
            sentences.append({
                "start": cur_start or 0,
                "end": end_ms,
                "text": "".join(cur_chars).strip(),
            })
            cur_chars, cur_start = [], None

    if cur_chars:
        end_ms = int(timestamps[min(char_idx - 1, n_ts - 1)][1]) if char_idx > 0 else (cur_start or 0)
        sentences.append({
            "start": cur_start or 0,
            "end": end_ms,
            "text": "".join(cur_chars).strip(),
        })
    return sentences


def assign_speaker(sentences, turns):
    """把 turn 的 spk 染色到句子。"""
    if not turns:
        return [{**s, "speaker": None} for s in sentences]

    out = []
    for s in sentences:
        mid = (s["start"] + s["end"]) / 2.0
        best_spk = None
        best_dist = float("inf")
        for ts, te, spk in turns:
            if ts <= mid <= te:
                best_spk, best_dist = spk, 0
                break
            d = min(abs(mid - ts), abs(mid - te))
            if d < best_dist:
                best_spk, best_dist = spk, d
        out.append({**s, "speaker": best_spk})
    return out


def group_paragraphs(sentences_with_spk, max_gap_ms: int = 800, max_paragraph_dur_ms: int = 60_000, max_paragraph_chars: int = 600):
    paragraphs = []
    for s in sentences_with_spk:
        if paragraphs:
            p = paragraphs[-1]
            same_spk = p["speaker"] == s["speaker"]
            gap_ok = (s["start"] - p["end"]) <= max_gap_ms
            dur_ok = (s["end"] - p["start"]) <= max_paragraph_dur_ms
            chars_ok = len(p["text"]) < max_paragraph_chars
            if same_spk and gap_ok and dur_ok and chars_ok:
                p["end"] = s["end"]
                p["text"] += s["text"]
                p["overlap"] = p["overlap"] or s.get("overlap", False)
                continue
        paragraphs.append({
            "start": s["start"],
            "end": s["end"],
            "speaker": s["speaker"],
            "text": s["text"],
            "overlap": s.get("overlap", False),
        })
    return paragraphs


def merge_consecutive_same_spk(paragraphs, max_gap_ms: int = 5000, max_dur_ms: int = 180_000, max_chars: int = 1500):
    if not paragraphs:
        return paragraphs
    out = [dict(paragraphs[0])]
    for p in paragraphs[1:]:
        last = out[-1]
        same_spk = last["speaker"] == p["speaker"]
        neither_overlap = not last.get("overlap", False) and not p.get("overlap", False)
        gap = p["start"] - last["end"]
        merged_dur = p["end"] - last["start"]
        merged_chars = len(last["text"]) + len(p["text"])
        if same_spk and neither_overlap and gap <= max_gap_ms and merged_dur <= max_dur_ms and merged_chars <= max_chars:
            last["end"] = p["end"]
            last["text"] += p["text"]
        else:
            out.append(dict(p))
    return out


# ============================================================
# 纠错 / 热词
# ============================================================

def load_corrections(path: str = "corrections.json") -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("rules"), list):
        rules = [r for r in data["rules"] if "wrong" in r and "right" in r]
        rules.sort(key=lambda r: -len(r["wrong"]))
        return {r["wrong"]: r["right"] for r in rules}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and not k.startswith("_")}


def load_hotwords(path: str = "corrections.json") -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = data.get("hotwords", []) if isinstance(data, dict) else []
    return " ".join(str(w) for w in words if w)


def apply_corrections(text: str, table: dict) -> str:
    for w, r in table.items():
        if w and w != r:
            text = text.replace(w, r)
    return text


# ============================================================
# 声纹聚类相关
# ============================================================

def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)) + 1e-10)
    return 20.0 * np.log10(rms)


def extract_chunk_embs(wav, chunks, min_dbfs: float = -50.0):
    embs, kept, dropped = [], [], 0
    for s_ms, e_ms in chunks:
        s, e = int(s_ms / 1000 * SR), int(e_ms / 1000 * SR)
        seg = wav[s:e]
        if len(seg) < SR * 0.5:
            continue
        if _rms_dbfs(seg) < min_dbfs:
            dropped += 1
            continue
        emb = extract_embedding_from_wave(seg, sr=SR)
        embs.append(emb)
        kept.append((s_ms, e_ms))
    if dropped:
        print(f"  [NOISE] 丢弃低能量 chunk {dropped} 个 (dBFS<{min_dbfs})")
    if embs:
        return np.stack(embs), kept
    return np.zeros((0, 192), dtype=np.float32), kept


def cluster_embs(embs: np.ndarray, num_spk: int = None, threshold: float = 0.7, whiten: bool = False):
    from sklearn.cluster import AgglomerativeClustering
    if len(embs) == 0:
        return np.array([], dtype=int)
    if len(embs) == 1:
        return np.array([0], dtype=int)

    if whiten:
        mean = embs.mean(axis=0, keepdims=True)
        embs = embs - mean
        print("  [WHITEN] 减去全局均值")

    normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    if num_spk and int(num_spk) > 0:
        clf = AgglomerativeClustering(n_clusters=int(num_spk), metric="cosine", linkage="average")
    else:
        clf = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average")
    return clf.fit_predict(normed)


def compute_centroids(embs: np.ndarray, labels: np.ndarray) -> dict:
    centroids = {}
    for lab in set(int(x) for x in labels):
        members = embs[labels == lab]
        c = members.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        centroids[lab] = c
    return centroids


def match_centroid(emb: np.ndarray, centroids: dict):
    if not centroids:
        return None, -1.0
    e = emb / (np.linalg.norm(emb) + 1e-8)
    best_spk, best_sim = None, -1.0
    for spk, c in centroids.items():
        sim = float(np.dot(e, c))
        if sim > best_sim:
            best_spk, best_sim = spk, sim
    return best_spk, best_sim


def segments_to_turns(segments, chunks, kept_chunks, labels):
    chunk_labels_by_seg = {}
    for (cs, ce), lab in zip(kept_chunks, labels):
        for i, (ss, se) in enumerate(segments):
            if cs >= ss and ce <= se + 1:
                chunk_labels_by_seg.setdefault(i, []).append(int(lab))
                break

    seg_turns = []
    for i, (ss, se) in enumerate(segments):
        labs = chunk_labels_by_seg.get(i, [])
        if not labs:
            continue
        top = Counter(labs).most_common(1)[0][0]
        seg_turns.append((ss, se, top))

    turns = []
    for s, e, spk in seg_turns:
        if turns and turns[-1][2] == spk and s - turns[-1][1] <= 500:
            turns[-1] = (turns[-1][0], e, spk)
        else:
            turns.append((s, e, spk))
    return turns


def chunks_to_fine_turns(kept_chunks, labels, smooth: bool = True):
    if not labels.size:
        return []
    raw = [(int(cs), int(ce), int(lab)) for (cs, ce), lab in zip(kept_chunks, labels)]

    if smooth and len(raw) >= 3:
        smoothed = list(raw)
        for i in range(1, len(raw) - 1):
            if raw[i - 1][2] == raw[i + 1][2] and raw[i][2] != raw[i - 1][2]:
                smoothed[i] = (raw[i][0], raw[i][1], raw[i - 1][2])
        raw = smoothed

    turns = [list(raw[0])]
    for cs, ce, lab in raw[1:]:
        if lab == turns[-1][2]:
            turns[-1][1] = ce
        else:
            turns.append([cs, ce, lab])
    return [tuple(t) for t in turns]


# ============================================================
# 主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    file_nm = "钱部长数据融合沟通"

    ap.add_argument("--wav", default=f"data/{file_nm}.mp3", help="输入音频")
    ap.add_argument("--num-spk", type=int, default=5, help="已知人数。pyannote/local 都会用到。设 0 则改用 min/max 或 threshold")
    ap.add_argument("--threshold", type=float, default=0.6, help="local AHC cosine 距离阈值")
    ap.add_argument("--enroll-db", default=f"{file_nm}_db.npz", help="可选: 声纹库, 把 spk_X 替换为真名")
    ap.add_argument("--match-threshold", type=float, default=0.55)
    ap.add_argument("--hotword", default="", help="额外热词，空格分隔")

    ap.add_argument("--itn", action=argparse.BooleanOptionalAction, default=True, help="ITN 中文数字 → 阿拉伯数字")
    ap.add_argument("--wetext-itn", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=False, help="是否走 FRCRN 降噪")
    ap.add_argument("--vad", choices=["fsmn", "silero"], default="fsmn", help="local 模式 VAD 引擎")
    ap.add_argument("--vad-fsmn-max-seg-ms", type=int, default=10000)
    ap.add_argument("--vad-fsmn-end-sil-ms", type=int, default=800)
    ap.add_argument("--vad-show-n", type=int, default=20)

    # pyannote 新参数
    ap.add_argument("--diar-backend", choices=["local", "pyannote", "pyannote-seg-vad"], default="pyannote",
                    help="local=原 FSMN/Silero+ERes2Net；pyannote=完整说话人分离；pyannote-seg-vad=只用 pyannote segmentation 做 VAD")
    ap.add_argument("--pyannote-model", default="pyannote/speaker-diarization-3.1")
    ap.add_argument("--pyannote-seg-model", default="pyannote/segmentation-3.0")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or "")
    ap.add_argument("--pyannote-device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-spk", type=int, default=None)
    ap.add_argument("--max-spk", type=int, default=None)
    ap.add_argument("--pyannote-merge-gap-ms", type=int, default=250)
    ap.add_argument("--pyannote-min-turn-ms", type=int, default=250)
    ap.add_argument("--pyannote-vad-onset", type=float, default=0.50)
    ap.add_argument("--pyannote-vad-offset", type=float, default=0.35)
    ap.add_argument("--pyannote-vad-min-on", type=float, default=0.25)
    ap.add_argument("--pyannote-vad-min-off", type=float, default=0.20)

    # SCD / local diarization 参数
    ap.add_argument("--scd", action="store_true")
    ap.add_argument("--scd-min-seg-s", type=float, default=5.0)
    ap.add_argument("--scd-window-s", type=float, default=0.75)
    ap.add_argument("--scd-hop-s", type=float, default=0.1)
    ap.add_argument("--scd-threshold", type=float, default=0.5)
    ap.add_argument("--scd-min-spk-dur-s", type=float, default=0.8)

    ap.add_argument("--embedder-model", default="iic/speech_eres2net_large_200k_sv_zh-cn_16k-common")
    ap.add_argument("--whiten", action="store_true")
    ap.add_argument("--diar-mode", choices=["segment", "chunk"], default="chunk")
    ap.add_argument("--diar-smooth", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-dbfs", type=float, default=-45.0)
    ap.add_argument("--merge-gap", type=int, default=300)
    ap.add_argument("--min-dur", type=int, default=500)
    ap.add_argument("--chunk-max", type=int, default=2000)
    ap.add_argument("--chunk-hop", type=int, default=1000)

    # BSS
    ap.add_argument("--no-bss", action="store_true", help="跳过重叠检测/分离")
    ap.add_argument("--bss-min-dur", type=float, default=5.0)
    ap.add_argument("--bss-max-dur", type=float, default=20.0)
    ap.add_argument("--bss-dump-dir", default="bss_out")

    # output / paragraph
    ap.add_argument("--output", default=f"result/{file_nm}_pyannote.json")
    ap.add_argument("--output-txt", default=f"result/{file_nm}_pyannote.txt")
    ap.add_argument("--para-gap", type=int, default=500)
    ap.add_argument("--para-max-dur", type=int, default=10000)
    ap.add_argument("--para-max-chars", type=int, default=600)
    ap.add_argument("--post-merge-gap", type=int, default=500)
    ap.add_argument("--post-merge-max-dur", type=int, default=20000)
    ap.add_argument("--post-merge-max-chars", type=int, default=1500)
    ap.add_argument("--debug-dir", default=f"result/debug/{file_nm}_pyannote")

    args = ap.parse_args()

    if args.embedder_model and args.diar_backend != "pyannote":
        from speaker_db import set_embedder_model
        set_embedder_model(args.embedder_model)

    print("\n=== [1/5] 加载音频 ===")
    wav, _ = librosa.load(args.wav, sr=SR, mono=True)
    dur_s = len(wav) / SR
    print(f"  文件: {args.wav}")
    print(f"  时长: {dur_s:.1f}s ({dur_s/60:.1f} min)")

    if not args.denoise:
        print("\n=== [2/5] 降噪已跳过 ===")
    else:
        print("\n=== [2/5] FRCRN 降噪 ===")
        rms_before = _rms_dbfs(wav)
        wav = run_denoise(wav, sr=SR)
        rms_after = _rms_dbfs(wav)
        print(f"  RMS dBFS: {rms_before:.1f} → {rms_after:.1f}")
        _release_model(main_bss, "_denoise_pipe")
        _free_gpu("after denoise")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, SR)
        clean_path = tmp.name

    try:
        # ─── 3. VAD + diarization / clustering ───
        print(f"\n=== [3/5] Diarization backend: {args.diar_backend} ===")
        centroids = {}
        spk_ids = []

        if args.diar_backend == "pyannote":
            turns, pyannote_label_map = run_pyannote_diarization(
                clean_path,
                num_spk=args.num_spk,
                min_spk=args.min_spk,
                max_spk=args.max_spk,
                model_id=args.pyannote_model,
                hf_token=args.hf_token,
                device=args.pyannote_device,
                merge_gap_ms=args.pyannote_merge_gap_ms,
                min_turn_ms=args.pyannote_min_turn_ms,
            )
            spk_ids = sorted(set(int(spk) for _, _, spk in turns))
            print_turn_stats(turns, title="PYANNOTE")

            _dump_debug(args.debug_dir, "03_turns_pyannote.json", {
                "backend": "pyannote",
                "model": args.pyannote_model,
                "num_spk_requested": args.num_spk,
                "min_spk": args.min_spk,
                "max_spk": args.max_spk,
                "speaker_count": len(spk_ids),
                "turn_count": len(turns),
                "label_map": {str(k): int(v) for k, v in pyannote_label_map.items()},
                "turns": [
                    {"start_s": round(s / 1000, 2), "end_s": round(e / 1000, 2),
                     "dur_s": round((e - s) / 1000, 2), "speaker": f"spk_{spk}"}
                    for s, e, spk in turns
                ],
            })

        else:
            if args.diar_backend == "pyannote-seg-vad":
                raw_segments = run_vad_pyannote_segmentation(
                    clean_path,
                    model_id=args.pyannote_seg_model,
                    hf_token=args.hf_token,
                    device=args.pyannote_device,
                    onset=args.pyannote_vad_onset,
                    offset=args.pyannote_vad_offset,
                    min_duration_on=args.pyannote_vad_min_on,
                    min_duration_off=args.pyannote_vad_min_off,
                )
            else:
                raw_segments = run_vad(
                    clean_path,
                    engine=args.vad,
                    fsmn_max_seg_ms=args.vad_fsmn_max_seg_ms,
                    fsmn_end_sil_ms=args.vad_fsmn_end_sil_ms,
                )

            print(f"  [VAD]   {len(raw_segments)} segments")
            show_count = len(raw_segments) if args.vad_show_n == 0 else max(0, args.vad_show_n)
            if args.vad_show_n >= 0:
                for i, (s_ms, e_ms) in enumerate(raw_segments[:show_count]):
                    print(f"    seg{i:03d}: {_fmt_time_range(s_ms, e_ms)}  ({(e_ms - s_ms) / 1000:.2f}s)")
                if args.vad_show_n > 0 and len(raw_segments) > args.vad_show_n:
                    print(f"    ... 还有 {len(raw_segments) - args.vad_show_n} 段未列出 (--vad-show-n 控制)")

            _dump_debug(args.debug_dir, "01_vad_raw.json", {
                "engine": args.diar_backend,
                "count": len(raw_segments),
                "segments": [
                    {"start_s": round(s / 1000, 2), "end_s": round(e / 1000, 2),
                     "dur_s": round((e - s) / 1000, 2)}
                    for s, e in raw_segments
                ],
            })

            segments = merge_segments(raw_segments, gap_threshold_ms=args.merge_gap, min_duration_ms=args.min_dur)
            print(f"  [MERGE] {len(segments)} segments (gap<{args.merge_gap}ms 合并, <{args.min_dur}ms 丢弃)")

            if args.scd:
                print(
                    f"  [SCD]   开始检测 (window={args.scd_window_s}s, hop={args.scd_hop_s}s, "
                    f"threshold={args.scd_threshold}, 只处理 > {args.scd_min_seg_s}s 的段)"
                )
                segments, scd_stats = split_segments_by_scd(
                    wav,
                    segments,
                    min_segment_for_scd_s=args.scd_min_seg_s,
                    window_s=args.scd_window_s,
                    hop_s=args.scd_hop_s,
                    distance_threshold=args.scd_threshold,
                    min_speaker_dur_s=args.scd_min_spk_dur_s,
                    sr=SR,
                    verbose=True,
                )
                print(
                    f"  [SCD]   {scd_stats['n_segments_in']} → {scd_stats['n_segments_out']} 段 "
                    f"(共 {scd_stats['n_scd_run']} 段被分析, 找到 {scd_stats['n_total_change_points']} 个切点)"
                )
                _dump_debug(args.debug_dir, "02b_scd.json", scd_stats)

            chunks = chunk_long_segments(segments, max_dur_ms=args.chunk_max, hop_ms=args.chunk_hop)
            print(f"  [CHUNK] {len(chunks)} chunks (>{args.chunk_max}ms 按 {args.chunk_hop}ms hop 切)")

            _dump_debug(args.debug_dir, "02_vad_merged.json", {
                "merged_count": len(segments),
                "chunk_count": len(chunks),
                "merge_gap_ms": args.merge_gap,
                "min_dur_ms": args.min_dur,
                "merged_segments": [
                    {"start_s": round(s / 1000, 2), "end_s": round(e / 1000, 2),
                     "dur_s": round((e - s) / 1000, 2)}
                    for s, e in segments
                ],
                "chunks": [{"start_s": round(s / 1000, 2), "end_s": round(e / 1000, 2)} for s, e in chunks],
            })

            embs, kept_chunks = extract_chunk_embs(wav, chunks, min_dbfs=args.min_dbfs)
            print(f"  [EMB]   {len(embs)} embeddings")

            if len(embs) < 2:
                print("  embedding 不足 2 个, 退出")
                return

            labels = cluster_embs(embs, num_spk=args.num_spk, threshold=args.threshold, whiten=args.whiten)
            centroids = compute_centroids(embs, labels)
            spk_ids = sorted(centroids.keys())
            print(f"  [CLUS]  发现 {len(spk_ids)} 个说话人簇: {spk_ids}")

            if args.diar_mode == "chunk":
                turns = chunks_to_fine_turns(kept_chunks, labels, smooth=args.diar_smooth)
                print(f"  [TURNS] {len(turns)} turns (chunk-level, smooth={args.diar_smooth})")
            else:
                turns = segments_to_turns(segments, chunks, kept_chunks, labels)
                print(f"  [TURNS] {len(turns)} turns (segment-vote)")

            print_turn_stats(turns, title="LOCAL")

            spk_dur_ms = Counter()
            for _ts, _te, _spk in turns:
                spk_dur_ms[int(_spk)] += (_te - _ts)

            _dump_debug(args.debug_dir, "03_turns.json", {
                "diar_mode": args.diar_mode,
                "backend": args.diar_backend,
                "num_spk_requested": args.num_spk,
                "threshold": args.threshold,
                "spk_count": len(spk_ids),
                "spk_total_duration_s": {f"spk_{spk}": round(dur / 1000, 1) for spk, dur in spk_dur_ms.most_common()},
                "turn_count": len(turns),
                "turns": [
                    {"start_s": round(s / 1000, 2), "end_s": round(e / 1000, 2),
                     "dur_s": round((e - s) / 1000, 2), "speaker": f"spk_{spk}"}
                    for s, e, spk in turns
                ],
            })

        # ─── 4. BSS 重叠区检测 ───
        overlap_ranges = []
        if args.no_bss:
            print("\n=== [4/5] BSS 已跳过 ===")
        elif not centroids:
            print("\n=== [4/5] BSS 已跳过：当前 pyannote 完整模式没有本地 centroids，无法把 BSS 分离轨道匹配回 speaker ===")
        else:
            print("\n=== [4/5] BSS 重叠检测 ===")
            n_overlap = n_skip = n_too_long = 0
            for idx, (s_ms, e_ms, spk) in enumerate(turns):
                dur = (e_ms - s_ms) / 1000.0
                seg = wav[int(s_ms / 1000 * SR):int(e_ms / 1000 * SR)]
                if dur < args.bss_min_dur:
                    n_skip += 1
                    continue
                if dur > args.bss_max_dur:
                    n_too_long += 1
                    continue
                try:
                    with torch.no_grad():
                        wavs = run_bss(seg, sr_in=SR)
                        chk = check_bss_output(wavs)
                except torch.cuda.OutOfMemoryError:
                    print(f"  [BSS-OOM] {s_ms/1000:.1f}-{e_ms/1000:.1f}s 显存不足")
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

            n_checked = len(turns) - n_skip - n_too_long
            print(f"  检测 {n_checked} turns, 重叠 {n_overlap} 段, 短<{args.bss_min_dur}s 跳过 {n_skip} 段, 长>{args.bss_max_dur}s 跳过 {n_too_long} 段")
            _release_model(main_bss, "_bss_pipe")
            _free_gpu("after bss")

            _dump_debug(args.debug_dir, "04_bss.json", {
                "checked": n_checked,
                "overlap": n_overlap,
                "skipped_short": n_skip,
                "skipped_long": n_too_long,
                "overlap_ranges": [
                    {"start_s": round(s / 1000, 2), "end_s": round(e / 1000, 2),
                     "spk_pair": [(f"spk_{x}" if x is not None else None) for x in spk_pair]}
                    for s, e, spk_pair, _wavs in overlap_ranges
                ],
            })

        # ─── 5. 整段 ASR + 染色 + 段落化 ───
        print("\n=== [5/5] 整段 ASR (Paraformer + VAD + Punc) ===")
        effective_hotword = " ".join(filter(None, [args.hotword, load_hotwords()])).strip()
        if effective_hotword:
            print(f"  [HOTWORD] {len(effective_hotword)} chars")

        sentences = asr_full(clean_path, hotword=effective_hotword)
        print(f"  [ASR] 得到 {len(sentences)} 个句子")

        if args.itn:
            n_changed = 0
            for s in sentences:
                new_text = apply_itn(s["text"], use_wetext=args.wetext_itn)
                if new_text != s["text"]:
                    n_changed += 1
                s["text"] = new_text
            print(f"  [ITN] {n_changed}/{len(sentences)} 句包含数字, 已转阿拉伯数字")

        corrections = load_corrections()
        if corrections:
            for s in sentences:
                s["text"] = apply_corrections(s["text"], corrections)
            print(f"  纠错表 {len(corrections)} 条已应用")

        _dump_debug(args.debug_dir, "05_asr_raw.json", {
            "hotword": effective_hotword,
            "itn_enabled": args.itn,
            "correction_count": len(corrections) if corrections else 0,
            "sentence_count": len(sentences),
            "sentences": [
                {"start_s": round(s["start"] / 1000, 2), "end_s": round(s["end"] / 1000, 2), "text": s["text"]}
                for s in sentences
            ],
        })

        sentences = assign_speaker(sentences, turns)
        for s in sentences:
            s["overlap"] = False

        # BSS 重叠区替换
        if overlap_ranges:
            replaced = 0
            for os_ms, oe_ms, spk_pair, sep_wavs in overlap_ranges:
                if spk_pair[0] == spk_pair[1] or spk_pair[0] is None or spk_pair[1] is None:
                    continue
                before = len(sentences)
                sentences = [s for s in sentences if not (os_ms <= (s["start"] + s["end"]) / 2.0 <= oe_ms)]
                replaced += before - len(sentences)
                for idx, w in enumerate(sep_wavs):
                    try:
                        with torch.no_grad():
                            text = asr_wav(w, sr=SR, hotword=effective_hotword)
                    except Exception as e:
                        print(f"  [OVERLAP-ASR-FAIL] spk_{spk_pair[idx]}: {e}")
                        text = ""
                    if not text.strip():
                        continue
                    if args.itn:
                        text = apply_itn(text, use_wetext=args.wetext_itn)
                    if corrections:
                        text = apply_corrections(text, corrections)
                    sentences.append({
                        "start": os_ms,
                        "end": oe_ms,
                        "text": text,
                        "speaker": spk_pair[idx],
                        "overlap": True,
                    })
                    _free_gpu()
            if replaced:
                print(f"  [OVERLAP-RE-ASR] 替换 {replaced} 句")
            sentences.sort(key=lambda r: (r["start"], r.get("speaker") if r.get("speaker") is not None else -1))

        # SpeakerDB 真名映射
        if args.enroll_db and os.path.exists(args.enroll_db) and centroids:
            db = SpeakerDB(args.enroll_db)
            spk_names = {}
            for spk, c in centroids.items():
                name, sim = db.match(c, threshold=args.match_threshold)
                spk_names[spk] = name if name else f"spk_{spk}"
            print(f"  [ENROLL] spk → 真名: {spk_names}")
        else:
            spk_names = {spk: f"spk_{spk}" for spk in sorted(set(t[2] for t in turns))}

        for s in sentences:
            if s["speaker"] is None:
                s["speaker"] = "spk_unknown"
            else:
                s["speaker"] = spk_names.get(s["speaker"], f"spk_{s['speaker']}")

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
                {"start_s": round(p["start"] / 1000, 2),
                 "end_s": round(p["end"] / 1000, 2),
                 "dur_s": round((p["end"] - p["start"]) / 1000, 2),
                 "speaker": p["speaker"],
                 "overlap": p.get("overlap", False),
                 "char_count": len(p["text"]),
                 "text": p["text"]}
                for p in paragraphs
            ],
        })

        def _header(p):
            tag = " [可能重叠]" if p.get("overlap") else ""
            return f"{p['speaker']} - {_fmt_time_range(p['start'], p['end'])}{tag}"

        print(f"\n=== 转写结果 ({len(paragraphs)} 段) ===")
        for p in paragraphs:
            print(f"\n{_header(p)}")
            print(p["text"])

        if args.output:
            final_json_path = get_next_filepath(args.output)
            os.makedirs(os.path.dirname(final_json_path) or ".", exist_ok=True)
            results_json = [
                {"start": round(p["start"] / 1000, 2),
                 "end": round(p["end"] / 1000, 2),
                 "speaker": p["speaker"],
                 "overlap": p.get("overlap", False),
                 "text": p["text"]}
                for p in paragraphs
            ]
            with open(final_json_path, "w", encoding="utf-8") as f:
                json.dump(results_json, f, ensure_ascii=False, indent=2)
            print(f"\n[saved json] {final_json_path}")

        if args.output_txt:
            final_txt_path = get_next_filepath(args.output_txt)
            os.makedirs(os.path.dirname(final_txt_path) or ".", exist_ok=True)
            with open(final_txt_path, "w", encoding="utf-8") as f:
                f.write(f"{os.path.basename(args.wav)}\n\n")
                for p in paragraphs:
                    f.write(f"{_header(p)}\n")
                    f.write(f"{p['text']}\n\n")
            print(f"[saved txt ] {os.path.abspath(final_txt_path)}")

    finally:
        try:
            os.unlink(clean_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
