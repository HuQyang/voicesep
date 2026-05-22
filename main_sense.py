"""
端到端语音转文字 demo (已修复 ASR 断层与代差问题)

管线:
  音频
   → [0] 音量标准化 (可手动放大或自动峰值标准化)
   → [1] FRCRN 降噪          (可关 --no-denoise)
   → [2] FSMN/Silero VAD 切段 + SCD 细粒度切分
   → [3] ERes2NetV2 emb + 聚类 → 每段说话人 spk_id
   → [4] BSS 重叠检测分离      (可关 --no-bss)
   → [5] 逐段投喂 SenseVoiceSmall 进行 ASR + ITN + 纠错
   → 输出 [(start, end, speaker, text), ...]
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

import main_bss  
from speaker_db import extract_embedding_from_wave, SpeakerDB
from main_diarization import merge_segments, chunk_long_segments
from main_bss import run_denoise, run_bss, check_bss_output
from scd import split_segments_by_scd


SR = 16000


def _free_gpu(tag: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if tag:
            free, total = torch.cuda.mem_get_info()
            print(f"  [gpu] {tag}: 空闲 {free/1e9:.2f} / {total/1e9:.2f} GB")


def _release_model(module, attr: str):
    if getattr(module, attr, None) is not None:
        setattr(module, attr, None)
        _free_gpu()


# ─────────── 懒加载 VAD / ASR ───────────
_vad_fsmn = None
_vad_silero = None
_asr = None


def run_vad_fsmn(wav_path: str, max_segment_ms: int = 60000, max_end_silence_ms: int = 800):
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
    global _vad_silero
    import torch
    if _vad_silero is None:
        print("[vad] 加载 Silero-VAD...")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
        _vad_silero = (model, utils)
    model, utils = _vad_silero
    get_speech_timestamps, _save, read_audio, *_ = utils
    wav = read_audio(wav_path, sampling_rate=SR)
    ts = get_speech_timestamps(wav, model, sampling_rate=SR)
    return [[int(t["start"] / SR * 1000), int(t["end"] / SR * 1000)] for t in ts]


def run_vad(wav_path: str, engine: str = "fsmn",
            fsmn_max_seg_ms: int = 60000, fsmn_end_sil_ms: int = 800):
    if engine == "silero":
        return run_vad_silero(wav_path)
    return run_vad_fsmn(wav_path,
                        max_segment_ms=fsmn_max_seg_ms,
                        max_end_silence_ms=fsmn_end_sil_ms)


def get_asr():
    """
    [修改点]: 升级为 SenseVoiceSmall，抗噪和会议场景识别率远超基础版 Paraformer。
    内置标点模型，不需要额外挂载 CT-Punc。
    """
    global _asr
    if _asr is None:
        print("[asr] 加载 SenseVoiceSmall (高抗噪, 自带标点)...")
        _asr = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            trust_remote_code=True,
            disable_update=True,
        )
    return _asr


_CJK_RE = re.compile(r"[一-鿿]")
_SPACE_BETWEEN_CJK = re.compile(r"(?<=[一-鿿])\s+(?=[一-鿿])")


def _clean_text(text: str) -> str:
    """清理 SenseVoice 返回的特殊标签 <|zh|><|NEUTRAL|><|Speech|> 等"""
    text = re.sub(r"<\|.*?\|>", "", text)
    return _SPACE_BETWEEN_CJK.sub("", text).strip()


# ─────────── ITN (Inverse Text Normalization) ───────────
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


def _parse_formal_cn_num(s: str):
    if not s: return None
    total = section = digit = 0
    has_num = False
    for ch in s:
        if ch in _DIGIT_MAP:
            digit = int(_DIGIT_MAP[ch])
            has_num = True
        elif ch in _UNIT_VAL:
            unit = _UNIT_VAL[ch]
            if unit >= 10_000:
                section = (section + (digit if digit > 0 else 1)) * unit
                total += section
                section = 0
            else:
                if digit == 0: digit = 1
                section += digit * unit
            digit = 0
            has_num = True
        else:
            return None
    return (total + section + digit) if has_num else None

_wetext_itn = None

def get_wetext_itn():
    global _wetext_itn
    if _wetext_itn is None:
        try:
            from itn.chinese.inverse_normalizer import InverseNormalizer
            _wetext_itn = InverseNormalizer()
            return _wetext_itn
        except Exception:
            pass
        try:
            from wetext import Normalizer
            try:
                _wetext_itn = Normalizer(lang="zh", operator="itn")
            except TypeError:
                _wetext_itn = Normalizer(remove_interjections=False)
            return _wetext_itn
        except Exception:
            _wetext_itn = False
    return _wetext_itn if _wetext_itn is not False else None


_APPROX_PREFIX_2 = ("好几",)
_APPROX_PREFIX_1 = ("几", "数", "上")

def _has_approx_prefix(text: str, pos: int) -> bool:
    if pos >= 2 and text[pos - 2:pos] in _APPROX_PREFIX_2: return True
    if pos >= 1 and text[pos - 1] in _APPROX_PREFIX_1: return True
    return False

def quick_itn(text: str) -> str:
    def repl_formal(m):
        if _has_approx_prefix(m.string, m.start()): return m.group()
        n = _parse_formal_cn_num(m.group())
        return str(n) if n is not None else m.group()
    text = _FORMAL_NUM.sub(repl_formal, text)

    def repl_run(m):
        if _has_approx_prefix(m.string, m.start()): return m.group()
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
        if start > 0 and (full[start - 1].isalnum() or full[start - 1] == "."): return s
        if end < len(full) and (full[end].isalnum() or full[end] == "."): return s
        if end < len(full) and full[end] in KEEP_DIGIT_UNITS: return s
        if start > 0 and full[start - 1] in "几数上多": return s
        if start > 1 and full[start - 2:start] == "好几": return s

        if n < 10000: return s
        if n >= 100000000:
            yi = n / 100000000
            return f"{int(yi)}亿" if yi == int(yi) else f"{yi:.1f}亿"
        wan = n / 10000
        return f"{int(wan)}万" if wan == int(wan) else f"{wan:.1f}万"
    return re.sub(r"\d{5,}", repl, text)

def apply_itn(text: str, use_wetext: bool = True) -> str:
    if use_wetext:
        n = get_wetext_itn()
        if n is not None:
            try: text = n.normalize(text)
            except Exception: pass
    text = quick_itn(text)
    text = smart_large_number_format(text)
    return text


def asr_wav(wav: np.ndarray, sr: int = SR, hotword: str = "") -> str:
    """[核心修改]: 专用于对单独切好的音频片段跑 ASR"""
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
        try: os.unlink(tmp_path)
        except OSError: pass


def group_paragraphs(
    sentences_with_spk,
    max_gap_ms: int = 800,
    max_paragraph_dur_ms: int = 60_000,
    max_paragraph_chars: int = 600,
):
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
            "start": s["start"], "end": s["end"],
            "speaker": s["speaker"], "text": s["text"],
            "overlap": s.get("overlap", False),
        })
    return paragraphs


def merge_consecutive_same_spk(paragraphs, max_gap_ms: int = 5000, max_dur_ms: int = 180_000, max_chars: int = 1500):
    if not paragraphs: return paragraphs
    out = [dict(paragraphs[0])]
    for p in paragraphs[1:]:
        last = out[-1]
        same_spk = last["speaker"] == p["speaker"]
        neither_overlap = not last.get("overlap", False) and not p.get("overlap", False)
        gap = p["start"] - last["end"]
        merged_dur = p["end"] - last["start"]
        merged_chars = len(last["text"]) + len(p["text"])
        if (same_spk and neither_overlap and gap <= max_gap_ms 
            and merged_dur <= max_dur_ms and merged_chars <= max_chars):
            last["end"] = p["end"]
            last["text"] += p["text"]
        else:
            out.append(dict(p))
    return out


def _fmt_time(ms: int) -> str:
    s = ms // 1000
    return f"{s//60}:{s%60:02d}"

def _fmt_time_range(start_ms: int, end_ms: int) -> str:
    return f"{_fmt_time(start_ms)}-{_fmt_time(end_ms)}"

def _dump_debug(debug_dir, name: str, data):
    if not debug_dir: return
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_corrections(path: str = "corrections.json") -> dict:
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("rules"), list):
        rules = [r for r in data["rules"] if "wrong" in r and "right" in r]
        rules.sort(key=lambda r: -len(r["wrong"]))
        return {r["wrong"]: r["right"] for r in rules}
    return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str) and not k.startswith("_")}

def load_hotwords(path: str = "corrections.json") -> str:
    if not os.path.exists(path): return ""
    with open(path, "r", encoding="utf-8") as f: data = json.load(f)
    words = data.get("hotwords", []) if isinstance(data, dict) else []
    return " ".join(str(w) for w in words if w)

def apply_corrections(text: str, table: dict) -> str:
    for w, r in table.items():
        if w and w != r: text = text.replace(w, r)
    return text


def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)) + 1e-10)
    return 20.0 * np.log10(rms)

def extract_chunk_embs(wav, chunks, min_dbfs: float = -50.0):
    embs, kept, dropped = [], [], 0
    for s_ms, e_ms in chunks:
        s, e = int(s_ms / 1000 * SR), int(e_ms / 1000 * SR)
        seg = wav[s:e]
        if len(seg) < SR * 0.5: continue
        if _rms_dbfs(seg) < min_dbfs:
            dropped += 1
            continue
        emb = extract_embedding_from_wave(seg, sr=SR)
        embs.append(emb)
        kept.append((s_ms, e_ms))
    if dropped: print(f"  [NOISE] 丢弃低能量 chunk {dropped} 个 (dBFS<{min_dbfs})")
    return (np.stack(embs) if embs else np.zeros((0, 192), dtype=np.float32)), kept

def cluster_embs(embs: np.ndarray, num_spk: int = None, threshold: float = 0.7, whiten: bool = False):
    from sklearn.cluster import AgglomerativeClustering
    if len(embs) == 0: return np.array([], dtype=int)
    if len(embs) == 1: return np.array([0], dtype=int)
    if whiten:
        mean = embs.mean(axis=0, keepdims=True)
        embs = embs - mean
    normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    if num_spk:
        clf = AgglomerativeClustering(n_clusters=num_spk, metric="cosine", linkage="average")
    else:
        clf = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, metric="cosine", linkage="average")
    return clf.fit_predict(normed)

def compute_centroids(embs: np.ndarray, labels: np.ndarray) -> dict:
    centroids = {}
    for lab in set(int(x) for x in labels):
        members = embs[labels == lab]
        c = members.mean(axis=0)
        centroids[lab] = c / (np.linalg.norm(c) + 1e-8)
    return centroids

def match_centroid(emb: np.ndarray, centroids: dict):
    e = emb / (np.linalg.norm(emb) + 1e-8)
    best_spk, best_sim = None, -1.0
    for spk, c in centroids.items():
        sim = float(np.dot(e, c))
        if sim > best_sim: best_spk, best_sim = spk, sim
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
        if not labs: continue
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
    if not labels.size: return []
    raw = [(int(cs), int(ce), int(lab)) for (cs, ce), lab in zip(kept_chunks, labels)]
    if smooth and len(raw) >= 3:
        smoothed = list(raw)
        for i in range(1, len(raw) - 1):
            if raw[i-1][2] == raw[i+1][2] and raw[i][2] != raw[i-1][2]:
                smoothed[i] = (raw[i][0], raw[i][1], raw[i-1][2])
        raw = smoothed
    turns = [list(raw[0])]
    for cs, ce, lab in raw[1:]:
        if lab == turns[-1][2]: turns[-1][1] = ce
        else: turns.append([cs, ce, lab])
    return [tuple(t) for t in turns]


# ─────────── 主流程 ───────────
def main():
    ap = argparse.ArgumentParser()
    file_nm = "04.21公交数据要素比赛决赛培训"
    ap.add_argument("--wav",default=f"data/{file_nm}.mp3", help="输入音频")
    ap.add_argument("--num-spk", type=int, default=3, help="已知人数")
    
    # [新增] 音量提升机制
    ap.add_argument("--volume-boost", type=float, default=0.0,
                    help="放大音量倍数。设为 0.0 时执行自动峰值标准化(解决声音过小被静音的问题)。")
    
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--enroll-db", default=None)
    ap.add_argument("--match-threshold", type=float, default=0.55)
    ap.add_argument("--hotword", default="")
    ap.add_argument("--itn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wetext-itn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--vad", choices=["fsmn", "silero"], default="silero")
    ap.add_argument("--vad-fsmn-max-seg-ms", type=int, default=60000)
    ap.add_argument("--vad-fsmn-end-sil-ms", type=int, default=800)
    ap.add_argument("--scd", action="store_true", default=False)
    ap.add_argument("--scd-min-seg-s", type=float, default=5.0)
    ap.add_argument("--scd-window-s", type=float, default=0.75)
    ap.add_argument("--scd-hop-s", type=float, default=0.1)
    ap.add_argument("--scd-threshold", type=float, default=0.5)
    ap.add_argument("--scd-min-spk-dur-s", type=float, default=0.8)
    ap.add_argument("--embedder-model", default="iic/speech_eres2net_large_200k_sv_zh-cn_16k-common")
    ap.add_argument("--whiten", action="store_true", default=True)
    ap.add_argument("--no-bss", action="store_true")
    ap.add_argument("--bss-min-dur", type=float, default=1.0)
    ap.add_argument("--bss-max-dur", type=float, default=30.0)
    ap.add_argument("--bss-dump-dir", default=None)
    ap.add_argument("--vad-show-n", type=int, default=20)
    ap.add_argument("--diar-mode", choices=["segment", "chunk"], default="chunk")
    ap.add_argument("--diar-smooth", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-dbfs", type=float, default=-65.0)
    ap.add_argument("--merge-gap", type=int, default=400)
    ap.add_argument("--min-dur", type=int, default=400)
    ap.add_argument("--chunk-max", type=int, default=2000)
    ap.add_argument("--chunk-hop", type=int, default=1000)
    ap.add_argument("--output", default=f"result/{file_nm}_sensevoice.json")
    ap.add_argument("--output-txt", default=f"result/{file_nm}_sensevoice.txt")
    ap.add_argument("--para-gap", type=int, default=800)
    ap.add_argument("--para-max-dur", type=int, default=60000)
    ap.add_argument("--para-max-chars", type=int, default=600)
    ap.add_argument("--post-merge-gap", type=int, default=5000)
    ap.add_argument("--post-merge-max-dur", type=int, default=180000)
    ap.add_argument("--post-merge-max-chars", type=int, default=1500)
    ap.add_argument("--debug-dir", default=f"result/debug/{file_nm}_sense")
    args = ap.parse_args()

    if args.embedder_model:
        from speaker_db import set_embedder_model
        set_embedder_model(args.embedder_model)

    # ─── 1. 加载 ───
    print(f"\n=== [1/5] 加载音频 ===")
    wav, _ = librosa.load(args.wav, sr=SR, mono=True)
    dur_s = len(wav) / SR
    print(f"  时长: {dur_s:.1f}s ({dur_s/60:.1f} min)")

    # ─────────── [新增] 放大原声音量逻辑 ───────────
    if args.volume_boost != 1.0:
        if args.volume_boost == 0.0:
            peak = np.max(np.abs(wav))
            if peak > 0:
                wav = wav / peak
            print(f"  [VOLUME] 已执行自动峰值标准化，原峰值 {peak:.4f}，现已整体拉满到最大音量")
        else:
            wav = wav * args.volume_boost
            wav = np.clip(wav, -1.0, 1.0)
            print(f"  [VOLUME] 音量已强制放大 {args.volume_boost} 倍，并执行防爆音压限")
    # ──────────────────────────────────────────────

    # ─── 2. 降噪 ───
    if not args.denoise:
        print(f"\n=== [2/5] 降噪已跳过 ===")
    else:
        print(f"\n=== [2/5] FRCRN 降噪 ===")
        wav = run_denoise(wav, sr=SR)
        _release_model(main_bss, "_denoise_pipe")
        _free_gpu("after denoise")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, SR)
        clean_path = tmp.name

    try:
        # ─── 3. VAD + 切段 + 聚类 ───
        print(f"\n=== [3/5] VAD ({args.vad}) + 声纹聚类 ===")
        raw_segments = run_vad(
            clean_path, engine=args.vad,
            fsmn_max_seg_ms=args.vad_fsmn_max_seg_ms,
            fsmn_end_sil_ms=args.vad_fsmn_end_sil_ms,
        )
        segments = merge_segments(raw_segments, gap_threshold_ms=args.merge_gap, min_duration_ms=args.min_dur)

        if args.scd:
            segments, _ = split_segments_by_scd(
                wav, segments,
                min_segment_for_scd_s=args.scd_min_seg_s,
                window_s=args.scd_window_s, hop_s=args.scd_hop_s,
                distance_threshold=args.scd_threshold,
                min_speaker_dur_s=args.scd_min_spk_dur_s,
                sr=SR, verbose=False
            )

        chunks = chunk_long_segments(segments, max_dur_ms=args.chunk_max, hop_ms=args.chunk_hop)
        embs, kept_chunks = extract_chunk_embs(wav, chunks, min_dbfs=args.min_dbfs)

        if len(embs) < 2: return

        labels = cluster_embs(embs, num_spk=args.num_spk, threshold=args.threshold, whiten=args.whiten)
        centroids = compute_centroids(embs, labels)

        if args.diar_mode == "chunk":
            turns = chunks_to_fine_turns(kept_chunks, labels, smooth=args.diar_smooth)
        else:
            turns = segments_to_turns(segments, chunks, kept_chunks, labels)

        # ─── 4. BSS 重叠区检测 ───
        overlap_ranges = []
        if not args.no_bss:
            print(f"\n=== [4/5] BSS 重叠检测 ===")
            for idx, (s_ms, e_ms, spk) in enumerate(turns):
                dur = (e_ms - s_ms) / 1000.0
                seg = wav[int(s_ms/1000*SR):int(e_ms/1000*SR)]
                if dur < args.bss_min_dur or dur > args.bss_max_dur: continue
                try:
                    with torch.no_grad():
                        wavs = run_bss(seg, sr_in=SR)
                        chk = check_bss_output(wavs)
                except: continue
                if chk["mode"] == "overlap":
                    with torch.no_grad():
                        spk_pair = []
                        for w in wavs:
                            emb = extract_embedding_from_wave(w, sr=SR)
                            mspk, _ = match_centroid(emb, centroids)
                            spk_pair.append(mspk)
                    overlap_ranges.append((s_ms, e_ms, spk_pair, [w.copy() for w in wavs]))
            _release_model(main_bss, "_bss_pipe")
            _free_gpu("after bss")

        # ─── 5. 逐段 ASR + 染色 + 段落化 ───
        # [核心修改]: 不再用 asr_full 读整段音频，而是按 turns 切好的音频块逐个翻译
        print(f"\n=== [5/5] 按 Turn 逐段 ASR (SenseVoiceSmall) ===")
        effective_hotword = " ".join(filter(None, [args.hotword, load_hotwords()])).strip()
        sentences = []
        corrections = load_corrections()

        # 正常回合 ASR
        for idx, (s_ms, e_ms, spk) in enumerate(turns):
            seg = wav[int(s_ms/1000*SR):int(e_ms/1000*SR)]
            if len(seg) < int(SR * 0.2): continue  # 过滤极短段
            
            try:
                text = asr_wav(seg, sr=SR, hotword=effective_hotword)
            except Exception as e:
                text = ""

            if not text.strip(): continue
            
            if args.itn: text = apply_itn(text, use_wetext=args.wetext_itn)
            if corrections: text = apply_corrections(text, corrections)

            sentences.append({
                "start": s_ms, "end": e_ms,
                "text": text, "speaker": spk,
                "overlap": False
            })
            print(f"  [ASR] {idx+1}/{len(turns)}: {_fmt_time_range(s_ms, e_ms)} spk_{spk} | {text}")

        # 处理重叠区: 从正常结果中剔除被重叠覆盖的句子，并补充分离出来的音频结果
        if overlap_ranges:
            replaced = 0
            for os_ms, oe_ms, spk_pair, sep_wavs in overlap_ranges:
                if spk_pair[0] == spk_pair[1] or spk_pair[0] is None or spk_pair[1] is None: continue
                before = len(sentences)
                sentences = [s for s in sentences if not (os_ms <= (s["start"] + s["end"]) / 2.0 <= oe_ms)]
                replaced += before - len(sentences)
                
                for idx, w in enumerate(sep_wavs):
                    try:
                        text = asr_wav(w, sr=SR, hotword=effective_hotword)
                    except:
                        text = ""
                    if not text.strip(): continue
                    if args.itn: text = apply_itn(text, use_wetext=args.wetext_itn)
                    if corrections: text = apply_corrections(text, corrections)
                    sentences.append({
                        "start": os_ms, "end": oe_ms,
                        "text": text, "speaker": spk_pair[idx],
                        "overlap": True,
                    })
            print(f"  [OVERLAP-RE-ASR] 已替换 {replaced} 句重叠区单路文本，追加双路并行 ASR")
            sentences.sort(key=lambda r: r["start"])

        # 真名映射
        if args.enroll_db and os.path.exists(args.enroll_db):
            db = SpeakerDB(args.enroll_db)
            spk_names = {}
            for spk, c in centroids.items():
                name, sim = db.match(c, threshold=args.match_threshold)
                spk_names[spk] = name if name else f"spk_{spk}"
        else:
            spk_names = {spk: f"spk_{spk}" for spk in centroids}

        # 整理段落
        for s in sentences:
            s["speaker"] = spk_names.get(s["speaker"], f"spk_{s['speaker']}" if s["speaker"] is not None else "spk_unknown")
            
        paragraphs = group_paragraphs(sentences, max_gap_ms=args.para_gap, max_paragraph_dur_ms=args.para_max_dur, max_paragraph_chars=args.para_max_chars)
        paragraphs = merge_consecutive_same_spk(paragraphs, max_gap_ms=args.post_merge_gap, max_dur_ms=args.post_merge_max_dur, max_chars=args.post_merge_max_chars)

        def _header(p):
            tag = " [可能重叠]" if p.get("overlap") else ""
            return f"{p['speaker']} - {_fmt_time_range(p['start'], p['end'])}{tag}"

        print(f"\n=== 转写结果 ({len(paragraphs)} 段) ===")
        for p in paragraphs:
            print(f"\n{_header(p)}")
            print(p["text"])

        # ─── 输出文件 ───
        if args.output:
            out_dir = os.path.dirname(args.output)
            if out_dir: os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump([{"start": round(p["start"]/1000, 2), "end": round(p["end"]/1000, 2), "speaker": p["speaker"], "overlap": p.get("overlap", False), "text": p["text"]} for p in paragraphs], f, ensure_ascii=False, indent=2)

        if args.output_txt:
            out_dir = os.path.dirname(args.output_txt)
            if out_dir: os.makedirs(out_dir, exist_ok=True)
            with open(args.output_txt, "w", encoding="utf-8") as f:
                f.write(f"{os.path.basename(args.wav)}\n\n")
                for p in paragraphs:
                    f.write(f"{_header(p)}\n")
                    f.write(f"{p['text']}\n\n")

    finally:
        try: os.unlink(clean_path)
        except OSError: pass

if __name__ == "__main__":
    main()