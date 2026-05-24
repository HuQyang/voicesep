"""
端到端语音转文字 demo

管线:
  音频
   → [1] FRCRN 降噪          (可关 --no-denoise)
   → [2] FSMN-VAD 切段 + 合并 + 滑窗 chunk
   → [3] ERes2NetV2 emb + 聚类 → 每段说话人 spk_id
   → [4] 对长 turn 跑 Mossformer2 BSS + post-check:
         - overlap → 两路, 各自匹配回聚类 centroid 拿 spk_id
         - single/noise → 用原始 turn
       (可关 --no-bss)
   → [5] Paraformer ASR + corrections.json 纠错
   → 输出 [(start, end, speaker, text), ...]

用法:
  python demo_pipeline.py --wav data/xxx.mp3
  python demo_pipeline.py --wav xxx.mp3 --num-spk 3 --output result.json
  python demo_pipeline.py --wav xxx.mp3 --enroll-db speakers/db.npz   # 替换为真名
  python demo_pipeline.py --wav xxx.mp3 --no-denoise --no-bss         # 关掉额外步骤
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
from model import FunASRNano


import main_bss  # 需要操作其内部 _denoise_pipe / _bss_pipe 引用以释放显存
from speaker_db import extract_embedding_from_wave, SpeakerDB
from main_diarization import merge_segments, chunk_long_segments
from main_bss import run_denoise, run_bss, check_bss_output
from scd import split_segments_by_scd


SR = 16000


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


# ─────────── 懒加载 VAD / ASR ───────────
_vad_fsmn = None
_vad_silero = None
_asr = None


def run_vad_fsmn(wav_path: str, max_segment_ms: int = 60000, max_end_silence_ms: int = 800):
    """
    FSMN-VAD, 返回 [[start_ms, end_ms], ...]

    max_segment_ms:      单段上限(ms). 默认 60000=60s 太宽容, 长对话里两人快速轮替会被
                         整段并入. 会议场景建议 5000-8000 强制切, 配合 cam++ 重聚类
                         能减少 speaker 被吞.
    max_end_silence_ms:  尾静音超过这个就关段(ms). 默认 800, 调小 (e.g. 300) 对快速轮替
                         更敏感, 但会切得碎.
    """
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
    """整段 ASR: Paraformer-large + 内置 VAD + CT-Punc + 时间戳"""
    global _asr
    if _asr is None:
        print("[asr] 加载 Paraformer + FSMN-VAD + CT-Punc...")
        _asr = AutoModel(
            # model="paraformer-zh",
            # model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            model = "FunAudioLLM/Fun-ASR-Nano-2512",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            disable_update=True,
        )
    return _asr

# def get_asr():
#     """
#     换回 Paraformer-zh，但【绝对不要】加 vad_model 参数！
#     纯中文底座，配合 CT-Punc，对 BSS 分离后的电音伪影抵抗力极强，绝不输出外语。
#     """
#     global _asr
#     if _asr is None:
#         print("[asr] 加载 Paraformer-zh (纯中文稳定底座)...")
#         _asr = AutoModel(
#             model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
#             punc_model="ct-punc",       # 加上标点模型
#             disable_update=True,
#             # 注意：千万不要写 vad_model="fsmn-vad"
#         )
#     return _asr

_CJK_RE = re.compile(r"[一-鿿]")
_SPACE_BETWEEN_CJK = re.compile(r"(?<=[一-鿿])\s+(?=[一-鿿])")
_SENT_SPLIT = re.compile(r"([。！？!?；;]+)")


def _clean_text(text: str) -> str:
    """Paraformer 默认 token 间带空格, 中文要去掉; 英文/数字间空格保留"""
    return _SPACE_BETWEEN_CJK.sub("", text).strip()


# ─────────── ITN (Inverse Text Normalization) ───────────
_DIGIT_CHARS = "零一二三四五六七八九幺两"
_DIGIT_MAP = {"零": "0", "一": "1", "二": "2", "三": "3", "四": "4",
              "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
              "幺": "1", "两": "2"}
_DIGIT_RUN = re.compile(f"[{_DIGIT_CHARS}]{{2,}}")

_UNIT_VAL = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100_000_000}
# 正式数词: 必含至少一个单位 (十/百/千/万/亿), 前后可有数字
_FORMAL_NUM = re.compile(
    r"[零一二三四五六七八九两幺]?"
    r"(?:[十百千万亿][零一二三四五六七八九两幺]?)+"
)


def _parse_formal_cn_num(s: str):
    """解析'四十三'→43, '一百五十'→150, '三千八百七十六'→3876. 失败返 None.

    重要修正:
      - 孤立的 '万' / '亿' 不视为数字, 返回 None (避免 '一千多万' → '1000多10000')
      - 孤立的 '十' / '百' / '千' 仍按经典语意处理 ('十块' → '10块')
    """
    if not s:
        return None
    total = 0
    section = 0     # 当前万段内的累积
    digit = 0
    has_digit = False    # 是否见过真正的数字字符 (零~九/两/幺)
    for ch in s:
        if ch in _DIGIT_MAP:
            digit = int(_DIGIT_MAP[ch])
            has_digit = True
        elif ch in _UNIT_VAL:
            unit = _UNIT_VAL[ch]
            # 关键修正: 孤立的 万/亿 (前面没数字, section 也空) → 视为单位字
            # 这样 "一千多万" 中的 "万" 不会被当成 10000
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
    # 必须有真正的数字字符才视为数字
    return (total + section + digit) if has_digit else None

_wetext_itn = None


def get_wetext_itn():
    """
    懒加载中文 ITN.
    优先级:
      1) WeTextProcessing -> itn.chinese.inverse_normalizer.InverseNormalizer
      2) wetext (轻量替代) -> Normalizer(lang='zh', operator='itn')
    都装不上 -> 返 None, 走正则 quick_itn 兜底
    """
    global _wetext_itn
    if _wetext_itn is None:
        # 1) WeTextProcessing
        try:
            from itn.chinese.inverse_normalizer import InverseNormalizer
            _wetext_itn = InverseNormalizer()
            print("[itn] 加载 WeTextProcessing.InverseNormalizer (itn.chinese)")
            return _wetext_itn
        except Exception:
            pass
        # 2) wetext (轻量包, 与上面是不同 PyPI 包)
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


# 模糊量词前缀: 出现在数词前表"大约/一些", 此时数词不该转阿拉伯数字
# 例: "几十万人" 不能变 "几100000人"; "上百" 不能变 "上100"; "好几千" 不能变 "好几1000"
_APPROX_PREFIX_2 = ("好几",)     # 2 字前缀
_APPROX_PREFIX_1 = ("几", "数", "上")   # 1 字前缀


def _has_approx_prefix(text: str, pos: int) -> bool:
    """检查 text[pos] 这个位置前面是否是模糊量词修饰"""
    if pos >= 2 and text[pos - 2:pos] in _APPROX_PREFIX_2:
        return True
    if pos >= 1 and text[pos - 1] in _APPROX_PREFIX_1:
        return True
    return False


def quick_itn(text: str) -> str:
    """
    简单 ITN:
      1) 正式数词 ("四十三"→"43", "一百五十"→"150", "三千八百七十六"→"3876")
      2) 逐位读数 ("三八七三"→"3873", "幺二零三九"→"12039")

    跳过转换的情况:
      - 前面是模糊量词 ("几十万"→保持, "上百"→保持, "好几千"→保持)

    顺序很重要: 先正式数词, 再逐位 (否则"四十三"的"四"会被逐位规则吃掉).
    """
    # 1) 正式数词
    def repl_formal(m):
        if _has_approx_prefix(m.string, m.start()):
            return m.group()   # 跳过 (几十万 / 上百 等)
        n = _parse_formal_cn_num(m.group())
        return str(n) if n is not None else m.group()
    text = _FORMAL_NUM.sub(repl_formal, text)

    # 2) 剩余的连续中文数字 (逐位读数)
    def repl_run(m):
        if _has_approx_prefix(m.string, m.start()):
            return m.group()
        return "".join(_DIGIT_MAP.get(c, c) for c in m.group())
    text = _DIGIT_RUN.sub(repl_run, text)
    return text


def smart_large_number_format(text: str) -> str:
    """
    可读性优化: 把"裸"的大整数转回"X万 / X亿"形式.
    示例:
      5010000人  →  501万人
      1200000元  →  120万元
      230000000  →  2.3亿
      12345        →  1.2万
      1234       →  保持不变 (小于 1 万)

    注意:
      - 跳过紧贴前/后字母数字的 (避免破坏 ID/电话/错误码)
      - 跳过紧贴 "年/号/期/章/节/楼/层/室" 等单位 (这些数字保留原样)
      - 保留前缀 "几/数/上/好几" 的模糊量词
    """
    # 后面紧贴的"应保留为纯阿拉伯数字"的单位
    KEEP_DIGIT_UNITS = "年号期章节楼层室届任季度版页页码秒分时第"

    def repl(m):
        s = m.group()
        n = int(s)

        # 边界检查: 前后是字母/数字/小数点 → 跳过 (像 ID/电话号)
        start, end = m.start(), m.end()
        full = m.string
        if start > 0 and (full[start - 1].isalnum() or full[start - 1] == "."):
            return s
        if end < len(full) and (full[end].isalnum() or full[end] == "."):
            return s
        # 后面紧贴特定单位 → 保留数字原样 (1986年, 第3章)
        if end < len(full) and full[end] in KEEP_DIGIT_UNITS:
            return s
        # 前面是模糊量词 → 跳过 (理论上前面 quick_itn 已处理, 这里再保险)
        if start > 0 and full[start - 1] in "几数上多":
            return s
        if start > 1 and full[start - 2:start] == "好几":
            return s

        if n < 10000:
            return s
        if n >= 100000000:
            yi = n / 100000000
            return f"{int(yi)}亿" if yi == int(yi) else f"{yi:.1f}亿"
        # 1 万 ~ 1 亿 之间
        wan = n / 10000
        return f"{int(wan)}万" if wan == int(wan) else f"{wan:.1f}万"

    return re.sub(r"\d{5,}", repl, text)


_POST_ITN_UNIT_FIXES = [
    # 兜底: 修复 "数字多10000" → "数字多万" (其他模块/旧数据残留)
    (re.compile(r"(\d+(?:\.\d+)?)\s*多\s*10000(?!\d)"), r"\1多万"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*多\s*100000000(?!\d)"), r"\1多亿"),
    # "数字10000" 中间无修饰 (兜底, 但 smart_large_number_format 应该已经处理掉)
    (re.compile(r"(?<![\d.])10000(?![\d.])"), r"万"),
    (re.compile(r"(?<![\d.])100000000(?![\d.])"), r"亿"),
]


def _post_fix_itn_units(text: str) -> str:
    """ITN 后兜底: 修 '数字多10000' / 孤立 10000 这类粘连错误"""
    for pat, repl in _POST_ITN_UNIT_FIXES:
        text = pat.sub(repl, text)
    return text


def apply_itn(text: str, use_wetext: bool = True) -> str:
    """
    ITN 链式:
      1) WeTextProcessing FST: 精准转规范数词 (一百五十→150, 三千八百七十六→3876)
         不抛异常但部分场景会"原样返回"
      2) quick_itn 正则: 扫剩下的中文数字 (尤其逐位读数 三八七三→3873)
         + 跳过模糊量词修饰 (几十万 / 上百 保留中文)
      3) smart_large_number_format: 把"裸"大整数转回"X万 / X亿"提升可读性
         (5010000 → 501万, 1234567 → 123.5万)
      4) _post_fix_itn_units: 兜底修复 "数字多10000" / 孤立 10000 等粘连错误
    """
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


def asr_wav(wav: np.ndarray, sr: int = SR, hotword: str = "") -> str:
    """对一段 numpy 波形跑 ASR, 返回清理后的纯文本 (用于 BSS 分离后的单路重转)"""
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

    优先用 FunASR 自带的 sentence_info; 退化到按标点切 + token timestamp 推算时间.
    """
    asr = get_asr()
    res = asr.generate(input=wav_path, batch_size_s=300, hotword=hotword)
    if not res:
        return []
    r = res[0]

    # 1) FunASR 新版直接给 sentence_info
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

    # 2) 退化: 按 timestamp + 标点切
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
        # 标点 / 其他
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
    """
    把 turn 的 spk 染色到句子.
    句子按它的中点落在哪个 turn 决定 spk; 找不到时按最近 turn 兜底.
    """
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


def group_paragraphs(
    sentences_with_spk,
    max_gap_ms: int = 800,
    max_paragraph_dur_ms: int = 60_000,
    max_paragraph_chars: int = 600,
):
    """
    把相邻同 spk 句子合并成段落. 触发新段落:
      - spk 变化
      - 时间间隙 > max_gap_ms  (真实静音)
      - 当前段落已达 max_paragraph_dur_ms (~60s 节奏)
      - 当前段落已达 max_paragraph_chars (太长一段读不下去)
    """
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


def merge_consecutive_same_spk(
    paragraphs,
    max_gap_ms: int = 5000,
    max_dur_ms: int = 180_000,
    max_chars: int = 1500,
):
    """
    后处理: 把相邻同 spk 段融合.
    条件 (全部满足):
      - speaker 相同
      - 两段都 overlap=False (重叠段不参与合并, 保留独立显示)
      - 时间间隔 <= max_gap_ms
      - 合并后总时长 <= max_dur_ms   (硬上限)
      - 合并后总字数 <= max_chars    (硬上限)
    后两条是硬上限, 避免 cam++ 把多人坍缩成同一个 spk 时, post-merge 无脑
    把整段会议串成一段超长文字 (人眼读不下来, 下游 LLM 也吃不消).
    """
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
        if (same_spk and neither_overlap
                and gap <= max_gap_ms
                and merged_dur <= max_dur_ms
                and merged_chars <= max_chars):
            last["end"] = p["end"]
            last["text"] += p["text"]
        else:
            out.append(dict(p))
    return out


def _fmt_time(ms: int) -> str:
    s = ms // 1000
    return f"{s//60}:{s%60:02d}"


def _fmt_time_range(start_ms: int, end_ms: int) -> str:
    """0:00-0:15 形式, 展示时段而非只是起点"""
    return f"{_fmt_time(start_ms)}-{_fmt_time(end_ms)}"


# ─────────── 调试: 阶段性中间结果落盘 ───────────
def _dump_debug(debug_dir, name: str, data):
    """
    把任意可序列化的 dict/list 写到 debug_dir/<name>.
    debug_dir 为 None/空 时跳过. 用于阶段性展示 (VAD / turns / BSS / ASR / final).
    """
    if not debug_dir:
        return
    os.makedirs(debug_dir, exist_ok=True)
    path = os.path.join(debug_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [debug] {path}")


# ─────────── 纠错 ───────────
def load_corrections(path: str = "corrections.json") -> dict:
    """
    解析 corrections.json. 支持两种结构:
      1) 新结构 (推荐): {"rules": [{"wrong": "X", "right": "Y"}, ...], "hotwords": [...]}
      2) 旧扁平 dict: {"X": "Y", ...}
    返回 {wrong: right} 字典 (供 apply_corrections 用).
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("rules"), list):
        # 新结构: 长串优先排序, 避免 "电池建康" 在 "建康" 之前被吃掉
        rules = [r for r in data["rules"] if "wrong" in r and "right" in r]
        rules.sort(key=lambda r: -len(r["wrong"]))
        return {r["wrong"]: r["right"] for r in rules}
    # 旧扁平 dict (忽略 _ 开头注释 key)
    return {k: v for k, v in data.items()
            if isinstance(k, str) and isinstance(v, str) and not k.startswith("_")}


def load_hotwords(path: str = "corrections.json") -> str:
    """从 corrections.json 的 hotwords[] 读热词列表, 拼成空格分隔的字符串供 ASR 用."""
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


# ─────────── 声纹相关 ───────────
def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)) + 1e-10)
    return 20.0 * np.log10(rms)


def extract_chunk_embs(wav, chunks, min_dbfs: float = -50.0):
    """对 chunk 列表提 emb, 同时按 dBFS 过滤"""
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
    return (np.stack(embs) if embs else np.zeros((0, 192), dtype=np.float32)), kept


def cluster_embs(embs: np.ndarray, num_spk: int = None, threshold: float = 0.7,
                 whiten: bool = False):
    """
    AHC 聚类.
    whiten=True: 聚类前先减全局均值. 远场/同房间多人录音, embedding 被房间声学
                 染色成共同分量, 减均值能拉开说话人差异 (speaker verification 经典 trick).
    """
    from sklearn.cluster import AgglomerativeClustering
    if len(embs) == 0:
        return np.array([], dtype=int)
    if len(embs) == 1:
        return np.array([0], dtype=int)

    if whiten:
        # 减去全局均值, 再归一化
        mean = embs.mean(axis=0, keepdims=True)
        embs = embs - mean
        print(f"  [WHITEN] 减去全局均值 (移除房间/通道共同分量)")

    normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    if num_spk:
        clf = AgglomerativeClustering(n_clusters=num_spk, metric="cosine", linkage="average")
    else:
        clf = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold,
            metric="cosine", linkage="average",
        )
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
    e = emb / (np.linalg.norm(emb) + 1e-8)
    best_spk, best_sim = None, -1.0
    for spk, c in centroids.items():
        sim = float(np.dot(e, c))
        if sim > best_sim:
            best_spk, best_sim = spk, sim
    return best_spk, best_sim


def segments_to_turns(segments, chunks, kept_chunks, labels):
    """
    [传统模式] 每段内 chunk label 投票 → 单段单 spk. 段内快速轮替会被多数吞掉.
    """
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
    """
    [细粒度模式] 按 chunk 顺序扫描, 同 spk 合并、不同 spk 切分.
    能 catch 段内快速轮替 (e.g. A说一句B接一句, gap<300ms 也能分开).

    smooth=True: 平滑掉孤立的单 chunk 异常岛 (X Y X → X X X), 防止 CAM++ 单点误识造成假切换.
    """
    if not labels.size:
        return []
    raw = [(int(cs), int(ce), int(lab)) for (cs, ce), lab in zip(kept_chunks, labels)]

    # 单点平滑: 中间 chunk 与左右都不同, 且左右相同 → 改成左右的 spk
    if smooth and len(raw) >= 3:
        smoothed = list(raw)
        for i in range(1, len(raw) - 1):
            if raw[i-1][2] == raw[i+1][2] and raw[i][2] != raw[i-1][2]:
                smoothed[i] = (raw[i][0], raw[i][1], raw[i-1][2])
        raw = smoothed

    # 合并相邻同 spk chunks 为 turns
    turns = [list(raw[0])]
    for cs, ce, lab in raw[1:]:
        if lab == turns[-1][2]:
            turns[-1][1] = ce
        else:
            turns.append([cs, ce, lab])
    return [tuple(t) for t in turns]


# ─────────── 主流程 ───────────
def main():
    ap = argparse.ArgumentParser()
    # file_nm = "2026-03-18 14_28 记录"
    # file_nm = "车辆管理业务研讨"
    file_nm = "04.21公交数据要素比赛决赛培训"
    # file_nm = "2025-09-30 15_56 记录"
    # file_nm = "钱部长数据融合沟通"
    ap.add_argument("--wav",default=f"data/{file_nm}.mp3", help="输入音频")
    ap.add_argument("--num-spk", type=int, default=5, help="已知人数 (最稳)")
    ap.add_argument("--threshold", type=float, default=0.65, help="AHC cosine 距离阈值")
    ap.add_argument("--enroll-db", default=None, help="可选: 声纹库, 把 spk_X 替换成真名")
    ap.add_argument("--match-threshold", type=float, default=0.55)
    ap.add_argument("--hotword", default="", help="")
    ap.add_argument("--itn", action=argparse.BooleanOptionalAction, default=True,
                    help="ITN: 中文数字 → 阿拉伯数字 (会议纪要刚需, 默认开)")
    ap.add_argument("--wetext-itn", action=argparse.BooleanOptionalAction, default=True,
                    help="优先 WeTextProcessing (装了就用, 没装自动退回 quick_itn). --no-wetext-itn 强制 quick_itn")
    ap.add_argument("--denoise", action=argparse.BooleanOptionalAction, default=False,
                    help="是否走 FRCRN 降噪 (--denoise / --no-denoise)")
    ap.add_argument("--vad", choices=["fsmn", "silero"], default="fsmn",
                    help="VAD 引擎: fsmn (中文会议默认) / silero (远场/低 SNR 更鲁棒)")
    ap.add_argument("--vad-fsmn-max-seg-ms", type=int, default=60000,
                    help="FSMN-VAD 单段上限(ms). 默认 60000 太宽容, 多人快速轮替会"
                         "整段并入. 会议场景建议 5000-8000 强切")
    ap.add_argument("--vad-fsmn-end-sil-ms", type=int, default=800,
                    help="FSMN-VAD 尾静音阈值(ms). 默认 800, 调小 (e.g. 300) "
                         "对快速轮替更敏感, 但会切得碎")
    ap.add_argument("--scd", action="store_true",
                    help="VAD 后追加 SCD: 用滑窗 cam++ embedding 距离检测说话人切换点, "
                         "把长 segment 切成单说话人子段. 解决 VAD 把交替对话并成一段的问题.")
    ap.add_argument("--scd-min-seg-s", type=float, default=5.0,
                    help="SCD 只处理 > 此时长(s) 的 segment, 短段不浪费算力")
    ap.add_argument("--scd-window-s", type=float, default=0.75,
                    help="SCD 滑窗大小(s). 默认 0.75, 小=对快速轮替敏感")
    ap.add_argument("--scd-hop-s", type=float, default=0.1,
                    help="SCD 滑窗 hop(s). 越小越精细, 但计算量大")
    ap.add_argument("--scd-threshold", type=float, default=0.5,
                    help="SCD 切点 cosine 距离阈值. 0.35 灵敏 / 0.5 默认 / 0.65 保守")
    ap.add_argument("--scd-min-spk-dur-s", type=float, default=0.8,
                    help="SCD 切出来的子段最短时长(s), 防过度切碎")
    ap.add_argument("--embedder-model", default="iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
                    help="声纹模型 (覆盖默认 ERes2NetV2). 推荐: "
                         "iic/speech_eres2net_base_200k_sv_zh-cn_16k-common (200k 训练, 192-d); "
                         "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common (最强, 512-d)")
    ap.add_argument("--whiten", action="store_true",
                    help="聚类前对所有 embedding 减全局均值. 远场/同房间多人录音, "
                         "embedding 被共同声学染色, 减均值能拉开说话人差异. 推荐打开.")
    ap.add_argument("--no-bss", action="store_true", help="跳过重叠检测/分离")
    ap.add_argument("--bss-min-dur", type=float, default=1.0,
                    help="turn 时长 >= 此值(s) 才跑 BSS 检测重叠")
    ap.add_argument("--bss-max-dur", type=float, default=30.0,
                    help="turn 时长 > 此值(s) 跳过 BSS (Mossformer2 长输入 O(L²) 吃显存)")
    ap.add_argument("--bss-dump-dir", default="bss_out",
                    help="BSS 检测到重叠时, 把原始混合 + 分离两路 wav 落盘到此目录 (展示用)")
    ap.add_argument("--vad-show-n", type=int, default=20,
                    help="VAD 打印前 N 个段的 start-end (0=全打印, -1=不打印)")
    ap.add_argument("--diar-mode", choices=["segment", "chunk"], default="segment",
                    help="diar 模式: segment=段内投票(传统稳); chunk=每个 chunk 独立投票(能 catch 快速轮替)")
    ap.add_argument("--diar-smooth", action=argparse.BooleanOptionalAction, default=True,
                    help="chunk 模式时是否平滑孤立点 (X Y X → X X X)")
    ap.add_argument("--min-dbfs", type=float, default=-45.0,
                    help="emb 提取前丢弃低能量 chunk (远场设更松, 比如 -55)")
    ap.add_argument("--merge-gap", type=int, default=300,
                    help="VAD 后相邻段间隙 < 此值(ms) 则合并 (调大 → 段更连续, 同人不易裂)")
    ap.add_argument("--min-dur", type=int, default=600,
                    help="合并后短于此值(ms) 的段丢弃 (声纹不稳)")
    ap.add_argument("--chunk-max", type=int, default=3000,
                    help="长段滑窗最大长度(ms), 聚类粒度")
    ap.add_argument("--chunk-hop", type=int, default=1500,
                    help="长段滑窗 hop(ms)")
    ap.add_argument("--output", default=f"result/{file_nm}_seacopara2.json", help="可选: 保存 .json")
    ap.add_argument("--output-txt", default=f"result/{file_nm}_seacopara2.txt", help="可选: 保存可读 .txt (一行一条 turn)")
    ap.add_argument("--para-gap", type=int, default=800,
                    help="段落分割: 句子间隙(ms) > 此值时另起一段")
    ap.add_argument("--para-max-dur", type=int, default=60000,
                    help="段落最大时长(ms), 超过强制另起")
    ap.add_argument("--para-max-chars", type=int, default=600,
                    help="段落最大字符数, 超过强制另起")
    ap.add_argument("--post-merge-gap", type=int, default=5000,
                    help="后合并: 同 spk + 都非 overlap, 间隔(ms)<=此值则合并")
    ap.add_argument("--post-merge-max-dur", type=int, default=180000,
                    help="后合并硬上限: 合并后段最大时长(ms), 默认 3 分钟")
    ap.add_argument("--post-merge-max-chars", type=int, default=1500,
                    help="后合并硬上限: 合并后段最大字符数, 默认 1500")
    ap.add_argument("--debug-dir", default=f"result/debug/{file_nm}_para",
                    help="若指定, 每个阶段 dump 一份 JSON 到此目录 "
                         "(vad/turns/bss/asr/final), 用于演示和调参定位")
    args = ap.parse_args()

    # 在任何 embedding 调用之前切换模型
    if args.embedder_model:
        from speaker_db import set_embedder_model
        set_embedder_model(args.embedder_model)

    # ─── 1. 加载 ───
    print(f"\n=== [1/5] 加载音频 ===")
    wav, _ = librosa.load(args.wav, sr=SR, mono=True)
    dur_s = len(wav) / SR
    print(f"  文件: {args.wav}")
    print(f"  时长: {dur_s:.1f}s ({dur_s/60:.1f} min)")

    # ─── 2. 降噪 ───
    if not args.denoise:
        print(f"\n=== [2/5] 降噪已跳过 ===")
    else:
        print(f"\n=== [2/5] FRCRN 降噪 ===")
        rms_before = _rms_dbfs(wav)
        wav = run_denoise(wav, sr=SR)
        rms_after = _rms_dbfs(wav)
        print(f"  RMS dBFS: {rms_before:.1f} → {rms_after:.1f}")
        # 降噪后 FRCRN 不再用, 释放显存
        _release_model(main_bss, "_denoise_pipe")
        _free_gpu("after denoise")

    # 把降噪后的 wav 写临时文件供 VAD / ASR 用
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
        print(f"  [VAD]   {len(raw_segments)} segments")
        for i, (s_ms, e_ms) in enumerate(raw_segments[: args.vad_show_n if args.vad_show_n > 0 else len(raw_segments)]):
            print(f"    seg{i:03d}: {_fmt_time_range(s_ms, e_ms)}  ({(e_ms-s_ms)/1000:.2f}s)")
        if args.vad_show_n > 0 and len(raw_segments) > args.vad_show_n:
            print(f"    ... 还有 {len(raw_segments) - args.vad_show_n} 段未列出 (--vad-show-n 控制)")

        _dump_debug(args.debug_dir, "01_vad_raw.json", {
            "engine": args.vad,
            "count": len(raw_segments),
            "segments": [{"start_s": round(s/1000, 2), "end_s": round(e/1000, 2),
                          "dur_s": round((e-s)/1000, 2)}
                         for s, e in raw_segments],
        })

        segments = merge_segments(raw_segments, gap_threshold_ms=args.merge_gap, min_duration_ms=args.min_dur)
        print(f"  [MERGE] {len(segments)} segments (gap<{args.merge_gap}ms 合并, <{args.min_dur}ms 丢弃)")

        # ── SCD: 长段按 embedding 距离切说话人切换点 ──
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
            # 距离统计帮你定位阈值
            if scd_stats["per_segment"]:
                d_maxs = [s["distance_max"] for s in scd_stats["per_segment"] if s.get("distance_max") is not None]
                d_means = [s["distance_mean"] for s in scd_stats["per_segment"] if s.get("distance_mean") is not None]
                if d_maxs:
                    print(f"  [SCD]   距离分布: max 中位 {sorted(d_maxs)[len(d_maxs)//2]:.3f} / "
                          f"mean 中位 {sorted(d_means)[len(d_means)//2]:.3f} "
                          f"(当前阈值 {args.scd_threshold:.2f}; "
                          f"如果 0 切点请把阈值调到约 max 中位的 80%)")
            # SCD 后段长前 20
            show_n = min(20, len(segments))
            print(f"  [SCD]   切分后前 {show_n} 段:")
            for i, (s, e) in enumerate(segments[:show_n]):
                print(f"    sub{i:03d}: {_fmt_time_range(s, e)}  ({(e-s)/1000:.2f}s)")
            _dump_debug(args.debug_dir, "02b_scd.json", scd_stats)

        chunks = chunk_long_segments(segments, max_dur_ms=args.chunk_max, hop_ms=args.chunk_hop)
        print(f"  [CHUNK] {len(chunks)} chunks (>{args.chunk_max}ms 按 {args.chunk_hop}ms hop 切)")

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

        labels = cluster_embs(embs, num_spk=args.num_spk, threshold=args.threshold,
                              whiten=args.whiten)
        centroids = compute_centroids(embs, labels)
        spk_ids = sorted(centroids.keys())
        print(f"  [CLUS]  发现 {len(spk_ids)} 个说话人簇: {spk_ids}")

        if args.diar_mode == "chunk":
            turns = chunks_to_fine_turns(kept_chunks, labels, smooth=args.diar_smooth)
            print(f"  [TURNS] {len(turns)} turns (chunk-level, smooth={args.diar_smooth})")
        else:
            turns = segments_to_turns(segments, chunks, kept_chunks, labels)
            print(f"  [TURNS] {len(turns)} turns (segment-vote)")

        # cam++ 输出: 每个 spk 的总时长 (用来快速定位坍缩问题, 例如 spk_0 吃掉 40 分钟)
        spk_dur_ms = Counter()
        for _ts, _te, _spk in turns:
            spk_dur_ms[int(_spk)] += (_te - _ts)
        print(f"  [SPK-DUR] " + " | ".join(
            f"spk_{spk}: {dur/1000:.1f}s"
            for spk, dur in spk_dur_ms.most_common()
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

        # ─── 4. BSS 重叠区检测 (只记录 overlap 时间窗, 不改文字) ───
        overlap_ranges = []   # [(start_ms, end_ms, [spk_a, spk_b]), ...]
        if args.no_bss:
            print(f"\n=== [4/5] BSS 已跳过 ===")
        else:
            print(f"\n=== [4/5] BSS 重叠检测 ===")
            n_overlap = n_skip = n_too_long = 0
            for idx, (s_ms, e_ms, spk) in enumerate(turns):
                dur = (e_ms - s_ms) / 1000.0
                seg = wav[int(s_ms/1000*SR):int(e_ms/1000*SR)]
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
                    # 保留分离出来的两路波形, 供 stage 5 单独 ASR
                    overlap_ranges.append((s_ms, e_ms, spk_pair, [w.copy() for w in wavs]))
                    print(f"  [OVERLAP] {_fmt_time_range(s_ms, e_ms)} → spk_{spk_pair}")

                    # 落盘: 原始混合 + 分离两路
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
            print(f"  检测 {n_checked} turns, 重叠 {n_overlap} 段, "
                  f"短<{args.bss_min_dur}s 跳过 {n_skip} 段, 长>{args.bss_max_dur}s 跳过 {n_too_long} 段")
            _release_model(main_bss, "_bss_pipe")
            _free_gpu("after bss")

            _dump_debug(args.debug_dir, "04_bss.json", {
                "checked": n_checked,
                "overlap": n_overlap,
                "skipped_short": n_skip,
                "skipped_long": n_too_long,
                "bss_min_dur_s": args.bss_min_dur,
                "bss_max_dur_s": args.bss_max_dur,
                "overlap_ranges": [
                    {"start_s": round(s/1000, 2), "end_s": round(e/1000, 2),
                     "spk_pair": [(f"spk_{x}" if x is not None else None) for x in spk_pair]}
                    for s, e, spk_pair, _wavs in overlap_ranges
                ],
            })

        # ─── 5. 整段 ASR + 染色 + 段落化 ───
        print(f"\n=== [5/5] 整段 ASR (Paraformer + VAD + Punc) ===")
        # 命令行 --hotword 跟 corrections.json hotwords[] 合并
        effective_hotword = " ".join(filter(None, [args.hotword, load_hotwords()])).strip()
        if effective_hotword:
            print(f"  [HOTWORD] {effective_hotword}")
        sentences = asr_full(clean_path, hotword=effective_hotword)
        print(f"  [ASR] 得到 {len(sentences)} 个句子")

        # ITN: 中文数字 → 阿拉伯数字 (会议纪要刚需)
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
                {"start_s": round(s["start"]/1000, 2),
                 "end_s": round(s["end"]/1000, 2),
                 "text": s["text"]}
                for s in sentences
            ],
        })

        # spk 染色
        sentences = assign_speaker(sentences, turns)

        # 重叠区: 删除原 ASR 在该区段的句子, 对两路 BSS 波形分别重新 ASR
        for s in sentences:
            s["overlap"] = False
        if overlap_ranges:
            replaced = 0
            for os_ms, oe_ms, spk_pair, sep_wavs in overlap_ranges:
                if spk_pair[0] == spk_pair[1] or spk_pair[0] is None or spk_pair[1] is None:
                    continue
                # 从 sentences 里剔除 midpoint 落在这个区间的原始句子
                before = len(sentences)
                sentences = [
                    s for s in sentences
                    if not (os_ms <= (s["start"] + s["end"]) / 2.0 <= oe_ms)
                ]
                replaced += before - len(sentences)
                # 对每路单独 ASR
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
                print(f"  [OVERLAP-RE-ASR] 替换 {replaced} 句, 对 {len(overlap_ranges)} 个重叠区各跑 2 路 ASR")
            sentences.sort(key=lambda r: (r["start"], r.get("speaker") if r.get("speaker") is not None else -1))

        # 可选: SpeakerDB 替换为真名
        if args.enroll_db and os.path.exists(args.enroll_db):
            db = SpeakerDB(args.enroll_db)
            spk_names = {}
            for spk, c in centroids.items():
                name, sim = db.match(c, threshold=args.match_threshold)
                spk_names[spk] = name if name else f"spk_{spk}"
            print(f"  [ENROLL] spk → 真名: {spk_names}")
        else:
            spk_names = {spk: f"spk_{spk}" for spk in centroids}

        # 合段落
        for s in sentences:
            s["speaker"] = spk_names.get(s["speaker"], f"spk_{s['speaker']}" if s["speaker"] is not None else "spk_unknown")
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
            print(f"  [POST-MERGE] {n_before} → {len(paragraphs)} 段 "
                  f"(同 spk + 无 overlap + gap≤{args.post_merge_gap}ms + "
                  f"<{args.post_merge_max_dur//1000}s + <{args.post_merge_max_chars} 字)")

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
            results_json = [
                {"start": round(p["start"]/1000, 2),
                 "end": round(p["end"]/1000, 2),
                 "speaker": p["speaker"],
                 "overlap": p.get("overlap", False),
                 "text": p["text"]}
                for p in paragraphs
            ]
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results_json, f, ensure_ascii=False, indent=2)
            print(f"\n[saved json] {args.output}")

        if args.output_txt:
            # os.makedirs(args.output_dir, exist_ok=True)
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
        try:
            os.unlink(clean_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
