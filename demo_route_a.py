"""
路线 A: SenseVoice 全程 + 文本对齐到 Paraformer 时间轴

思路:
  Pass 1 (Paraformer+cam++): 拿到说话人骨架 [(speaker, start, end, text_p)]
  Pass 2 (SenseVoice 全程):   拿到高质量完整 transcript (长上下文, 同音字纠正)
  Pass 3 (difflib 对齐):      把 SenseVoice 文本切回到 Paraformer 时间段

输出: <audio>.A.txt / <audio>.A.json
"""

import os
import re
import json
import argparse
import difflib
from funasr import AutoModel


SV_TAG_RE = re.compile(r"<\|[^|]*\|>")


def clean_sv(text: str) -> str:
    return SV_TAG_RE.sub("", text).strip()


def align_sv_to_segments(sv_full: str, para_segments: list) -> list:
    """
    用 difflib 把 SenseVoice 全文按字符位置映射回 Paraformer 的段切分。

    实现:
      1. 拼接所有 Paraformer 段文本 -> para_concat, 记录每段的字符区间
      2. SequenceMatcher 找 para_concat <-> sv_full 的匹配块
      3. 构建 para_pos -> sv_pos 的映射 (未匹配的位置用最近的匹配位置)
      4. 对每段, 用 sv 中对应的子串作为最终文本
    """
    if not para_segments:
        return []

    # 拼接 + 记录边界
    para_concat = ""
    bounds = []  # [(seg_idx, char_start, char_end)]
    for i, seg in enumerate(para_segments):
        start = len(para_concat)
        para_concat += seg.get("text_para", "")
        bounds.append((i, start, len(para_concat)))

    if not para_concat or not sv_full:
        return [
            {**seg, "text": seg.get("text_para", "")} for seg in para_segments
        ]

    # 字符级对齐
    matcher = difflib.SequenceMatcher(None, para_concat, sv_full, autojunk=False)
    blocks = matcher.get_matching_blocks()

    para_to_sv = [-1] * (len(para_concat) + 1)
    for a, b, size in blocks:
        for k in range(size):
            if a + k < len(para_to_sv):
                para_to_sv[a + k] = b + k
        if a + size < len(para_to_sv):
            para_to_sv[a + size] = b + size

    # 前向 + 后向填充未匹配位置
    last = 0
    for i in range(len(para_to_sv)):
        if para_to_sv[i] < 0:
            para_to_sv[i] = last
        else:
            last = para_to_sv[i]

    # 切出每段
    out = []
    for seg_idx, cs, ce in bounds:
        sv_s = para_to_sv[cs]
        sv_e = para_to_sv[ce] if ce < len(para_to_sv) else len(sv_full)
        # 把紧跟标点带上
        while sv_e < len(sv_full) and sv_full[sv_e] in "，。！？、；：,.!?;:":
            sv_e += 1
        text = sv_full[sv_s:sv_e].strip()
        seg = dict(para_segments[seg_idx])
        seg["text"] = text or seg.get("text_para", "")
        seg.pop("text_para", None)
        out.append(seg)
    return out


class RouteAPipeline:
    def __init__(self):
        print("[init] Pass1: Paraformer + cam++ ...")
        self.diar = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            disable_update=True,
        )
        print("[init] Pass2: SenseVoiceSmall ...")
        self.sv = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            punc_model="ct-punc",
            disable_update=True,
        )
        print("[init] 完成")

    def process(self, audio: str, hotword: str = "") -> list:
        print(f"[pass1] 说话人骨架: {audio}")
        res_p = self.diar.generate(input=audio, batch_size_s=300, hotword=hotword)
        para_segs = []
        for it in res_p or []:
            for s in it.get("sentence_info", []):
                para_segs.append({
                    "speaker": f"Speaker_{s.get('spk', 'X')}",
                    "start": round(s.get("start", 0) / 1000.0, 2),
                    "end": round(s.get("end", 0) / 1000.0, 2),
                    "text_para": s.get("text", ""),
                })
        print(f"[pass1] {len(para_segs)} 段")

        print(f"[pass2] SenseVoice 全程转写 (含 VAD + 30s 上下文)")
        res_sv = self.sv.generate(
            input=audio,
            cache={},
            language="zh",
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
            merge_length_s=30,
            hotword=hotword,
        )
        sv_full = ""
        for it in res_sv or []:
            sv_full += clean_sv(it.get("text", ""))
        print(f"[pass2] SenseVoice 总字数 {len(sv_full)}")

        print("[pass3] 文本对齐...")
        return align_sv_to_segments(sv_full, para_segs)

    @staticmethod
    def merge(segments: list) -> list:
        if not segments:
            return []
        m = [dict(segments[0])]
        for s in segments[1:]:
            if s["speaker"] == m[-1]["speaker"]:
                m[-1]["end"] = s["end"]
                m[-1]["text"] += s["text"]
            else:
                m.append(dict(s))
        return m

    @staticmethod
    def to_readable(segments: list) -> str:
        out = ["========== 会议记录 (路线A) ==========\n"]
        for seg in segments:
            s, e = seg["start"], seg["end"]
            ts = f"[{int(s//60):02d}:{int(s%60):02d}-{int(e//60):02d}:{int(e%60):02d}]"
            out.append(f"{ts} {seg['speaker']}:\n{seg['text']}\n")
        return "\n".join(out)

    @staticmethod
    def to_json(segments: list, audio: str) -> dict:
        speakers = sorted({s["speaker"] for s in segments})
        dur = max((s["end"] for s in segments), default=0.0)
        return {
            "source": os.path.basename(audio),
            "route": "A_sensevoice_full_align",
            "duration_sec": dur,
            "speakers": [{"id": s, "name": None} for s in speakers],
            "segments": segments,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--hotword", default="")
    args = ap.parse_args()

    pipe = RouteAPipeline()
    segs = pipe.process(args.audio, hotword=args.hotword)
    if not segs:
        print("无输出")
        return
    merged = pipe.merge(segs)
    readable = pipe.to_readable(merged)
    print(readable)

    base = os.path.splitext(args.audio)[0]
    with open(f"{base}.A.txt", "w", encoding="utf-8") as f:
        f.write(readable)
    with open(f"{base}.A.json", "w", encoding="utf-8") as f:
        json.dump(pipe.to_json(segs, args.audio), f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {base}.A.txt / {base}.A.json")


if __name__ == "__main__":
    main()
