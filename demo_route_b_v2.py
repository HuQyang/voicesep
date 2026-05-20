"""
路线 B v2: Paraformer + 热词/词典 + 声纹后处理

相比 v1 (demo_route_b.py), 多做的事:
  ① 对每个 VAD 段提 cam++ embedding (重新计算, 不用 cam++ 在线聚类结果)
  ② 与已注册声纹库匹配 -> 命中者直接用真实姓名
  ③ 未命中的 embeddings 重新做 AHC 聚类 (阈值更松) -> 合并被切碎的同一人
  ④ 未注册的人按首次发言顺序分配 Speaker_A / Speaker_B / ...

用法:
  # 假设已经用 enroll.py 注册过 张三/李四
  python demo_route_b_v2.py <audio>
  python demo_route_b_v2.py <audio> --extra-hotword "钱部长 数据要素"

可调参数 (代码顶部常量):
  MATCH_THRESHOLD       与注册库匹配的 cos_sim 阈值, 越严越不容易误识别
  CLUSTER_THRESHOLD     未匹配段重聚类的 cos_sim 阈值, 越松越倾向合并
  MIN_SEG_LEN_FOR_EMB   小于此秒数的段不参与 embedding (跟随邻居)

输出: <audio>.B2.txt / <audio>.B2.json
"""

import os
import json
import argparse

import numpy as np
import librosa

from funasr import AutoModel

from speaker_db import (
    SpeakerDB,
    extract_embedding_from_wave,
    cluster_embeddings,
)


# ------- 可调参数 -------
MATCH_THRESHOLD = 0.65
CLUSTER_THRESHOLD = 0.55
MIN_SEG_LEN_FOR_EMB = 0.5  # 秒
# -----------------------


def load_corrections(path: str) -> dict:
    if not os.path.exists(path):
        return {"rules": [], "hotwords": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["rules"] = sorted(data.get("rules", []), key=lambda r: -len(r["wrong"]))
    return data


def apply_corrections(text: str, rules: list) -> str:
    for r in rules:
        text = text.replace(r["wrong"], r["right"])
    return text


class RouteB2Pipeline:
    def __init__(self, db_path: str = "speakers/db.npz"):
        print("[init] Paraformer + VAD + 标点 + cam++ ...")
        self.model = AutoModel(
            model="paraformer-zh",
            vad_model="fsmn-vad",
            punc_model="ct-punc",
            spk_model="cam++",
            disable_update=True,
        )
        self.db = SpeakerDB(db_path)
        print("[init] 完成")

    def process(self, audio: str, hotwords: list, rules: list) -> list:
        hot_str = " ".join(hotwords) if hotwords else ""
        if hot_str:
            print(f"[asr] 热词: {hot_str}")

        # ---- 预加载音频 (避免 FunASR 内部 mp3 解码偶发失败) ----
        print(f"[load] librosa 加载音频: {audio}")
        wav, sr = librosa.load(audio, sr=16000, mono=True)
        wav = wav.astype(np.float32)
        print(f"[load] 时长 {len(wav)/sr:.1f}s, 采样率 {sr}")

        # ---- Pass 1: ASR + 原始 cam++ 切分 ----
        res = self.model.generate(input=wav, batch_size_s=300, hotword=hot_str)
        raw_segs = []
        for it in res or []:
            for s in it.get("sentence_info", []):
                raw = s.get("text", "")
                raw_segs.append({
                    "raw_spk": str(s.get("spk", "X")),
                    "start": round(s.get("start", 0) / 1000.0, 2),
                    "end": round(s.get("end", 0) / 1000.0, 2),
                    "text": apply_corrections(raw, rules),
                })
        if not raw_segs:
            return []
        raw_n = len({s["raw_spk"] for s in raw_segs})
        print(f"[pass1] cam++ 原始切分 {len(raw_segs)} 段, raw 说话人 {raw_n} 个")

        # ---- Pass 2: 逐段提 embedding (复用已加载的 wav) ----
        print("[pass2] 提取每段 embedding...")

        emb_list = []
        emb_to_seg = []  # 第 k 个 embedding 对应 raw_segs 的下标
        for i, seg in enumerate(raw_segs):
            dur = seg["end"] - seg["start"]
            if dur < MIN_SEG_LEN_FOR_EMB:
                continue
            s_idx = max(0, int(seg["start"] * sr))
            e_idx = min(len(wav), int(seg["end"] * sr))
            chunk = wav[s_idx:e_idx]
            try:
                emb = extract_embedding_from_wave(chunk, sr=16000)
                emb_list.append(emb)
                emb_to_seg.append(i)
            except Exception as e:
                print(f"  [warn] 段 {i} embedding 失败: {e}")

        embeddings = (
            np.stack(emb_list, axis=0) if emb_list
            else np.zeros((0, SpeakerDB.EMB_DIM), dtype=np.float32)
        )
        print(f"[pass2] 提取到 {len(embeddings)} 个 embedding")

        # ---- Pass 3: 匹配注册库 ----
        speaker_name = [None] * len(raw_segs)
        unmatched_k = []
        for k, i in enumerate(emb_to_seg):
            name, sim = self.db.match(embeddings[k], threshold=MATCH_THRESHOLD)
            if name is not None:
                speaker_name[i] = name
            else:
                unmatched_k.append(k)
        print(f"[pass3] 匹配到注册声纹的段: "
              f"{sum(1 for n in speaker_name if n is not None)}/{len(emb_to_seg)}")

        # ---- Pass 4: 未匹配段重聚类, 合并被 cam++ 切碎的同一人 ----
        if unmatched_k:
            unmatched_embs = embeddings[unmatched_k]
            labels = cluster_embeddings(unmatched_embs, threshold=CLUSTER_THRESHOLD)
            n_clusters = len(set(labels))
            print(f"[pass4] {len(unmatched_k)} 段未匹配 -> 重聚类为 {n_clusters} 人")

            # 按各 cluster 首次出现时间排序 -> Speaker_A/B/C
            first_t = {}
            for k_idx, lbl in enumerate(labels):
                seg_i = emb_to_seg[unmatched_k[k_idx]]
                t = raw_segs[seg_i]["start"]
                if lbl not in first_t or t < first_t[lbl]:
                    first_t[lbl] = t

            sorted_lbls = sorted(first_t.keys(), key=lambda l: first_t[l])
            lbl_to_name = {
                l: f"Speaker_{chr(ord('A') + idx)}"
                for idx, l in enumerate(sorted_lbls)
            }

            for k_idx, lbl in enumerate(labels):
                seg_i = emb_to_seg[unmatched_k[k_idx]]
                speaker_name[seg_i] = lbl_to_name[lbl]
        else:
            print("[pass4] 全部命中注册库, 无需聚类")

        # ---- Pass 5: 短段 (< MIN_SEG_LEN_FOR_EMB) 跟随邻居 ----
        for i in range(len(speaker_name)):
            if speaker_name[i] is None:
                # 优先看后面相邻有标签的, 否则往前找
                nb = None
                for j in range(i + 1, len(speaker_name)):
                    if speaker_name[j]:
                        nb = speaker_name[j]
                        break
                if nb is None:
                    for j in range(i - 1, -1, -1):
                        if speaker_name[j]:
                            nb = speaker_name[j]
                            break
                speaker_name[i] = nb or "Speaker_X"

        # ---- 组装输出 ----
        return [
            {
                "speaker": speaker_name[i],
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for i, seg in enumerate(raw_segs)
        ]

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
        lines = ["========== 会议记录 (路线 B v2) ==========\n"]
        for seg in segments:
            s, e = seg["start"], seg["end"]
            ts = f"[{int(s//60):02d}:{int(s%60):02d}-{int(e//60):02d}:{int(e%60):02d}]"
            lines.append(f"{ts} {seg['speaker']}:\n{seg['text']}\n")
        return "\n".join(lines)

    @staticmethod
    def to_json(segments: list, audio: str) -> dict:
        speakers = sorted({s["speaker"] for s in segments})
        dur = max((s["end"] for s in segments), default=0.0)
        return {
            "source": os.path.basename(audio),
            "route": "B2_paraformer_speaker_postproc",
            "duration_sec": dur,
            "speakers": [
                {"id": s, "registered": not s.startswith("Speaker_")}
                for s in speakers
            ],
            "segments": segments,
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--corrections", default="corrections.json")
    ap.add_argument("--extra-hotword", default="")
    ap.add_argument("--db", default="speakers/db.npz")
    args = ap.parse_args()

    conf = load_corrections(args.corrections)
    hotwords = list(conf.get("hotwords", []))
    if args.extra_hotword:
        hotwords += args.extra_hotword.split()

    pipe = RouteB2Pipeline(db_path=args.db)
    segs = pipe.process(args.audio, hotwords, conf["rules"])
    if not segs:
        print("无输出")
        return
    merged = pipe.merge(segs)
    readable = pipe.to_readable(merged)
    print(readable)

    base = os.path.splitext(args.audio)[0]
    with open(f"{base}.B2.txt", "w", encoding="utf-8") as f:
        f.write(readable)
    with open(f"{base}.B2.json", "w", encoding="utf-8") as f:
        json.dump(pipe.to_json(segs, args.audio), f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {base}.B2.txt / {base}.B2.json")


if __name__ == "__main__":
    main()
