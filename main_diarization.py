"""
说话人分离 demo
路线 A: VAD 切段 -> CAM++ embedding -> AHC 聚类 -> 输出 [start, end, spk_id]

用法:
    python demo_diarization.py test_audio.wav
    python demo_diarization.py test_audio.wav --num-spk 2     # 已知人数
    python demo_diarization.py test_audio.wav --enroll-db speakers.npz   # 已知参会人匹配
"""
import argparse
import numpy as np
import librosa
from funasr import AutoModel
from sklearn.cluster import AgglomerativeClustering

from speaker_db import extract_embedding_from_wave, SpeakerDB


SR = 16000


def run_vad(wav_path: str):
    """FSMN-VAD 切分语音段, 返回 [(start_ms, end_ms), ...]"""
    vad = AutoModel(model="fsmn-vad", model_revision="v2.0.4", disable_update=True)
    res = vad.generate(input=wav_path)
    return res[0]["value"]  # [[start_ms, end_ms], ...]


def merge_segments(segments, gap_threshold_ms: int = 300, min_duration_ms: int = 800):
    """相邻间隙 < gap_threshold 合并; 合并后短于 min_duration 丢弃"""
    if not segments:
        return []
    merged = [list(segments[0])]
    for s, e in segments[1:]:
        if s - merged[-1][1] <= gap_threshold_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= min_duration_ms]


def chunk_long_segments(segments, max_dur_ms: int = 3000, hop_ms: int = 1500):
    """长段按滑窗强切, diarization 标准做法 (3s 窗 + 1.5s hop)"""
    out = []
    for s, e in segments:
        if e - s <= max_dur_ms:
            out.append((s, e))
        else:
            t = s
            while t < e:
                out.append((t, min(t + max_dur_ms, e)))
                t += hop_ms
    return out


def _rms_dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2)) + 1e-10)
    return 20.0 * np.log10(rms)


def extract_segment_embeddings(
    wav: np.ndarray,
    segments,
    min_dbfs: float = -45.0,
    avg_subwindows: bool = False,
    subwin_s: float = 1.5,
    subhop_s: float = 0.75,
):
    """
    对每个段:
      1) RMS dBFS < min_dbfs 视为噪音, 丢弃
      2) avg_subwindows=True 时, 段长 > subwin_s 做滑窗 emb 求 L2 平均
         (可能拉近不同说话人, 默认关)
    """
    embs, kept, dropped_noise = [], [], 0
    for s_ms, e_ms in segments:
        s, e = int(s_ms / 1000 * SR), int(e_ms / 1000 * SR)
        seg = wav[s:e]
        if len(seg) < SR * 0.5:
            continue
        if _rms_dbfs(seg) < min_dbfs:
            dropped_noise += 1
            continue

        if not avg_subwindows:
            emb = extract_embedding_from_wave(seg, sr=SR)
        else:
            win = int(subwin_s * SR)
            hop = int(subhop_s * SR)
            if len(seg) <= win:
                emb = extract_embedding_from_wave(seg, sr=SR)
            else:
                sub_embs = []
                for i in range(0, len(seg) - win + 1, hop):
                    e_i = extract_embedding_from_wave(seg[i : i + win], sr=SR)
                    e_i = e_i / (np.linalg.norm(e_i) + 1e-8)
                    sub_embs.append(e_i)
                emb = np.mean(np.stack(sub_embs), axis=0)
                emb = emb / (np.linalg.norm(emb) + 1e-8)

        embs.append(emb)
        kept.append((s_ms, e_ms))
    if dropped_noise:
        print(f"[NOISE]  丢弃低能量段 {dropped_noise} 个 (dBFS<{min_dbfs})")
    return (np.stack(embs) if embs else np.zeros((0, 192), dtype=np.float32)), kept


def cluster(embs: np.ndarray, num_spk: int = None, threshold: float = 0.7):
    # cosine 距离: 1 - cos_sim
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    if num_spk:
        clf = AgglomerativeClustering(n_clusters=num_spk, metric="cosine", linkage="average")
    else:
        clf = AgglomerativeClustering(
            n_clusters=None, distance_threshold=threshold,
            metric="cosine", linkage="average",
        )
    return clf.fit_predict(embs)


def match_to_db(embs: np.ndarray, db_path: str, threshold: float = 0.55):
    db = SpeakerDB()
    db.load(db_path)
    labels = []
    for emb in embs:
        name, score = db.match(emb)
        labels.append(name if score >= threshold else f"unknown")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav",default="data/2025-09-30 15_56 记录.mp3", help="输入音频路径")
    ap.add_argument("--num-spk", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=0.7, help="AHC cosine 距离阈值, 越大越倾向合并")
    ap.add_argument("--enroll-db", default=None, help="已注册声纹库 .npz")
    ap.add_argument("--merge-gap", type=int, default=300, help="<这个间隙(ms)的相邻段合并")
    ap.add_argument("--min-dur", type=int, default=800, help="合并后短于这个(ms)的段丢弃")
    ap.add_argument("--chunk-max", type=int, default=3000, help="长段滑窗最大长度(ms)")
    ap.add_argument("--chunk-hop", type=int, default=1500, help="长段滑窗 hop(ms)")
    ap.add_argument("--min-dbfs", type=float, default=-40.0, help="段 RMS dBFS 低于此值视为噪音丢弃 (噪音治本)")
    ap.add_argument("--min-cluster-size", type=int, default=1, help="簇内 emb 数 < 此值标 noise (1=不过滤)")
    ap.add_argument("--avg-subwindows", action="store_true", help="段内滑窗 emb 平均 (会拉近说话人, 默认关)")
    args = ap.parse_args()

    wav, _ = librosa.load(args.wav, sr=SR, mono=True)
    segments = run_vad(args.wav)
    print(f"[VAD]    {len(segments)} segments")

    segments = merge_segments(segments, args.merge_gap, args.min_dur)
    print(f"[MERGE]  {len(segments)} segments (gap<{args.merge_gap}ms 合并, <{args.min_dur}ms 丢弃)")

    segments = chunk_long_segments(segments, args.chunk_max, args.chunk_hop)
    print(f"[CHUNK]  {len(segments)} segments (>{args.chunk_max}ms 按 {args.chunk_hop}ms hop 滑窗)")

    embs, kept = extract_segment_embeddings(
        wav, segments,
        min_dbfs=args.min_dbfs,
        avg_subwindows=args.avg_subwindows,
    )
    print(f"[EMB]    {len(embs)} embeddings ({embs.shape[1] if len(embs) else 0}-d)")

    if len(embs) < 2:
        print("[!] embedding 数量不足 2 个, 无法聚类")
        return

    if args.enroll_db:
        labels = match_to_db(embs, args.enroll_db)
    else:
        labels = cluster(embs, num_spk=args.num_spk, threshold=args.threshold)

    # 标记小簇为 noise (麦克风噪声常自成一类)
    if args.enroll_db is None:
        from collections import Counter
        counts = Counter(labels.tolist())
        noise_clusters = {c for c, n in counts.items() if n < args.min_cluster_size}
        if noise_clusters:
            print(f"[NOISE]  簇 {sorted(noise_clusters)} 大小<{args.min_cluster_size}, 标 noise")
            labels = np.array(["noise" if c in noise_clusters else str(c) for c in labels])
        else:
            labels = np.array([str(c) for c in labels])

    # 合并相邻同 spk 的小窗成 "说话人轮次"
    turns = []
    for (s, e), lab in zip(kept, labels):
        if turns and turns[-1][2] == lab and s - turns[-1][1] <= 500:
            turns[-1][1] = e
        else:
            turns.append([s, e, lab])
    print(f"[TURNS]  {len(turns)} turns (相邻同 spk 且 gap<=500ms 合并)")

    print("\n=== diarization ===")
    for s, e, lab in turns:
        print(f"[{s/1000:7.2f}s -> {e/1000:7.2f}s]  spk_{lab}")


if __name__ == "__main__":
    main()
