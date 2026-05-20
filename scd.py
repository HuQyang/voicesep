"""
Speaker Change Detection (SCD) - 滑窗 embedding 距离检测说话人切换点

原理:
  VAD 只看静音, 检测不到同一段连续语音里的说话人切换 (e.g. 两人快速对话, gap<300ms).
  本模块对每个长 VAD segment:
    1. 用细滑窗 (默认 0.75s 窗 + 0.1s hop) 提 cam++ embedding
    2. 算相邻 embedding 的 cosine 距离, 距离越大说话人越可能换
    3. 距离的局部峰 > threshold 判为切点
    4. 在切点切分原 segment

  这是 pyannote.audio 内部用的同一类算法 (segmentation-3.0), 但不依赖 HF 模型,
  完全用你现有的 ERes2NetV2 embedder.

性能:
  21s segment, 0.1s hop → 约 210 个 embedding
  batch 提 (64/batch) 在 GPU 上 ~1-2s 完成
  只对 > min_segment_for_scd_s 的段跑, 总开销可控

调参建议:
  距离峰阈值 distance_threshold:
    0.35-0.45  灵敏 (会切碎, 但找回率高)
    0.50-0.60  适中 (默认 0.5, 多数会议场景 OK)
    0.65+      保守 (只切大变化)
  滑窗 window_s:
    0.5  快速对话敏感, 但 embedding 不稳
    0.75 默认, 平衡
    1.0  embedding 更稳, 但短轮替会漏
"""
from typing import List, Tuple

import numpy as np

from speaker_db import extract_embeddings_batch_from_waves


SR = 16000


def _sliding_embeddings(
    wav: np.ndarray, segment_start_ms: int, segment_end_ms: int,
    window_s: float = 0.75, hop_s: float = 0.1, sr: int = SR,
) -> Tuple[np.ndarray, List[int]]:
    """
    在 [segment_start_ms, segment_end_ms] 内做滑窗, 返回 (embeddings, centers_ms).
    """
    s_idx = int(segment_start_ms / 1000 * sr)
    e_idx = int(segment_end_ms / 1000 * sr)
    seg_wav = wav[s_idx:e_idx]

    window = int(window_s * sr)
    hop = int(hop_s * sr)

    waves = []
    centers_ms = []
    pos = 0
    while pos + window <= len(seg_wav):
        waves.append(seg_wav[pos:pos + window])
        center_sample = pos + window // 2
        centers_ms.append(int(segment_start_ms + center_sample / sr * 1000))
        pos += hop

    if not waves:
        return np.zeros((0, 192), dtype=np.float32), []

    embs = extract_embeddings_batch_from_waves(waves, sr=sr)
    return embs, centers_ms


def _smooth_distances(distances: np.ndarray, smooth_window: int = 3) -> np.ndarray:
    """对距离序列做小窗滑动平均, 抑制单点噪声"""
    if len(distances) < smooth_window:
        return distances
    kernel = np.ones(smooth_window) / smooth_window
    return np.convolve(distances, kernel, mode="same")


def detect_speaker_changes(
    wav: np.ndarray,
    segment_start_ms: int,
    segment_end_ms: int,
    window_s: float = 0.75,
    hop_s: float = 0.1,
    distance_threshold: float = 0.5,
    min_speaker_dur_s: float = 0.8,
    smooth: bool = True,
    sr: int = SR,
) -> Tuple[List[int], dict]:
    """
    检测 [segment_start_ms, segment_end_ms] 内的说话人切换点.

    返回:
      change_points_ms: [ms, ms, ...] 切点(在 segment 内部, 不含端点)
      detail:           调试信息 dict (centers, distances 等)

    算法: cosine 距离的局部峰 + 强制最小切片长度
    """
    embs, centers = _sliding_embeddings(
        wav, segment_start_ms, segment_end_ms,
        window_s=window_s, hop_s=hop_s, sr=sr,
    )
    if len(embs) < 3:
        return [], {"reason": "embeddings<3", "n_emb": len(embs)}

    # cosine 距离
    embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    cos_sims = np.sum(embs_n[1:] * embs_n[:-1], axis=1)
    distances = 1.0 - cos_sims   # shape (N-1,)
    if smooth:
        distances = _smooth_distances(distances, smooth_window=3)

    # 找峰: distance > threshold, 且是局部最大, 且距离上一个切点 >= min_speaker_dur
    min_gap = int(min_speaker_dur_s / hop_s)
    change_points_ms = []
    last_change_idx = -min_gap   # 允许从开头就切
    for i in range(1, len(distances) - 1):
        if (distances[i] > distance_threshold
                and distances[i] >= distances[i - 1]
                and distances[i] >= distances[i + 1]
                and i - last_change_idx >= min_gap):
            # i 是 distance 索引, 对应 centers[i+1] 处的切点
            change_points_ms.append(centers[i + 1])
            last_change_idx = i

    return change_points_ms, {
        "n_emb": int(len(embs)),
        "distance_max": float(distances.max()) if len(distances) else 0.0,
        "distance_mean": float(distances.mean()) if len(distances) else 0.0,
        "n_changes": len(change_points_ms),
        "centers_ms": centers,
        "distances": distances.tolist(),
    }


def split_segments_by_scd(
    wav: np.ndarray,
    segments: List[Tuple[int, int]],
    min_segment_for_scd_s: float = 5.0,
    window_s: float = 0.75,
    hop_s: float = 0.1,
    distance_threshold: float = 0.5,
    min_speaker_dur_s: float = 0.8,
    sr: int = SR,
    verbose: bool = True,
) -> Tuple[List[Tuple[int, int]], dict]:
    """
    对 VAD 输出的 segments 列表, 把每个 > min_segment_for_scd_s 的段
    用 SCD 切成更细的子 segment.

    返回:
      new_segments: 切分后的 segment 列表
      stats:        统计信息 (跑了几段, 共找出多少切点)
    """
    out = []
    n_scd_run = 0
    n_total_cps = 0
    per_segment_detail = []

    for s_ms, e_ms in segments:
        dur_s = (e_ms - s_ms) / 1000.0
        if dur_s < min_segment_for_scd_s:
            out.append((s_ms, e_ms))
            continue

        cps, detail = detect_speaker_changes(
            wav, s_ms, e_ms,
            window_s=window_s, hop_s=hop_s,
            distance_threshold=distance_threshold,
            min_speaker_dur_s=min_speaker_dur_s,
            sr=sr,
        )
        n_scd_run += 1
        n_total_cps += len(cps)
        per_segment_detail.append({
            "start_s": round(s_ms/1000, 2),
            "end_s": round(e_ms/1000, 2),
            "dur_s": round(dur_s, 2),
            "n_changes": len(cps),
            "change_points_s": [round(cp/1000, 2) for cp in cps],
            "distance_max": detail.get("distance_max"),
            "distance_mean": detail.get("distance_mean"),
        })

        if not cps:
            out.append((s_ms, e_ms))
            continue

        # 在切点处切分
        boundaries = [s_ms] + cps + [e_ms]
        for i in range(len(boundaries) - 1):
            sub_s = boundaries[i]
            sub_e = boundaries[i + 1]
            if sub_e - sub_s >= 200:   # 最小 200ms 防止空段
                out.append((sub_s, sub_e))

        if verbose:
            print(f"  [SCD] {s_ms/1000:7.2f}-{e_ms/1000:7.2f}s "
                  f"({dur_s:.1f}s) → {len(cps)} 切点 → {len(boundaries)-1} 子段")

    return out, {
        "n_segments_in": len(segments),
        "n_segments_out": len(out),
        "n_scd_run": n_scd_run,
        "n_total_change_points": n_total_cps,
        "per_segment": per_segment_detail,
    }
