"""
声纹库 + embedding 工具

- get_embedder():                  懒加载 modelscope cam++ sv pipeline
- extract_embedding_from_wave():   从 numpy 波形提 embedding
- extract_embedding_from_file():   从 wav 文件提 embedding
- SpeakerDB:                       注册声纹库 (load/save/add/remove/match)
- cluster_embeddings():            AHC 聚类, 用于合并 cam++ 切碎的同一人

embedding 维度: 192 (cam++)
相似度: cosine
"""

import os
import tempfile
from typing import Optional, Tuple, List

import numpy as np
import librosa
import soundfile as sf


_embedder = None


# 默认 ERes2NetV2 (192-d, 通用中文). 可用 set_embedder_model() 或环境变量
# SPEAKER_EMB_MODEL 覆盖. 推荐备选 (已验证存在):
#   iic/speech_eres2net_base_200k_sv_zh-cn_16k-common  (ERes2Net 200k 小时训练, 192-d)
#   iic/speech_eres2net_large_200k_sv_zh-cn_16k-common (ERes2Net-large, 512-d, 最强)
#   iic/speech_eres2netv2_sv_zh-cn_16k-common          (ERes2NetV2, 192-d, 默认)
#   iic/speech_campplus_sv_zh-cn_16k-common            (cam++, 192-d, 经典基线)
_EMBEDDER_MODEL = os.environ.get(
    "SPEAKER_EMB_MODEL",
    # "iic/speech_eres2netv2_sv_zh-cn_16k-common",
    "iic/speech_eres2net_large_200k_sv_zh-cn_16k-common",
)


def set_embedder_model(model_name: str):
    """运行时切换声纹模型. 调用后下次 get_embedder() 会重新加载"""
    global _EMBEDDER_MODEL, _embedder
    if model_name and model_name != _EMBEDDER_MODEL:
        print(f"[speaker_db] 切换声纹模型: {_EMBEDDER_MODEL} → {model_name}")
        _EMBEDDER_MODEL = model_name
        _embedder = None


def get_embedder():
    """懒加载, 避免每次 import 都跑模型初始化"""
    global _embedder
    if _embedder is None:
        from modelscope.pipelines import pipeline as ms_pipeline
        print(f"[speaker_db] 加载声纹模型: {_EMBEDDER_MODEL}")
        _embedder = ms_pipeline(
            task="speaker-verification",
            model=_EMBEDDER_MODEL,
        )
    return _embedder


def extract_embedding_from_wave(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """从 numpy 波形 (1D float, mono) 提取 cam++ embedding"""
    embedder = get_embedder()
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    # modelscope sv pipeline 走文件路径最稳, 写临时 wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, 16000)
        tmp_path = tmp.name
    try:
        result = embedder([tmp_path], output_emb=True)
        emb = result["embs"][0]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return np.asarray(emb, dtype=np.float32)


def extract_embedding_from_file(path: str) -> np.ndarray:
    embedder = get_embedder()
    result = embedder([path], output_emb=True)
    return np.asarray(result["embs"][0], dtype=np.float32)


def extract_embeddings_batch_from_waves(
    waves: list, sr: int = 16000, batch_size: int = 64
) -> np.ndarray:
    """
    批量提 embedding (SCD 这种需要密集滑窗时必用, 避免逐个写临时文件).
    返回 shape (N, 192) float32.
    """
    if not waves:
        return np.zeros((0, 192), dtype=np.float32)
    embedder = get_embedder()
    all_embs = []
    # 一批批写临时文件 → 一次 pipeline 调用 → 删文件
    for batch_start in range(0, len(waves), batch_size):
        batch_waves = waves[batch_start:batch_start + batch_size]
        tmp_paths = []
        try:
            for w in batch_waves:
                if sr != 16000:
                    w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
                f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(f.name, w, 16000)
                f.close()
                tmp_paths.append(f.name)
            result = embedder(tmp_paths, output_emb=True)
            all_embs.append(np.asarray(result["embs"], dtype=np.float32))
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
    return np.vstack(all_embs)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class SpeakerDB:
    """
    .npz 文件存储:
      names:       object array of str
      embeddings:  float32, shape (N, D)   D 由首条 emb 决定 (cam++/ERes2Net=192, large=512)
    """

    EMB_DIM = 192  # 仅作初始占位, 实际维度由首条 emb / 加载文件决定

    def __init__(self, path: str = "speakers/db.npz"):
        self.path = path
        self.names: List[str] = []
        # 用 None 表示"还没有维度信息", 第一次 add/load 时自动确定
        self.embeddings: np.ndarray = np.zeros((0, self.EMB_DIM), dtype=np.float32)
        self.load()

    def load(self):
        if os.path.exists(self.path):
            data = np.load(self.path, allow_pickle=True)
            self.names = list(data["names"])
            self.embeddings = data["embeddings"].astype(np.float32)
            print(f"[db] 已加载 {len(self.names)} 个声纹: {self.names}")
        else:
            print(f"[db] 声纹库为空 (路径 {self.path})")

    def save(self):
        dir_ = os.path.dirname(self.path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        np.savez(
            self.path,
            names=np.array(self.names, dtype=object),
            embeddings=self.embeddings,
        )
        print(f"[db] 已保存到 {self.path} ({len(self.names)} 个声纹)")

    def add(self, name: str, emb: np.ndarray):
        emb = emb.astype(np.float32).reshape(1, -1)
        if name in self.names:
            idx = self.names.index(name)
            self.embeddings[idx] = emb[0]
            print(f"[db] 覆盖更新: {name}")
        else:
            self.names.append(name)
            if self.embeddings.shape[0] == 0:
                self.embeddings = emb
            else:
                self.embeddings = np.vstack([self.embeddings, emb])
            print(f"[db] 新增: {name}")

    def remove(self, name: str) -> bool:
        if name not in self.names:
            return False
        idx = self.names.index(name)
        self.names.pop(idx)
        self.embeddings = np.delete(self.embeddings, idx, axis=0)
        return True

    def match(
        self, emb: np.ndarray, threshold: float = 0.65
    ) -> Tuple[Optional[str], float]:
        """与库内全员比对, 返回 (姓名, 相似度) 或 (None, 最佳相似度)"""
        if len(self.names) == 0:
            return None, 0.0
        e = emb.astype(np.float32)
        sims = (self.embeddings @ e) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(e) + 1e-8
        )
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= threshold:
            return self.names[best_idx], best_sim
        return None, best_sim


def cluster_embeddings(
    embeddings: np.ndarray, threshold: float = 0.55
) -> np.ndarray:
    """
    AHC 聚类. threshold = cosine_sim, >= 视为同一人.
    返回 labels (int array, 长度 = embeddings.shape[0])
    """
    from sklearn.cluster import AgglomerativeClustering

    n = embeddings.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normed = embeddings / norms

    clu = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clu.fit_predict(normed)
    return labels
