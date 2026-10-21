"""基于启发式自然语言解释的 EoH 候选聚类筛选。

该模块不依赖 scikit-learn：使用 NumPy 实现文本 TF-IDF、K-means 和
凝聚层次聚类，便于在项目现有环境中直接运行。每个簇只保留效用最高的
候选程序作为代表元。
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Sequence

import numpy as np


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*|[0-9]+|[\u4e00-\u9fff]")
_KEYWORD_GROUPS = (
    ("cost", "成本", "费用", "cost", "economic", "价格"),
    ("path", "路径", "最短路", "路由", "distance", "shortest"),
    ("capacity", "容量", "拥堵", "负载", "capacity", "congestion", "load"),
    ("flow", "流量", "需求", "商品", "flow", "demand", "commodity"),
    ("runtime", "时间", "效率", "速度", "runtime", "time", "efficient"),
)


# 统一使用可读的中英文关键词，保证 keyword 方法能够识别中文解释。
# 这里覆盖早期版本中可能因编码转换而损坏的关键词表。
_KEYWORD_GROUPS = (
    ("cost", "成本", "费用", "价格", "economic"),
    ("path", "路径", "最短路", "路由", "distance", "shortest"),
    ("capacity", "容量", "拥堵", "负载", "capacity", "congestion", "load"),
    ("flow", "流量", "需求", "商品", "flow", "demand", "commodity"),
    ("runtime", "时间", "效率", "速度", "runtime", "time", "efficient"),
)


def _tokens(text: str) -> list[str]:
    """提取中英文词元，并为中文连续字符补充二元词元。"""
    text = str(text or "").lower()
    basic = _TOKEN_RE.findall(text)
    cjk = "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")
    bigrams = [cjk[i:i + 2] for i in range(len(cjk) - 1)]
    return basic + bigrams


def _tfidf(texts: Sequence[str]) -> np.ndarray:
    """构造无外部依赖的 L2 归一化 TF-IDF 矩阵。"""
    documents = [_tokens(text) for text in texts]
    vocabulary = sorted({token for doc in documents for token in doc})
    if not vocabulary:
        return np.zeros((len(texts), 1), dtype=float)
    index = {token: i for i, token in enumerate(vocabulary)}
    matrix = np.zeros((len(texts), len(vocabulary)), dtype=float)
    document_frequency = np.zeros(len(vocabulary), dtype=float)
    for row, doc in enumerate(documents):
        counts = Counter(doc)
        for token, count in counts.items():
            col = index[token]
            matrix[row, col] = 1.0 + math.log(float(count))
            document_frequency[col] += 1.0
    matrix *= np.log((1.0 + len(texts)) / (1.0 + document_frequency))[None, :]
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, np.maximum(norms, 1e-12), out=np.zeros_like(matrix), where=norms > 0)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """返回两个已归一化或普通向量的余弦相似度。"""
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def _initial_centers(matrix: np.ndarray, k: int) -> list[int]:
    """使用确定性的最远点策略初始化 K-means 中心。"""
    n = len(matrix)
    centers = [0]
    while len(centers) < k:
        distances = [min(1.0 - _cosine(matrix[i], matrix[j]) for j in centers)
                     if i not in centers else -1.0 for i in range(n)]
        candidate = max(range(n), key=lambda i: (distances[i], -i))
        centers.append(candidate)
    return centers


def _kmeans(matrix: np.ndarray, k: int) -> list[int]:
    """确定性余弦 K-means；空簇会由最远样本补齐。"""
    n = len(matrix)
    labels = np.zeros(n, dtype=int)
    centers = matrix[_initial_centers(matrix, k)].copy()
    for _ in range(40):
        new_labels = np.array([
            max(range(k), key=lambda j: (_cosine(row, centers[j]), -j))
            for row in matrix
        ], dtype=int)
        new_centers = np.zeros_like(centers)
        for j in range(k):
            members = matrix[new_labels == j]
            if len(members):
                center = members.mean(axis=0)
                norm = np.linalg.norm(center)
                new_centers[j] = center / norm if norm > 1e-12 else center
            else:
                distances = [1.0 - _cosine(matrix[i], centers[new_labels[i]]) for i in range(n)]
                index = max(range(n), key=lambda i: (distances[i], -i))
                new_centers[j] = matrix[index]
                new_labels[index] = j
        if np.array_equal(labels, new_labels):
            labels = new_labels
            break
        labels, centers = new_labels, new_centers
    return labels.tolist()


def _agglomerative(matrix: np.ndarray, k: int) -> list[int]:
    """使用平均余弦相似度的凝聚层次聚类。"""
    clusters: list[list[int]] = [[i] for i in range(len(matrix))]

    def similarity(left: list[int], right: list[int]) -> float:
        return sum(_cosine(matrix[i], matrix[j]) for i in left for j in right) / (len(left) * len(right))

    while len(clusters) > k:
        pair = max(
            ((i, j) for i in range(len(clusters)) for j in range(i + 1, len(clusters))),
            key=lambda ij: (similarity(clusters[ij[0]], clusters[ij[1]]),
                            -ij[0], -ij[1]),
        )
        i, j = pair
        clusters[i].extend(clusters.pop(j))
    labels = [0] * len(matrix)
    for label, members in enumerate(clusters):
        for index in members:
            labels[index] = label
    return labels


def _keyword(matrix: np.ndarray, texts: Sequence[str], k: int) -> list[int]:
    """按解释中的成本、路径、容量等关键词分组，并确定性补齐空簇。"""
    scores = np.zeros((len(texts), len(_KEYWORD_GROUPS)), dtype=float)
    for row, text in enumerate(texts):
        lowered = str(text or "").lower()
        for col, group in enumerate(_KEYWORD_GROUPS):
            scores[row, col] = sum(lowered.count(keyword) for keyword in group)
    labels = np.argmax(scores, axis=1).astype(int) % k
    # 关键词重复或缺失时，按文本向量的距离将样本分配到缺失簇。
    missing = [cluster for cluster in range(k) if cluster not in set(labels.tolist())]
    for cluster in missing:
        counts = Counter(labels.tolist())
        donor = max(range(len(labels)), key=lambda i: (scores[i, labels[i]], -counts[labels[i]], -i))
        labels[donor] = cluster
    return labels.tolist()


def select_representatives(
    candidates: Sequence[Any],
    scores: dict[str, float],
    limit: int,
    *,
    mode: str = "topk",
    method: str = "kmeans",
    min_clusters: int = 2,
) -> tuple[list[Any], dict[str, Any]]:
    """筛选候选并返回代表元及可持久化的聚类诊断。

    ``mode='topk'`` 与旧实现一致。``mode='cluster'`` 时，簇数为
    ``min(limit, 候选数)``，因此在候选数足够时至少为 ``min_clusters``；
    候选不足时返回全部候选并记录实际簇数，避免 EoH 早期阶段丢失可行种群。
    """
    if mode not in {"topk", "cluster"}:
        raise ValueError("mode must be 'topk' or 'cluster'")
    if method not in {"kmeans", "agglomerative", "keyword"}:
        raise ValueError("method must be 'kmeans', 'agglomerative' or 'keyword'")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    if type(min_clusters) is not int or min_clusters < 1:
        raise ValueError("min_clusters must be a positive integer")
    unique: dict[str, Any] = {}
    for candidate in candidates:
        unique.setdefault(candidate.code_hash, candidate)
    items = list(unique.values())
    ordered = lambda rows: sorted(rows, key=lambda p: (-scores[p.code_hash], p.id))
    if not items:
        raise ValueError("at least one candidate is required")
    if mode == "topk":
        selected = ordered(items)[:limit]
        return selected, {"mode": mode, "method": None, "cluster_count": 0,
                          "candidate_count": len(items), "representatives": [p.id for p in selected]}

    cluster_count = min(limit, len(items))
    if cluster_count < min_clusters:
        selected = ordered(items)
        return selected, {"mode": mode, "method": method, "cluster_count": len(items),
                          "minimum_clusters": min_clusters, "fallback": "insufficient_candidates",
                          "candidate_count": len(items), "representatives": [p.id for p in selected]}
    texts = [getattr(item, "explanation", "") for item in items]
    matrix = _tfidf(texts)
    if method == "kmeans":
        labels = _kmeans(matrix, cluster_count)
    elif method == "agglomerative":
        labels = _agglomerative(matrix, cluster_count)
    else:
        labels = _keyword(matrix, texts, cluster_count)
    representatives = []
    clusters: dict[int, list[Any]] = {}
    for item, label in zip(items, labels):
        clusters.setdefault(int(label), []).append(item)
    for label in sorted(clusters):
        representatives.append(ordered(clusters[label])[0])
    selected = ordered(representatives)[:limit]
    return selected, {"mode": mode, "method": method, "cluster_count": len(clusters),
                      "minimum_clusters": min_clusters, "candidate_count": len(items),
                      "clusters": {str(label): [p.id for p in members] for label, members in clusters.items()},
                      "representatives": [p.id for p in selected]}
