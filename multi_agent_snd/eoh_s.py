"""EoH-S：面向互补启发式集合的核心算法组件。

本模块不依赖 LLM 或具体求解器，便于单元测试和替换评价函数。
性能向量中的数值均按“越小越好”解释，例如每个训练实例上的 gap。
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from statistics import fmean
from typing import Any, Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EoHSConfig:
    """EoH-S 搜索配置。"""

    population_size: int = 3
    max_iterations: int = 1
    cpi_mode: str = "mean"
    memetic_candidates: int = 2
    cs_enabled: bool = True
    ls_enabled: bool = True
    ls_parent_mode: str = "best_average"

    def __post_init__(self) -> None:
        if self.population_size < 1:
            raise ValueError("population_size 必须为正整数")
        if self.max_iterations < 1:
            raise ValueError("max_iterations 必须为正整数")
        if self.cpi_mode not in {"mean", "weighted"}:
            raise ValueError("cpi_mode 只能是 mean 或 weighted")
        if self.memetic_candidates < 1:
            raise ValueError("memetic_candidates 必须为正整数")
        if not self.cs_enabled and not self.ls_enabled:
            raise ValueError("CS 与 LS 至少启用一个")


def _key(program: Any) -> str:
    """获取程序稳定键；同时兼容 Program 和测试用的简单对象。"""
    return str(getattr(program, "code_hash", getattr(program, "id", program)))


def _vector(vectors: Mapping[str, Sequence[float]], program: Any) -> list[float]:
    try:
        return [float(x) for x in vectors[_key(program)]]
    except KeyError as exc:
        raise KeyError(f"缺少程序 {_key(program)} 的实例级性能向量") from exc


def finite_failure_vectors(vectors: Mapping[str, Sequence[float]],
                           margin: float = 1.0,
                           failure_types: Mapping[str, Sequence[str | None]] | None = None) -> dict[str, list[float]]:
    """逐实例将失败值替为高于最差可行 Gap 的有限惩罚。

    相同场景的所有候选须同时传入，以免不同批次采用不同失败惩罚。
    有失败类型时依次区分行数超限、超时和其他失败；无类型时保持旧版惩罚。
    """
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("margin 必须为正有限数")
    if not vectors:
        return {}
    width = len(next(iter(vectors.values())))
    if width == 0 or any(len(row) != width for row in vectors.values()):
        raise ValueError("性能向量必须非空且维度一致")
    if failure_types is not None and any(
            key not in failure_types or len(failure_types[key]) != width for key in vectors):
        raise ValueError("失败类型向量必须与性能向量同维")
    replacements = []
    for i in range(width):
        valid = [float(row[i]) for row in vectors.values() if math.isfinite(float(row[i]))]
        replacements.append((max(valid) if valid else 0.0) + margin)
    def replace(key: str, index: int, value: float) -> float:
        if math.isfinite(float(value)):
            return float(value)
        if failure_types is None:
            return replacements[index]
        kind = failure_types[key][index]
        severity = {"line_limit": 1.0, "timeout": 2.0}.get(kind, 3.0)
        return replacements[index] + (severity - 1.0) * margin

    return {key: [replace(key, i, value) for i, value in enumerate(row)]
            for key, row in vectors.items()}


def cpi(performance_vectors: Mapping[str, Sequence[float]],
        selected_ids: Sequence[str], weights: Sequence[float] | None = None) -> float:
    """计算 CPI：逐实例取集合中最优值，再对实例求平均。

    ``CPI(H) = sum_i w_i * min_{h in H} f_i(h)``，缺省权重为均匀权重。
    """
    if not selected_ids:
        return float("inf")
    rows = [[float(x) for x in performance_vectors[item]] for item in selected_ids]
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("性能向量必须非空且维度一致")
    best = [min(row[i] for row in rows) for i in range(width)]
    if weights is None:
        return fmean(best)
    if len(weights) != width or any(not math.isfinite(float(w)) or float(w) < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError("weights 必须是非负且总和为正的同维向量")
    total = float(sum(weights))
    return sum(float(w) * value for w, value in zip(weights, best)) / total


def marginal_cpi_gain(candidate_vector: Sequence[float],
                      selected_vectors: Sequence[Sequence[float]],
                      weights: Sequence[float] | None = None) -> float:
    """计算候选程序相对于当前集合的 CPI 改善量（越大越好）。"""
    candidate = [float(x) for x in candidate_vector]
    if not candidate:
        return 0.0
    importance = [1.0] * len(candidate) if weights is None else list(weights)
    if len(importance) != len(candidate):
        raise ValueError("weights 与性能向量维度不一致")
    if not selected_vectors:
        return -sum(w * value for w, value in zip(importance, candidate) if w > 0) / sum(importance)
    current = [min(float(row[i]) for row in selected_vectors)
               for i in range(len(candidate))]
    return sum(importance[i] * (max(current[i] - candidate[i], 0.0)
                if math.isfinite(current[i]) and math.isfinite(candidate[i]) else
                (float("inf") if not math.isfinite(current[i]) and math.isfinite(candidate[i]) else 0.0))
               for i in range(len(candidate)) if importance[i] > 0)


def cpm_select(candidates: Sequence[Any],
               vectors: Mapping[str, Sequence[float]],
               population_size: int,
               weights: Sequence[float] | None = None) -> tuple[list[Any], dict[str, Any]]:
    """互补种群管理（CPM）的贪心选择。

    第一个代表是平均性能最好的程序；随后每次选择能最大幅度改善当前
    逐实例最优值的程序，因此不会只保留同一种行为模式。
    """
    unique: dict[str, Any] = {_key(item): item for item in candidates}
    if not unique:
        return [], {"method": "cpm", "selected": [], "cpi": None}
    items = list(unique.values())
    if any(_key(item) not in vectors for item in items):
        missing = [_key(item) for item in items if _key(item) not in vectors]
        raise KeyError(f"CPM 缺少性能向量: {missing}")
    width = len(_vector(vectors, items[0]))
    if weights is not None and (len(weights) != width or
            any(not math.isfinite(float(w)) or float(w) < 0 for w in weights) or sum(weights) <= 0):
        raise ValueError("weights 必须与性能向量等长、非负且总和为正")
    k = min(max(1, int(population_size)), len(items))
    def average(item: Any) -> float:
        row = _vector(vectors, item)
        return fmean(row) if weights is None else sum(w * value for w, value in zip(weights, row) if w > 0) / sum(weights)

    first = min(items, key=lambda item: (average(item), _key(item)))
    selected = [first]
    selected_keys = {_key(first)}
    gains: list[dict[str, Any]] = []
    while len(selected) < k:
        selected_vectors = [_vector(vectors, item) for item in selected]
        remaining = [item for item in items if _key(item) not in selected_keys]
        candidate = max(
            remaining,
            key=lambda item: (marginal_cpi_gain(_vector(vectors, item), selected_vectors, weights),
                               -average(item), _key(item)),
        )
        gain = marginal_cpi_gain(_vector(vectors, candidate), selected_vectors, weights)
        gains.append({"program_id": getattr(candidate, "id", _key(candidate)), "gain": gain})
        selected.append(candidate)
        selected_keys.add(_key(candidate))
    selected_ids = [_key(item) for item in selected]
    diagnostic = {
        "method": "cpm",
        "population_size": k,
        "selected": [getattr(item, "id", _key(item)) for item in selected],
        "selected_keys": selected_ids,
        "first_selection": getattr(first, "id", _key(first)),
        "marginal_gains": gains,
        "cpi": cpi(vectors, selected_ids, weights),
        "performance_vectors": {key: list(map(float, vectors[key])) for key in unique},
    }
    return selected, diagnostic


def complementary_distance(first: Sequence[float], second: Sequence[float]) -> float:
    """Manhattan 互补距离。"""
    if len(first) != len(second):
        raise ValueError("性能向量维度不一致")
    return sum(abs(float(a) - float(b)) for a, b in zip(first, second))


def select_complementary_parents(population: Sequence[Any],
                                 vectors: Mapping[str, Sequence[float]]) -> tuple[Any, Any]:
    """选择实例级行为差异最大的两个 CS 父代。"""
    if len(population) < 2:
        raise ValueError("CS 至少需要两个不同父代")
    return max(combinations(population, 2), key=lambda pair: (
        complementary_distance(_vector(vectors, pair[0]), _vector(vectors, pair[1])),
        _key(pair[0]), _key(pair[1])))


def select_local_parent(population: Sequence[Any], vectors: Mapping[str, Sequence[float]],
                        mode: str = "best_average") -> Any:
    """选择 LS 父代；默认采用训练实例平均性能最佳者。"""
    if not population:
        raise ValueError("LS 需要至少一个父代")
    if mode != "best_average":
        raise ValueError("当前仅支持 best_average LS 父代策略")
    return min(population, key=lambda item: (fmean(_vector(vectors, item)), _key(item)))


def performance_vector(program: Any, instances: Iterable[Mapping[str, Any]],
                       evaluator: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]],
                       failure_types: list[str | None] | None = None) -> list[float]:
    """记录逐实例性能。

    失败保留为 ``inf`` 并可同时记录失败类型；随后按同批次可行 Gap
    统一设置惩罚，保证任意失败都不会优于可行程序。
    """
    result: list[float] = []
    for instance in instances:
        try:
            row = evaluator(program, instance)
            value = row.get("gap")
            if row.get("status") == "ok" and value is not None:
                score = float(value)
            else:
                score = float("inf")
            if failure_types is not None:
                failure_types.append(None if math.isfinite(score) else row.get("failure_type") or "other")
            result.append(score if math.isfinite(score) else float("inf"))
        except Exception:
            result.append(float("inf"))
            if failure_types is not None:
                failure_types.append("other")
    return result
