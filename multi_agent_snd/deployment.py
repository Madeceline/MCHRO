"""最终启发式的可调用包装；部署时不再访问 LLM，也不偷偷改用 MIP。"""
from __future__ import annotations

import time
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from snd_solver import SNDSolver
from .evaluation import validate_instance, validate_solution
from .execution import execute


def _effective_line_limit(data: dict[str, Any], fixed: int, scale: int | None) -> int:
    """与训练评估器一致的实例规模执行上限。"""
    if scale is None:
        return fixed
    complexity = (len(data.get("nodes", ())) + len(data.get("arcs", ()))
                  + len(data.get("commodities", ())))
    if data.get("time_expansion"):
        complexity += len(data["time_expansion"].get("time_points", ()))
    return max(1, min(fixed, scale * max(1, complexity)))


def _validate_runtime_limits(timeout: float, line_limit: int, scale: int | None) -> None:
    if not math.isfinite(float(timeout)) or timeout <= 0:
        raise ValueError("timeout 必须为正有限数")
    if type(line_limit) is not int or line_limit < 1:
        raise ValueError("line_limit 必须为正整数")
    if scale is not None and (type(scale) is not int or scale < 1):
        raise ValueError("line_limit_scale 必须为正整数或 None")


class ProgramSNDSolver:
    """ProgramSNDSolver(code_path).solve(instance)，兼容实例对象/字典/JSON。"""

    def __init__(self, code_path: str | Path, timeout: float = 30, line_limit: int = 2000000,
                 line_limit_scale: int | None = None) -> None:
        _validate_runtime_limits(timeout, line_limit, line_limit_scale)
        self.code = Path(code_path).read_text(encoding="utf-8")
        self.timeout, self.line_limit, self.line_limit_scale = timeout, line_limit, line_limit_scale

    def solve(self, instance: Any) -> dict[str, Any]:
        started = time.perf_counter()
        data = SNDSolver._load_instance(instance)
        validate_instance(data)
        result = execute(self.code, "solve_snd", data, timeout=self.timeout,
                         line_limit=_effective_line_limit(data, self.line_limit, self.line_limit_scale))
        objective, checks = validate_solution(data, result["result"])
        if not checks["feasible"]:
            raise ValueError(f"生成启发式返回不可行方案：{checks}")
        return {**result["result"], "objective": objective, "objective_value": objective,
                "optimal": False, "solve_time": time.perf_counter() - started,
                "gap": None, "constraint_satisfaction": checks, "executed_lines": result["lines"],
                "solver_status": "heuristic_feasible", "solver_used": "generated_heuristic",
                "instance_format": data.get("format", "static-snd-v2"),
                "time_expansion": data.get("time_expansion")}


class EnsembleSNDSolver:
    """运行导出集合的全部成员，验解后选目标值最小的可行方案。"""

    def __init__(self, member_paths: list[str | Path], timeout: float = 30,
                 line_limit: int = 2000000, line_limit_scale: int | None = None) -> None:
        if not member_paths:
            raise ValueError("启发式集合不能为空")
        _validate_runtime_limits(timeout, line_limit, line_limit_scale)
        self.member_paths = [Path(path) for path in member_paths]
        self.timeout, self.line_limit, self.line_limit_scale = timeout, line_limit, line_limit_scale

    def solve(self, instance: Any) -> dict[str, Any]:
        started = time.perf_counter()
        data = SNDSolver._load_instance(instance)
        validate_instance(data)
        feasible = []
        failures = []
        for index, path in enumerate(self.member_paths):
            try:
                row = ProgramSNDSolver(path, self.timeout, self.line_limit,
                                       self.line_limit_scale).solve(data)
                feasible.append((row["objective"], index, row))
            except Exception as exc:
                failures.append({"member": path.name, "error": str(exc)})
        if not feasible:
            raise RuntimeError(f"集合中没有可行求解器：{failures}")
        _, index, chosen = min(feasible, key=lambda row: (row[0], row[1]))
        return {**chosen, "solve_time": time.perf_counter() - started,
                "selected_member": self.member_paths[index].name,
                "attempted_members": [path.name for path in self.member_paths],
                "failed_members": failures, "solver_used": "generated_heuristic_ensemble"}


class MixedSNDSolver:
    """M-PSRO 部署：按混合策略抽样，失败时按权重顺序回退到其他成员。

    训练阶段仍按概率评估；部署阶段的回退只在抽中的程序异常、超时或返回不可行解
    时触发，避免单个脆弱成员让整个混合求解器失效。
    """

    def __init__(self, member_paths: list[str | Path], weights: list[float], *,
                 seed: int = 42, strategy_hash: str | None = None,
                 timeout: float = 30, line_limit: int = 2000000,
                 line_limit_scale: int | None = None) -> None:
        if (not member_paths or len(member_paths) != len(weights)
                or any(not math.isfinite(w) or w <= 0 for w in weights)
                or not math.isclose(sum(weights), 1.0, abs_tol=1e-9)):
            raise ValueError("混合求解器需要同数量的成员路径和归一化正权重")
        _validate_runtime_limits(timeout, line_limit, line_limit_scale)
        self.member_paths = [Path(path) for path in member_paths]
        self.weights = list(weights)
        self.strategy_hash = strategy_hash
        self.seed, self.timeout, self.line_limit, self.line_limit_scale = seed, timeout, line_limit, line_limit_scale

    def solve(self, instance: Any) -> dict[str, Any]:
        data = SNDSolver._load_instance(instance)
        validate_instance(data)
        # 与训练评估一致：随机源只取决于 seed、混合支持集和实例内容。
        identity = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
        mix_id = self.strategy_hash or hashlib.sha256(json.dumps({"members": [
            hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.member_paths], "weights": self.weights}).encode()).hexdigest()
        instance_hash = hashlib.sha256(identity.encode()).hexdigest()
        key = hashlib.sha256(json.dumps([self.seed, mix_id, instance_hash],
                                        sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        draw = random.Random(int(key[:16], 16)).random()
        cumulative = 0.0
        index = len(self.weights) - 1
        for i, weight in enumerate(self.weights):
            cumulative += weight
            if draw < cumulative:
                index = i
                break
        # 先尝试确定性抽中的成员，再按权重从高到低回退；结果完全由实例哈希决定。
        order = [index] + [i for i, _ in sorted(
            enumerate(self.weights), key=lambda item: (-item[1], item[0])) if i != index]
        failures = []
        attempted = []
        for candidate_index in order:
            attempted.append(self.member_paths[candidate_index].stem)
            try:
                result = ProgramSNDSolver(self.member_paths[candidate_index],
                                          self.timeout, self.line_limit,
                                          self.line_limit_scale).solve(data)
                return {**result, "selected_member": self.member_paths[candidate_index].stem,
                        "mixture_weights": self.weights,
                        "attempted_members": attempted,
                        "fallback_used": candidate_index != index,
                        "failed_members": failures,
                        "solver_used": "generated_mpsro_mixture"}
            except Exception as exc:
                failures.append({"member": self.member_paths[candidate_index].stem,
                                 "error": str(exc)})
        raise RuntimeError(f"混合策略所有成员均失败：{failures}")


class MetaSNDSolver:
    """部署求解器元策略边际：先抽一个历史响应，再执行它的纯/混合程序。"""

    def __init__(self, policies: list[dict[str, Any]], probabilities: list[float], *,
                 seed: int = 42, timeout: float = 30, line_limit: int = 2000000,
                 line_limit_scale: int | None = None) -> None:
        if (not policies or len(policies) != len(probabilities)
                or any(not math.isfinite(p) or p < 0 for p in probabilities)
                or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)):
            raise ValueError("元策略须为非空策略表及归一化非负概率")
        _validate_runtime_limits(timeout, line_limit, line_limit_scale)
        self.policies, self.probabilities = policies, probabilities
        self.seed, self.timeout, self.line_limit, self.line_limit_scale = seed, timeout, line_limit, line_limit_scale

    def solve(self, instance: Any) -> dict[str, Any]:
        data = SNDSolver._load_instance(instance)
        validate_instance(data)
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False)
        instance_hash = hashlib.sha256(payload.encode()).hexdigest()
        distribution_hash = hashlib.sha256(json.dumps(
            {"policies": [p["id"] for p in self.policies], "probabilities": self.probabilities},
            sort_keys=True).encode()).hexdigest()
        key = hashlib.sha256(json.dumps([self.seed, distribution_hash, instance_hash]).encode()).hexdigest()
        draw = random.Random(int(key[:16], 16)).random()
        cumulative = 0.0
        index = len(self.probabilities) - 1
        for i, probability in enumerate(self.probabilities):
            cumulative += probability
            if draw < cumulative:
                index = i
                break
        policy = self.policies[index]
        if policy.get("mixture_weights") is None:
            solver = ProgramSNDSolver(policy["member_paths"][0], self.timeout, self.line_limit,
                                      self.line_limit_scale)
        else:
            solver = MixedSNDSolver(policy["member_paths"], policy["mixture_weights"],
                                    seed=self.seed, strategy_hash=policy["strategy_hash"],
                                    timeout=self.timeout, line_limit=self.line_limit,
                                    line_limit_scale=self.line_limit_scale)
        result = solver.solve(data)
        return {**result, "selected_meta_policy_id": policy["id"],
                "meta_probabilities": list(self.probabilities),
                "solver_used": "generated_mpsro_meta_strategy"}
