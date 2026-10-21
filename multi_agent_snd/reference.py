"""限时、可复现的参考解构造；默认不枚举弧组合。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import heapq
import math
import random
import time

from .constraints import check_arc_flow_solution

REFERENCE_VERSION = "reference-v3"
HEURISTIC_METHODS = (
    "randomized_residual_dijkstra",
    "variable_cost",
    "fixed_cost_greedy",
    "capacity_aware",
)


@dataclass(frozen=True)
class ReferenceConfig:
    """参考解选择与预算；启发式失败不等价于证明实例不可行。"""

    method: str = "auto"
    complexity_threshold: int = 1000
    heuristic: str = "randomized_residual_dijkstra"
    restarts: int = 4
    max_paths: int = 16
    timeout: float = 15.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.method not in {"auto", "heuristic", "mip"}:
            raise ValueError("reference_method 必须为 auto、heuristic 或 mip")
        if type(self.complexity_threshold) is not int or self.complexity_threshold < 1:
            raise ValueError("reference_complexity_threshold 必须为正整数")
        if self.heuristic not in HEURISTIC_METHODS:
            raise ValueError(f"reference_heuristic 必须是 {HEURISTIC_METHODS}")
        for name in ("restarts", "max_paths"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"reference_{name} 必须为正整数")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("reference_timeout 必须为正有限数")
        if type(self.seed) is not int:
            raise ValueError("reference seed 必须为整数")

    def identity(self) -> dict:
        return {"version": REFERENCE_VERSION, **asdict(self)}


def instance_complexity(data: dict) -> int:
    """综合规模：节点数 + 弧数 + 商品数（时间展开点数也计入）。"""
    value = len(data.get("nodes", ())) + len(data.get("arcs", ()))
    value += len(data.get("commodities", ()))
    if data.get("time_expansion"):
        value += len(data["time_expansion"].get("time_points", ()))
    return value


def select_reference_method(data: dict, config: ReferenceConfig) -> str:
    if config.method == "auto":
        return "mip" if instance_complexity(data) <= config.complexity_threshold else "heuristic"
    return config.method


def _weight(arc: dict, arc_index: int, residual: float, opened: set[int],
            remaining: float, mode: str, factor: float) -> float:
    variable = float(arc["variable_cost"])
    fixed = float(arc["fixed_cost"])
    if mode == "variable_cost":
        base = variable
    elif mode == "fixed_cost_greedy":
        base = variable + fixed * (arc_index not in opened) / max(remaining, 1e-8)
    elif mode == "capacity_aware":
        ratio = residual / max(float(arc["capacity"]), 1.0)
        base = variable * (1.0 + 0.5 / max(ratio, 0.05))
        base += fixed * (arc_index not in opened) / max(remaining, 1e-8)
    else:  # randomized_residual_dijkstra
        base = variable + fixed * (arc_index not in opened) / max(remaining, 1e-8)
    return max(base * factor, 1e-12)


def heuristic_reference(data: dict, config: ReferenceConfig) -> dict:
    """多次残余容量最短路构造，返回预算内费用最低的可行解。"""
    started = time.perf_counter()
    deadline = started + config.timeout
    arcs, commodities = data["arcs"], data["commodities"]
    adjacency = {node: [] for node in data["nodes"]}
    for index, arc in enumerate(arcs):
        adjacency[arc["tail"]].append(index)
    rng = random.Random(config.seed)
    best = None
    completed = searches = 0
    interrupted = False
    mode = config.heuristic
    for restart in range(config.restarts):
        if time.perf_counter() >= deadline:
            interrupted = True
            break
        order = list(commodities)
        if restart == 0:
            order.sort(key=lambda item: (-float(item["demand"]), str(item["id"])))
        else:
            rng.shuffle(order)
        residual = [float(arc["capacity"]) for arc in arcs]
        opened: set[int] = set()
        flows = {}
        randomized = mode == "randomized_residual_dijkstra"
        factors = [1.0 if restart == 0 and not randomized else rng.uniform(.85, 1.15)
                   for _ in arcs]
        if mode == "fixed_cost_greedy":
            fixed_weight = 1.0
        else:
            fixed_weight = 1.0 if restart == 0 else rng.uniform(.25, 1.75)
        success = True
        for commodity in order:
            values = {}
            remaining = float(commodity["demand"])
            source, target = commodity["origin"], commodity["destination"]
            for _ in range(config.max_paths):
                if remaining <= 1e-8:
                    break
                if time.perf_counter() >= deadline:
                    interrupted = True
                    break
                searches += 1
                distances = {source: 0.0}
                previous = {}
                queue = [(0.0, source)]
                while queue:
                    distance, node = heapq.heappop(queue)
                    if distance != distances[node]:
                        continue
                    if node == target:
                        break
                    if time.perf_counter() >= deadline:
                        interrupted = True
                        break
                    for index in adjacency[node]:
                        if residual[index] <= 1e-8:
                            continue
                        arc = arcs[index]
                        factor = factors[index] if randomized else 1.0
                        weight = _weight(arc, index, residual[index], opened, remaining, mode, factor)
                        candidate = distance + weight
                        if candidate < distances.get(arc["head"], math.inf):
                            distances[arc["head"]] = candidate
                            previous[arc["head"]] = index
                            heapq.heappush(queue, (candidate, arc["head"]))
                if interrupted or target not in previous:
                    break
                path = []
                node = target
                while node != source:
                    index = previous[node]
                    path.append(index)
                    node = arcs[index]["tail"]
                amount = min(remaining, *(residual[index] for index in path))
                for index in path:
                    key = str(arcs[index]["id"])
                    values[key] = values.get(key, 0.0) + amount
                    residual[index] -= amount
                    opened.add(index)
                remaining -= amount
            if remaining > 1e-8:
                success = False
                break
            flows[str(commodity["id"])] = values
        completed += 1
        if success:
            solution = {"arc_open": {str(arcs[index]["id"]): 1.0 for index in sorted(opened)},
                        "flows": flows}
            objective, checks = check_arc_flow_solution(data, solution)
            if checks["feasible"] and (best is None or objective < best["value"]):
                best = {**solution, "value": objective, "objective_value": objective,
                        "constraint_satisfaction": checks}
        if interrupted:
            break
    if best is None:
        raise ValueError("参考启发式未找到可行解（不是不可行证明）；可增加 restarts/max_paths/timeout，或选择 mip")
    return {**best, "kind": "heuristic", "is_optimal": False, "mip_gap": None,
            "status": "heuristic_feasible", "solver_used": mode,
            "reference_heuristic": mode, "solve_time": time.perf_counter() - started,
            "restarts_completed": completed, "path_searches": searches,
            "budget_exhausted": interrupted}


def mip_reference(data: dict, solver_type: str, time_limit: float,
                  mip_gap: float, log_output: bool = False) -> dict:
    from snd_solver import SNDSolver
    result = SNDSolver(data, solver_type, time_limit, mip_gap, log_output).solve()
    row = result.to_dict()
    row["kind"] = "optimal" if result.is_optimal and (result.mip_gap or 0) <= 1e-9 else "incumbent"
    return row
