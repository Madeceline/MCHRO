"""SND 数据验证、参考解、六指标评估及持久化缓存。"""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
import math
import time
import threading
import random
from dataclasses import replace
from pathlib import Path
from typing import Any
from concurrent.futures import Future

from .constraints import check_arc_flow_solution
from snd_solver import SNDSolver

from .execution import ExecutionError, execute
from .storage import Program, ProgramArchive, write_json
from .utilities import UtilityConfig
from .reference import ReferenceConfig, instance_complexity, select_reference_method
from .performance import TimingRecorder


def digest(data: Any) -> str:
    """内容寻址与执行顺序无关；禁止非有限 JSON 数值。"""
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False,
                                    ensure_ascii=False).encode()).hexdigest()


def validate_instance(data: dict) -> None:
    """验证静态或时间展开的可拆分多商品弧流实例。

    时间相关实例必须把服务时间编码到时间展开网络中；评估器不会悄悄忽略
    ``time_expansion`` 的拓扑元数据，也不会把其还原为静态 OD。
    """
    digest(data)
    SNDSolver._normalize(data)
    for key in ("paths", "service_constraints", "path_constraints"):
        if data.get(key):
            raise ValueError(f"当前弧流评估器不支持 {key}，请扩展评估器后使用")
    for arc in data["arcs"]:
        if any(not math.isfinite(float(arc[k])) for k in ("capacity", "fixed_cost", "variable_cost")):
            raise ValueError("弧属性必须为有限数")
    if any(not math.isfinite(float(k["demand"])) for k in data["commodities"]):
        raise ValueError("需求必须为有限数")


def validate_perturbation(base: dict, generated: dict) -> float:
    """D=mean |x'/x-1|；固定拓扑/OD，每个数值只能取基础值的 [0.5,2] 倍。"""
    validate_instance(generated)
    if generated.get("format") != base.get("format"):
        raise ValueError("生成器不能改变实例格式")
    if generated["nodes"] != base["nodes"]:
        raise ValueError("生成器不能改变节点集合或顺序")
    base_time = base.get("time_expansion")
    generated_time = generated.get("time_expansion")
    if (base_time is None) != (generated_time is None):
        raise ValueError("生成器不能把静态实例与时间展开实例互相转换")
    if base_time is not None:
        # 时间拓扑、服务时刻和商品时间窗属于实例结构，生成器只能扰动成本、容量和需求。
        for key in ("time_points", "node_map", "arc_schedule", "physical_nodes", "physical_arcs"):
            if generated_time.get(key) != base_time.get(key):
                raise ValueError(f"生成器不能改变 time_expansion.{key}")
    changes = []
    for collection, identity, values in (
        ("arcs", ("id", "tail", "head"), ("capacity", "fixed_cost", "variable_cost")),
        ("commodities", ("id", "origin", "destination"), ("demand",)),
    ):
        before, after = base[collection], generated[collection]
        if len(before) != len(after):
            raise ValueError("生成器不能增删弧或商品")
        for old, new in zip(before, after):
            if any(old[key] != new[key] for key in identity):
                raise ValueError("生成器不能改变弧/商品 ID、端点或排列")
            for key in values:
                x, y = float(old[key]), float(new[key])
                if not 0.5 * x <= y <= 2 * x:
                    raise ValueError(f"{collection}.{key} 超出允许扰动范围")
                changes.append(abs(y / x - 1) if x else 0.0)
    if base_time is not None:
        # 等待弧是无收费的跨期持有边，不能让生成器把它变成付费服务。
        scheduled = {str(x["arc_id"]): x for x in base_time["arc_schedule"]}
        for arc in generated["arcs"]:
            if scheduled[str(arc["id"])]["kind"] == "wait":
                if float(arc["fixed_cost"]) != 0 or float(arc["variable_cost"]) != 0:
                    raise ValueError("生成器不能给等待弧增加费用")
    if base_time is not None:
        # commodity_windows 的 demand 可随 commodity demand 一起扰动，其余时间字段必须固定。
        if len(generated_time["commodity_windows"]) != len(base_time["commodity_windows"]):
            raise ValueError("生成器不能增删时间商品窗口")
        before_windows = {str(x["commodity_id"]): x for x in base_time["commodity_windows"]}
        after_windows = {str(x["commodity_id"]): x for x in generated_time["commodity_windows"]}
        if set(before_windows) != set(after_windows):
            raise ValueError("生成器不能增删时间商品窗口")
        generated_demands = {str(x["id"]): float(x["demand"]) for x in generated["commodities"]}
        for key, old in before_windows.items():
            new = after_windows[key]
            for field in ("origin", "destination", "release_time", "deadline"):
                if new.get(field) != old.get(field):
                    raise ValueError(f"生成器不能改变商品时间字段 {field}")
            if float(new.get("demand")) != generated_demands[key]:
                raise ValueError("time_expansion 商品需求必须与 commodities 一致")
    return sum(changes) / len(changes)


def validate_solution(data: dict, solution: Any) -> tuple[float, dict]:
    """重算目标与守恒/容量/二元/非负约束，拒绝 NaN、Inf 和陌生 ID。"""
    if not isinstance(solution, dict):
        raise ValueError("求解器必须返回字典")
    y, flows = solution.get("arc_open"), solution.get("flows")
    if not isinstance(y, dict) or not isinstance(flows, dict):
        raise ValueError("结果必须含 arc_open 和 flows 字典")
    arcs = {str(a["id"]) for a in data["arcs"]}
    commodities = {str(k["id"]) for k in data["commodities"]}

    def check(mapping: dict, allowed: set[str]) -> None:
        if len({str(k) for k in mapping}) != len(mapping) or not {str(k) for k in mapping} <= allowed:
            raise ValueError("结果包含未知或重复 ID")
        if any(not math.isfinite(float(v)) for v in mapping.values()):
            raise ValueError("结果变量必须为有限数")

    check(y, arcs)
    if len({str(k) for k in flows}) != len(flows) or not {str(k) for k in flows} <= commodities:
        raise ValueError("流量中存在未知或重复商品 ID")
    for mapping in flows.values():
        if not isinstance(mapping, dict):
            raise ValueError("每个商品的流量必须是字典")
        check(mapping, arcs)
    objective, checks = check_arc_flow_solution(data, solution)
    if not math.isfinite(objective):
        raise ValueError("目标值溢出或非有限")
    return objective, checks


def gap(value: float, reference: float) -> float:
    """零参考值单独约定：同为零取 0，否则分母取 1，并在文档中说明。"""
    return (value - reference) / (abs(reference) if reference else 1.0)


def lower_bound(data: dict, field: str | None) -> float:
    """无容量、无固定费用的逐商品最短路；field=None 表示需求加权服务弧数。"""
    waiting = ({str(item["arc_id"]) for item in data["time_expansion"]["arc_schedule"]
                if item["kind"] == "wait"} if data.get("time_expansion") else set())
    nodes = {node: i for i, node in enumerate(data["nodes"])}
    adjacency = [[] for _ in nodes]
    for arc in data["arcs"]:
        weight = float(arc[field]) if field is not None else (
            0.0 if str(arc["id"]) in waiting else 1.0)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("最短路下界要求非负有限弧权")
        adjacency[nodes[arc["tail"]]].append((nodes[arc["head"]], weight))
    # 同起点商品共用一次 Dijkstra；等待弧的零权和并行弧均保留。
    by_source = {}
    for commodity in data["commodities"]:
        by_source.setdefault(commodity["origin"], []).append(commodity)
    distances_by_source = {}
    for source in by_source:
        distances = [math.inf] * len(nodes)
        origin = nodes[source]
        distances[origin] = 0.0
        queue = [(0.0, origin)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            for target, weight in adjacency[node]:
                candidate = distance + weight
                if candidate < distances[target]:
                    distances[target] = candidate
                    heapq.heappush(queue, (candidate, target))
        distances_by_source[source] = distances
    # 按原商品顺序求和，尽量保持旧版浮点聚合顺序。
    return sum(distances_by_source[k["origin"]][nodes[k["destination"]]] * float(k["demand"])
               for k in data["commodities"])


class SNDEvaluator:
    """持久化全部生成实例、参考值及候选执行记录；同内容仅计算一次。"""

    def __init__(self, base: dict, archive: ProgramArchive, utility: UtilityConfig,
                 solver_type: str, time_limit: float, mip_gap: float,
                 timeout: float, samples: int, seed: int, log_output: bool = False,
                 training_bases: list[dict] | None = None, *,
                 reference_config: ReferenceConfig | None = None,
                 reference_cache_dir: str | Path | None = None,
                 parallel_workers: int = 4, timing: TimingRecorder | None = None) -> None:
        validate_instance(base)
        self.base, self.archive, self.utility = base, archive, utility
        self.solver_type, self.time_limit, self.mip_gap = solver_type, time_limit, mip_gap
        self.timeout, self.samples, self.seed, self.log_output = timeout, samples, seed, log_output
        self.reference_config = reference_config or ReferenceConfig(seed=seed)
        self.reference_cache_dir = Path(reference_cache_dir) if reference_cache_dir is not None else None
        self.reference_signature = digest({**self.reference_config.identity(),
                                           "solver_type": solver_type,
                                           "time_limit": time_limit, "mip_gap": mip_gap})
        # 范围采样模式下，每个训练样本可以使用不同节点规模和拓扑；
        # 未提供时退化为旧行为：所有生成器样本都从同一个基础实例出发。
        self.training_bases = list(training_bases or [base])
        if not self.training_bases:
            raise ValueError("training_bases 不能为空")
        for item in self.training_bases:
            validate_instance(item)
        self.references: dict[str, dict] = {}
        self.generated: dict[tuple, dict] = {}
        self.evaluations: dict[tuple, dict] = {}
        self.program_index: dict[str, Program] = {}
        # 并行评估时保护缓存和 JSONL 归档；锁不包围耗时的候选执行。
        self._cache_lock = threading.RLock()
        self.lower_bounds: dict[str, dict] = {}
        self._inflight: dict[tuple, Future] = {}
        if type(parallel_workers) is not int or parallel_workers < 1:
            raise ValueError("parallel_workers 必须为正整数")
        # 所有评估线程、主体搜索线程共享进程预算；不在持有此预算时等待缓存。
        self._process_slots = threading.BoundedSemaphore(parallel_workers)
        self.timing = timing or TimingRecorder(archive)

    def _cached(self, kind: str, key, cache: dict, compute):
        """同键 single-flight：只允许一个线程计算，其他线程共享结果或异常。"""
        flight_key = (kind, key)
        with self._cache_lock:
            if key in cache:
                self.timing.count(f"{kind}_cache_hit")
                return cache[key]
            pending = self._inflight.get(flight_key)
            owner = pending is None
            if owner:
                pending = self._inflight[flight_key] = Future()
                self.timing.count(f"{kind}_cache_miss")
            else:
                self.timing.count(f"{kind}_cache_wait")
        if not owner:
            return pending.result()
        try:
            value = compute()
            with self._cache_lock:
                cache[key] = value
            pending.set_result(value)
            return value
        except BaseException as exc:
            pending.set_exception(exc)
            raise
        finally:
            with self._cache_lock:
                self._inflight.pop(flight_key, None)

    def instance_lower_bounds(self, data: dict) -> dict:
        key = digest(data)
        def compute():
            with self.timing.measure("lower_bounds", instance_hash=key):
                return {"path": lower_bound(data, None),
                        "transport": lower_bound(data, "variable_cost")}
        return self._cached("lower_bounds", key, self.lower_bounds, compute)

    def _execute(self, *args, **kwargs):
        with self._process_slots:
            with self.timing.measure("candidate_process" if args[1] == "solve_snd"
                                     else "generator_process"):
                return execute(*args, **kwargs)

    def _execution_line_limit(self, data: dict) -> int:
        """返回候选程序硬上限；可选地随实例规模增长。"""
        fixed = self.utility.execution_line_limit
        if fixed is None:
            fixed = self.utility.work_budget * 10
        scale = self.utility.execution_line_limit_scale
        if scale is None:
            return fixed
        complexity = (len(data.get("nodes", ())) + len(data.get("arcs", ()))
                      + len(data.get("commodities", ())))
        if data.get("time_expansion"):
            complexity += len(data["time_expansion"].get("time_points", ()))
        return max(1, min(fixed, scale * max(1, complexity)))

    def reference(self, data: dict) -> dict:
        return self._cached("reference", digest(data), self.references,
                            lambda: self._reference(data))

    def _reference(self, data: dict) -> dict:
        key = digest(data)
        with self._cache_lock:
            cached = self.references.get(key)
        if cached is not None:
            return cached
        method = select_reference_method(data, self.reference_config)
        cache_path = (self.reference_cache_dir / self.reference_signature / f"{key}.json"
                      if self.reference_cache_dir is not None else None)
        row = None
        if cache_path is not None and cache_path.is_file():
            try:
                saved = json.loads(cache_path.read_text(encoding="utf-8"))
                objective, checks = validate_solution(data, saved)
                if (checks["feasible"] and saved.get("reference_signature") == self.reference_signature
                        and saved.get("instance_hash") == key
                        and saved.get("selected_method") == method):
                    row = saved
                    self.timing.count("reference_disk_cache_hit")
            except (ValueError, TypeError, KeyError, OSError):
                row = None
        if row is None:
            cfg = self.reference_config
            arguments = ((data, replace(cfg, timeout=cfg.timeout * .8))
                         if method == "heuristic" else
                         (data, self.solver_type, self.time_limit, self.mip_gap, self.log_output))
            from .execution import isolated_call
            with self._process_slots:
                with self.timing.measure("reference_process", method=method, instance_hash=key):
                    output = isolated_call("reference_heuristic" if method == "heuristic" else "reference_mip",
                                           arguments, cfg.timeout)
            row = output["result"]
            objective, checks = validate_solution(data, row)
            if not checks["feasible"]:
                raise ValueError("未获得通过独立检查的参考可行解；启发式失败不证明不可行")
            row.update(value=objective, constraint_satisfaction=checks,
                       instance_hash=key, selected_method=method,
                       instance_complexity=instance_complexity(data),
                       complexity_threshold=self.reference_config.complexity_threshold,
                       reference_signature=self.reference_signature,
                       reference_config=self.reference_config.identity(),
                       wall_time=output["wall_time"])
            if cache_path is not None:
                write_json(cache_path, row)
            with self._cache_lock:
                if key in self.references:
                    return self.references[key]
                self.references[key] = row
                write_json(self.archive.root / "instances" / f"{key}.json", data)
                write_json(self.archive.root / "references" / f"{key}.json", row)
        with self._cache_lock:
            if key not in self.references:
                self.references[key] = row
            return self.references[key]

    def sample(self, generator: Program, index: int) -> dict:
        key = (generator.code_hash, generator.player, index)
        return self._cached("sample", key, self.generated,
                            lambda: self._sample(generator, index))

    def _sample(self, generator: Program, index: int) -> dict:
        # 所有候选使用共同随机数：同一玩家/样本固定 seed，避免排序影响难度。
        if generator.mixture_weights is not None:
            key = (generator.code_hash, generator.player, index)
            with self._cache_lock:
                cached = self.generated.get(key)
            if cached is not None:
                return cached
            selected = self._draw_member(generator, [self.seed, generator.code_hash, index])
            row = {**self.sample(selected, index), "selected_generator_member_id": selected.id,
                   "mixture_weights": list(generator.mixture_weights)}
            with self._cache_lock:
                if key not in self.generated:
                    self.generated[key] = row
                    self.archive.append("samples.jsonl", {
                        "program_id": generator.id, "sample": index,
                        **{k: v for k, v in row.items() if k != "instance"}})
                return self.generated[key]
        key = (generator.code_hash, generator.player, index)
        with self._cache_lock:
            cached = self.generated.get(key)
        if cached is not None:
            return cached
        if key not in self.generated:
            seed = int(digest([self.seed, generator.player, index])[:8], 16)
            started = time.perf_counter()
            source_base = self.training_bases[index % len(self.training_bases)]
            try:
                result = self._execute(generator.code, "generate_snd", copy.deepcopy(source_base), seed,
                                 timeout=self.timeout,
                                 line_limit=self._execution_line_limit(source_base),
                                 max_code_lines=self.utility.max_code_lines,
                                 max_code_chars=self.utility.max_code_chars,
                                 max_ast_nodes=self.utility.max_ast_nodes)
                data = result["result"]
                distortion = validate_perturbation(source_base, data)
                reference = self.reference(data)
                row = {"status": "ok", "instance": data, "instance_hash": digest(data),
                       "source_instance_hash": digest(source_base),
                       "distortion": distortion, "reference": reference, "seed": seed}
            except Exception as exc:
                row = {"status": "invalid_instance", "error": str(exc),
                       "source_instance_hash": digest(source_base),
                       "distortion": 0, "seed": seed}
                if isinstance(exc, ExecutionError):
                    row.update({"failure_type": exc.failure_type, "lines": exc.lines,
                                "elapsed": exc.elapsed,
                                "line_limit": self._execution_line_limit(source_base)})
                elif "源码超过" in str(exc):
                    row["failure_type"] = "code_limit"
            row["sample_wall_time"] = time.perf_counter() - started
            with self._cache_lock:
                if key in self.generated:
                    return self.generated[key]
                self.generated[key] = row
                self.archive.append("samples.jsonl", {"program_id": generator.id, "sample": index,
                                                        **{k: v for k, v in row.items() if k != "instance"}})
        return self.generated[key]

    def evaluate(self, solver: Program, data: dict) -> dict:
        if solver.member_ids is not None:
            return self._evaluate(solver, data)
        key = (solver.code_hash, digest(data))
        def compute():
            with self.timing.measure("evaluation", instance_hash=key[1], code_hash=key[0]):
                return self._evaluate(solver, data)
        return self._cached("evaluation", key, self.evaluations,
                            compute)

    def _evaluate(self, solver: Program, data: dict) -> dict:
        if solver.mixture_weights is not None:
            # 元博弈收益在 pair() 中精确加权；实际部署先按确定性抽样，失败时
            # 按权重顺序尝试其余成员，避免一个脆弱成员让整个混合策略失效。
            selected = self._draw_member(solver, [self.seed, solver.code_hash, digest(data)])
            # 与导出类一致：未抽中的成员按权重降序、原顺序破平局回退。
            remaining = sorted(
                ((index, self.program_index[mid]) for index, mid in enumerate(solver.member_ids)
                 if mid != selected.id),
                key=lambda item: (-solver.mixture_weights[item[0]], item[0]),
            )
            order = [selected] + [member for _, member in remaining]
            failures = []
            for member in order:
                row = self.evaluate(member, data)
                if row["status"] == "ok":
                    return {**row, "selected_member_id": member.id,
                            "mixture_weights": list(solver.mixture_weights),
                            "fallback_used": member.id != selected.id,
                            "failed_members": failures}
                failures.append({"member_id": member.id, "error": row.get("error")})
            return {"status": "solver_failure", "gap": None,
                    "constraint_satisfaction": {"feasible": False},
                    "error": "混合策略所有成员均失败", "failed_members": failures,
                    "selected_member_id": selected.id,
                    "mixture_weights": list(solver.mixture_weights),
                    "reference_kind": self.reference(data)["kind"]}
        # 集合成员逐个独立执行并验解，按实际目标选最低成本可行解；
        # 无需参考最优值，因此训练收益和部署行为一致。
        if solver.member_ids is not None:
            self.program_index.update({p.id: p for p in self.archive.programs})
            rows = [self.evaluate(self.program_index[member_id], data)
                    for member_id in solver.member_ids]
            feasible = [(i, row) for i, row in enumerate(rows) if row["status"] == "ok"]
            if feasible:
                index, best = min(feasible, key=lambda pair: (pair[1]["objective"], pair[0]))
                metrics = dict(best["metrics"])
                # 集合部署确实运行了全部成员，工作量必须按实际总执行量计费。
                metrics["work"] = min(1.0, sum(row.get("lines", 0) for row in rows)
                                      / self.utility.work_budget)
                return {**best, "metrics": metrics,
                        "lines": sum(row.get("lines", 0) for row in rows),
                        "selected_member_id": solver.member_ids[index],
                        "attempted_member_ids": list(solver.member_ids),
                        "ensemble_wall_time": sum(row.get("wall_time", 0) for row in rows)}
            return {"status": "solver_failure", "gap": None,
                    "constraint_satisfaction": {"feasible": False},
                    "error": "集合中所有程序均失败", "reference_kind": self.reference(data)["kind"],
                    "attempted_member_ids": list(solver.member_ids),
                    "wall_time": sum(row.get("wall_time", 0) for row in rows)}
        key = (solver.code_hash, digest(data))
        with self._cache_lock:
            cached = self.evaluations.get(key)
        if cached is not None:
            return cached
        if key not in self.evaluations:
            reference = self.reference(data)
            started = time.perf_counter()
            try:
                execution = self._execute(solver.code, "solve_snd", copy.deepcopy(data), timeout=self.timeout,
                                    line_limit=self._execution_line_limit(data),
                                    max_code_lines=self.utility.max_code_lines,
                                    max_code_chars=self.utility.max_code_chars,
                                    max_ast_nodes=self.utility.max_ast_nodes)
                solution = execution["result"]
                objective, checks = validate_solution(data, solution)
                if not checks["feasible"]:
                    raise ValueError(f"不可行方案：{checks}")
                opened = {str(k): float(v) for k, v in solution["arc_open"].items()}
                flows = {str(k): {str(a): float(v) for a, v in f.items()} for k, f in solution["flows"].items()}
                loads = {str(a["id"]): sum(f.get(str(a["id"]), 0) for f in flows.values()) for a in data["arcs"]}
                # 时间展开网络中，等待弧表示库存/滞留，不是运输服务；
                # 路径长度和服务拥堵指标只统计真实运输弧。
                waiting = ({str(item["arc_id"]) for item in data["time_expansion"]["arc_schedule"]
                            if item["kind"] == "wait"} if data.get("time_expansion") else set())
                service_arcs = [a for a in data["arcs"] if str(a["id"]) not in waiting]
                fixed = sum(a["fixed_cost"] * opened.get(str(a["id"]), 0) for a in data["arcs"])
                transport = sum(a["variable_cost"] * loads[str(a["id"])] for a in data["arcs"])
                bounds = self.instance_lower_bounds(data)

                def bounded(z: float) -> float:
                    z = max(0.0, z)
                    return z / (1 + z)

                metrics = {"cost": bounded(gap(objective, reference["value"])),
                           "path": bounded(gap(sum(loads[str(a["id"])] for a in service_arcs),
                                               bounds["path"])),
                           "transport": bounded(gap(transport, bounds["transport"])),
                           "fixed": fixed / max(sum(a["fixed_cost"] for a in data["arcs"]), 1e-12),
                           "congestion": min(1.0, max((loads[str(a["id"])] / max(a["capacity"], 1e-12)
                                                       for a in service_arcs), default=0.0)),
                           "work": min(1.0, execution["lines"] / self.utility.work_budget)}
                row = {"status": "ok", "objective": objective, "gap": gap(objective, reference["value"]),
                       "reference_kind": reference["kind"], "reference_value": reference["value"],
                       "metrics": metrics, "arc_open": opened, "flows": flows,
                       "constraint_satisfaction": checks, "lines": execution["lines"],
                       "elapsed": execution["elapsed"], "wall_time": execution["wall_time"]}
            except Exception as exc:
                row = {"status": "solver_failure", "error": str(exc), "gap": None,
                       "reference_kind": reference["kind"], "constraint_satisfaction": {"feasible": False},
                       "wall_time": time.perf_counter() - started, "elapsed": None}
                if isinstance(exc, ExecutionError):
                    row.update({"failure_type": exc.failure_type, "lines": exc.lines,
                                "elapsed": exc.elapsed,
                                "line_limit": self._execution_line_limit(data)})
                else:
                    message = str(exc)
                    row["failure_type"] = ("code_limit" if "源码超过" in message
                                            else "invalid_solution")
            with self._cache_lock:
                if key in self.evaluations:
                    return self.evaluations[key]
                self.evaluations[key] = row
                self.archive.append("evaluations.jsonl", {"program_id": solver.id, "code_hash": solver.code_hash,
                                                           "instance_hash": key[1], **row})
        return self.evaluations[key]

    def mixture_distribution(self, solver: Program, data: dict) -> dict:
        """精确列出单实例上各成员的执行分布，不用一次抽样估计可行率。"""
        if solver.mixture_weights is None:
            raise ValueError("mixture_distribution 仅适用于混合求解器")
        self.program_index.update({p.id: p for p in self.archive.programs})
        members = []
        for member_id, weight in zip(solver.member_ids, solver.mixture_weights):
            row = self.evaluate(self.program_index[member_id], data)
            members.append({"member_id": member_id, "weight": weight,
                            "status": row["status"], "gap": row.get("gap"),
                            "objective": row.get("objective")})
        successful = [item for item in members if item["status"] == "ok"]
        success_rate = sum(item["weight"] for item in successful)
        return {"instance_hash": digest(data), "members": members,
                "expected_success_rate": success_rate,
                "expected_gap_given_success": (
                    sum(item["weight"] * item["gap"] for item in successful) / success_rate
                    if success_rate > 0 else None),
                "expected_gap": (
                    sum(item["weight"] * item["gap"] for item in members)
                    if len(successful) == len(members) else None)}

    def pair(self, solver: Program, generator: Program) -> tuple[float, float]:
        # 混合响应作为一个元博弈动作；线性展开成员收益，避免 Monte Carlo 噪声。
        if solver.mixture_weights is not None or generator.mixture_weights is not None:
            self.program_index.update({p.id: p for p in self.archive.programs})
        if solver.mixture_weights is not None:
            rows = [self.pair(self.program_index[member_id], generator)
                    for member_id in solver.member_ids]
            return tuple(sum(weight * row[i] for weight, row in zip(solver.mixture_weights, rows))
                         for i in range(2))
        if generator.mixture_weights is not None:
            rows = [self.pair(solver, self.program_index[member_id])
                    for member_id in generator.member_ids]
            return tuple(sum(weight * row[i] for weight, row in zip(generator.mixture_weights, rows))
                         for i in range(2))
        values = []
        for index in range(self.samples):
            sample = self.sample(generator, index)
            if sample["status"] == "ok":
                observation = {**self.evaluate(solver, sample["instance"]), "distortion": sample["distortion"]}
            else:
                observation = sample
            values.append(self.utility.pair(observation, generator.player))
        return tuple(sum(row[j] for row in values) / len(values) for j in range(2))

    def _draw_member(self, strategy: Program, key: list[Any]) -> Program:
        """对相同实例/样本和全局种子固定抽取，避免评估顺序改变部署结果。"""
        self.program_index.update({p.id: p for p in self.archive.programs})
        draw = random.Random(int(digest(key)[:16], 16)).random()
        cumulative = 0.0
        for member_id, weight in zip(strategy.member_ids, strategy.mixture_weights):
            cumulative += weight
            if draw < cumulative:
                return self.program_index[member_id]
        return self.program_index[strategy.member_ids[-1]]
