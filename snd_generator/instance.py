"""静态及时间展开多商品 SND 实例的数据结构与生成器。"""

from __future__ import annotations

import json
import heapq
import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Arc:
    """有向网络弧及其设计和经济属性。"""
    id: int
    tail: int
    head: int
    capacity: int
    variable_cost: float
    fixed_cost: float

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "tail": self.tail, "head": self.head,
                "capacity": self.capacity, "variable_cost": self.variable_cost,
                "fixed_cost": self.fixed_cost}


@dataclass(frozen=True, slots=True)
class Commodity:
    """需要在网络中进行路由的起讫点需求。"""
    id: int
    origin: int
    destination: int
    demand: int

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "origin": self.origin,
                "destination": self.destination, "demand": self.demand}


@dataclass(frozen=True, slots=True)
class SNDParameterBounds:
    """SND 实例生成参数的闭区间。

    ``nodes`` 使用整数上下界；``coverage`` 和 ``fixed_cost`` 使用
    ``[0, 1]`` 内的浮点上下界。调用 :meth:`sample` 时，参数使用
    给定随机种子采样。``time_periods=None`` 保持静态格式；
    提供区间时生成时间展开网络，``max_travel_time`` 限制物理弧运输时长。
    区间端点都是“时间段数”，不是现实中的小时数；每段等长，具体单位由用户约定。
    """

    nodes: tuple[int, int] = (2, 2)
    coverage: tuple[float, float] = (0.5, 0.5)
    fixed_cost: tuple[float, float] = (0.5, 0.5)
    time_periods: tuple[int, int] | None = None
    max_travel_time: int = 2

    def __post_init__(self) -> None:
        # 复制为规范 tuple，避免列表等可变对象进入配置和哈希过程。
        nodes = tuple(self.nodes)
        coverage = tuple(self.coverage)
        fixed_cost = tuple(self.fixed_cost)
        if len(nodes) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in nodes):
            raise TypeError("nodes bounds must contain two integers")
        if nodes[0] < 2 or nodes[0] > nodes[1]:
            raise ValueError("nodes bounds must satisfy 2 <= min <= max")
        for name, values in (("coverage", coverage), ("fixed_cost", fixed_cost)):
            if len(values) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                        or not math.isfinite(float(x)) for x in values):
                raise TypeError(f"{name} bounds must contain two finite numbers")
            if not 0.0 <= float(values[0]) <= float(values[1]) <= 1.0:
                raise ValueError(f"{name} bounds must satisfy 0 <= min <= max <= 1")
        object.__setattr__(self, "nodes", (nodes[0], nodes[1]))
        object.__setattr__(self, "coverage", (float(coverage[0]), float(coverage[1])))
        object.__setattr__(self, "fixed_cost", (float(fixed_cost[0]), float(fixed_cost[1])))
        if self.time_periods is not None:
            periods = tuple(self.time_periods)
            if (len(periods) != 2 or any(type(x) is not int for x in periods)
                    or periods[0] < 1 or periods[0] > periods[1]):
                raise ValueError("time_periods bounds must satisfy 1 <= min <= max")
            object.__setattr__(self, "time_periods", periods)
        if type(self.max_travel_time) is not int or self.max_travel_time < 1:
            raise ValueError("max_travel_time must be a positive integer")

    def sample(self, *, seed: int | None = None) -> "SNDInstance":
        """按范围采样并返回一份可复现的实例。"""
        rng = random.Random(seed)
        node_count = rng.randint(*self.nodes)
        coverage = rng.uniform(*self.coverage)
        fixed_cost = rng.uniform(*self.fixed_cost)
        periods = rng.randint(*self.time_periods) if self.time_periods is not None else None
        return SNDInstance(node_count, coverage, fixed_cost, seed=seed,
                           time_periods=periods, max_travel_time=self.max_travel_time)

    def to_dict(self) -> dict[str, Any]:
        """返回适合写入 JSON 的范围配置。"""
        result = {"nodes": list(self.nodes), "coverage": list(self.coverage),
                  "fixed_cost": list(self.fixed_cost)}
        if self.time_periods is not None:
            result.update(time_periods=list(self.time_periods),
                          max_travel_time=self.max_travel_time)
        return result


class SNDInstance:
    """一个可复现的多商品 SND 实例；默认静态，可选时间展开。

    ``time_periods=H`` 表示 H 个等长时间段、H+1 个边界时刻 ``0..H``；
    例如 H=3 对应 ``0→1→2→3``。它不表示生成 H 个实例。
    时间段在现实中等于多少分钟/小时由调用方统一约定，本类不做单位换算；
    ``max_travel_time``、释放时刻、截止时刻均以这个时间段为单位。
    ``time_periods=None`` 时保留原来的静态实例结构。
    """

    def __init__(self, node_count: int, arc_coverage: float,
                 fixed_cost_adjustment: float, *, seed: int | None = None,
                 time_periods: int | None = None, max_travel_time: int = 2) -> None:
        # 先校验参数再消耗随机数，使无效调用的失败行为保持确定性。
        self._validate(node_count, arc_coverage, fixed_cost_adjustment, seed)
        if time_periods is not None and (type(time_periods) is not int or time_periods < 1):
            raise ValueError("time_periods must be a positive integer or None")
        if type(max_travel_time) is not int or max_travel_time < 1:
            raise ValueError("max_travel_time must be a positive integer")
        self.seed = seed
        rng = random.Random(seed)
        # 使用局部随机数生成器，避免影响应用程序的全局随机状态。
        self.nodes = tuple(range(node_count))
        self.node_coordinates = {node: (round(rng.uniform(0.0, 100.0), 3),
                                         round(rng.uniform(0.0, 100.0), 3))
                                 for node in self.nodes}
        all_pairs = [(tail, head) for tail in self.nodes for head in self.nodes
                     if tail != head]
        # 即使覆盖率很低，也保留一个有向环作为连通骨架。
        ring = [(node, (node + 1) % node_count) for node in self.nodes]
        arc_count = min(len(all_pairs), max(node_count,
                                            round(arc_coverage * len(all_pairs))))
        extras = [pair for pair in all_pairs if pair not in ring]
        rng.shuffle(extras)
        selected_pairs = ring + extras[:arc_count - node_count]

        commodity_pairs = all_pairs.copy()
        rng.shuffle(commodity_pairs)
        commodity_count = rng.randint(max(1, node_count // 2),
                                      min(3 * node_count, len(commodity_pairs)))
        self.commodities = tuple(
            Commodity(i, origin, destination, rng.randint(5, 30))
            for i, (origin, destination) in enumerate(commodity_pairs[:commodity_count])
        )
        total_demand = sum(item.demand for item in self.commodities)
        # 调节参数 [0, 1] 映射到 [0.5, 2.0]，从而保证成本始终为正。
        multiplier = 0.5 + 1.5 * fixed_cost_adjustment
        arcs = []
        for arc_id, (tail, head) in enumerate(selected_pairs):
            x1, y1 = self.node_coordinates[tail]
            x2, y2 = self.node_coordinates[head]
            distance = math.hypot(x1 - x2, y1 - y2)
            variable_cost = round(1.0 + 0.05 * distance + rng.uniform(0.0, 5.0), 2)
            capacity = math.ceil(total_demand * rng.uniform(1.0, 1.8))
            fixed_cost = round((10.0 + 0.4 * capacity + 8.0 * variable_cost)
                               * multiplier, 2)
            arcs.append(Arc(arc_id, tail, head, capacity, variable_cost, fixed_cost))
        self.arcs = tuple(arcs)
        self.metadata = {
            "format": "static-snd-v2", "node_count": node_count,
            "requested_arc_coverage": arc_coverage,
            "actual_arc_coverage": round(len(self.arcs) / len(all_pairs), 6),
            "fixed_cost_adjustment": fixed_cost_adjustment,
            "fixed_cost_multiplier": round(multiplier, 6),
            "total_demand": total_demand,
        }
        self.time_expansion: dict[str, Any] | None = None
        if time_periods is not None:
            self._expand_time(time_periods, max_travel_time, rng)

    def _expand_time(self, periods: int, max_travel_time: int,
                     rng: random.Random) -> None:
        """把物理网络展开到 t=0..H；等待弧允许提前到达的商品留在目的地。

        相邻整数时刻相隔一个等长时间段；弧旅行时间是整数段数，
        生成器不保存或推断每段对应的现实分钟数。
        """
        physical_nodes = self.nodes
        physical_arcs = self.arcs
        physical_coordinates = self.node_coordinates
        original_commodities = self.commodities
        physical_count = len(physical_nodes)
        def node_id(node: int, t: int) -> int:
            """时间展开节点 ID；物理节点 ID 为 0..N-1。"""
            return t * physical_count + node
        durations = {arc.id: rng.randint(1, min(max_travel_time, periods))
                     for arc in physical_arcs}

        # 使用物理弧旅行时间计算最早到达时刻，保证每条生成需求都有可行路径。
        distances: dict[tuple[int, int], int] = {}
        outgoing = {node: [] for node in physical_nodes}
        for arc in physical_arcs:
            outgoing[arc.tail].append((arc.head, durations[arc.id]))
        for origin in physical_nodes:
            best = {origin: 0}
            queue = [(0, origin)]
            while queue:
                distance, node = heapq.heappop(queue)
                if distance != best[node]:
                    continue
                for head, travel in outgoing[node]:
                    new_distance = distance + travel
                    if new_distance < best.get(head, math.inf):
                        best[head] = new_distance
                        heapq.heappush(queue, (new_distance, head))
            distances.update({(origin, dest): duration for dest, duration in best.items()
                              if dest != origin and duration <= periods})
        eligible = sorted(distances)
        rng.shuffle(eligible)
        if not eligible:
            raise RuntimeError("time horizon contains no reachable OD pair")

        # 每个商品只注入一次需求：源=(物理起点, release)，汇=(物理终点, deadline)。
        # 每条展开弧的容量至少为总需求，因此逐商品可达即可保证存在联合可行流。
        commodity_windows = []
        expanded_commodities = []
        for index, item in enumerate(original_commodities):
            origin, dest = eligible[index % len(eligible)]
            shortest = distances[(origin, dest)]
            release = rng.randint(0, periods - shortest)
            deadline = rng.randint(release + shortest, periods)
            expanded_commodities.append(Commodity(item.id, node_id(origin, release),
                                                  node_id(dest, deadline), item.demand))
            commodity_windows.append({"commodity_id": item.id, "origin": origin,
                                      "destination": dest, "release_time": release,
                                      "deadline": deadline, "demand": item.demand})

        total_demand = sum(item.demand for item in original_commodities)
        expanded_arcs = []
        arc_schedule = []
        for t in range(periods):
            for arc in physical_arcs:
                travel = durations[arc.id]
                if t + travel > periods:
                    continue
                # 固定费针对每次服务出发收取；容量、费用随出发时刻变化。
                capacity = max(total_demand, math.ceil(arc.capacity * rng.uniform(.85, 1.15)))
                variable = round(arc.variable_cost * rng.uniform(.8, 1.2), 2)
                fixed = round(arc.fixed_cost * rng.uniform(.85, 1.15), 2)
                arc_id = len(expanded_arcs)
                expanded_arcs.append(Arc(arc_id, node_id(arc.tail, t),
                                         node_id(arc.head, t + travel), capacity,
                                         variable, fixed))
                arc_schedule.append({"arc_id": arc_id, "kind": "service",
                                     "physical_arc_id": arc.id,
                                     "departure_time": t, "arrival_time": t + travel})
            for node in physical_nodes:
                arc_id = len(expanded_arcs)
                expanded_arcs.append(Arc(arc_id, node_id(node, t), node_id(node, t + 1),
                                         total_demand, 0.0, 0.0))
                arc_schedule.append({"arc_id": arc_id, "kind": "wait", "physical_node": node,
                                     "departure_time": t, "arrival_time": t + 1})

        self.nodes = tuple(node_id(node, t) for t in range(periods + 1)
                           for node in physical_nodes)
        self.node_coordinates = {
            node_id(node, t): physical_coordinates[node]
            for t in range(periods + 1) for node in physical_nodes
        }
        self.arcs = tuple(expanded_arcs)
        self.commodities = tuple(expanded_commodities)
        self.time_expansion = {
            "time_points": list(range(periods + 1)),
            "physical_nodes": list(physical_nodes),
            "physical_arcs": [arc.to_dict() for arc in physical_arcs],
            "node_map": [{"node_id": node_id(node, t), "physical_node": node,
                          "time": t} for t in range(periods + 1) for node in physical_nodes],
            "arc_schedule": arc_schedule,
            "commodity_windows": commodity_windows,
        }
        self.metadata.update(format="time-expanded-snd-v1", time_periods=periods,
                             physical_node_count=physical_count,
                             max_travel_time=max_travel_time,
                             expanded_node_count=len(self.nodes),
                             expanded_arc_count=len(self.arcs))

    @staticmethod
    def _validate(node_count: int, arc_coverage: float,
                  fixed_cost_adjustment: float, seed: int | None) -> None:
        if not isinstance(node_count, int) or isinstance(node_count, bool):
            raise TypeError("node_count must be an integer")
        if node_count < 2:
            raise ValueError("node_count must be at least 2")
        for name, value in (("arc_coverage", arc_coverage),
                            ("fixed_cost_adjustment", fixed_cost_adjustment)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a number")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
            raise TypeError("seed must be an integer or None")

    def to_dict(self) -> dict[str, Any]:
        result = {"format": self.metadata["format"], "seed": self.seed,
                "nodes": list(self.nodes),
                "node_coordinates": {str(n): {"x": xy[0], "y": xy[1]}
                                     for n, xy in self.node_coordinates.items()},
                "arcs": [arc.to_dict() for arc in self.arcs],
                "commodities": [item.to_dict() for item in self.commodities],
                "metadata": dict(self.metadata)}
        if self.time_expansion is not None:
            result["time_expansion"] = self.time_expansion
        return result

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
