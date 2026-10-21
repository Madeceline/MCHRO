"""静态与时间展开 SND 弧流目标、容量和守恒约束的统一数值检查。"""
from __future__ import annotations

from typing import Any


def _value(mapping: dict[Any, Any], raw_id: Any, default: Any = None) -> Any:
    return mapping.get(str(raw_id), mapping.get(raw_id, default))


def check_arc_flow_solution(data: dict[str, Any],
                       solution: Any) -> tuple[float, dict[str, float | bool]]:
    """重算守恒、共享容量和变量域；时间展开实例直接按展开节点检查。

    因此时间弧的先后关系由实例拓扑保证，检查逻辑无需额外引入时间索引。
    调用前须检查有限数。
    """
    if not isinstance(solution, dict):
        raise TypeError("solve_snd 必须返回字典")
    arc_open, flows = solution.get("arc_open"), solution.get("flows")
    if not isinstance(arc_open, dict) or not isinstance(flows, dict):
        raise ValueError("结果必须包含字典 arc_open 和 flows")
    max_balance = max_capacity = max_binary = max_nonnegative = 0.0
    objective = 0.0
    for arc in data["arcs"]:
        y = float(_value(arc_open, arc["id"], 0.0))
        max_binary = max(max_binary, min(abs(y), abs(y - 1.0)))
        objective += float(arc["fixed_cost"]) * y
        load = 0.0
        for commodity in data["commodities"]:
            commodity_flow = _value(flows, commodity["id"], {})
            if not isinstance(commodity_flow, dict):
                raise ValueError("每个商品的 flows 值必须是字典")
            value = float(_value(commodity_flow, arc["id"], 0.0))
            max_nonnegative = max(max_nonnegative, max(0.0, -value))
            load += value
            objective += float(arc["variable_cost"]) * value
        max_capacity = max(max_capacity,
                           load - float(arc["capacity"]) * y)
    for commodity in data["commodities"]:
        commodity_flow = _value(flows, commodity["id"], {})
        for node in data["nodes"]:
            outgoing = sum(float(_value(commodity_flow, arc["id"], 0.0))
                           for arc in data["arcs"] if arc["tail"] == node)
            incoming = sum(float(_value(commodity_flow, arc["id"], 0.0))
                           for arc in data["arcs"] if arc["head"] == node)
            expected = (float(commodity["demand"])
                        if node == commodity["origin"] else
                        -float(commodity["demand"])
                        if node == commodity["destination"] else 0.0)
            max_balance = max(max_balance, abs(outgoing - incoming - expected))
    tolerance = 1e-6
    checks = {
        "feasible": (max_balance <= tolerance and max_capacity <= tolerance
                     and max_binary <= tolerance and max_nonnegative <= tolerance),
        "max_flow_balance_violation": max_balance,
        "max_capacity_violation": max(0.0, max_capacity),
        "max_binary_violation": max_binary,
        "max_nonnegativity_violation": max_nonnegative,
    }
    return objective, checks
