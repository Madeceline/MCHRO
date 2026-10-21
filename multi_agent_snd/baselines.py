"""可运行的暖启动及无 API 离线客户端；不冒充 LLM 生成结果。"""
from __future__ import annotations

import hashlib


BASE_SOLVER = '''def solve_snd(instance):
    # 残余容量最短路构造；可拆分商品，共享已开启弧的固定费用。
    # 对时间展开实例，nodes/arcs 已编码出发和到达时刻，普通最短路会自然遵守时间顺序。
    nodes, arcs = instance["nodes"], instance["arcs"]
    residual = {str(a["id"]): float(a["capacity"]) for a in arcs}
    opened = {str(a["id"]): 0.0 for a in arcs}
    flows = {}
    for commodity in sorted(instance["commodities"], key=lambda k: (-k["demand"], str(k["id"]))):
        demand = float(commodity["demand"])
        remaining = demand
        values = {str(a["id"]): 0.0 for a in arcs}
        while remaining > 0.00000001:
            distance = {node: float("inf") for node in nodes}
            previous = {}
            unused = list(nodes)
            distance[commodity["origin"]] = 0.0
            while unused:
                node = min(unused, key=lambda v: (distance[v], v))
                unused.remove(node)
                if node == commodity["destination"] or distance[node] == float("inf"):
                    break
                for arc in arcs:
                    key = str(arc["id"])
                    if arc["tail"] == node and arc["head"] in unused and residual[key] > 0.00000001:
                        weight = arc["variable_cost"] + 1.0 * arc["fixed_cost"] * (1.0-opened[key]) / demand
                        candidate = distance[node] + weight
                        if candidate < distance[arc["head"]]:
                            distance[arc["head"]] = candidate
                            previous[arc["head"]] = arc
            if commodity["destination"] not in previous:
                return {"arc_open": opened, "flows": flows}
            path = []
            node = commodity["destination"]
            while node != commodity["origin"]:
                arc = previous[node]
                path.append(str(arc["id"]))
                node = arc["tail"]
            amount = min(remaining, min(residual[key] for key in path))
            for key in path:
                values[key] += amount
                residual[key] -= amount
                opened[key] = 1.0
            remaining -= amount
        flows[str(commodity["id"])] = values
    return {"arc_open": opened, "flows": flows}
'''

BASE_GENERATOR = '''def generate_snd(base_instance, seed):
    return dict(base_instance)
'''


class OfflineMultiAgentClient:
    """根据 prompt 哈希产生确定性代码变体，仅用于集成测试和演示。"""

    def generate(self, prompt: str, temperature: float = 1.0, n: int = 1) -> list[str]:
        result = []
        for index in range(n):
            value = int(hashlib.sha256((prompt + str(index)).encode()).hexdigest()[:8], 16)
            if "PLAYER=solver\n" in prompt:
                factor = (value % 11 + 1) / 5
                code = BASE_SOLVER.replace("1.0 * arc", f"{factor} * arc")
                thought = f"残余容量最短路逐商品构造，固定费用权重 {factor}，优先共享已开启弧。可能因贪心路由失败。"
            else:
                factor = 0.75 + (value % 51) / 100
                # 只改固定成本，保证基础实例的可行性不变。
                code = ('def generate_snd(base_instance, seed):\n'
                        '    result = dict(base_instance)\n'
                        '    result["arcs"] = [dict(a) for a in base_instance["arcs"]]\n'
                        '    for arc in result["arcs"]:\n'
                        f'        if arc["fixed_cost"] > 0 and (int(arc["id"])+seed)%2:\n'
                        f'            arc["fixed_cost"] *= {factor}\n'
                        '    return result\n')
                thought = f"按弧 ID 和 seed 确定性扰动固定费用，倍率 {factor}，不改拓扑与可行域。"
            result.append(f"EXPLANATION: {thought}\n```python\n{code}\n```")
        return result
