"""只演示生成器：输出可复现的时间相关多商品 SND 实例。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from snd_generator import SNDInstance, SNDParameterBounds
from snd_solver import SNDSolver


def main() -> None:
    # 不传 time_periods 时，保留既有 static-snd-v2 格式。
    static = SNDInstance(5, 0.5, 0.7, seed=42)

    # 假定每段 = 1 小时：H=4 表示未来 4 小时，有 0、1、2、3、4 五个边界时刻。
    # max_travel_time=2 表示单条物理弧随机旅行时间最多 2 段，即最多 2 小时。
    # 若改为每段 = 30 分钟，同样参数只代表未来 2 小时、最多运输 1 小时；
    # 现实时间粒度由用户约定，生成器本身不做单位换算。
    temporal = SNDInstance(5, 0.5, 0.7, seed=42,
                           time_periods=4, max_travel_time=2)

    # 也可以在闭区间内采样时间跨度、节点数及弧覆盖率。
    bounds = SNDParameterBounds(nodes=(4, 7), coverage=(0.3, 0.7),
                                fixed_cost=(0.4, 0.9), time_periods=(3, 5),
                                max_travel_time=2)
    sampled = bounds.sample(seed=123)

    print("static:", static.metadata["format"], len(static.nodes), len(static.arcs))
    print("temporal:", temporal.metadata["format"], len(temporal.nodes),
          len(temporal.arcs), len(temporal.commodities))
    print("first commodity window:", temporal.time_expansion["commodity_windows"][0])
    print("sampled horizon:", sampled.metadata["time_periods"])

    # 现有弧流参考求解器直接使用时间展开后的 nodes/arcs/commodities。
    solved = SNDSolver(temporal, time_limit=30, mip_gap=0).solve()
    print("reference status/objective:", solved.status, solved.objective_value)

    # 只写实例数据，不执行求解器；同一 seed 可逐字节重现 JSON。
    output = PROJECT_ROOT / "temporal_instance_example.json"
    output.write_text(temporal.to_json(), encoding="utf-8")
    print("saved:", output)


if __name__ == "__main__":
    main()
