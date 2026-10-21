"""六种生成器目标和求解器的综合损失；效用系数均可外部配置。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


DIRECTIONS = {
    "cost": "总成本：增大固定成本与运输成本之和相对参考值的差距",
    "path": "路径长度：增大需求加权服务弧跳数相对无容量最短路下界的差距；时间模式不计等待弧",
    "transport": "运输成本：增大运输费用相对无容量最小运输费用下界的差距",
    "fixed": "固定成本：提高已开启弧固定费用占全网固定费用的比例",
    "congestion": "容量拥堵：提高最大服务弧容量利用率；时间模式不计等待弧",
    "work": "执行工作量：提高候选程序 Python 行执行次数（耗时的确定性代理）",
}


@dataclass
class UtilityConfig:
    """所有玩家最大化效用；求解器最小化多指标综合损失。"""

    solver_weights: dict[str, float] = field(default_factory=lambda: {
        "cost": 0.55, "path": 0.10, "transport": 0.10,
        "fixed": 0.10, "congestion": 0.10, "work": 0.05,
    })
    distortion_penalty: float = 0.10
    failure_penalty: float = 2.0
    work_budget: int = 200000
    # 候选程序的硬执行行数上限。None 保持旧版 work_budget*10 的兼容行为；
    # 新实验应显式设置该值，使“工作量效用预算”和“安全执行上限”彼此独立。
    execution_line_limit: int | None = None
    # 可选的实例规模缩放系数。启用后，实际上限为
    # min(execution_line_limit, execution_line_limit_scale * complexity(instance))。
    # None 表示固定上限，保持旧行为。
    execution_line_limit_scale: int | None = None
    # LLM 候选源码的静态质量门槛。它们与运行时行数上限不同：前者
    # 拦截异常冗长/复杂的程序，后者用于阻止死循环或低效循环。
    max_code_lines: int | None = 240
    max_code_chars: int | None = 24000
    max_ast_nodes: int | None = 2000

    def __post_init__(self) -> None:
        if set(self.solver_weights) != set(DIRECTIONS):
            raise ValueError("solver_weights 必须包含六个方向")
        if any(not math.isfinite(v) or v < 0 for v in self.solver_weights.values()):
            raise ValueError("效用权重必须非负有限")
        total = sum(self.solver_weights.values())
        if total <= 0 or self.work_budget < 1:
            raise ValueError("权重之和和 work_budget 必须为正")
        if self.execution_line_limit is not None and (
                type(self.execution_line_limit) is not int or self.execution_line_limit < 1):
            raise ValueError("execution_line_limit 必须为正整数或 None")
        if self.execution_line_limit_scale is not None and (
                type(self.execution_line_limit_scale) is not int or self.execution_line_limit_scale < 1):
            raise ValueError("execution_line_limit_scale 必须为正整数或 None")
        for name, value in (("max_code_lines", self.max_code_lines),
                            ("max_code_chars", self.max_code_chars),
                            ("max_ast_nodes", self.max_ast_nodes)):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} 必须为正整数或 None")
        if not math.isfinite(self.distortion_penalty) or self.distortion_penalty < 0:
            raise ValueError("distortion_penalty 必须为非负有限数")
        if not math.isfinite(self.failure_penalty) or self.failure_penalty <= 1:
            raise ValueError("failure_penalty 必须为大于 1 的有限数")
        self.solver_weights = {k: v / total for k, v in self.solver_weights.items()}

    def pair(self, observation: dict, direction: str) -> tuple[float, float]:
        """返回 (求解器效用, 该方向生成器效用)，一般不互为相反数。"""
        if observation["status"] == "invalid_instance":
            return 0.0, -self.failure_penalty
        if observation["status"] != "ok":
            # 生成器的运行有效性由独立校验与参考可行解认证。
            return -self.failure_penalty, 1.0 - self.distortion_penalty * observation["distortion"]
        metrics = observation["metrics"]
        solver = -sum(w * metrics[k] for k, w in self.solver_weights.items())
        generator = metrics[direction] - self.distortion_penalty * observation["distortion"]
        return solver, generator

    def description(self, player: str) -> str:
        if player == "solver":
            return (f"最大化 U0=-平均_j E[综合损失]；六项权重={self.solver_weights}。"
                    "综合损失包含成本差距、绕行、运输费用、固定开销、拥堵和执行工作量。"
                    f"候选源码不得超过 {self.max_code_lines or '不限'} 行、"
                    f"{self.max_code_chars or '不限'} 字符和 {self.max_ast_nodes or '不限'} 个 AST 节点。")
        return (f"你的唯一方向为 {player}：{DIRECTIONS[player]}。最大化 "
                f"U=E[m_{player}]-{self.distortion_penalty}*D；D 是成本/容量/需求相对基础实例的平均绝对比例扰动。"
                "只能在基础值 [0.5,2] 倍范围内改变数值，节点、弧端点与商品 OD 不得改变。"
                "总成本和路径/运输差距用 max(0,z)/(1+max(0,z)) 归一化；"
                "固定费用占比及容量利用率位于 [0,1]，工作量除以预算后截断为 1。"
                "生成非法或不可行实例会得到负效用，求解器失败则奖励难例。")
