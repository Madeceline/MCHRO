"""一般和星形多方博弈：显式联合策略 + 线性规划粗相关均衡。"""

from __future__ import annotations

import itertools
import math
from typing import Any, Protocol

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix


class MetaSolver(Protocol):
    """可替换的元求解器接口，所有玩家都最大化各自效用。"""

    def solve(self, solver_tables: list[np.ndarray],
              generator_tables: list[np.ndarray]) -> dict[str, Any]: ...


class CCEMetaSolver:
    """有限一般和博弈的 epsilon-CCE；不把边际分布冒充独立纳什均衡。"""

    def __init__(self, max_profiles: int = 50000) -> None:
        if max_profiles < 1:
            raise ValueError("max_profiles 必须为正整数")
        self.max_profiles = max_profiles

    def solve(self, solver_tables: list[np.ndarray],
              generator_tables: list[np.ndarray]) -> dict[str, Any]:
        if not solver_tables or len(solver_tables) != len(generator_tables):
            raise ValueError("必须为每个生成器提供两方各自的收益表")
        a = [np.asarray(x, dtype=float) for x in solver_tables]
        b = [np.asarray(x, dtype=float) for x in generator_tables]
        if any(x.ndim != 2 or min(x.shape) < 1 for x in a + b):
            raise ValueError("收益表必须为非空二维数组")
        rows = a[0].shape[0]
        if any(x.ndim != 2 or x.shape[0] != rows or x.shape != y.shape
               or not np.isfinite(x).all() or not np.isfinite(y).all()
               for x, y in zip(a, b)):
            raise ValueError("收益表尺寸不一致或含非有限值")
        sizes = [rows] + [x.shape[1] for x in a]
        profile_count = math.prod(sizes)
        if profile_count > self.max_profiles:
            raise ValueError(f"联合策略数 {profile_count} 超过 max_profiles={self.max_profiles}；"
                             "请减少策略池/迭代次数，或替换为采样式元求解器")
        profiles = np.array(list(itertools.product(*(range(n) for n in sizes))))
        # u0 = sum_j A_j(s,g_j)，uj = B_j(s,g_j)；共享同一个 solver 动作。
        utilities = np.zeros((len(sizes), profile_count))
        for j, (x, y) in enumerate(zip(a, b), 1):
            utilities[0] += x[profiles[:, 0], profiles[:, j]]
            utilities[j] = y[profiles[:, 0], profiles[:, j]]
        deviations = []
        owners = []
        for player, size in enumerate(sizes):
            for action in range(size):
                if player == 0:
                    changed = sum(x[action, profiles[:, j]] for j, x in enumerate(a, 1))
                else:
                    changed = b[player - 1][profiles[:, 0], action]
                deviations.append(changed - utilities[player])
                owners.append(player)
        gain = np.asarray(deviations)
        # min epsilon, E_pi[u_i(a_i',a_-i)-u_i(a)] <= epsilon。
        lp = linprog(
            np.r_[np.zeros(profile_count), 1.0],
            A_ub=csr_matrix(np.c_[gain, -np.ones(len(gain))]),
            b_ub=np.zeros(len(gain)),
            A_eq=csr_matrix([np.r_[np.ones(profile_count), 0.0]]), b_eq=[1.0],
            bounds=[(0.0, None)] * (profile_count + 1), method="highs",
            options={"presolve": True, "primal_feasibility_tolerance": 1e-9,
                     "dual_feasibility_tolerance": 1e-9},
        )
        if not lp.success:
            raise RuntimeError(f"CCE 线性规划失败：{lp.message}")
        joint = np.maximum(lp.x[:-1], 0.0)
        joint /= joint.sum()
        marginals = [np.bincount(profiles[:, i], weights=joint, minlength=n)
                     for i, n in enumerate(sizes)]
        gain_values = gain @ joint
        regrets = [max(0.0, max(gain_values[k] for k, p in enumerate(owners) if p == i))
                   for i in range(len(sizes))]
        # 额外报告边际乘积分布的 NashConv；CCE 的相关性不能在部署时无条件丢掉。
        product = np.prod([marginals[i][profiles[:, i]] for i in range(len(sizes))], axis=0)
        product_gains = gain @ product
        nash_conv = sum(max(0.0, max(product_gains[k] for k, p in enumerate(owners) if p == i))
                        for i in range(len(sizes)))
        return {
            "method": "epsilon_CCE_LP", "profile_count": profile_count,
            "profiles": profiles.tolist(), "joint": joint.tolist(),
            "marginals": [x.tolist() for x in marginals],
            "expected_utilities": (utilities @ joint).tolist(),
            "deviation_gains": regrets, "cce_gap": max(regrets),
            "total_deviation_gain": sum(regrets),
            "product_nash_conv": float(nash_conv),
        }
