"""实验诊断：不参与训练收益计算，避免把有限测试结果冒充收敛证明。"""
from __future__ import annotations

import math
from statistics import fmean, stdev
from typing import Any


def summarize_evaluations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """报告成功率及仅对成功样本有定义的 Gap；不隐藏失败样本。"""
    successful = [float(row["gap"]) for row in rows
                  if row.get("status") == "ok" and row.get("gap") is not None]
    count = len(rows)
    mean = fmean(successful) if successful else None
    # 95% Wilson 区间不依赖成功率正态近似；小样本仍需谨慎解释。
    z = 1.959963984540054
    rate = len(successful) / count if count else None
    if count:
        denominator = 1 + z*z/count
        center = (rate + z*z/(2*count)) / denominator
        half = z*math.sqrt(rate*(1-rate)/count + z*z/(4*count*count)) / denominator
        success_ci = [max(0.0, center-half), min(1.0, center+half)]
    else:
        success_ci = None
    # Gap 区间仅适用于成功样本，且正态近似不是跨分布泛化保证。
    gap_ci = ([mean - z*stdev(successful)/math.sqrt(len(successful)),
               mean + z*stdev(successful)/math.sqrt(len(successful))]
              if len(successful) >= 2 else None)
    return {"count": count, "successful": len(successful), "success_rate": rate,
            "success_rate_wilson_95": success_ci,
            "mean_gap": mean if len(successful) == count else None,
            "mean_gap_successful": mean, "mean_gap_successful_approx_95": gap_ci}


def instance_scale(data: dict[str, Any]) -> str:
    """按物理节点分层；时间展开图不把复制的时间节点误当物理规模。"""
    nodes = int(data.get("metadata", {}).get("physical_node_count", len(data["nodes"])))
    if nodes < 10:
        return "small_lt10"
    if nodes < 30:
        return "medium_10_29"
    return "large_ge30"


def summarize_by_scale(instances: list[dict[str, Any]],
                       rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if len(instances) != len(rows):
        raise ValueError("实例与评估结果数量不一致")
    buckets: dict[str, list[dict[str, Any]]] = {}
    for instance, row in zip(instances, rows):
        buckets.setdefault(instance_scale(instance), []).append(row)
    return {key: summarize_evaluations(value) for key, value in buckets.items()}
