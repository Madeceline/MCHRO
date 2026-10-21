"""离线下界微基准；不调用 LLM、候选程序或参考求解器。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
from statistics import median
import sys
from tempfile import TemporaryDirectory
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multi_agent_snd.evaluation import SNDEvaluator, lower_bound, validate_instance
from multi_agent_snd.storage import ProgramArchive
from multi_agent_snd.utilities import UtilityConfig
from snd_generator import SNDInstance


def legacy_lower_bound(data: dict, field: str | None) -> float:
    """保留优化前逐商品、V-1 轮全弧扫描，仅用来核对和计时。"""
    waiting = {str(a["arc_id"]) for a in data.get("time_expansion", {}).get("arc_schedule", [])
               if a["kind"] == "wait"}
    total = 0.0
    for commodity in data["commodities"]:
        distances = dict.fromkeys(data["nodes"], math.inf)
        distances[commodity["origin"]] = 0.0
        for _ in range(len(distances) - 1):
            for arc in data["arcs"]:
                weight = float(arc[field]) if field is not None else (
                    0.0 if str(arc["id"]) in waiting else 1.0)
                distances[arc["head"]] = min(distances[arc["head"]],
                                              distances[arc["tail"]] + weight)
        total += distances[commodity["destination"]] * float(commodity["demand"])
    return total


def bound_pair(function, data: dict) -> dict:
    return {"path": function(data, None), "transport": function(data, "variable_cost")}


def timed(function, repeats: int):
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        value = function()
        durations.append(time.perf_counter() - started)
    return value, median(durations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, help="已有实例 JSON；省略则固定种子生成时间展开实例")
    parser.add_argument("--repeats", type=int, default=3, help="冷计算重复次数，报告中位数")
    parser.add_argument("--cache-repeats", type=int, default=50, help="热缓存查询次数，报告中位数")
    args = parser.parse_args()
    if args.repeats < 1 or args.cache_repeats < 1:
        parser.error("重复次数必须为正整数")
    data = (json.loads(args.instance.read_text(encoding="utf-8")) if args.instance is not None
            else SNDInstance(10, .5, .7, seed=42, time_periods=3).to_dict())
    validate_instance(data)
    old, old_seconds = timed(lambda: bound_pair(legacy_lower_bound, data), args.repeats)
    new, new_seconds = timed(lambda: bound_pair(lower_bound, data), args.repeats)
    if not all(math.isclose(old[key], new[key], rel_tol=1e-10, abs_tol=1e-8) for key in old):
        raise AssertionError(f"下界数值不一致：old={old}, new={new}")
    with TemporaryDirectory(prefix="snd-bound-benchmark-") as tmp:
        evaluator = SNDEvaluator(data, ProgramArchive(Path(tmp)), UtilityConfig(),
                                 "scipy", 10, 0, 10, 1, 42, parallel_workers=1)
        cold, cold_seconds = timed(lambda: evaluator.instance_lower_bounds(data), 1)
        warm, warm_seconds = timed(lambda: evaluator.instance_lower_bounds(data), args.cache_repeats)
        assert cold == warm == new
        report = {
            "scope": "lower_bounds_only_not_full_training_speedup",
            "python": platform.python_version(),
            "instance": str(args.instance) if args.instance is not None else "synthetic_seed_42",
            "nodes": len(data["nodes"]), "arcs": len(data["arcs"]),
            "commodities": len(data["commodities"]),
            "distinct_origins": len({k["origin"] for k in data["commodities"]}),
            "repeats": args.repeats, "cache_repeats": args.cache_repeats,
            "bounds_match": True, "bounds": new,
            "legacy_pair_median_seconds": old_seconds,
            "optimized_pair_median_seconds": new_seconds,
            "cached_pair_first_seconds": cold_seconds,
            "cached_pair_hit_median_seconds": warm_seconds,
            "compute_speedup": old_seconds / max(new_seconds, 1e-12),
            "warm_cache_speedup": old_seconds / max(warm_seconds, 1e-12),
            "cache_counters": evaluator.timing.summary()["counters"],
        }
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
