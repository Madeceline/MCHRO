"""编辑配置后直接运行：用 10–200 节点公开实例对照算法，仅生成四张论文图的 PNG/PDF。"""
from __future__ import annotations

import importlib.util
import hashlib
import inspect
import html
import json
import heapq
import math
from pathlib import Path
import random
import re
import sys
import time
import uuid
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
RUN_DIR = PROJECT_ROOT / "runs" / "multi_20261020_210456_9714585b"
SOLVER_FILE = RUN_DIR / "final_solver" / "solver.py"
OUTPUT_FILE = PROJECT_ROOT / "final_solution.json"
COMPARISON_FILE = PROJECT_ROOT / "final_comparison.json"
# True 时只读取已有 final_comparison.json 绘图，不重新运行测试实例。
# False 时先运行完整对照，结束后自动绘图。论文出图改样式时建议设为 True。
PLOT_ONLY = False
GENERATE_FIGURES = True
FIGURE_OUTPUT_DIR = PROJECT_ROOT / "benchmark_figures"
FIGURE_FORMATS = ("png", "pdf")
FIGURE_DPI = 600
# 每次绘图独立存档，目录、文件名和图内脚注均标注时间，不复用历史图片。
# 绘图函数源码和样式设置进入指纹，便于追溯相同数据。
FIGURE_STYLE_VERSION = 4
MAX_BENCHMARK_NODES = 200
FIGURE_GROUP_LABELS = {
    "fixed_charge_snd": "Fixed-charge SND",
    "linear_flow_special_case": "Linear multicommodity flow (zero fixed costs)",
}
# 原作者发布的静态基准包；运行测试不联网、不调用 LLM，不随机修改原始数据。
# 每项为 (解析器类型, 本地原始压缩包名, 包内实例名)。删去某行即可减少测试数量。
# Canad: F/V 表示固定/变动成本占优，L/T 表示容量宽松/紧张。
# 下方规模来自实际文件头/作者索引，不使用官网表格中的近似弧数。
BENCHMARK_CASES = (
    ("canad", "R.tgz", "r01.1.dow"),               # 10 节点，35 弧，10 商品。
    ("canad", "R.tgz", "r03.1.dow"),               # 10 节点，35 弧，50 商品。
    ("canad", "R.tgz", "r10.1.dow"),               # 20 节点，120 弧，40 商品。
    ("canad", "R.tgz", "r15.1.dow"),               # 20 节点，220 弧，200 商品。
    ("canad", "CPlus.tgz", "c25_100_10_F_L_5.dow"), # 25 节点，100 弧，10 商品。
    ("canad", "CPlus.tgz", "c25_100_30_F_T_5.dow"), # 25 节点，100 弧，30 商品。
    ("canad", "C.tgz", "c49.dow"),                 # 30 节点，518 弧，100 商品，V/L。
    ("canad", "C.tgz", "c52.dow"),                 # 30 节点，517 弧，100 商品，F/T。
    ("holmberg", "cexall.zip", "cexe6"),            # 40 节点，500 弧，78 商品。
    ("holmberg", "cexall.zip", "cexe7"),            # 40 节点，1000 弧，78 商品，稠密网络。
    ("holmberg", "cexall.zip", "cexc4"),            # 50 节点，290 弧，40 商品。
    ("holmberg", "cexall.zip", "cexe11"),           # 60 节点，421 弧，118 商品。
    ("holmberg", "cexall.zip", "cexe12"),           # 70 节点，492 弧，138 商品。
    ("holmberg", "cexall.zip", "cexe13"),           # 75 节点，546 弧，148 商品。
    ("canad", "CPlus.tgz", "c100_400_10_F_L_10.dow"), # 100 节点，400 弧，10 商品。
    ("canad", "CPlus.tgz", "c100_400_10_F_T_10.dow"), # 100 节点，400 弧，10 商品，紧容量。
    ("canad", "CPlus.tgz", "c100_400_10_V_L_10.dow"), # 100 节点，400 弧，10 商品，变动费用占优。
    ("canad", "CPlus.tgz", "c100_400_30_F_T_10.dow"), # 100 节点，400 弧，30 商品，固定费用占优。
    ("canad", "CPlus.tgz", "c100_400_30_V_T_10.dow"), # 100 节点，400 弧，30 商品。
    ("holmberg", "cexall.zip", "cexc2"),            # 100 节点，394 弧，50 商品。
    ("holmberg", "cexall.zip", "cexc3"),            # 100 节点，980 弧，50 商品；较稠密。
    ("holmberg", "cexall.zip", "cexb3"),            # 103 节点，814 弧，12 商品。
    ("holmberg", "cexall.zip", "cexf6"),            # 150 节点，410 弧，10 商品。
    ("holmberg", "cexall.zip", "cexf7"),            # 150 节点，415 弧，15 商品。
    # 以下为公开线性多商品流特例：原数据没有固定费用，转换时明确设 fixed_cost=0。
    # 保留全部 1528 个商品，不截取 OD，不添加固定费用；报告单列，不能冒充完整 SND 基准。
    # 已核实该原文件违反单节点容量必要条件；保留用于输入审计，默认记录证据并跳过求解。
    # 因此本清单的完整固定费用 SND 可行解对照目前最高为 150 节点，不宣称验证了 200 节点性能。
    ("fnss", "waxman.tgz", "w1_200_04"),           # 200 节点，1550 弧，1528 商品。
    # 新增原作者发布的 Grid / Planar 网络；只做无损格式转换，不是随机合成实例。
    # 同属零固定费用特例，与上面的固定费用 SND 分开统计和展示。
    ("mnetgen", "gridmnet.tgz", "grid1"),          # 25 节点，80 弧，50 商品。
    ("mnetgen", "gridmnet.tgz", "grid4"),          # 100 节点，360 弧，100 商品。
    ("mnetgen", "planarmnet.tgz", "planar30"),     # 30 节点，150 弧，92 商品。
    ("mnetgen", "planarmnet.tgz", "planar50"),     # 50 节点，250 弧，267 商品。
    ("mnetgen", "planarmnet.tgz", "planar80"),     # 80 节点，440 弧，543 商品。
    ("mnetgen", "planarmnet.tgz", "planar100"),    # 100 节点，532 弧，1085 商品。
    ("mnetgen", "planarmnet.tgz", "planar150"),    # 150 节点，850 弧，2239 商品。
)
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
# 原始 .dow 格式转换为项目标准 JSON，便于直接传给其他求解入口。
# 只做无损格式适配；不改需求、费用、容量，不额外添加时间约束。
CONVERTED_INSTANCE_DIR = PROJECT_ROOT / "benchmarks" / "converted"
SOLVER_TIMEOUT = 1200.0       # 每成员秒数上限；防止单实例无限等待。
SOLVER_LINE_LIMIT = 30_000_0000000
# 上面两个值保留你的现有设置；200 节点且 1528 商品可能明显更慢。
# 要限制整批时长，可将 SOLVER_TIMEOUT 改为 120；超时会记录失败并继续，不提升或伪造目标值。
BASELINE_TIMEOUT = 120.0      # 每种基线、每个实例的总预算，含全部重启；循环内检查。
GRASP_RESTARTS = 10           # 随机多启动构造次数；并非完整 GRASP 实现。
RANDOM_SEED = 1000            # 各实例/方法使用相同固定重启种子，避免依赖全局随机状态。

METHOD_LABELS = {
    "exported_llm_heuristic": "LLM heuristic",
    "variable_cost": "Variable-cost greedy",
    "fixed_cost_greedy": "Fixed-cost greedy",
    "capacity_aware": "Capacity-aware greedy",
    "grasp": "Randomized multistart",
}
METHOD_COLORS = {
    "exported_llm_heuristic": "#0072B2",
    "variable_cost": "#D55E00",
    "fixed_cost_greedy": "#009E73",
    "capacity_aware": "#CC79A7",
    "grasp": "#E69F00",
}
METHOD_ORDER = tuple(METHOD_LABELS)


def write_json(path: Path, data) -> None:
    """仅供本测试脚本：原子写入，短暂文件占用时重试；最终失败保留临时文件。"""
    from multi_agent_snd.storage import clean_json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(clean_json(data), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            if attempt == 5:
                raise PermissionError(f"无法更新 {path}；完整数据已保存在 {temporary}") from exc
            time.sleep(0.05 * 2 ** attempt)


def _baseline_solution(data: dict, mode: str, seed: int = 0,
                       *, deadline: float | None = None) -> dict:
    """用逐商品增广路构造一个可行解。

    这是经典的容量约束逐次最短路基线：每次为剩余需求寻找一条残余网络路径，
    直到该商品需求全部发送。mode 分别对应变动成本、固定费用感知和容量感知权重。
    它们用于对照，不会调用 LLM 或 MIP。
    """
    arcs = data["arcs"]
    nodes = data["nodes"]
    residual = {str(a["id"]): float(a["capacity"]) for a in arcs}
    opened: dict[str, float] = {}
    outgoing = {node: [] for node in nodes}
    for arc in arcs:
        outgoing[arc["tail"]].append(arc)
    rng = random.Random(seed)
    commodities = sorted(data["commodities"], key=lambda x: (-float(x["demand"]), str(x["id"])))
    flows: dict[str, dict[str, float]] = {}

    for commodity in commodities:
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("基线超过每实例总时间预算")
        source, target = commodity["origin"], commodity["destination"]
        remaining = float(commodity["demand"])
        commodity_flow: dict[str, float] = {}
        while remaining > 1e-9:
            distance = {node: math.inf for node in nodes}
            previous: dict[int, dict] = {}
            distance[source] = 0.0
            queue = [(0.0, source)]
            while queue:
                if deadline is not None and time.perf_counter() >= deadline:
                    raise TimeoutError("基线超过每实例总时间预算")
                current, node = heapq.heappop(queue)
                if current > distance[node] + 1e-12:
                    continue
                if node == target:
                    break
                for arc in outgoing.get(node, ()):
                    key = str(arc["id"])
                    capacity = residual[key]
                    if capacity <= 1e-9:
                        continue
                    if mode == "variable_cost":
                        weight = float(arc["variable_cost"])
                    elif mode == "capacity_aware":
                        ratio = capacity / max(float(arc["capacity"]), 1.0)
                        weight = float(arc["variable_cost"]) * (1.0 + 0.5 / max(ratio, 0.05))
                    else:  # fixed_cost_aware / grasp
                        weight = float(arc["variable_cost"])
                        if key not in opened:
                            weight += float(arc["fixed_cost"]) / max(remaining, 1.0)
                    if mode == "grasp":
                        # 小幅随机扰动形成不同的候选路径，仍保持成本为正。
                        weight *= 1.0 + rng.uniform(-0.08, 0.08)
                    candidate = current + max(weight, 1e-9)
                    if candidate < distance[arc["head"]] - 1e-12:
                        distance[arc["head"]] = candidate
                        previous[arc["head"]] = arc
                        heapq.heappush(queue, (candidate, arc["head"]))
            if target not in previous:
                raise RuntimeError(f"{mode} 无法为商品 {commodity['id']} 找到可行残余路径")
            path = []
            node = target
            while node != source:
                arc = previous[node]
                path.append(str(arc["id"]))
                node = arc["tail"]
            amount = min(remaining, *(residual[key] for key in path))
            for key in path:
                residual[key] -= amount
                commodity_flow[key] = commodity_flow.get(key, 0.0) + amount
                opened[key] = 1.0
            remaining -= amount
        flows[str(commodity["id"])] = commodity_flow
    return {"arc_open": opened, "flows": flows}


def run_baseline(data: dict, name: str, *, restarts: int = 1,
                 timeout: float = BASELINE_TIMEOUT) -> dict:
    """运行并验证一个基线；GRASP 取多次随机构造中的最好可行解。"""
    from multi_agent_snd.evaluation import validate_solution

    started = time.perf_counter()
    candidates = []
    failures = []
    attempted = 0
    timed_out = False
    modes = {"variable_cost": "variable_cost", "fixed_cost_greedy": "fixed_cost_aware",
             "capacity_aware": "capacity_aware", "grasp": "grasp"}
    for attempt in range(max(1, restarts)):
        attempted += 1
        try:
            solution = _baseline_solution(data, modes[name], seed=RANDOM_SEED + attempt,
                                          deadline=started + timeout)
            objective, checks = validate_solution(data, solution)
            if checks["feasible"]:
                candidates.append((objective, solution, checks))
            else:
                failures.append(f"重启 {attempt} 的方案未通过独立可行性校验")
        except TimeoutError:
            timed_out = True
            break  # 保留此前完成的可行重启，不丢弃已找到的 incumbent。
        except RuntimeError as exc:
            failures.append(str(exc))  # 一次构造失败不能丢弃其他重启得到的可行解。
    if not candidates:
        if timed_out:
            raise TimeoutError(f"基线 {name} 超时，未找到经过校验的可行解")
        raise RuntimeError(f"基线 {name} 没有生成可行解：{failures[:1]}")
    objective, solution, checks = min(candidates, key=lambda x: x[0])
    return {**solution, "objective": objective, "objective_value": objective,
            "solver_status": "baseline_feasible", "solver_used": name,
            "constraint_satisfaction": checks, "solve_time": time.perf_counter() - started,
            "restarts": max(1, restarts), "attempted_restarts": attempted,
            "successful_restarts": len(candidates), "terminated_by_budget": timed_out,
            "restart_failures": failures}


def load_solver_class():
    if not SOLVER_FILE.exists():
        raise FileNotFoundError(f"找不到最终程序：{SOLVER_FILE}")
    sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location("exported_final_solver", SOLVER_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载：{SOLVER_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FinalSNDSolver


def summarize_comparison(instances: list[dict]) -> dict:
    """按模型类别与节点数统计；只对可行解计算差距，同时明确报告失败率。

    成功样本平均差距不能独立衡量整体表现，失败的实例不填 0，也不填虚假目标值。
    固定费用 SND 与零固定费用的线性特例绝不放在同一个总体平均值中。
    """
    def methods_for(cases):
        methods = sorted({row["method"] for case in cases for row in case["results"]})
        result = {}
        for method in methods:
            rows = [row for case in cases for row in case["results"] if row["method"] == method]
            skipped = [row for row in rows if row["status"] == "skipped_infeasible_input"]
            attempted = len(rows) - len(skipped)
            feasible = [row for row in rows if row["status"] == "ok" and row.get("feasible")]
            gaps = [row["best_observed_relative_gap"] for row in feasible]
            result[method] = {
                "attempted_instances": attempted, "feasible_instances": len(feasible),
                "skipped_proven_infeasible_inputs": len(skipped),
                "failed_instances": attempted - len(feasible),
                "success_rate": len(feasible) / attempted if attempted else None,
                "mean_best_observed_relative_gap_on_success": sum(gaps) / len(gaps) if gaps else None,
                "best_or_tied_instances": sum(gap <= 1e-9 for gap in gaps),
                "total_wall_seconds": sum(row["solve_time"] for row in rows),
            }
        return result

    groups = sorted({case["metadata"]["comparison_group"] for case in instances})
    return {group: {
        "instance_count": len(cases),
        "proven_infeasible_inputs": sum(case.get("input_audit", {}).get("status") == "proven_infeasible"
                                       for case in cases),
        "methods": methods_for(cases),
        "by_node_count": {str(n): methods_for([case for case in cases if case["nodes"] == n])
                          for n in sorted({case["nodes"] for case in cases})},
    } for group in groups
        for cases in [[case for case in instances if case["metadata"]["comparison_group"] == group]]}


def _publication_style():
    """集中管理论文图风格；使用英文标签避免中文字体缺失导致的方框。"""
    return {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }


def _figure_cases(report: dict) -> dict[str, list[dict]]:
    """保存全部实例与方法状态；失败不冒充零差距，未执行不冒充失败。"""
    groups: dict[str, list[dict]] = {}
    seen = set()
    for instance in report.get("instances", []):
        group = instance.get("metadata", {}).get("comparison_group", "unclassified")
        identity = (group, instance["name"])
        if identity in seen:
            raise ValueError(f"绘图报告包含重复实例：{identity}")
        seen.add(identity)
        raw_results = {}
        for row in instance.get("results", []):
            if row["method"] in raw_results:
                raise ValueError(f"实例 {identity} 包含重复方法：{row['method']}")
            raw_results[row["method"]] = row
        methods = list(METHOD_ORDER) + sorted(set(raw_results) - set(METHOD_ORDER))
        skipped = (instance.get("input_audit", {}).get("status") == "proven_infeasible"
                   or instance.get("status") == "skipped_infeasible_input")
        results = []
        for method in methods:
            row = raw_results.get(method, {})
            objective = row.get("objective")
            if objective is not None:
                objective = float(objective)
                if not math.isfinite(objective):
                    objective = None
            elapsed = row.get("solve_time")
            if elapsed is not None:
                elapsed = float(elapsed)
                if not math.isfinite(elapsed) or elapsed < 0:
                    elapsed = None
            if skipped or row.get("status") == "skipped_infeasible_input":
                status = "skipped"
            elif row.get("status") == "ok" and row.get("feasible") and objective is not None:
                status = "ok"
            elif row.get("status") in ("ok", "failed"):
                status = "failed"
            else:
                status = "pending"
            results.append({"method": method, "status": status,
                            "objective": objective if status == "ok" else None,
                            "solve_time": elapsed if status in ("ok", "failed") else None})
        groups.setdefault(group, []).append({
            "name": instance["name"], "nodes": int(instance["nodes"]),
            "arcs": int(instance["arcs"]), "commodities": int(instance["commodities"]),
            "input_proven_infeasible": skipped, "results": results,
        })
    return groups


def _figure_rows(report: dict, *, group: str = "fixed_charge_snd") -> list[dict]:
    """展开成功结果；分母仅为该实例已观察到的最好可行目标值。"""
    rows = []
    for instance in _figure_cases(report).get(group, []):
        successful = [row for row in instance["results"] if row["status"] == "ok"]
        best = min((row["objective"] for row in successful), default=None)
        for row in successful:
            rows.append({"instance": instance["name"], "nodes": instance["nodes"],
                         "arcs": instance["arcs"], "commodities": instance["commodities"],
                         "complexity": instance["nodes"] + instance["arcs"] + instance["commodities"],
                         "method": row["method"], "objective": row["objective"],
                         "gap_pct": 100.0 * (row["objective"] - best) / (abs(best) or 1.0),
                         "solve_time": row["solve_time"]})
    return rows


def _figure_hash(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def _quality_cases(cases: list[dict]) -> list[dict]:
    """目标值图不依赖耗时，不能因为耗时抖动就重复保存完全相同的图。"""
    return [{**case, "results": [{key: value for key, value in row.items() if key != "solve_time"}
                                 for row in case["results"]]} for case in cases]


def _feasibility_by_scale(cases: list[dict], method: str) -> dict[int, dict]:
    counts = {}
    for case in cases:
        item = counts.setdefault(case["nodes"], {"attempted": 0, "feasible": 0})
        for row in case["results"]:
            if row["method"] == method and row["status"] in ("ok", "failed"):
                item["attempted"] += 1
                item["feasible"] += int(row["status"] == "ok")
    return {nodes: {**item, "rate": 100.0 * item["feasible"] / item["attempted"]
                    if item["attempted"] else None} for nodes, item in counts.items()}


def _save_publication_figure(fig, output_dir: Path, stem: str) -> dict[str, str]:
    """同时保存矢量格式和高分辨率位图，返回实际路径。"""
    paths = {}
    for extension in FIGURE_FORMATS:
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if extension == "png":
            kwargs["dpi"] = FIGURE_DPI
        fig.savefig(path, **kwargs)
        paths[extension] = str(path)
    return paths


def _draw_group_figure(kind: str, group: str, cases: list[dict], *, ax) -> None:
    """在指定子图绘制一个模型类别，不混合不同模型的目标或统计分母。"""
    import numpy as np

    methods = list(dict.fromkeys(row["method"] for case in cases for row in case["results"]))
    names = [case["name"] for case in cases]
    x = np.arange(len(cases))
    by_instance = {case["name"]: {row["method"]: row for row in case["results"]} for case in cases}
    color = lambda method: METHOD_COLORS.get(method, "#777777")
    label = lambda method: METHOD_LABELS.get(method, method)
    markers = ("o", "s", "^", "D", "v", "P", "X")

    if kind == "fig01_objective_by_instance":
        width = 0.8 / max(len(methods), 1)
        best_values = [min((row["objective"] for row in case["results"]
                           if row["status"] == "ok"), default=None) for case in cases]
        for i, method in enumerate(methods):
            positions = x + (i - (len(methods) - 1) / 2) * width
            rows = [by_instance[name][method] for name in names]
            gaps = [100.0 * (row["objective"] - best) / (abs(best) or 1.0)
                    if row["status"] == "ok" else np.nan
                    for row, best in zip(rows, best_values)]
            ax.bar(positions, gaps, width=width, color=color(method),
                   label=label(method), zorder=3)
            for position, gap, row in zip(positions, gaps, rows):
                if row["status"] != "ok":
                    ax.annotate({"failed": "F", "skipped": "S", "pending": "P"}[row["status"]],
                                (position, 0), xytext=(0, 4), textcoords="offset points",
                                ha="center", fontsize=6, color=color(method))
                elif gap == 0:
                    # 最优柱高度为零，用圆点区分真实零差距与缺失结果。
                    ax.scatter(position, 0, s=12, color=color(method), zorder=4, clip_on=False)
        ax.set_ylim(bottom=0)
        if not any(best is not None for best in best_values):
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "No feasible objectives (skipped, failed or pending)",
                    transform=ax.transAxes, ha="center")
        ax.set_ylabel("Gap to best observed (%) / lower is better")
        ax.set_xlabel("Benchmark instance (dot: best; F: failed; S: skipped; P: pending)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)
        ax.legend(frameon=False, ncol=min(3, len(methods)), fontsize=7)
    elif kind == "fig02_runtime_by_instance":
        field = "solve_time"
        values_present = []
        for i, method in enumerate(methods):
            rows = [by_instance[name][method] for name in names]
            # 费用只连可行结果；耗时保留失败尝试，并用叉号明确区分。
            values = [row.get(field) if field == "solve_time" or row["status"] == "ok" else None
                      for row in rows]
            values_present.extend(value for value in values if value is not None)
            marker = markers[i % len(markers)]
            ax.plot(x, [np.nan if value is None else value for value in values],
                    marker=marker, markersize=3.5, color=color(method), label=label(method))
            if field == "solve_time":
                failed = [(j, value) for j, (row, value) in enumerate(zip(rows, values))
                          if row["status"] == "failed" and value is not None]
                ax.scatter([p[0] for p in failed], [p[1] for p in failed],
                           marker="x", s=45, color=color(method), zorder=4)
        if values_present and min(values_present) > 0:
            ax.set_yscale("log")
        ax.set_ylabel("Total cost (feasible only)" if field == "objective" else "Wall-clock time (s)")
        ax.set_xlabel("Benchmark instance" if field == "objective"
                      else "Benchmark instance (x: failed attempt)")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right")
        if not values_present:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "No feasible objectives (skipped, failed or pending)" if field == "objective"
                    else "No timed attempts (skipped or pending)", transform=ax.transAxes, ha="center")
        ax.grid(which="both", axis="y", color="#D9D9D9", linewidth=0.5)
        ax.legend(frameon=False, ncol=min(3, len(methods)), fontsize=7)
    elif kind == "fig03_relative_gap_heatmap":
        matrix = np.full((len(methods), len(cases)), np.nan)
        for j, case in enumerate(cases):
            successful = [row["objective"] for row in case["results"] if row["status"] == "ok"]
            best = min(successful, default=None)
            for i, method in enumerate(methods):
                row = by_instance[case["name"]][method]
                if row["status"] == "ok":
                    matrix[i, j] = 100.0 * (row["objective"] - best) / (abs(best) or 1.0)
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap("YlGnBu").with_extremes(bad="#EEEEEE")
        finite = matrix[np.isfinite(matrix)]
        image = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap,
                          interpolation="nearest", vmin=0,
                          vmax=max(1.0, float(finite.max()) if finite.size else 1.0))
        for i, method in enumerate(methods):
            for j, name in enumerate(names):
                status = by_instance[name][method]["status"]
                if status != "ok":
                    ax.text(j, i, {"failed": "F", "skipped": "S", "pending": "P"}[status],
                            ha="center", va="center", fontsize=7)
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels([label(method) for method in methods])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=60, ha="right")
        ax.set_xlabel("Instance (F: failed; S: skipped input; P: pending)")
        ax.figure.colorbar(image, ax=ax, pad=0.02).set_label("Gap to best observed (%)")
    elif kind == "fig04_feasibility_by_scale":
        nodes = sorted({case["nodes"] for case in cases})
        width = 0.78 / max(len(methods), 1)
        all_stats = {method: _feasibility_by_scale(cases, method) for method in methods}
        for i, method in enumerate(methods):
            stats = all_stats[method]
            rates = [stats[n]["rate"] if stats[n]["rate"] is not None else np.nan for n in nodes]
            ax.bar(np.arange(len(nodes)) + (i - (len(methods) - 1) / 2) * width,
                   rates, width=width, color=color(method), label=label(method))
        for i, n in enumerate(nodes):
            if not any(all_stats[method][n]["attempted"] for method in methods):
                ax.text(i, 3, "N/A", ha="center", fontsize=7)
        ax.set_ylim(0, 105)
        ax.set_xticks(np.arange(len(nodes)))
        ax.set_xticklabels([str(n) for n in nodes])
        ax.set_xlabel("Number of nodes (N/A: no attempted runs)")
        ax.set_ylabel("Feasible runs (%)")
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
        ax.legend(frameon=False, ncol=min(3, len(methods)), fontsize=7)
    else:
        raise ValueError(f"未知图表类型：{kind}")
    ax.set_title(FIGURE_GROUP_LABELS.get(group, group), fontsize=10)


def _draw_publication_figure(kind: str, groups: dict[str, list[dict]]):
    """全批次仅四张图；不同模型类别在各图内分面，不产生逐实例图。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(groups), 1, squeeze=False,
                             figsize=(max(10.0, 0.5 * max(map(len, groups.values()))),
                                      3.6 * len(groups)), constrained_layout=True)
    for ax, (group, cases) in zip(axes[:, 0], groups.items()):
        _draw_group_figure(kind, group, cases, ax=ax)
    return fig


def _render_figure(root: Path, key: str, payload: dict, style: dict, draw,
                   run_stamp: str) -> dict:
    """在本次时间戳目录中绘制实际数据，保存每张图的数据和文件哈希。"""
    import matplotlib.pyplot as plt

    data_fingerprint = _figure_hash(payload)
    fingerprint = _figure_hash({"key": key, "data": data_fingerprint, "style": style})
    components = [re.sub(r"[^A-Za-z0-9_-]+", "_", part) for part in key.split("/")]
    directory = root.joinpath(*components[:-1])
    stem = f"{components[-1]}_{run_stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    fig = draw()
    try:
        fig.text(0.995, 0.002, f"Generated: {run_stamp}", ha="right", va="bottom",
                 fontsize=6, color="#777777")
        files = _save_publication_figure(fig, directory, stem)
    finally:
        plt.close(fig)
    asset = {"render_fingerprint": fingerprint, "data_fingerprint": data_fingerprint,
             "figure_key": key, "plotted_data": payload, "files": files,
             "file_sha256": {extension: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                             for extension, path in files.items()}}
    write_json(directory / f"{stem}.data.json", asset)
    return {**asset, "reused": False}


def generate_benchmark_figures(report: dict, output_dir: str | Path | None = None,
                               *, run_stamp: str | None = None) -> dict:
    """全批次固定四张图，各自输出 PNG/PDF；不同模型类别在图内独立分面。

    相对差距只相对于本次最好可行解，不是最优性 Gap。保存实际绘图数据与
    文件哈希；重绘已有报告不代表进行了一次新的独立实验。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # 重画历史报告也遵守节点上限，不把旧的更大实例悄悄画回论文图。
    filtered = {**report, "instances": [case for case in report.get("instances", [])
                                       if int(case["nodes"]) <= MAX_BENCHMARK_NODES]}
    groups = {group: sorted(cases, key=lambda case: (case["nodes"], case["name"]))
              for group, cases in sorted(_figure_cases(filtered).items())}
    if not groups:
        raise ValueError("报告中没有可绘图的实例")
    root = (Path(output_dir) if output_dir is not None else FIGURE_OUTPUT_DIR).resolve()
    run_stamp = run_stamp or datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[0-9]{8}_[0-9]{6}(?:_[0-9]{6})?", run_stamp):
        raise ValueError("run_stamp 须为 YYYYMMDD_HHMMSS 或 YYYYMMDD_HHMMSS_ffffff")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_stamp
    suffix = 1
    while True:
        try:
            destination.mkdir()
            break
        except FileExistsError:
            suffix += 1
            destination = root / f"{run_stamp}_{suffix:02d}"
    style = {"version": FIGURE_STYLE_VERSION, "rc": _publication_style(),
             "labels": METHOD_LABELS, "colors": METHOD_COLORS, "groups": FIGURE_GROUP_LABELS,
             "formats": FIGURE_FORMATS, "dpi": FIGURE_DPI,
             "matplotlib": matplotlib.__version__, "numpy": np.__version__,
             "render_code": hashlib.sha256("".join(inspect.getsource(fn) for fn in (
                 _draw_group_figure, _draw_publication_figure, _feasibility_by_scale,
                 _save_publication_figure, _render_figure)).encode("utf-8")).hexdigest()}
    fingerprint = _figure_hash({"groups": groups, "style": style})
    assets = {}
    with plt.rc_context(_publication_style()):
        for kind in ("fig01_objective_by_instance", "fig02_runtime_by_instance",
                     "fig03_relative_gap_heatmap", "fig04_feasibility_by_scale"):
            plot_groups = (groups if kind == "fig02_runtime_by_instance" else
                           {group: _quality_cases(cases) for group, cases in groups.items()})
            assets[kind] = _render_figure(destination, kind, {"groups": plot_groups}, style,
                lambda kind=kind, data=plot_groups: _draw_publication_figure(kind, data), run_stamp)
    figures = {key: asset["files"] for key, asset in assets.items()}
    manifest_path = destination / "figure_manifest.json"
    sources = []
    source = {"source_report": str(Path(report["result_dir"]) / "comparison.json")
              if report.get("result_dir") else str(COMPARISON_FILE),
              "solver_file": report.get("solver_file"), "started_at": report.get("started_at"),
              "status": report.get("status"), "runtime": report.get("runtime"),
              "seed": report.get("seed"),
              "report_fingerprint": _figure_hash({key: value for key, value in report.items()
                  if not key.startswith("figure_") and key != "visualization_error"})}
    if source not in sources:
        sources.append(source)

    manifest = {
        "schema": "snd-benchmark-figures-v4",
        "data_fingerprint": fingerprint,
        "style_version": FIGURE_STYLE_VERSION,
        "last_plot_stamp": run_stamp,
        "updated_at": datetime.now().astimezone().isoformat(),
        "identity_note": "Self-contained timestamped snapshot; replotting is not an independent experiment.",
        "source_report_schema": report.get("schema"),
        "sources": sources, "scope": "Four figures with separate model-group panels; at most 200 nodes.",
        "excluded_over_node_limit": [case["name"] for case in report.get("instances", [])
                                     if int(case["nodes"]) > MAX_BENCHMARK_NODES],
        "gap_definition": "100 * (objective - best_observed_feasible_objective) / abs(best_observed_feasible_objective)",
        "zero_best_denominator": 1.0,
        "feasibility_denominator": "ok + failed attempts, including all-failed instances; excludes pending/skipped inputs",
        "groups": groups,
        "formats": list(FIGURE_FORMATS), "dpi_for_png": FIGURE_DPI,
        "files": figures,
        "assets": {key: {field: value for field, value in asset.items() if field != "plotted_data"}
                   for key, asset in assets.items()},
        "generated_assets": sum(not asset["reused"] for asset in assets.values()),
        "reused_assets": sum(asset["reused"] for asset in assets.values()),
        "note": "Relative gap is not a gap to a proven optimum or BKS.",
    }
    write_json(manifest_path, manifest)
    write_json(destination / "plotted_data.json", groups)
    parts = ['<!doctype html><meta charset="utf-8"><title>Benchmark figures</title>',
             '<style>body{font-family:system-ui;max-width:1100px;margin:2em auto}img{max-width:100%}</style>',
             '<h1>Benchmark figures / 算法对照图</h1>',
             '<p>全批次仅四张图，仅 PNG/PDF；各模型类别在图内单独展示。F=失败，S=输入不可行跳过，P=尚未执行。'
             '每次绘图独立存档；重绘次数不代表独立实验次数。相对差距不等于最优性 Gap。</p>',
             '<p>绘图时间戳：' + html.escape(run_stamp) + '</p>',
             '<p>来源报告：' + html.escape(source['source_report']) + '</p>',
             '<p>求解器：' + html.escape(str(source['solver_file'])) + '</p>',
             '<p><a href="figure_manifest.json">来源与指纹清单</a> | '
             '<a href="plotted_data.json">本次实际绘图数据</a></p>']
    for key, paths in figures.items():
        links = {extension: Path(path).relative_to(destination).as_posix()
                 for extension, path in paths.items()}
        parts.append(f'<h2>{html.escape(key)}</h2><p>' + " | ".join(
            f'<a href="{html.escape(link, quote=True)}">{extension.upper()}</a>'
            for extension, link in links.items()) + '</p>')
        if "png" in links:
            parts.append(f'<img loading="lazy" src="{html.escape(links["png"], quote=True)}" '
                         f'alt="{html.escape(key, quote=True)}">')
    index_path = destination / "index.html"
    index_path.write_text("\n".join(parts), encoding="utf-8")
    # 提供固定入口，用户无需从历史时间戳/指纹目录中猜测哪份是最新结果。
    latest_link = destination.name + "/index.html"
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Latest benchmark figures</title>'
        '<h1>最新绘图结果</h1><p><a href="' + latest_link + '">打开最新四张算法对照图</a></p>'
        '<p>旧目录是历史输出；文件夹数量不代表独立实验次数。图表来源见链接中的清单。</p>',
        encoding="utf-8")
    print(f"图表资源：新生成 {manifest['generated_assets']} 张，复用 {manifest['reused_assets']} 张；"
          f"浏览索引：{index_path}", flush=True)
    return {"output_dir": str(destination), "files": figures, "manifest": str(manifest_path),
            "index": str(index_path), "data_fingerprint": fingerprint,
            "reused": manifest["generated_assets"] == 0}


def main() -> None:
    from multi_agent_snd.benchmarks import (load_benchmark, CANAD_URLS, HOLMBERG_URL, FNSS_URL,
                                          capacity_infeasibility_certificates)
    from multi_agent_snd.evaluation import validate_solution

    # 绘图模式只读取已有报告，不重新执行求解器。适合修改图样式后反复出图。
    if PLOT_ONLY:
        if not COMPARISON_FILE.exists():
            raise FileNotFoundError(f"找不到已有对照报告：{COMPARISON_FILE}")
        report = json.loads(COMPARISON_FILE.read_text(encoding="utf-8"))
        if report.get("schema") != "snd-benchmark-comparison-v1":
            raise ValueError("对照报告 schema 不匹配，无法生成论文图")
        figure_stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        result = generate_benchmark_figures(
            report, FIGURE_OUTPUT_DIR, run_stamp=figure_stamp)
        report["figure_output_dir"] = result["output_dir"]
        report["figure_manifest"] = result["manifest"]
        report["figure_files"] = result["files"]
        report["figure_index"] = result["index"]
        report["figure_data_fingerprint"] = result["data_fingerprint"]
        # 记录重画结果，后续打开 final_comparison.json 即可定位图表。
        write_json(COMPARISON_FILE, report)
        print(f"论文级图表目录：{result['output_dir']}", flush=True)
        for stem, paths in result["files"].items():
            print(f"{stem}: {paths}", flush=True)
        return

    if not BENCHMARK_CASES:
        raise ValueError("BENCHMARK_CASES 不能为空")
    if len(set(BENCHMARK_CASES)) != len(BENCHMARK_CASES):
        raise ValueError("BENCHMARK_CASES 包含重复项，会导致统计偏差")
    if not math.isfinite(BASELINE_TIMEOUT) or BASELINE_TIMEOUT <= 0:
        raise ValueError("BASELINE_TIMEOUT 必须为正有限数")
    urls = {"R.tgz": CANAD_URLS["R"], "C.tgz": CANAD_URLS["C"],
            "CPlus.tgz": CANAD_URLS["CPlus"], "cexall.zip": HOLMBERG_URL,
            "waxman.tgz": FNSS_URL,
            "gridmnet.tgz": "https://commalab.di.unipi.it/files/Data/MMCF/gridmnet.tgz",
            "planarmnet.tgz": "https://commalab.di.unipi.it/files/Data/MMCF/planarmnet.tgz"}
    # 先验证全部输入，格式错误不可误记成算法求解失败。
    datasets = []
    for kind, archive, name in BENCHMARK_CASES:
        path = BENCHMARK_DIR / archive
        if not path.exists():
            raise FileNotFoundError(f"请从 {urls.get(archive, '原作者网站')} 下载实例包到 {path}")
        data = load_benchmark(path, name, kind=kind)
        if len(data["nodes"]) > MAX_BENCHMARK_NODES:
            raise ValueError(f"实例 {name} 超过 {MAX_BENCHMARK_NODES} 节点，请从 BENCHMARK_CASES 移除")
        instance_file = CONVERTED_INSTANCE_DIR / kind / f"{name}.json"
        write_json(instance_file, data)
        datasets.append((name, archive, instance_file, data))
    # 每次测试有独立归档；根目录两个 JSON 仍是本次最新结果，历史报告不会被覆盖。
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    result_dir = BENCHMARK_DIR / "results" / stamp
    report = {"schema": "snd-benchmark-comparison-v1", "solver_file": str(SOLVER_FILE),
              "started_at": datetime.now().astimezone().isoformat(), "status": "running",
              "total_instances": len(datasets), "completed_instances": 0,
              "result_dir": str(result_dir), "seed": RANDOM_SEED,
              "runtime": {"timeout_per_member": SOLVER_TIMEOUT,
                          "line_limit_per_member": SOLVER_LINE_LIMIT, "line_limit_scale": None,
                          "baseline_timeout_per_method": BASELINE_TIMEOUT,
                          "grasp_restarts": GRASP_RESTARTS},
              "notes": "grasp 是随机扰动多启动构造基线，非完整 GRASP 文献实现；"
                       "best_observed_relative_gap 仅对本次成功方法最小值，不是最优性 Gap；"
                       "各方法预算不同，运行时间包含各自实现及 LLM 隔离验解开销；"
                       "linear_flow_special_case 无固定开弧费用，单独统计；"
                       "成功样本平均差距需结合失败率，失败不证明实例不可行。",
              "instances": []}
    solutions = {"solver_file": str(SOLVER_FILE), "instances": []}

    def save_progress():
        report["summary_by_model"] = summarize_comparison(report["instances"])
        report["updated_at"] = datetime.now().astimezone().isoformat()
        for comparison_path, solution_path in (
                (result_dir / "comparison.json", result_dir / "solutions.json"),
                (COMPARISON_FILE, OUTPUT_FILE)):
            write_json(comparison_path, report)
            write_json(solution_path, solutions)

    write_json(result_dir / "manifest.json", {
        "schema": "snd-public-benchmark-manifest-v1",
        "instances": [{"name": name, "archive": archive, "instance_file": str(path),
                       "nodes": len(data["nodes"]), "arcs": len(data["arcs"]),
                       "commodities": len(data["commodities"]), "metadata": data["metadata"]}
                      for name, archive, path, data in datasets],
    })
    save_progress()
    baselines = [("variable_cost", 1), ("fixed_cost_greedy", 1),
                 ("capacity_aware", 1), ("grasp", GRASP_RESTARTS)]
    for index, (name, archive, instance_file, data) in enumerate(datasets, start=1):
        print(f"\n[{index}/{len(datasets)}] 基准 {name}：节点={len(data['nodes'])}，弧={len(data['arcs'])}，"
              f"商品={len(data['commodities'])}，分组={data['metadata']['comparison_group']}", flush=True)
        rows, plans = [], {}
        instance_report = {"name": name, "archive": archive, "metadata": data['metadata'],
            "instance_file": str(instance_file), "status": "running",
            "nodes": len(data['nodes']), "arcs": len(data['arcs']),
            "commodities": len(data['commodities']), "best_observed_objective": None,
            "results": rows}
        report['instances'].append(instance_report)
        solutions['instances'].append({"name": name, "solutions": plans})
        certificates = capacity_infeasibility_certificates(data)
        instance_report['input_audit'] = {
            "status": "proven_infeasible" if certificates else "necessary_conditions_passed",
            "certificates": certificates,
            "note": "仅检查单节点割容量必要条件；通过不代表一定可行，不调用 LP/MIP。",
        }
        if certificates:
            first = certificates[0]
            print(f"输入不可行：节点 {first['node']} 的 {first['direction']} 净需求 "
                  f"{first['required_net_flow']:.3f} > 可用容量 {first['available_capacity']:.3f}；"
                  "保留原始数据与证据，跳过全部算法，不计入算法失败率。", flush=True)
            for method, _ in [("exported_llm_heuristic", 1), *baselines]:
                rows.append({"method": method, "status": "skipped_infeasible_input",
                             "objective": None, "feasible": None, "solve_time": 0.0,
                             "best_observed_relative_gap": None,
                             "error": "原始实例违反容量必要条件；不修改数据，不运行算法。"})
            instance_report['status'] = 'skipped_infeasible_input'
            report['completed_instances'] = index
            save_progress()
            continue
        for method, restarts in [("exported_llm_heuristic", 1), *baselines]:
            started = time.perf_counter()
            try:
                if method == "exported_llm_heuristic":
                    solver = load_solver_class()(timeout=SOLVER_TIMEOUT,
                        line_limit=SOLVER_LINE_LIMIT, line_limit_scale=None)
                    result = solver.solve(data)
                else:
                    result = run_baseline(data, method, restarts=restarts, timeout=BASELINE_TIMEOUT)
                # 即使部署包装或基线返回了 objective，仍针对当前输入重新验解与重算。
                objective, checks = validate_solution(data, result)
                if not checks["feasible"]:
                    raise ValueError(f"{method} 返回不可行方案：{checks}")
                result.update(objective=objective, objective_value=objective, constraint_satisfaction=checks)
                elapsed = time.perf_counter() - started
                rows.append({"method": method, "status": "ok", "objective": result['objective'],
                    "solve_time": elapsed, "executed_lines": result.get('executed_lines'),
                    "restarts": result.get('restarts'),
                    "successful_restarts": result.get('successful_restarts'),
                    "terminated_by_budget": result.get('terminated_by_budget', False),
                    "feasible": result['constraint_satisfaction']['feasible']})
                plans[method] = result
                print(f"{method}：目标值={result['objective']:.6f}，耗时={elapsed:.3f}s", flush=True)
            except Exception as exc:
                rows.append({"method": method, "status": "failed",
                    "solve_time": time.perf_counter() - started,
                    "feasible": False, "objective": None,
                    "failure_type": getattr(exc, "failure_type", type(exc).__name__),
                    "executed_lines": getattr(exc, "lines", None),
                    "error": f"{type(exc).__name__}: {exc}"})
                print(f"{method} 失败（继续其他方法）：{exc}", flush=True)
            successful = [row['objective'] for row in rows if row['status'] == 'ok' and row['feasible']]
            best = min(successful) if successful else None
            instance_report['best_observed_objective'] = best
            for row in rows:
                row['best_observed_relative_gap'] = (
                    (row['objective'] - best) / (abs(best) or 1.0)
                    if row['status'] == 'ok' and best is not None else None)
            # 每个方法结束立即落盘；某次失败或后续中断都不会丢掉此前完成结果。
            save_progress()
        instance_report['status'] = 'completed'
        report['completed_instances'] = index
        save_progress()
    report['status'] = 'completed'
    report['finished_at'] = datetime.now().astimezone().isoformat()
    if GENERATE_FIGURES:
        try:
            figure_result = generate_benchmark_figures(
                report, FIGURE_OUTPUT_DIR, run_stamp=stamp)
            report["figure_output_dir"] = figure_result["output_dir"]
            report["figure_manifest"] = figure_result["manifest"]
            report["figure_files"] = figure_result["files"]
            report["figure_index"] = figure_result["index"]
            report["figure_data_fingerprint"] = figure_result["data_fingerprint"]
            print(f"论文级图表目录：{figure_result['output_dir']}", flush=True)
        except Exception as exc:
            # 对照结果已经完整时，绘图失败不应掩盖实验结果；报告保留明确诊断。
            report["visualization_error"] = f"{type(exc).__name__}: {exc}"
            print(f"绘图失败（对照结果仍已保存）：{exc}", flush=True)
    save_progress()
    print(f"对照结果文件：{COMPARISON_FILE}\n本次独立归档：{result_dir}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows 编辑器/重定向也统一输出 UTF-8。
    main()
