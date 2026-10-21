"""静态与时间展开多商品 SND 的弧流 MIP 求解器。"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class BackendUnavailableError(RuntimeError):
    """可选求解器后端不可用时抛出的异常。"""


@dataclass(slots=True)
class SNDSolveResult:
    """与具体求解器无关的 SND 求解结果。

    ``arc_open`` contains binary design decisions and ``flows`` contains one
    每个商品一组流量向量。约束诊断信息便于在交给后续实验前独立验证结果。
    """

    status: str
    objective_value: float | None
    is_optimal: bool
    solve_time: float
    requested_solver: str
    solver_used: str
    mip_gap: float | None
    arc_open: dict[str, float]
    flows: dict[str, dict[str, float]]
    constraint_satisfaction: dict[str, float | bool]
    message: str
    fallback_reason: str | None = None
    # 时间展开实例的物理节点/弧及商品时间窗；静态实例为 None。
    time_expansion: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass(frozen=True, slots=True)
class _ArcData:
    id: str
    tail: int
    head: int
    capacity: float
    variable_cost: float
    fixed_cost: float


@dataclass(frozen=True, slots=True)
class _CommodityData:
    id: str
    origin: int
    destination: int
    demand: float


@dataclass(slots=True)
class _RawSolution:
    status: str
    objective: float | None
    is_optimal: bool
    gap: float | None
    y: list[float]
    x: list[list[float]]
    message: str


class SNDSolver:
    """构建并求解静态或时间展开的多商品 SND 实例。

    模型是带容量限制的多商品弧流 MIP。``y[a]`` 表示开启弧，``x[k,a]``
    表示商品 ``k`` 在弧上的流量；目标函数最小化固定开启成本和单位流量成本。

    Parameters
    ----------
    instance:
        An object exposing ``to_dict()``, an instance dictionary, or a JSON
        file path.
    solver_type:
        ``"scipy"``, ``"gurobi"``, ``"cplex"`` or ``"ortools"``.
        Optional backends automatically fall back to SciPy when unavailable.
    time_limit:
        Maximum solve time in seconds; ``None`` means no explicit limit.
    mip_gap:
        Relative MIP optimality gap in ``[0, 1]``.
    log_output:
        Whether the selected backend should print its optimization log.
    """

    SUPPORTED_SOLVERS = frozenset({"scipy", "gurobi", "cplex", "ortools"})

    def __init__(
        self,
        instance: Any,
        solver_type: str = "scipy",
        time_limit: float | None = 300.0,
        mip_gap: float = 1e-4,
        log_output: bool = False,
    ) -> None:
        # 统一后端名称，确保后续分派逻辑确定。
        solver_type = solver_type.lower().strip()
        if solver_type not in self.SUPPORTED_SOLVERS:
            names = ", ".join(sorted(self.SUPPORTED_SOLVERS))
            raise ValueError(f"solver_type must be one of: {names}")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive or None")
        if not 0.0 <= mip_gap <= 1.0:
            raise ValueError("mip_gap must be in [0, 1]")

        self.requested_solver = solver_type
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.log_output = log_output
        # 在输入边界兼容多种格式，进入内部后统一使用一种数据结构。
        self._data = self._load_instance(instance)
        self.nodes, self.arcs, self.commodities = self._normalize(self._data)
        self.time_expansion = self._data.get("time_expansion")
        self._node_index = {node: i for i, node in enumerate(self.nodes)}

    def solve(self) -> SNDSolveResult:
        """构建、求解、校验并返回统一格式的结果。"""
        started = time.perf_counter()
        solver_used = self.requested_solver
        fallback_reason = None
        try:
            raw = self._dispatch(self.requested_solver)
        except BackendUnavailableError as exc:
            if self.requested_solver == "scipy":
                raise
            fallback_reason = str(exc)
            solver_used = "scipy"
            raw = self._solve_scipy()

        elapsed = time.perf_counter() - started
        # 将后端数组转换为输入实例中的稳定弧 ID 和商品 ID。
        arc_open = {arc.id: self._clean(raw.y[a])
                    for a, arc in enumerate(self.arcs)}
        flows = {
            commodity.id: {
                arc.id: self._clean(raw.x[k][a])
                for a, arc in enumerate(self.arcs)
            }
            for k, commodity in enumerate(self.commodities)
        }
        checks = self._check_constraints(raw.y, raw.x)
        return SNDSolveResult(
            status=raw.status,
            objective_value=raw.objective,
            is_optimal=raw.is_optimal,
            solve_time=elapsed,
            requested_solver=self.requested_solver,
            solver_used=solver_used,
            mip_gap=raw.gap,
            arc_open=arc_open,
            flows=flows,
            constraint_satisfaction=checks,
            message=raw.message,
            fallback_reason=fallback_reason,
            time_expansion=self.time_expansion,
        )

    def _dispatch(self, name: str) -> _RawSolution:
        # 延迟导入可选后端，普通 SciPy 运行不会要求安装商业求解器。
        return {
            "scipy": self._solve_scipy,
            "gurobi": self._solve_gurobi,
            "cplex": self._solve_cplex,
            "ortools": self._solve_ortools,
        }[name]()

    @staticmethod
    def _load_instance(instance: Any) -> dict[str, Any]:
        # JSON 显式按 UTF-8 读取；生成器对象通过公开的 to_dict() 适配。
        if isinstance(instance, (str, Path)):
            path = Path(instance)
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"cannot read SND instance from {path}: {exc}") from exc
        if isinstance(instance, dict):
            return instance
        to_dict = getattr(instance, "to_dict", None)
        if callable(to_dict):
            data = to_dict()
            if isinstance(data, dict):
                return data
        raise TypeError("instance must be a dictionary, JSON path, or object with to_dict()")

    @staticmethod
    def _normalize(data: dict[str, Any]) -> tuple[tuple[int, ...],
                                                  tuple[_ArcData, ...],
                                                  tuple[_CommodityData, ...]]:
        # 先转换为紧凑的类型化记录，在建模前尽早发现端点或成本错误。
        try:
            nodes = tuple(int(node) for node in data["nodes"])
            arcs = tuple(
                _ArcData(str(a["id"]), int(a["tail"]), int(a["head"]),
                         float(a["capacity"]), float(a["variable_cost"]),
                         float(a["fixed_cost"]))
                for a in data["arcs"]
            )
            commodities = tuple(
                _CommodityData(str(k["id"]), int(k["origin"]),
                               int(k["destination"]), float(k["demand"]))
                for k in data["commodities"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid SND instance schema: {exc}") from exc
        if len(nodes) < 2 or len(set(nodes)) != len(nodes):
            raise ValueError("nodes must contain at least two unique identifiers")
        node_set = set(nodes)
        if not arcs or not commodities:
            raise ValueError("instance must contain at least one arc and one commodity")
        if len({a.id for a in arcs}) != len(arcs):
            raise ValueError("arc ids must be unique")
        if len({k.id for k in commodities}) != len(commodities):
            raise ValueError("commodity ids must be unique")
        if any(a.tail not in node_set or a.head not in node_set or
               a.tail == a.head or a.capacity < 0 or
               a.variable_cost < 0 or a.fixed_cost < 0 or
               not all(math.isfinite(v) for v in (a.capacity, a.variable_cost, a.fixed_cost)) for a in arcs):
            raise ValueError("arcs contain invalid endpoints, capacity, or costs")
        if any(k.origin not in node_set or k.destination not in node_set or
               k.origin == k.destination or k.demand <= 0 or not math.isfinite(k.demand) for k in commodities):
            raise ValueError("commodities contain invalid endpoints or demand")
        SNDSolver._validate_time_expansion(data, node_set, {a.id for a in arcs})
        return nodes, arcs, commodities

    @staticmethod
    def _validate_time_expansion(data: dict[str, Any], node_set: set[int],
                                 arc_ids: set[str]) -> None:
        """校验时间展开元数据；弧流模型仍直接使用展开后的顶层字段。"""
        expansion = data.get("time_expansion")
        if expansion is None:
            if data.get("format") == "time-expanded-snd-v1":
                raise ValueError("time-expanded-snd-v1 requires time_expansion")
            return
        if data.get("format") != "time-expanded-snd-v1":
            raise ValueError("time_expansion requires time-expanded-snd-v1 format")
        if not isinstance(expansion, dict):
            raise ValueError("time_expansion must be a dictionary")
        points = expansion.get("time_points")
        node_map = expansion.get("node_map")
        schedule = expansion.get("arc_schedule")
        windows = expansion.get("commodity_windows")
        if (not isinstance(points, list) or len(points) < 2 or
                points != list(range(len(points))) or
                not isinstance(node_map, list) or not isinstance(schedule, list) or
                not isinstance(windows, list)):
            raise ValueError("time_expansion contains invalid time metadata")
        mapped_nodes: dict[int, dict[str, Any]] = {}
        for item in node_map:
            try:
                node_id, physical, t = int(item["node_id"]), int(item["physical_node"]), int(item["time"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid time_expansion.node_map") from exc
            if node_id not in node_set or t not in points or node_id in mapped_nodes:
                raise ValueError("time_expansion.node_map is inconsistent with nodes")
            mapped_nodes[node_id] = {"physical_node": physical, "time": t}
        if set(mapped_nodes) != node_set:
            raise ValueError("time_expansion.node_map must cover all nodes")
        scheduled: dict[str, dict[str, Any]] = {}
        for item in schedule:
            try:
                aid = str(item["arc_id"])
                departure, arrival = int(item["departure_time"]), int(item["arrival_time"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid time_expansion.arc_schedule") from exc
            if aid not in arc_ids or aid in scheduled or departure not in points or arrival not in points:
                raise ValueError("time_expansion.arc_schedule is inconsistent with arcs")
            scheduled[aid] = {"departure_time": departure, "arrival_time": arrival,
                              "kind": item.get("kind")}
        if set(scheduled) != arc_ids:
            raise ValueError("time_expansion.arc_schedule must cover all arcs")
        physical_arcs = {str(item["id"]): item for item in expansion.get("physical_arcs", [])}
        if len(physical_arcs) != len(expansion.get("physical_arcs", [])):
            raise ValueError("time_expansion physical arc IDs must be unique")
        schedule_by_id = {str(item["arc_id"]): item for item in schedule}
        for arc in data["arcs"]:
            event = scheduled[str(arc["id"])]
            tail, head = mapped_nodes[int(arc["tail"])], mapped_nodes[int(arc["head"])]
            if (tail["time"] != event["departure_time"] or
                    head["time"] != event["arrival_time"] or head["time"] <= tail["time"]):
                raise ValueError("time_expansion arc times do not match arc endpoints")
            original = schedule_by_id[str(arc["id"])]
            if event["kind"] == "wait":
                if (tail["physical_node"] != head["physical_node"] or
                        head["time"] != tail["time"] + 1 or
                        int(original.get("physical_node", -1)) != tail["physical_node"] or
                        float(arc["fixed_cost"]) != 0 or float(arc["variable_cost"]) != 0):
                    raise ValueError("invalid temporal wait arc")
            elif event["kind"] == "service":
                physical_arc_id = str(original.get("physical_arc_id"))
                match = physical_arcs.get(physical_arc_id)
                if (match is None or int(match["tail"]) != tail["physical_node"] or
                        int(match["head"]) != head["physical_node"]):
                    raise ValueError("temporal service arc disagrees with physical arc")
            else:
                raise ValueError("temporal arc kind must be service or wait")
        commodities = {str(item["id"]): item for item in data["commodities"]}
        if len(windows) != len(commodities):
            raise ValueError("time_expansion commodity_windows count mismatch")
        seen_windows: set[str] = set()
        for window in windows:
            cid = str(window.get("commodity_id"))
            if cid not in commodities or cid in seen_windows:
                raise ValueError("unknown or duplicate commodity in time_expansion commodity_windows")
            seen_windows.add(cid)
            item = commodities[cid]
            origin, destination = mapped_nodes[int(item["origin"])], mapped_nodes[int(item["destination"])]
            try:
                release, deadline = int(window["release_time"]), int(window["deadline"])
                physical_origin = int(window["origin"])
                physical_destination = int(window["destination"])
                demand = float(window["demand"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid time_expansion commodity window") from exc
            if (origin != {"physical_node": physical_origin, "time": release} or
                    destination != {"physical_node": physical_destination, "time": deadline} or
                    not 0 <= release < deadline < len(points) or
                    not math.isclose(demand, float(item["demand"]), rel_tol=1e-9, abs_tol=1e-9)):
                raise ValueError("time_expansion commodity window disagrees with commodities")

    def _solve_scipy(self) -> _RawSolution:
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import lil_matrix
        except ImportError as exc:
            raise BackendUnavailableError(
                "当前 Python 解释器中无法使用 SciPy（"
                f"{exc}）。请切换到项目 `.venv`，或执行 "
                "`python -m pip install numpy scipy` 安装依赖。"
            ) from exc

        # 变量布局：先放 y[a]，再按商品分块放置 x[k,a]。
        arc_n, commodity_n, node_n = len(self.arcs), len(self.commodities), len(self.nodes)
        variable_n = arc_n + commodity_n * arc_n
        c = np.zeros(variable_n)
        for a, arc in enumerate(self.arcs):
            c[a] = arc.fixed_cost
        for k in range(commodity_n):
            for a, arc in enumerate(self.arcs):
                c[arc_n + k * arc_n + a] = arc.variable_cost

        # 约束行依次为每个（商品、节点）的流平衡，以及各弧的容量约束。
        row_n = commodity_n * node_n + arc_n
        matrix = lil_matrix((row_n, variable_n), dtype=float)
        lower = np.zeros(row_n)
        upper = np.zeros(row_n)
        row = 0
        for k, commodity in enumerate(self.commodities):
            for node in self.nodes:
                for a, arc in enumerate(self.arcs):
                    column = arc_n + k * arc_n + a
                    if arc.tail == node:
                        matrix[row, column] += 1.0
                    if arc.head == node:
                        matrix[row, column] -= 1.0
                # 起点 RHS 为正表示注入需求，终点 RHS 为负表示需求流出网络。
                balance = commodity.demand if node == commodity.origin else (
                    -commodity.demand if node == commodity.destination else 0.0
                )
                lower[row] = upper[row] = balance
                row += 1
        for a, arc in enumerate(self.arcs):
            # 弧上所有商品流量之和不能超过 capacity*y[a]。
            matrix[row, a] = -arc.capacity
            for k in range(commodity_n):
                matrix[row, arc_n + k * arc_n + a] = 1.0
            lower[row], upper[row] = -np.inf, 0.0
            row += 1

        # 只有开启变量是整数变量，流量变量保持连续。
        integrality = np.zeros(variable_n, dtype=int)
        integrality[:arc_n] = 1
        lb = np.zeros(variable_n)
        ub = np.full(variable_n, np.inf)
        ub[:arc_n] = 1.0
        options: dict[str, Any] = {"disp": self.log_output,
                                   "mip_rel_gap": self.mip_gap,
                                   "presolve": True, "random_seed": 0}
        if self.time_limit is not None:
            options["time_limit"] = self.time_limit
        # SciPy 将 HiGHS 专用参数原样转交；只屏蔽这条已知转交提示。
        # 不在此切换 threads：HiGHS 调度器在同一进程全局共享，切换会与旧 LP 冲突。
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Unrecognized options detected.*", category=RuntimeWarning)
            result = milp(c=c, integrality=integrality, bounds=Bounds(lb, ub),
                          constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                          options=options)
        statuses = {0: "optimal", 1: "time_limit", 2: "infeasible",
                    3: "unbounded", 4: "solver_error"}
        has_solution = result.x is not None
        values = result.x if has_solution else np.zeros(variable_n)
        y = values[:arc_n].tolist()
        x = [values[arc_n + k * arc_n:arc_n + (k + 1) * arc_n].tolist()
             for k in range(commodity_n)]
        gap = getattr(result, "mip_gap", None)
        return _RawSolution(statuses.get(result.status, "unknown"),
                            float(result.fun) if result.fun is not None else None,
                            result.status == 0,
                            float(gap) if gap is not None else None,
                            y, x, str(result.message))

    def _solve_gurobi(self) -> _RawSolution:
        # Gurobi 是可选依赖；导入、license 或建模失败都会转换为后端不可用，
        # 让 solve() 能够按配置自动回退。
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError as exc:
            raise BackendUnavailableError("Gurobi package is not installed") from exc
        try:
            model = gp.Model("static_snd")
            model.Params.OutputFlag = int(self.log_output)
            model.Params.MIPGap = self.mip_gap
            model.Params.Seed = 0
            model.Params.Threads = 1
            if self.time_limit is not None:
                model.Params.TimeLimit = self.time_limit
            y = model.addVars(len(self.arcs), vtype=GRB.BINARY, name="open")
            x = model.addVars(len(self.commodities), len(self.arcs), lb=0.0,
                              vtype=GRB.CONTINUOUS, name="flow")
            model.setObjective(
                gp.quicksum(a.fixed_cost * y[i] for i, a in enumerate(self.arcs)) +
                gp.quicksum(a.variable_cost * x[k, i]
                            for k in range(len(self.commodities))
                            for i, a in enumerate(self.arcs)), GRB.MINIMIZE)
            for k, commodity in enumerate(self.commodities):
                for node in self.nodes:
                    outgoing = gp.quicksum(x[k, a] for a, arc in enumerate(self.arcs)
                                           if arc.tail == node)
                    incoming = gp.quicksum(x[k, a] for a, arc in enumerate(self.arcs)
                                           if arc.head == node)
                    rhs = commodity.demand if node == commodity.origin else (
                        -commodity.demand if node == commodity.destination else 0.0)
                    model.addConstr(outgoing - incoming == rhs)
            for a, arc in enumerate(self.arcs):
                model.addConstr(gp.quicksum(x[k, a]
                                            for k in range(len(self.commodities)))
                                <= arc.capacity * y[a])
            model.optimize()
        except Exception as exc:
            raise BackendUnavailableError(f"Gurobi unavailable (package/license): {exc}") from exc
        statuses = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
                    GRB.INFEASIBLE: "infeasible", GRB.UNBOUNDED: "unbounded",
                    GRB.INF_OR_UNBD: "infeasible_or_unbounded"}
        has_solution = model.SolCount > 0
        return _RawSolution(statuses.get(model.Status, "solver_error"),
                            float(model.ObjVal) if has_solution else None,
                            model.Status == GRB.OPTIMAL,
                            float(model.MIPGap) if has_solution else None,
                            [float(y[a].X) for a in range(len(self.arcs))] if has_solution else [0.0] * len(self.arcs),
                            [[float(x[k, a].X) for a in range(len(self.arcs))]
                             for k in range(len(self.commodities))] if has_solution else self._zero_flows(),
                            f"Gurobi status {model.Status}")

    def _solve_cplex(self) -> _RawSolution:
        # DOcplex 同样是可选依赖，并通过相同的适配器接口隔离。
        try:
            from docplex.mp.model import Model
        except ImportError as exc:
            raise BackendUnavailableError("DOcplex/CPLEX package is not installed") from exc
        try:
            model = Model(name="static_snd", log_output=self.log_output)
            model.parameters.threads = 1
            model.parameters.randomseed = 0
            model.parameters.mip.tolerances.mipgap = self.mip_gap
            if self.time_limit is not None:
                model.parameters.timelimit = self.time_limit
            y = model.binary_var_list(len(self.arcs), name="open")
            x = {(k, a): model.continuous_var(lb=0, name=f"flow_{k}_{a}")
                 for k in range(len(self.commodities)) for a in range(len(self.arcs))}
            model.minimize(model.sum(arc.fixed_cost * y[a]
                                     for a, arc in enumerate(self.arcs)) +
                           model.sum(arc.variable_cost * x[k, a]
                                     for k in range(len(self.commodities))
                                     for a, arc in enumerate(self.arcs)))
            for k, commodity in enumerate(self.commodities):
                for node in self.nodes:
                    outgoing = model.sum(x[k, a] for a, arc in enumerate(self.arcs)
                                         if arc.tail == node)
                    incoming = model.sum(x[k, a] for a, arc in enumerate(self.arcs)
                                         if arc.head == node)
                    rhs = commodity.demand if node == commodity.origin else (
                        -commodity.demand if node == commodity.destination else 0.0)
                    model.add_constraint(outgoing - incoming == rhs)
            for a, arc in enumerate(self.arcs):
                model.add_constraint(model.sum(x[k, a]
                                               for k in range(len(self.commodities)))
                                     <= arc.capacity * y[a])
            solution = model.solve(log_output=self.log_output)
        except Exception as exc:
            raise BackendUnavailableError(f"CPLEX unavailable (package/license): {exc}") from exc
        details = model.solve_details
        status_text = str(details.status).lower()
        status = "optimal" if "optimal" in status_text else (
            "infeasible" if "infeasible" in status_text else
            "time_limit" if "limit" in status_text else "solver_error")
        has_solution = solution is not None
        return _RawSolution(status, float(solution.objective_value) if has_solution else None,
                            status == "optimal", getattr(details, "mip_relative_gap", None),
                            [float(solution.get_value(v)) for v in y] if has_solution else [0.0] * len(self.arcs),
                            [[float(solution.get_value(x[k, a])) for a in range(len(self.arcs))]
                             for k in range(len(self.commodities))] if has_solution else self._zero_flows(),
                            str(details.status))

    def _solve_ortools(self) -> _RawSolution:
        # 如果可用，OR-Tools 的 CBC 提供免费的整数规划后端。
        try:
            from ortools.linear_solver import pywraplp
        except ImportError as exc:
            raise BackendUnavailableError("OR-Tools package is not installed") from exc
        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if solver is None:
            raise BackendUnavailableError("OR-Tools CBC backend is unavailable")
        if self.time_limit is not None:
            solver.SetTimeLimit(round(self.time_limit * 1000))
        solver.SetNumThreads(1)
        solver.SetSolverSpecificParametersAsString(f"ratioGap={self.mip_gap}")
        if self.log_output:
            solver.EnableOutput()
        y = [solver.BoolVar(f"open_{a}") for a in range(len(self.arcs))]
        x = [[solver.NumVar(0.0, solver.infinity(), f"flow_{k}_{a}")
              for a in range(len(self.arcs))] for k in range(len(self.commodities))]
        objective = solver.Objective()
        for a, arc in enumerate(self.arcs):
            objective.SetCoefficient(y[a], arc.fixed_cost)
            for k in range(len(self.commodities)):
                objective.SetCoefficient(x[k][a], arc.variable_cost)
        objective.SetMinimization()
        for k, commodity in enumerate(self.commodities):
            for node in self.nodes:
                rhs = commodity.demand if node == commodity.origin else (
                    -commodity.demand if node == commodity.destination else 0.0)
                constraint = solver.Constraint(rhs, rhs)
                for a, arc in enumerate(self.arcs):
                    if arc.tail == node:
                        constraint.SetCoefficient(x[k][a], 1.0)
                    if arc.head == node:
                        constraint.SetCoefficient(x[k][a], -1.0)
        for a, arc in enumerate(self.arcs):
            constraint = solver.Constraint(-solver.infinity(), 0.0)
            constraint.SetCoefficient(y[a], -arc.capacity)
            for k in range(len(self.commodities)):
                constraint.SetCoefficient(x[k][a], 1.0)
        status_code = solver.Solve()
        statuses = {pywraplp.Solver.OPTIMAL: "optimal",
                    pywraplp.Solver.FEASIBLE: "feasible",
                    pywraplp.Solver.INFEASIBLE: "infeasible",
                    pywraplp.Solver.UNBOUNDED: "unbounded",
                    pywraplp.Solver.ABNORMAL: "solver_error",
                    pywraplp.Solver.NOT_SOLVED: "not_solved"}
        has_solution = status_code in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
        return _RawSolution(statuses.get(status_code, "unknown"),
                            objective.Value() if has_solution else None,
                            status_code == pywraplp.Solver.OPTIMAL, None,
                            [v.solution_value() for v in y] if has_solution else [0.0] * len(self.arcs),
                            [[v.solution_value() for v in row] for row in x] if has_solution else self._zero_flows(),
                            f"OR-Tools status {status_code}")

    def _check_constraints(self, y: list[float], x: list[list[float]]) -> dict[str, float | bool]:
        # 根据返回变量重新计算残差，不直接依赖后端状态判断可行性。
        max_balance = 0.0
        for k, commodity in enumerate(self.commodities):
            for node in self.nodes:
                balance = sum(x[k][a] for a, arc in enumerate(self.arcs) if arc.tail == node)
                balance -= sum(x[k][a] for a, arc in enumerate(self.arcs) if arc.head == node)
                expected = commodity.demand if node == commodity.origin else (
                    -commodity.demand if node == commodity.destination else 0.0)
                max_balance = max(max_balance, abs(balance - expected))
        max_capacity = max(
            (sum(x[k][a] for k in range(len(self.commodities))) - arc.capacity * y[a]
             for a, arc in enumerate(self.arcs)), default=0.0)
        max_capacity = max(0.0, max_capacity)
        max_binary = max((min(abs(value), abs(value - 1.0)) for value in y), default=0.0)
        min_flow = min((value for row in x for value in row), default=0.0)
        feasible = max_balance <= 1e-5 and max_capacity <= 1e-5 and max_binary <= 1e-5 and min_flow >= -1e-5
        return {"feasible": feasible,
                "max_flow_balance_violation": max_balance,
                "max_capacity_violation": max_capacity,
                "max_binary_violation": max_binary,
                "max_nonnegativity_violation": max(0.0, -min_flow)}

    def _zero_flows(self) -> list[list[float]]:
        return [[0.0] * len(self.arcs) for _ in self.commodities]

    @staticmethod
    def _clean(value: float) -> float:
        if abs(value) < 1e-9:
            return 0.0
        if abs(value - round(value)) < 1e-9:
            return float(round(value))
        return float(value)
