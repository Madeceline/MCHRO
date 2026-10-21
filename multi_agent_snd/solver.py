"""一般和多主体程序共演化：1 个求解器玩家 + 至多 6 个方向生成器玩家。"""
from __future__ import annotations

import ast
import json
import platform
import os
import random
import uuid
import time
import threading
from statistics import fmean
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .operators import OPERATOR_TEMPLATES, EVOLUTION_OPERATOR_CODES
from .eoh_s import (EoHSConfig, cpi as eohs_cpi, cpm_select, finite_failure_vectors,
                    performance_vector, select_complementary_parents,
                    select_local_parent)
from .code_parser import CodeParser
from .clustering import select_representatives
from .diagnostics import summarize_evaluations, summarize_by_scale
from snd_generator import SNDInstance, SNDParameterBounds
from snd_solver import SNDSolver

from .baselines import BASE_GENERATOR, BASE_SOLVER, OfflineMultiAgentClient
from .evaluation import SNDEvaluator, digest, validate_instance
from .execution import isolated_call, validate_code
from .game import CCEMetaSolver, MetaSolver
from .prompts import build_prompt
from .storage import Program, ProgramArchive, clean_json, write_json
from .utilities import DIRECTIONS, UtilityConfig
from .reference import ReferenceConfig
from .performance import ordered_map, TimingRecorder


_WARM_START_AST = ast.dump(ast.parse(BASE_SOLVER), include_attributes=False)


@dataclass
class MultiAgentResult:
    """可序列化统一结果；完整历史源码通过 programs 目录内容寻址。"""

    data: dict[str, Any]
    output_dir: Path

    @property
    def objective(self) -> float | None:
        return self.data.get("objective")

    def to_dict(self) -> dict[str, Any]:
        return clean_json(self.data)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, allow_nan=False)


class MultiAgentSNDSolver:
    """每个对象对应一次训练和基础 SND 实例求解。

    ``solver_type/time_limit/mip_gap`` 控制参考 MIP，不是进化总预算。
    ``outer_iterations`` 为外层扩池次数，``eoh_iterations`` 为每次响应搜索轮数。
    所有 LLM 客户端需支持 generate(prompt, temperature, n)，并可被 spawn 序列化。
    """

    def __init__(self, instance: Any, solver_type: str = "scipy", time_limit: float = 30,
                 mip_gap: float = 0.0, log_output: bool = False, *,
                 llm_client: Any | None = None,
                 generator_directions: tuple[str, ...] = ("cost", "path", "congestion"),
                 outer_iterations: int = 2, eoh_iterations: int = 1,
                 population_size: int = 3, n_candidates: int = 1,
                 instances_per_generator: int = 1, seed: int = 42,
                 temperature: float = 1.0, candidate_timeout: float = 15,
                 llm_timeout: float = 90, llm_retries: int = 2,
                 utility_config: UtilityConfig | None = None, meta_solver: MetaSolver | None = None,
                 test_instances: list[Any] | None = None,
                 validation_instances: list[Any] | None = None,
                 output_dir: str | Path | None = None,
                 progress_output: bool = True, make_plots: bool = True,
                 instance_bounds: SNDParameterBounds | Mapping[str, Any] | None = None,
                 test_instance_count: int = 3,
                 validation_instance_count: int = 0,
                 eoh_selection: str = "topk", cluster_method: str = "kmeans",
                 min_clusters: int = 2, eoh_framework: str = "eoh",
                 cpi_mode: str = "mean", cs_enabled: bool = True,
                 ls_enabled: bool = True, parallel_workers: int = 4,
                 strategy_pool_limit: int = 8,
                 external_deviation_tolerance: float | None = 1e-3,
                 early_stop_patience: int = 2,
                 stagnation_patience: int = 2,
                 stagnation_restarts: int = 1,
                 response_mode: str = "pure", mixture_support_size: int = 3,
                 mixture_epsilon: float = 0.05,
                 eohs_mixture_selection: str = "cpm",
                 cpi_failure_margin: float = 1.0,
                 minimum_mixture_success: float = 0.0,
                 deployment_selection: str = "cce",
                 reference_method: str = "auto", reference_complexity_threshold: int = 1000,
                 reference_heuristic: str = "randomized_residual_dijkstra",
                 reference_restarts: int = 4, reference_max_paths: int = 16,
                 reference_timeout: float = 15.0,
                 reference_cache_dir: str | Path | None = None,
                 response_workers: int = 1,
                 generator_operators: tuple[str, ...] = EVOLUTION_OPERATOR_CODES) -> None:
        directions = tuple(generator_directions)
        if not 1 <= len(directions) <= 6 or len(set(directions)) != len(directions) or any(d not in DIRECTIONS for d in directions):
            raise ValueError("generator_directions 必须为 1 至 6 个不重复的合法方向")
        for name, value in {"outer_iterations": outer_iterations, "eoh_iterations": eoh_iterations,
                            "population_size": population_size, "n_candidates": n_candidates,
                            "instances_per_generator": instances_per_generator}.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} 必须为正整数")
        if population_size < 2 or type(llm_retries) is not int or llm_retries < 0:
            raise ValueError("population_size 至少 2，llm_retries 为非负整数")
        if type(test_instance_count) is not int or test_instance_count < 1:
            raise ValueError("test_instance_count 必须为正整数")
        if type(validation_instance_count) is not int or validation_instance_count < 0:
            raise ValueError("validation_instance_count 必须为非负整数")
        if eoh_selection not in {"topk", "cluster"}:
            raise ValueError("eoh_selection 必须为 topk 或 cluster")
        if eoh_framework not in {"eoh", "eoh-s"}:
            raise ValueError("eoh_framework must be eoh or eoh-s")
        if response_mode not in {"pure", "mpsro"}:
            raise ValueError("response_mode 必须为 pure 或 mpsro")
        if type(mixture_support_size) is not int or mixture_support_size < 1:
            raise ValueError("mixture_support_size 必须为正整数")
        if not np.isfinite(mixture_epsilon) or mixture_epsilon < 0:
            raise ValueError("mixture_epsilon 必须为非负有限数")
        if eohs_mixture_selection not in {"cpm", "utility"}:
            raise ValueError("eohs_mixture_selection 必须为 cpm 或 utility")
        if not np.isfinite(cpi_failure_margin) or cpi_failure_margin <= 0:
            raise ValueError("cpi_failure_margin 必须为正有限数")
        if cpi_mode not in {"mean", "weighted"}:
            raise ValueError("cpi_mode must be mean or weighted")
        if not cs_enabled and not ls_enabled:
            raise ValueError("at least one of CS and LS must be enabled")
        if type(parallel_workers) is not int or parallel_workers < 1:
            raise ValueError("parallel_workers must be a positive integer")
        if type(response_workers) is not int or response_workers < 1:
            raise ValueError("response_workers 必须为正整数")
        generator_operators = tuple(generator_operators)
        if (not generator_operators or len(set(generator_operators)) != len(generator_operators)
                or any(code not in EVOLUTION_OPERATOR_CODES for code in generator_operators)):
            raise ValueError("generator_operators 必须是 E1/E2/M1/M2/M3 的非空无重复子集")
        if type(strategy_pool_limit) is not int or strategy_pool_limit < 2:
            raise ValueError("strategy_pool_limit 必须是至少 2 的整数")
        if external_deviation_tolerance is not None and (
                not np.isfinite(external_deviation_tolerance) or external_deviation_tolerance < 0):
            raise ValueError("external_deviation_tolerance 必须为非负有限数或 None")
        if type(early_stop_patience) is not int or early_stop_patience < 1:
            raise ValueError("early_stop_patience 必须为正整数")
        if type(stagnation_patience) is not int or stagnation_patience < 1:
            raise ValueError("stagnation_patience 必须为正整数")
        if type(stagnation_restarts) is not int or stagnation_restarts < 0:
            raise ValueError("stagnation_restarts 必须为非负整数")
        if not np.isfinite(minimum_mixture_success) or not 0 <= minimum_mixture_success <= 1:
            raise ValueError("minimum_mixture_success 必须属于 [0,1]")
        if deployment_selection not in {"cce", "validation_gap"}:
            raise ValueError("deployment_selection 必须为 cce 或 validation_gap")
        if cluster_method not in {"kmeans", "agglomerative", "keyword"}:
            raise ValueError("cluster_method 必须为 kmeans、agglomerative 或 keyword")
        if type(min_clusters) is not int or min_clusters < 1:
            raise ValueError("min_clusters 必须为正整数")
        if eoh_selection == "cluster" and min_clusters > population_size:
            raise ValueError("聚类模式下 min_clusters 不能大于 population_size")
        if type(seed) is not int or any(not np.isfinite(x) or x <= 0 for x in (candidate_timeout, llm_timeout, time_limit)):
            raise ValueError("seed 必须是整数，所有时间限制必须为正有限数")
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature 必须为非负有限数")
        self.base = SNDSolver(instance, solver_type, time_limit, mip_gap)._data
        self.reference_config = ReferenceConfig(
            method=reference_method, complexity_threshold=reference_complexity_threshold,
            heuristic=reference_heuristic, restarts=reference_restarts,
            max_paths=reference_max_paths, timeout=reference_timeout, seed=seed)
        validate_instance(self.base)
        if instance_bounds is None:
            self.instance_bounds = None
        elif isinstance(instance_bounds, SNDParameterBounds):
            self.instance_bounds = instance_bounds
        elif isinstance(instance_bounds, Mapping):
            try:
                self.instance_bounds = SNDParameterBounds(**dict(instance_bounds))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"instance_bounds 配置无效：{exc}") from exc
        else:
            raise TypeError("instance_bounds 必须是 SNDParameterBounds、字典或 None")
        # 同一实验中的训练基础网络必须保持静态/时间模式一致；
        # 若 H 可变，展开规模可以变化，但程序输入语义不能混用。
        if self.instance_bounds is not None:
            temporal_base = self.base.get("time_expansion") is not None
            temporal_bounds = self.instance_bounds.time_periods is not None
            if temporal_base != temporal_bounds:
                raise ValueError("基础实例与 instance_bounds 的静态/时间模式必须一致")
        self.directions, self.players = directions, ("solver", *directions)
        self.client = llm_client if llm_client is not None else OfflineMultiAgentClient()
        self.utility = utility_config or UtilityConfig()
        self.meta_solver = meta_solver or CCEMetaSolver()
        self.T, self.R, self.K, self.n = outer_iterations, eoh_iterations, population_size, n_candidates
        self.eoh_selection, self.cluster_method, self.min_clusters = (
            eoh_selection, cluster_method, min_clusters)
        self.eoh_framework = eoh_framework
        self.response_mode = response_mode
        self.mixture_support_size = mixture_support_size
        self.mixture_epsilon = float(mixture_epsilon)
        self.eohs_mixture_selection = eohs_mixture_selection
        self.cpi_failure_margin = float(cpi_failure_margin)
        self._response_candidates: dict[str, list[Program]] = {}
        # 记录最终部署选择是否遵循 CCE，以及是否因基础实例不可行而回退。
        self._last_selection_diagnostic: dict[str, Any] = {}
        self.strategy_pool_limit = strategy_pool_limit
        self.external_deviation_tolerance = external_deviation_tolerance
        self.early_stop_patience = early_stop_patience
        self.stagnation_patience = stagnation_patience
        self.stagnation_restarts = stagnation_restarts
        self.minimum_mixture_success = float(minimum_mixture_success)
        self.deployment_selection = deployment_selection
        self._solver_stagnation = 0
        self._previous_training_cpi: float | None = None
        # 并行度只控制独立候选/实例评估，不改变 LLM 调用和选择顺序。
        self.parallel_workers = parallel_workers
        self.response_workers = response_workers
        self.generator_operators = generator_operators
        self._call_lock = threading.Lock()
        self._next_call_index = 0
        self._search_context = threading.local()
        self._player_rng = {player: random.Random(int(digest([seed, player, "parents"])[:16], 16))
                            for player in self.players}
        self.eohs_config = EoHSConfig(population_size=population_size,
                                      max_iterations=eoh_iterations,
                                       cpi_mode=cpi_mode,
                                       cs_enabled=cs_enabled,
                                       ls_enabled=ls_enabled)
        self.seed, self.rng = seed, random.Random(seed)
        self.temperature, self.llm_timeout, self.retries = temperature, llm_timeout, llm_retries
        self.progress, self.make_plots, self.started = progress_output, make_plots, False
        self._eohs_sets: dict[str, dict[str, Any]] = {}
        self.output_base = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "runs"
        self.reference_args = (solver_type, time_limit, mip_gap, candidate_timeout,
                               instances_per_generator, seed, log_output)
        self.reference_cache_dir = (Path(reference_cache_dir) if reference_cache_dir is not None
                                    else self.output_base / "reference_cache")
        self.test_instance_count = test_instance_count
        if self.instance_bounds is None:
            # 未启用范围时保持原有行为：训练样本均由传入基础实例派生。
            self.training_bases = [self.base]
        else:
            # 使用共同随机数：同一个样本编号在所有生成器玩家中看到同一个网络。
            self.training_bases = [self.instance_bounds.sample(seed=seed + 20000 + i).to_dict()
                                   for i in range(instances_per_generator)]
        if test_instances is None:
            # 留出集只用于报告，不参与 fitness、变异或最终程序选择。
            if self.instance_bounds is None:
                n = len(self.base["nodes"])
                # 时间展开实例的 nodes/arcs 是放大后的集合；测试实例应按物理网络规模采样。
                base_metadata = self.base.get("metadata", {})
                physical_n = int(base_metadata.get("physical_node_count", n))
                physical_arcs = self.base.get("time_expansion", {}).get("physical_arcs", [])
                arc_count = len(physical_arcs) or len(self.base["arcs"])
                density = arc_count / (physical_n * (physical_n - 1))
                adjustment = self.base.get("metadata", {}).get("fixed_cost_adjustment", 0.5)
                horizon = base_metadata.get("time_periods")
                max_travel = int(base_metadata.get("max_travel_time", 2))
                self.tests = [SNDInstance(
                    physical_n, density, adjustment, seed=seed + 10000 + i,
                    time_periods=int(horizon) if horizon is not None else None,
                    max_travel_time=max_travel,
                ).to_dict() for i in range(test_instance_count)]
            else:
                self.tests = [self.instance_bounds.sample(seed=seed + 10000 + i).to_dict()
                              for i in range(test_instance_count)]
        else:
            if not test_instances:
                raise ValueError("test_instances 不能为空")
            self.tests = [SNDSolver._load_instance(x) for x in test_instances]
        for test in self.tests:
            validate_instance(test)
        if validation_instances is not None and not validation_instances:
            raise ValueError("validation_instances 不能为空")
        if validation_instances is not None:
            self.validation = [SNDSolver._load_instance(x) for x in validation_instances]
        elif validation_instance_count:
            if self.instance_bounds is None:
                n = len(self.base["nodes"])
                metadata = self.base.get("metadata", {})
                physical_n = int(metadata.get("physical_node_count", n))
                physical_arcs = self.base.get("time_expansion", {}).get("physical_arcs", [])
                arc_count = len(physical_arcs) or len(self.base["arcs"])
                density = arc_count / (physical_n * (physical_n - 1))
                adjustment = metadata.get("fixed_cost_adjustment", 0.5)
                horizon = metadata.get("time_periods")
                max_travel = int(metadata.get("max_travel_time", 2))
                self.validation = [SNDInstance(
                    physical_n, density, adjustment, seed=seed + 15000 + i,
                    time_periods=int(horizon) if horizon is not None else None,
                    max_travel_time=max_travel,
                ).to_dict() for i in range(validation_instance_count)]
            else:
                self.validation = [self.instance_bounds.sample(seed=seed + 15000 + i).to_dict()
                                   for i in range(validation_instance_count)]
        else:
            self.validation = []
        for item in self.validation:
            validate_instance(item)
            if (item.get("time_expansion") is not None) != (self.base.get("time_expansion") is not None):
                raise ValueError("validation_instances 与基础实例的静态/时间模式必须一致")
        self.validation_instance_count = len(self.validation)
        if self.deployment_selection == "validation_gap" and not self.validation:
            raise ValueError("validation_gap 部署选择需要非空独立验证集")
        self.config = {"seed": seed, "generator_directions": list(directions),
                       "outer_iterations": outer_iterations, "eoh_iterations": eoh_iterations,
                       "population_size": population_size, "n_candidates": n_candidates,
                       "instances_per_generator": instances_per_generator, "temperature": temperature,
                       "candidate_timeout": candidate_timeout, "llm_timeout": llm_timeout,
                       "llm_retries": llm_retries, "solver_type": solver_type, "time_limit": time_limit,
                       "mip_gap": mip_gap, "utility": asdict(self.utility), "client": type(self.client).__name__,
                       "meta_solver": type(self.meta_solver).__name__,
                       "max_profiles": getattr(self.meta_solver, "max_profiles", None),
                       "instance_bounds": (self.instance_bounds.to_dict()
                                           if self.instance_bounds is not None else None),
                       "test_instance_count": test_instance_count,
                       "validation_instance_count": len(self.validation),
                       "eoh_selection": eoh_selection, "cluster_method": cluster_method,
                       "min_clusters": min_clusters,
                       "eoh_framework": eoh_framework, "cpi_mode": cpi_mode,
                       "cs_enabled": cs_enabled, "ls_enabled": ls_enabled,
                       "parallel_workers": parallel_workers,
                       "response_workers": response_workers,
                       "generator_operators": list(generator_operators),
                       "llm_seed_scheme": ("player_iteration_call_v1" if response_workers > 1
                                           else "legacy_global_call_index"),
                       "strategy_pool_limit": strategy_pool_limit,
                       "external_deviation_tolerance": external_deviation_tolerance,
                       "early_stop_patience": early_stop_patience,
                       "stagnation_patience": stagnation_patience,
                       "stagnation_restarts": stagnation_restarts,
                       "minimum_mixture_success": minimum_mixture_success,
                       "deployment_selection": deployment_selection,
                       "response_mode": response_mode,
                       "mixture_support_size": mixture_support_size,
                        "mixture_epsilon": mixture_epsilon,
                        "eohs_mixture_selection": eohs_mixture_selection,
                        "cpi_failure_margin": cpi_failure_margin,
                       "reference": self.reference_config.identity(),
                       "reference_cache_dir": str(self.reference_cache_dir),
                       "llm_model": getattr(self.client, "model", None),
                       "candidate_hash_seed": 0,
                       "reference_random_seed": self.reference_config.seed,
                       "lp_method": "highs", "lp_feasibility_tolerance": 1e-9,
                       "python": platform.python_version(), "platform": platform.platform(),
                       "packages": {name: version(name) for name in ("numpy", "scipy", "matplotlib")}}
        # 记录基础实例是否为时间展开格式，便于复现实验配置和下游分析。
        self.config["base_instance_format"] = self.base.get("format", "static-snd-v2")
        self.config["base_time_periods"] = self.base.get("metadata", {}).get("time_periods")
        self.config["warm_start_policy"] = "training_only_never_export"

    def _report(self, text: str) -> None:
        if self.progress:
            print(text, flush=True)

    def _is_warm_start(self, program: Program) -> bool:
        """Return whether *program* is the solver warm-start baseline.

        Generator baselines are still useful as opponents during training.  The
        hard deployment restriction applies to the solver player only.
        """
        if program.player != "solver":
            return False
        if program.operator == "base":
            return True
        if program.member_ids is not None or not program.code:
            return False
        # Relabelling a copy as I1/LS must not bypass the deployment gate.
        # AST comparison also ignores comments, blank lines and formatting.
        if not hasattr(self, "_warm_start_matches"):
            self._warm_start_matches: dict[str, bool] = {}
        if program.code not in self._warm_start_matches:
            try:
                same = ast.dump(ast.parse(program.code), include_attributes=False) == _WARM_START_AST
            except SyntaxError:
                same = False  # Static validation handles invalid source separately.
            self._warm_start_matches[program.code] = same
        return self._warm_start_matches[program.code]

    def _contains_warm_start(self, program: Program, _seen: set[str] | None = None) -> bool:
        """Return whether a pure, ensemble, or mixed policy contains the baseline."""
        if self._is_warm_start(program):
            return True
        if program.member_ids is None:
            return False
        seen = set() if _seen is None else set(_seen)
        if program.id in seen:
            raise ValueError(f"策略成员存在循环引用，无法验证暖启动限制：{program.id}")
        seen.add(program.id)
        archive = getattr(self, "archive", None)
        index = {item.id: item for item in archive.programs} if archive is not None else {}
        missing = [member_id for member_id in program.member_ids if member_id not in index]
        if not program.member_ids or missing:
            raise ValueError(f"策略成员缺失，无法验证暖启动限制：{program.id}，{missing}")
        return any(self._contains_warm_start(index[member_id], seen)
                   for member_id in program.member_ids)

    def _without_warm_start(self, programs: list[Program]) -> list[Program]:
        """Filter solver deployment candidates while preserving archive order."""
        return [program for program in programs
                if not self._contains_warm_start(program)]

    def _training_candidates(self, programs: list[Program], player: str) -> list[Program]:
        """Retire the solver warm start once K distinct non-baseline sources exist."""
        unique: dict[str, Program] = {}
        for program in programs:
            unique.setdefault(program.code_hash, program)
        candidates = list(unique.values())
        if player != "solver":
            return candidates
        non_warm = self._without_warm_start(candidates)
        if len(non_warm) >= self.K and not getattr(self, "_warm_start_retired", False):
            self._warm_start_retired = True
            self.archive.append("events.jsonl", {
                "event": "warm_start_retired", "population_size": self.K,
                "non_warm_start_ids": [program.id for program in non_warm],
            })
            self._report(f"  solver：非暖启动种群已达到 {self.K}，永久移除本地搜索种群中的暖启动")
        return non_warm if getattr(self, "_warm_start_retired", False) else candidates

    def _write_selection_status(self, *, error: str, pools: dict[str, list[Program]] | None = None,
                                candidates: list[Program] | None = None) -> None:
        """Persist a failure diagnosis before refusing a warm-start deployment."""
        root = getattr(self, "root", None)
        if root is None:
            return
        pool_rows = pools or {}
        rows = candidates if candidates is not None else pool_rows.get("solver", [])
        non_warm = self._without_warm_start(rows)
        archive = getattr(self, "archive", None)
        archived = [program for program in getattr(archive, "programs", [])
                    if program.player == "solver"]
        payload = {
            "status": "rejected",
            "reason": error,
            "warm_start_forbidden": True,
            "solver_pool_ids": [program.id for program in pool_rows.get("solver", [])],
            "solver_pool_operators": [program.operator for program in pool_rows.get("solver", [])],
            "non_warm_start_candidate_ids": [program.id for program in non_warm],
            "non_warm_start_candidate_count": len(non_warm),
            "warm_start_retired": getattr(self, "_warm_start_retired", False),
            "excluded_warm_start_ids": [program.id for program in archived
                                        if self._contains_warm_start(program)],
            "archive_solver_ids": [program.id for program in archived],
            "rejected_candidates": [{"id": program.id, "error": program.error}
                                    for program in archived if program.error],
            "deployment_selection": dict(self._last_selection_diagnostic),
        }
        evaluator = getattr(self, "evaluator", None)
        if evaluator is not None:
            # 其他主体可能仍在评估；先取快照，避免诊断迭代被并发插入打断。
            with getattr(evaluator, "_cache_lock", threading.RLock()):
                evaluation_rows = list(evaluator.evaluations.items())
        else:
            evaluation_rows = []
        payload["failed_evaluations"] = [
            {"code_hash": key[0], "instance_hash": key[1],
             "failure_type": row.get("failure_type"), "error": row.get("error")}
            for key, row in evaluation_rows
            if row.get("status") != "ok"
        ]
        try:
            write_json(root / "selection_status.json", payload)
        except OSError as exc:
            self._report(f"无法写入 selection_status.json：{exc}")

    def _tables(self, pools: dict[str, list[Program]]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """每条星形边保留两张不同收益表；求解器贡献除以 N。"""
        a, b = [], []
        for direction in self.directions:
            def make_row(solver: Program) -> list[tuple[float, float]]:
                return [self.evaluator.pair(solver, generator) for generator in pools[direction]]
            rows = ordered_map(make_row, pools["solver"], self.parallel_workers)
            pairs = np.array(rows)
            a.append(pairs[:, :, 0] / len(self.directions))
            b.append(pairs[:, :, 1])
        return a, b

    def _fitness(self, program: Program, pools: dict, meta: dict) -> float:
        """固定单方偏离只需对手边际；当前联合期望收益仍须使用 CCE 联合分布。"""
        if program.player == "solver":
            return sum(weight * self.evaluator.pair(program, g)[0] / len(self.directions)
                       for j, d in enumerate(self.directions, 1)
                       for weight, g in zip(meta["marginals"][j], pools[d]))
        return sum(weight * self.evaluator.pair(s, program)[1]
                   for weight, s in zip(meta["marginals"][0], pools["solver"]))

    def _opponents(self, player: str, pools: dict, meta: dict) -> dict:
        names = self.directions if player == "solver" else ("solver",)
        archive_index = {p.id: p for p in self.archive.programs}
        def describe(p: Program, weight: float) -> dict:
            row = {"id": p.id, "probability": weight, "explanation": p.explanation,
                   "code": p.code}
            if p.member_ids is not None:
                row["member_ids"] = p.member_ids
                row["mixture_weights"] = p.mixture_weights
                row["member_codes"] = {member_id: archive_index[member_id].code
                                       for member_id in p.member_ids}
            return row
        return {name: [describe(p, weight)
                       for p, weight in zip(pools[name], meta["marginals"][self.players.index(name)])]
                for name in names}

    def _generate(self, player: str, operator: str, parents: list[Program], t: int, r: int,
                  pools: dict, meta: dict, eoh_s_context: dict | None = None) -> list[Program]:
        if not hasattr(self, "timing"):
            self.timing = TimingRecorder(self.archive)
        prompt = build_prompt(player, operator, parents, self.user_prompt, self.utility,
                              self._opponents(player, pools, meta), eoh_s_context,
                              temporal=self.base.get("time_expansion") is not None)
        responses = []
        received: list[tuple[int, str]] = []
        for attempt in range(self.retries + 1):
            with self._call_lock:
                call_index = self._next_call_index
                self._next_call_index += 1
            if self.response_workers > 1:
                sequence = getattr(self._search_context, "call_index", 0)
                self._search_context.call_index = sequence + 1
                # 不依赖线程抢占顺序，供 ReplayClient 按 seed 精确定位响应。
                seed = int(digest([self.seed, player, t, sequence])[:8], 16)
            else:
                seed = self.seed + call_index
            call = {"index": call_index, "player": player, "iteration": t,
                    "eoh_iteration": r, "operator": operator, "attempt": attempt, "seed": seed,
                    "prompt": prompt, "temperature": self.temperature, "n": self.n,
                    "model": getattr(self.client, "model", None)}
            started = time.perf_counter()
            # 请求发出前先写日志，意外中断也能追溯调用意图。
            self.archive.append("llm_requests.jsonl", call)
            try:
                with self.timing.measure("llm", player=player, operator=operator, call_index=call_index):
                    output = isolated_call("llm", (self.client, prompt, self.temperature, self.n, seed), self.llm_timeout)
                responses = output["result"]
                call["raw_outputs"] = responses
                # 返回数量不符也保留已有候选，重试不能把中间程序丢掉。
                texts = responses if isinstance(responses, list) else [responses]
                received.extend((call["index"], x) for x in texts if isinstance(x, str))
                if not isinstance(responses, list) or len(responses) != self.n or any(not isinstance(x, str) for x in responses):
                    raise ValueError("generate 必须返回 n 个字符串组成的列表")
                call["status"] = "ok"
            except Exception as exc:
                error = str(exc)
                for key in ("LLM_API_KEY", "API_KEY", "OPENAI_API_KEY"):
                    secret = os.environ.get(key)
                    if secret:
                        error = error.replace(secret, "[REDACTED]")
                call.update(status="failed", error=error)
                responses = []
            call["elapsed"] = time.perf_counter() - started
            with self._call_lock:
                self.archive.calls.append(call)
            self.archive.append("llm_calls.jsonl", call)
            if call["status"] == "ok":
                break
            # 子进程中的 API 异常以前只存入日志；在父进程显示已脱敏的真实原因，
            # 避免终端仅出现 LiteLLM 的通用反馈链接，也不把失败误报成生成成功。
            self._report(f"  LLM 调用失败 [{player}/{operator}，尝试 {attempt + 1}/{self.retries + 1}]："
                         f"{call['error']}")
            if attempt == self.retries:
                self._report(f"  本次请求重试已耗尽，将继续使用已有候选；详情：{self.root / 'llm_calls.jsonl'}")
        candidates = []
        signature = "solve_snd(instance)" if player == "solver" else "generate_snd(base_instance, seed)"
        for call_index, response in received:
            explanation, code, error = "", "", None
            try:
                explanation = CodeParser._extract_explanation(response)
                code = CodeParser._extract_code(response)
                validate_code(code, signature,
                              max_code_lines=self.utility.max_code_lines,
                              max_code_chars=self.utility.max_code_chars,
                              max_ast_nodes=self.utility.max_ast_nodes)
            except Exception as exc:
                error = str(exc)
                # 解释缺失或 AST 非法也保存能提取到的源码；原始响应另有完整日志。
                if not code:
                    try:
                        code = CodeParser._extract_code(response)
                    except ValueError:
                        pass
            program = self.archive.add(player=player, role="solver" if player == "solver" else "generator",
                                       explanation=explanation, code=code, parents=parents,
                                       iteration=t, eoh_iteration=r, operator=operator, error=error)
            self.archive.append("events.jsonl", {"event": "generated", "program_id": program.id,
                                                  "call_index": call_index})
            if not error:
                candidates.append(program)
        return candidates

    def _eohs_scenarios(self, pools: dict, meta: dict) -> tuple[list[dict], list[float], list[dict]]:
        """基础网络与当前生成器的合法难例组成共同训练集。"""
        scenarios = list(self.training_bases)
        weights = [1.0] * len(scenarios)
        labels = [{"source": "base", "hash": digest(data)} for data in scenarios]
        seen = {digest(data) for data in scenarios}
        self.evaluator.program_index.update({p.id: p for p in self.archive.programs})
        for j, direction in enumerate(self.directions, 1):
            for probability, generator in zip(meta["marginals"][j], pools[direction]):
                # M-PSRO 生成器的各纯程序分别提供难例；不能只取一次随机样本。
                support = ([self.evaluator.program_index[mid] for mid in generator.member_ids]
                           if generator.mixture_weights is not None else [generator])
                support_weights = (generator.mixture_weights if generator.mixture_weights is not None
                                   else [1.0])
                for member, member_weight in zip(support, support_weights):
                    for index in range(self.evaluator.samples):
                        sample = self.evaluator.sample(member, index)
                        if sample["status"] != "ok":
                            continue
                        data = sample["instance"]
                        key = digest(data)
                        if key in seen:
                            continue
                        seen.add(key)
                        scenarios.append(data)
                        # weighted 模式按生成器的外部与内部概率提高活跃难例权重。
                        weights.append(1.0 + float(probability) * member_weight
                                       if self.eohs_config.cpi_mode == "weighted" else 1.0)
                        labels.append({"source": direction, "generator_id": generator.id,
                                       "member_id": member.id,
                                       "sample": index, "hash": key})
        return scenarios, weights, labels

    def _eohs_vectors(self, population: list[Program], scenarios: list[dict]) -> dict[str, list[float]]:
        """在固定难例集上记录各求解程序的逐实例 gap 向量。"""
        def evaluate_one(program: Program) -> tuple[str, list[float], list[str | None]]:
            failures: list[str | None] = []
            vector = performance_vector(program, scenarios, self.evaluator.evaluate, failures)
            return program.code_hash, vector, failures

        rows = ordered_map(evaluate_one, population, self.parallel_workers)
        self._eohs_failure_types = {key: types for key, _, types in rows}
        return {key: vector for key, vector, _ in rows}

    def _cpi_vectors(self, population: list[Program], scenarios: list[dict]) -> dict[str, list[float]]:
        """把当前候选集的失败值按每个实例统一映射为有限惩罚。"""
        vectors = self._eohs_vectors(population, scenarios)
        return finite_failure_vectors(vectors, self.cpi_failure_margin,
                                      getattr(self, "_eohs_failure_types", None))

    def _search_eohs(self, player: str, t: int, pools: dict, meta: dict) -> Program:
        """EoH-S：CS/LS 生成候选，CPM 保留互补集合。"""
        if player != "solver":
            return self._search_eoh_original(player, t, pools, meta)
        population = list(getattr(self, "_eohs_population", []))
        # 长时间没有降低 CPI 时重启本地种群，避免 CS/LS 反复围绕同一批程序搜索。
        # 重启只影响下一轮的求解器响应，不修改已归档程序和元博弈历史。
        if (population and self._solver_stagnation >= self.stagnation_patience
                and getattr(self, "_solver_restarts", 0) < self.stagnation_restarts):
            self._solver_restarts = getattr(self, "_solver_restarts", 0) + 1
            self.archive.append("events.jsonl", {"event": "eoh_s_restart",
                                                  "iteration": t,
                                                  "reason": "cpi_stagnation",
                                                  "stagnation_rounds": self._solver_stagnation})
            self._report(f"  solver: CPI 连续 {self._solver_stagnation} 轮无改善，重启 EoH-S 种群")
            population = []
            self._solver_stagnation = 0
        if not population:
            # 重启从历史响应的纯成员恢复；已退役的暖启动会在 CPM 前过滤。
            index = {p.id: p for p in self.archive.programs}
            population = [index[mid] for policy in pools[player]
                          for mid in (policy.member_ids or [policy.id])
                          if index[mid].member_ids is None]
        scenarios, weights, labels = self._eohs_scenarios(pools, meta)
        selected_weights = weights if self.eohs_config.cpi_mode == "weighted" else None

        def select_population(candidates: list[Program], vectors: dict[str, list[float]]):
            # The baseline may temporarily fill an under-populated search pool,
            # but once K genuine candidates exist it is removed before CPM.
            eligible = self._training_candidates(candidates, player)
            selected, diagnostic = cpm_select(eligible, vectors, self.K, selected_weights)
            diagnostic["warm_start_candidates"] = sum(
                1 for candidate in candidates if self._contains_warm_start(candidate))
            diagnostic["warm_start_excluded"] = sum(
                1 for candidate in candidates if candidate not in eligible)
            return selected, diagnostic

        initial = self._generate(player, "I1", [], t, 0, pools, meta)
        population.extend(initial)
        if not population:
            raise RuntimeError("EoH-S 无可用求解器程序；请检查 LLM 调用与候选校验日志")
        vectors = self._cpi_vectors(population, scenarios)
        population, diagnostic = select_population(population, vectors)
        self.archive.append("events.jsonl", {"event": "eoh_s_selection", "player": player,
                                              "iteration": t, "selection": diagnostic})
        for r in range(1, self.R + 1):
            vectors = self._cpi_vectors(population, scenarios)
            generated: list[Program] = []
            if self.eohs_config.cs_enabled and len(population) >= 2:
                first, second = select_complementary_parents(population, vectors)
                context = {"strategy": "CS", "cpi": eohs_cpi(vectors, [p.code_hash for p in population], selected_weights),
                           "parent_a_vector": vectors[first.code_hash],
                           "parent_b_vector": vectors[second.code_hash]}
                generated.extend(self._generate(player, "CS", [first, second], t, r,
                                                pools, meta, context))
            if self.eohs_config.ls_enabled and population:
                parent = select_local_parent(population, vectors)
                context = {"strategy": "LS", "cpi": eohs_cpi(vectors, [p.code_hash for p in population], selected_weights),
                           "parent_vector": vectors[parent.code_hash]}
                generated.extend(self._generate(player, "LS", [parent], t, r,
                                                pools, meta, context))
            all_candidates = population + generated
            all_vectors = self._cpi_vectors(all_candidates, scenarios)
            population, diagnostic = select_population(all_candidates, all_vectors)
            self.archive.append("events.jsonl", {"event": "eoh_s_selection", "player": player,
                                                  "iteration": t, "eoh_iteration": r,
                                                  "selection": diagnostic})
            self._report(f"  {player}: EoH-S {r}/{self.R} complete, CPI={diagnostic['cpi']:.6f}")
        if player == "solver":
            original_population = list(population)
            population = self._without_warm_start(population)
            if not population:
                message = ("EoH-S 未生成可部署的非暖启动求解器程序；"
                           "暖启动仅用于初始化和训练诊断，禁止作为最终输出")
                self._write_selection_status(error=message, pools=pools, candidates=original_population)
                raise RuntimeError(message)
        final_vectors = self._cpi_vectors(population, scenarios)
        final_cpi = eohs_cpi(final_vectors, [p.code_hash for p in population], selected_weights)
        # 生成器难例集合会随轮次变化，不能直接用它比较停滞；用固定的基础训练
        # 子集作为可比诊断，同时保留包含难例的完整 CPI 供 EoH-S 报告。
        base_count = sum(label.get("source") == "base" for label in labels)
        base_vectors = {key: row[:base_count] for key, row in final_vectors.items()}
        base_weights = (selected_weights[:base_count] if selected_weights is not None else None)
        base_cpi = eohs_cpi(base_vectors, [p.code_hash for p in population], base_weights)
        if self._previous_training_cpi is not None:
            self._solver_stagnation = (self._solver_stagnation + 1
                                       if self._previous_training_cpi - base_cpi <= 1e-9 else 0)
        self._previous_training_cpi = base_cpi
        self._eohs_population = population
        self._response_candidates[player] = list(population)
        if self.response_mode == "mpsro":
            self._eohs_last = {"player": player, "iteration": t,
                               "selected_ids": [p.id for p in population],
                               "cpi": final_cpi, "base_cpi": base_cpi,
                               "performance_vectors": final_vectors,
                               "scenarios": labels, "weights": weights}
            self._eohs_sets[player] = self._eohs_last
            return max(population, key=lambda p: self._fitness(p, pools, meta))
        ensemble = self.archive.add(player="solver", role="solver",
            explanation="CPM 互补集合；运行全部成员并选择最低目标值的可行解",
            code="", parents=population, iteration=t, eoh_iteration=self.R,
            operator="CPM", members=population)
        self._eohs_last = {"player": player, "iteration": t,
                           "selected_ids": [p.id for p in population],
                           "ensemble_id": ensemble.id,
                            "cpi": final_cpi, "base_cpi": base_cpi,
                           "performance_vectors": final_vectors,
                           "scenarios": labels, "weights": weights}
        self._eohs_sets[player] = self._eohs_last
        return ensemble

    def _search_eoh_original(self, player: str, t: int, pools: dict, meta: dict) -> Program:
        """原有 EoH 搜索实现，保留 Top-K/文本聚类兼容行为。"""
        # 混合响应不能当作变异父代；取出其纯程序支持集继续进化。
        archive_index = {p.id: p for p in self.archive.programs}
        population = [archive_index[mid] for policy in pools[player]
                      for mid in (policy.member_ids if policy.mixture_weights is not None
                                  else [policy.id]) if archive_index[mid].member_ids is None]
        scores: dict[str, float] = {}

        def rank(candidates: list[Program]) -> list[Program]:
            unique = {}
            for p in candidates:
                if p.code_hash not in scores:
                    scores[p.code_hash] = self._fitness(p, pools, meta)
                self.archive.append("events.jsonl", {"event": "fitness", "program_id": p.id,
                                                      "iteration": t, "fitness": scores[p.code_hash]})
                unique.setdefault(p.code_hash, p)
            candidate_values = list(unique.values())
            eligible = self._training_candidates(candidate_values, player)
            selected, diagnostic = select_representatives(
                eligible, scores, self.K,
                mode=self.eoh_selection, method=self.cluster_method,
                min_clusters=self.min_clusters)
            diagnostic["warm_start_candidates"] = sum(
                1 for candidate in candidate_values if self._contains_warm_start(candidate))
            diagnostic["warm_start_excluded"] = sum(
                1 for candidate in candidate_values if candidate not in eligible)
            self.archive.append("events.jsonl", {
                "event": "population_selection", "player": player, "iteration": t,
                "selection": diagnostic,
            })
            return selected

        population = rank(population + self._generate(player, "I1", [], t, 0, pools, meta))
        for r in range(1, self.R + 1):
            operators = self.generator_operators if player != "solver" else EVOLUTION_OPERATOR_CODES
            for operator in operators:
                count = OPERATOR_TEMPLATES[operator].parent_count
                if len(population) < count:
                    self.archive.append("events.jsonl", {"event": "skipped_operator", "player": player,
                        "iteration": t, "eoh_iteration": r, "operator": operator, "reason": "insufficient_distinct_parents"})
                    continue
                rng = self._player_rng[player] if self.response_workers > 1 else self.rng
                parents = rng.sample(population, count)
                population = rank(population + self._generate(player, operator, parents, t, r, pools, meta))
            self._report(f"  {player}：EoH {r}/{self.R} 完成，最佳效用 {scores[population[0].code_hash]:.6f}")
        # M-PSRO 从本地候选池筛选混合支持集；原模式仍只取首名。
        if player == "solver":
            original_population = list(population)
            population = self._without_warm_start(population)
            if not population:
                message = ("EoH 未生成可部署的非暖启动求解器程序；"
                           "暖启动仅用于初始化和训练诊断，禁止作为最终输出")
                self._write_selection_status(error=message, pools=pools, candidates=original_population)
                raise RuntimeError(message)
        self._response_candidates[player] = list(population)
        return population[0]

    def _search(self, player: str, t: int, pools: dict, meta: dict) -> Program:
        best = (self._search_eohs(player, t, pools, meta) if self.eoh_framework == "eoh-s"
                else self._search_eoh_original(player, t, pools, meta))
        if self.response_mode != "mpsro":
            return best
        # M-PSRO 扩池单位为纯程序概率混合；EoH-S 默认保留 CPM 的互补选择。
        candidates = {p.code_hash: p for p in self._response_candidates[player]
                      if p.member_ids is None and p.error is None
                      and not (player == "solver" and self._contains_warm_start(p))}
        if not candidates:
            message = (f"{player} 的 M-PSRO 响应池中没有可评估的非暖启动纯程序"
                       if player == "solver" else
                       f"{player} 的 M-PSRO 响应池中没有可评估的纯程序")
            self._write_selection_status(error=message, pools=pools,
                                         candidates=self._response_candidates[player])
            raise RuntimeError(message)
        scored = sorted(((self._fitness(p, pools, meta), p) for p in candidates.values()),
                        key=lambda item: (-item[0], item[1].id))
        top = scored[0][0]
        if self.eoh_framework == "eoh-s" and player == "solver" and self.eohs_mixture_selection == "cpm":
            # CPM 已根据实例级 CPI 排序。容差筛选会删去补短板的成员，
            # 因而该模式保留前 m 个 CPM 代表；m<K 时须明确记录截断。
            # CPM 已经构造了互补集合；保留完整集合，避免丢失补短板成员。
            members = list(candidates.values())
            selection_rule = "cpm_complementary"
        else:
            members = [p for value, p in scored
                       if top - value <= self.mixture_epsilon + 1e-12][:self.mixture_support_size]
            selection_rule = "utility_epsilon"
        weights = [1.0 / len(members)] * len(members)
        mixture = self.archive.add(player=player, role="solver" if player == "solver" else "generator",
            explanation=f"M-PSRO 等权混合响应：{len(members)} 个程序；"
                        f"选择规则 {selection_rule}，本轮最佳效用 {top:.6f}",
            code="", parents=members, iteration=t, eoh_iteration=self.R,
            operator="MPSRO", members=members, mixture_weights=weights)
        self.archive.append("events.jsonl", {"event": "mixed_response", "player": player,
            "iteration": t, "strategy_id": mixture.id, "member_ids": mixture.member_ids,
             "weights": weights, "member_utilities": [value for value, p in scored if p in members],
             "epsilon": self.mixture_epsilon, "selection_rule": selection_rule,
             "cpm_member_count": len(candidates) if selection_rule == "cpm_complementary" else None,
             "cpm_truncated": False})
        return mixture

    def _search_responses(self, t: int, pools: dict, meta: dict) -> dict:
        """同一旧快照上响应，全部完成后才扩池；可选受限主体并发。"""
        def search(player):
            self._report(f"  {player}：开始响应搜索（外层 {t}/{self.T}）")
            with self.timing.measure("response_search", player=player, iteration=t):
                if self.response_workers == 1:
                    return self._search(player, t, pools, meta)
                self._search_context.call_index = 0
                with self.archive.scope(f"t{t:03d}_{player}"):
                    return self._search(player, t, pools, meta)
        return dict(zip(self.players, ordered_map(search, self.players, self.response_workers)))

    def _select(self, pools: dict, meta: dict, *, allow_warm_start: bool = False) -> Program:
        # 部署选择不使用留出测试集；先保证当前基础实例可行。
        choices = (list(pools["solver"]) if allow_warm_start
                   else self._without_warm_start(list(pools["solver"])))
        all_choices = list(choices)
        if not choices and not allow_warm_start:
            message = ("训练未产生可部署的非暖启动求解器程序；"
                       "暖启动仅用于初始化和训练兜底，禁止作为最终输出。"
                       "请检查 LLM 候选、代码校验、EoH 筛选和验证门槛。")
            self._write_selection_status(error=message, pools=pools,
                                         candidates=pools["solver"])
            raise RuntimeError(message)
        if self.response_mode == "mpsro":
            # M-PSRO 优先导出混合响应；门槛未通过时允许降级到纯程序，
            # 并在 deployment_selection 明确记录，避免误称为混合部署。
            mixed = [p for p in choices if p.mixture_weights is not None
                     and (allow_warm_start or not self._contains_warm_start(p))]
            # cce 模式严格遵循混合响应的 CCE 边际；验证模式允许把纯成员
            # 作为候选，直接选择验证 Gap 最低且可靠的最终程序。
            if mixed and self.deployment_selection == "cce":
                choices = mixed
        elif self.eoh_framework == "eoh-s":
            ensembles = [p for p in choices if p.member_ids is not None
                         and (allow_warm_start or not self._contains_warm_start(p))]
            if ensembles:
                choices = ensembles
        # M-PSRO 的 CCE 边际在可部署的混合响应子集中比较；不能再按收益
        # 替代该边际，否则 selected_solver 与 meta_strategy 会不一致。
        marginal = {p.id: float(probability)
                    for p, probability in zip(pools["solver"], meta.get("marginals", [[]])[0])}
        cce_preferred = None
        cce_global_preferred = None
        if self.response_mode == "mpsro" and any(p.mixture_weights is not None for p in choices):
            if meta.get("marginals"):
                index = int(np.argmax(np.asarray(meta["marginals"][0], dtype=float)))
                if index < len(pools["solver"]):
                    preferred = pools["solver"][index]
                    cce_global_preferred = preferred.id
                    if preferred in choices:
                        cce_preferred = preferred.id
                    else:
                        cce_preferred = max(choices, key=lambda p: (marginal.get(p.id, 0.0), p.id)).id
            if self.deployment_selection == "validation_gap" and self.validation:
                def validation_key(p: Program) -> tuple[float, float, str]:
                    rows = [self.evaluator.evaluate(p, data) for data in self.validation]
                    gaps = [float(row["gap"]) for row in rows if row["status"] == "ok"]
                    return (sum(gaps) / len(gaps) if len(gaps) == len(rows) else float("inf"),
                            -marginal.get(p.id, 0.0), p.id)
                ranked = sorted(choices, key=validation_key)
            else:
                ranked = sorted(choices, key=lambda p: (-marginal.get(p.id, 0.0),
                                                         -self._fitness(p, pools, meta), p.id))
        else:
            if self.deployment_selection == "validation_gap" and self.validation:
                def validation_key(p: Program) -> tuple[float, float, str]:
                    rows = [self.evaluator.evaluate(p, data) for data in self.validation]
                    gaps = [float(row["gap"]) for row in rows if row["status"] == "ok"]
                    return (sum(gaps) / len(gaps) if len(gaps) == len(rows) else float("inf"),
                            -self._fitness(p, pools, meta), p.id)
                ranked = sorted(choices, key=validation_key)
            else:
                ranked = sorted(choices, key=lambda p: (-self._fitness(p, pools, meta), p.id))
        feasibility = {p.id: self.evaluator.evaluate(p, self.base)["status"] == "ok"
                       for p in ranked}
        mixture_reliability: dict[str, float | None] = {}
        if self.response_mode == "mpsro":
            # 混合响应的“有一个成员成功”会掩盖脆弱成员；这里以基础实例和
            # 独立验证集上的最小成员成功概率作门槛，避免 91.25% 这类混合可靠性。
            reliability_instances = [self.base, *self.validation]
            for p in ranked:
                if p.mixture_weights is None:
                    continue
                values = [self.evaluator.mixture_distribution(p, data)["expected_success_rate"]
                          for data in reliability_instances]
                mixture_reliability[p.id] = min(values) if values else None
                if (mixture_reliability[p.id] is not None
                        and mixture_reliability[p.id] + 1e-12 < self.minimum_mixture_success):
                    feasibility[p.id] = False
        if self.validation:
            # validation 只用于最终部署候选的稳健性门槛，不参与训练收益矩阵。
            for p in ranked:
                if feasibility[p.id]:
                    feasibility[p.id] = all(
                        self.evaluator.evaluate(p, data)["status"] == "ok"
                        for data in self.validation
                    )
        feasible = next((p for p in ranked if feasibility[p.id]), None)
        if feasible is not None:
            self._last_selection_diagnostic = {
                "rule": ("validation_gap_then_feasibility" if self.deployment_selection == "validation_gap"
                          else "cce_marginal_then_feasibility" if cce_preferred else "utility"),
                "cce_global_preferred_id": cce_global_preferred,
                "cce_preferred_id": cce_preferred,
                "selected_id": feasible.id,
                "fallback_from_cce": cce_preferred is not None and feasible.id != cce_preferred,
                "feasible_ids": [p.id for p in ranked if feasibility[p.id]],
                "mixture_reliability": mixture_reliability,
                "minimum_mixture_success": self.minimum_mixture_success,
            }
            return feasible
        # 若混合响应均未过门槛，优先在纯策略里找通过基础与验证实例者；
        # 不能把“不满足混合可靠性门槛”悄悄当成已通过。
        fallback = next((p for p in all_choices
                         if p.mixture_weights is None
                         and self.evaluator.evaluate(p, self.base)["status"] == "ok"
                         and all(self.evaluator.evaluate(p, data)["status"] == "ok"
                                 for data in self.validation)), None)
        if fallback is None and allow_warm_start:
            # Checkpoints may display the incumbent baseline; solve() never
            # calls this permissive path for its final deployment choice.
            fallback = ranked[0]
        if fallback is None:
            message = ("没有满足部署门槛的非暖启动求解器程序；"
                       "拒绝回退到暖启动，训练结果未生成可部署最终程序。")
            self._write_selection_status(error=message, pools=pools,
                                         candidates=choices)
            raise RuntimeError(message)
        self._last_selection_diagnostic = {
            "rule": ("validation_gap_then_feasibility" if self.deployment_selection == "validation_gap"
                      else "cce_marginal_then_feasibility"),
            "cce_global_preferred_id": cce_global_preferred,
            "cce_preferred_id": cce_preferred,
            "selected_id": fallback.id,
            "fallback_from_cce": cce_preferred is not None and fallback.id != cce_preferred,
            "feasible_ids": [],
            "mixture_reliability": mixture_reliability,
            "minimum_mixture_success": self.minimum_mixture_success,
            "threshold_unmet": True,
        }
        return fallback

    def _prune_pools(self, pools: dict, meta: dict, responses: dict) -> None:
        """保留新响应和旧池中概率最高的策略，限制CCE联合动作数。"""
        maximum = getattr(self.meta_solver, "max_profiles", None)
        cap = self.strategy_pool_limit
        if maximum is not None:
            cap = min(cap, max(1, int(maximum ** (1 / len(self.players)))))
        if cap < 2:
            raise ValueError("max_profiles 太小，至少需容纳每个主体两个策略")
        for i, player in enumerate(self.players):
            old = pools[player]
            response = responses[player]
            if len(old) < cap or response.code_hash in {p.code_hash for p in old}:
                continue
            ranked = sorted(zip(meta["marginals"][i], old), key=lambda item: (-item[0], item[1].id))
            pools[player] = [p for _, p in ranked[:cap - 1]]
            self.archive.append("events.jsonl", {"event": "strategy_pruning", "player": player,
                                                  "retained_ids": [p.id for p in pools[player]],
                                                  "new_response_id": response.id})

    def _snapshot(self, t: int, pools: dict, a: list, b: list, meta: dict, deviations: list,
                   elapsed: float) -> dict:
        # Intermediate checkpoints may display the incumbent warm start before
        # the first generated response exists; final solve() selection remains
        # strict and never permits this path.
        selected = self._select(pools, meta, allow_warm_start=True)
        # 每轮只看训练/验证诊断；test_instances 是最终一次性报告集，避免反复查看造成泄漏。
        validation = [self.evaluator.evaluate(selected, data) for data in self.validation]
        distribution = ([self.evaluator.mixture_distribution(selected, data) for data in self.validation]
                        if selected.mixture_weights is not None else None)
        validation_summary = summarize_evaluations(validation)
        result = {"iteration": t, "pools": {p: [x.id for x in pool] for p, pool in pools.items()},
                  "solver_tables": [x.tolist() for x in a], "generator_tables": [x.tolist() for x in b],
                  "meta": meta, "external_deviation_gains": deviations,
                  "empirical_exploitability": sum(deviations), "selected_solver_id": selected.id,
                  "deployment_selection": dict(self._last_selection_diagnostic),
                  "validation_evaluations": validation,
                  "validation_summary": validation_summary,
                  "validation_mixture_distribution": distribution,
                  # 保留字段名以兼容旧可视化/分析脚本；每轮不读取测试集。
                  "test_evaluations": None, "test_success_rate": None,
                  "test_mean_gap": None, "test_mean_gap_successful": None,
                  "test_mixture_distribution": None,
                  "test_expected_success_rate": None, "test_expected_gap": None,
                   "elapsed": elapsed}
        write_json(self.root / "checkpoints" / f"iteration_{t:03d}.json", result)
        return result

    def solve(self, user_prompt: str) -> MultiAgentResult:
        """训练、评估、导出类、保存全程序谱系和绘图；每次必须传入用户提示词。"""
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError("每次运行必须提供非空 user_prompt")
        if self.started:
            raise RuntimeError("一个对象只运行一次；请新建 MultiAgentSNDSolver")
        self.started, self.user_prompt = True, user_prompt
        started = time.perf_counter()
        self.output_base.mkdir(parents=True, exist_ok=True)
        # 采用继承项目权限的普通目录；Windows tempfile 的私有 ACL 会妨碍 IDE 读取。
        # UUID 只区分文件夹，不参与随机数、程序 ID、评估或元策略。
        self.root = (self.output_base / (datetime.now().strftime("multi_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8])).resolve()
        self.root.mkdir(exist_ok=False)
        self.archive = ProgramArchive(self.root)
        self.timing = TimingRecorder(self.archive)
        # 使用本模块默认元求解器时，先检查最坏情况下的联合策略规模。
        predicted = min(self.T + 1, self.strategy_pool_limit) ** len(self.players)
        maximum = getattr(self.meta_solver, "max_profiles", None)
        if maximum is not None and predicted > maximum:
            self._report(f"提示：若每轮均增加新策略，联合动作数可达 {predicted}，超过 {maximum}；"
                         "请减小外层轮数或替换元求解器。达到上限时会保留档案并明确报错。")
        self.evaluator = SNDEvaluator(self.base, self.archive, self.utility, *self.reference_args,
                                      training_bases=self.training_bases,
                                      reference_config=self.reference_config,
                                      reference_cache_dir=self.reference_cache_dir,
                                      parallel_workers=self.parallel_workers, timing=self.timing)
        self._report(f"程序启动：1 个求解器 + {len(self.directions)} 个生成器\n输出目录：{self.root}")
        if getattr(self.client, "model", None):
            self._report(f"LLM 模型：{self.client.model}")
        write_json(self.root / "config.json", {**self.config, "user_prompt": user_prompt})
        write_json(self.root / "base_instance.json", self.base)
        write_json(self.root / "training_instances.json", self.training_bases)
        write_json(self.root / "validation_instances.json", self.validation)
        write_json(self.root / "test_instances.json", self.tests)
        # 初始化之前确认参考数据可解；缺失依赖或参考解不确定时明确失败。
        self._report(f"准备基础与 {len(self.tests)} 个测试实例的参考解（评估并发 {self.parallel_workers}）")
        with self.timing.measure("reference_preparation"):
            ordered_map(self.evaluator.reference, [self.base, *self.tests], self.parallel_workers)
        pools = {}
        for player in self.players:
            pools[player] = [self.archive.add(player=player, role="solver" if player == "solver" else "generator",
                explanation="残余容量最短路暖启动（不保证任意网络成功）" if player == "solver" else "不扰动基础实例的生成器暖启动",
                code=BASE_SOLVER if player == "solver" else BASE_GENERATOR, parents=[], iteration=0,
                eoh_iteration=0, operator="base")]
        with self.timing.measure("payoff_tables", iteration=0):
            a, b = self._tables(pools)
        with self.timing.measure("cce", iteration=0):
            meta = self.meta_solver.solve(a, b)
        with self.timing.measure("validation_selection", iteration=0):
            history = [self._snapshot(0, pools, a, b, meta, [0.0] * len(self.players), time.perf_counter()-started)]
        stable_rounds = 0
        for t in range(1, self.T + 1):
            # 所有玩家对同一个旧快照计算响应，避免顺序更新造成玩家次序偏差。
            with self.timing.measure("responses", iteration=t):
                responses = self._search_responses(t, pools, meta)
            deviations = [max(0.0, self._fitness(responses[p], pools, meta) - meta["expected_utilities"][i])
                          for i, p in enumerate(self.players)]
            self._prune_pools(pools, meta, responses)
            for player, response in responses.items():
                if response.code_hash not in {p.code_hash for p in pools[player]}:
                    pools[player].append(response)
                self.archive.append("events.jsonl", {"event": "pool_selection", "iteration": t,
                    "player": player, "response_id": response.id, "pool_ids": [p.id for p in pools[player]]})
            with self.timing.measure("payoff_tables", iteration=t):
                a, b = self._tables(pools)
            with self.timing.measure("cce", iteration=t):
                meta = self.meta_solver.solve(a, b)
            with self.timing.measure("validation_selection", iteration=t):
                history.append(self._snapshot(t, pools, a, b, meta, deviations, time.perf_counter()-started))
            write_json(self.root / "timing_summary.json", self.timing.summary())
            self._report(f"迭代 {t}/{self.T} 完成：CCE gap={meta['cce_gap']:.3g}，"
                         f"外部响应偏离={sum(deviations):.5f}，"
                         f"验证成功率={history[-1]['validation_summary']['success_rate']}")
            if self.external_deviation_tolerance is not None:
                stable_rounds = stable_rounds + 1 if sum(deviations) <= self.external_deviation_tolerance else 0
                if stable_rounds >= self.early_stop_patience:
                    self._report(f"连续 {stable_rounds} 轮外部响应偏离不超过阈值，提前停止")
                    break
        selected = self._select(pools, meta)
        if self._contains_warm_start(selected):
            message = ("拒绝最终输出：所选策略包含暖启动程序；"
                       "暖启动只能用于初始化和训练诊断，不能导出为最终求解器。")
            self._write_selection_status(error=message, pools=pools, candidates=[selected])
            raise RuntimeError(message)
        solution = self.evaluator.evaluate(selected, self.base)
        deployment_distribution = (self.evaluator.mixture_distribution(selected, self.base)
                                   if selected.mixture_weights is not None else None)
        # 测试集只在搜索结束后评估一次；它不进入任何外层选择、停滞判断或提示词。
        self._report(f"训练搜索结束，开始 {len(self.tests)} 个留出实例测试")
        with self.timing.measure("holdout_test"):
            selected_test_evaluations = ordered_map(
                lambda data: self.evaluator.evaluate(selected, data), self.tests, self.parallel_workers)
        selected_test_summary = summarize_evaluations(selected_test_evaluations)
        selected_test_scale_summary = summarize_by_scale(self.tests, selected_test_evaluations)
        # 失败统计按程序-实例评估对计数，同时给出不同程序数，避免把重复失败误读为不同源码数量。
        failure_counts: dict[str, int] = {}
        failed_programs: set[str] = set()
        failed_pairs = 0
        for key, row in self.evaluator.evaluations.items():
            if row.get("status") == "ok":
                continue
            kind = str(row.get("failure_type", "runtime"))
            failure_counts[kind] = failure_counts.get(kind, 0) + 1
            failed_programs.add(str(key[0]))
            failed_pairs += 1
        for key, row in self.evaluator.generated.items():
            if row.get("status") == "ok":
                continue
            kind = str(row.get("failure_type", "invalid_instance"))
            failure_counts[kind] = failure_counts.get(kind, 0) + 1
            failed_programs.add(str(key[0]))
            failed_pairs += 1
        # 所有生成过的求解器在训练结束后报告留出集表现，绝不反馈到选择或变异中。
        candidate_reports = []
        solver_programs = [p for p in self.archive.programs if p.player == "solver"]

        def make_candidate_report(program: Program) -> dict[str, Any]:
            # 候选诊断使用训练后独立验证集；不要因汇总所有历史候选而重复窥视测试集。
            report_instances = self.validation
            if program.error:
                tests = [{"status": "rejected", "error": program.error, "gap": None}
                         for _ in report_instances]
            else:
                tests = ordered_map(lambda data: self.evaluator.evaluate(program, data),
                                    report_instances, self.parallel_workers)
            successful = [row for row in tests if row["status"] == "ok"]
            mixture_distribution = ([self.evaluator.mixture_distribution(program, data)
                                     for data in report_instances]
                                    if program.error is None and program.mixture_weights is not None else None)
            return {"program_id": program.id, "code_hash": program.code_hash,
                    "evaluation_split": "validation",
                      "success_rate": len(successful) / len(tests) if tests else None, "per_instance": tests,
                      "mixture_distribution": mixture_distribution,
                      "expected_success_rate": (sum(row["expected_success_rate"] for row in mixture_distribution)
                                                / len(mixture_distribution) if mixture_distribution else None),
                      "mean_gap": sum(x["gap"] for x in successful) / len(tests)
                                  if tests and len(successful) == len(tests) else None,
                      "evaluation_time_successful": sum(x.get("elapsed", 0) for x in successful),
                      "evaluation_wall_time": sum(x.get("wall_time", 0) for x in tests)}

        with self.timing.measure("candidate_reports"):
            candidate_reports = ordered_map(make_candidate_report, solver_programs, self.parallel_workers)
        for report in candidate_reports:
            self.archive.append("candidate_tests.jsonl", report)
        final = {"schema": "multi-agent-snd-v1", "config": self.config, "user_prompt": user_prompt,
                 "instance_format": self.base.get("format", "static-snd-v2"),
                 "time_expansion": self.base.get("time_expansion"),
                 "objective": solution.get("objective"), "optimal": False, "gap": solution.get("gap"),
                 "reference_kind": solution.get("reference_kind"), "solve_time": time.perf_counter()-started,
                 "reference_value": solution.get("reference_value"),
                 "arc_open": solution.get("arc_open", {}), "flows": solution.get("flows", {}),
                 "constraint_satisfaction": solution["constraint_satisfaction"],
                 "solver_status": "heuristic_feasible" if solution["status"] == "ok" else "no_feasible_heuristic",
                  "solver_used": "multi_agent_generated_heuristic", "selected_solver": selected.to_dict(),
                  "selected_solver_set": selected.member_ids or [selected.id],
                 "selected_solver_distribution": deployment_distribution,
                 "deployment_selection": dict(self._last_selection_diagnostic),
                  "test_success_rate": selected_test_summary["success_rate"],
                  "test_mean_gap": selected_test_summary["mean_gap"],
                  "test_mean_gap_successful": selected_test_summary["mean_gap_successful"],
                  "deployment_kind": ("mixed" if selected.mixture_weights is not None else
                                      "ensemble" if selected.member_ids is not None else "single"),
                 "eohs": self._eohs_sets,
                 "programs": self.archive.catalog(), "players": list(self.players), "iterations": history,
                   "candidate_tests": candidate_reports,
                   "test_evaluations": selected_test_evaluations,
                   "test_summary": selected_test_summary,
                   "test_scale_summary": selected_test_scale_summary,
                  "failure_summary": {"failed_program_instance_pairs": failed_pairs,
                                      "distinct_failed_programs": len(failed_programs),
                                      "by_type": failure_counts},
                 "llm_summary": {"attempts": len(self.archive.calls),
                    "successful_calls": sum(c["status"] == "ok" for c in self.archive.calls),
                    "rejected_programs": sum(p.error is not None for p in self.archive.programs)},
                 "final_pools": {p: [x.id for x in pool] for p, pool in pools.items()},
                 "meta": meta,
                 "diagnostic_scope": {"cce": "finite_current_strategy_pool",
                    "external_response": "finite_llm_candidate_search",
                     "test": "finite_seeded_holdout_report_only_not_generalization_proof",
                     "test_evaluated_once_after_search": True,
                     "test_instance_count": len(self.tests),
                     "validation_instance_count": len(self.validation),
                    "reference": "incumbent_or_optimal_as_recorded",
                    "profile_enumeration_limit": getattr(self.meta_solver, "max_profiles", None)},
                 "artifacts": {"result": "result.json", "program_archive": "programs.jsonl",
                    "timings": "timings.jsonl", "timing_summary": "timing_summary.json",
                    "llm_calls": "llm_calls.jsonl", "events": "events.jsonl", "evaluations": "evaluations.jsonl",
                    "training_instances": "training_instances.json", "validation_instances": "validation_instances.json",
                      "test_instances": "test_instances.json",
                     "final_solver": "final_solver/solver.py",
                     "runtime_config": "final_solver/runtime_config.json", "figures": "figures"}}
        self._export(selected)
        if self.response_mode == "mpsro":
            self._export_meta(pools["solver"], meta["marginals"][0], selected=selected)
            final["artifacts"]["meta_solver"] = "final_solver/meta_solver.py"
        self.archive.write_browser(selected.id)
        write_json(self.root / "lineage.json", {"selected_id": selected.id, "programs": self.archive.catalog()})
        write_json(self.root / "result.json", final)
        write_json(self.output_base / "latest_run.json", {"output_dir": str(self.root), "result": str(self.root / "result.json")})
        if self.make_plots:
            # 可视化依赖单独模块，核心评估和元博弈不依赖绘图实现。
            from snd_visualization.multi_agent import MultiAgentVisualizer
            try:
                with self.timing.measure("plots"):
                    final["figure_files"] = MultiAgentVisualizer(final, self.root / "figures").plot_all()
            except Exception as exc:
                final["visualization_error"] = str(exc)
                write_json(self.root / "result.json", final)
                raise RuntimeError(f"算法结果已保存于 {self.root}，但绘图失败：{exc}") from exc
        final["total_wall_time"] = time.perf_counter() - started
        final["timing_summary"] = self.timing.summary()
        write_json(self.root / "timing_summary.json", final["timing_summary"])
        write_json(self.root / "result.json", final)
        self._report(f"运行完成：{self.root / 'result.json'}")
        return MultiAgentResult(final, self.root)

    def _export(self, selected: Program) -> None:
        """文件夹内保留独立类与源码；依赖项目运行时，但不依赖 LLM/API key。"""
        if self._contains_warm_start(selected):
            message = ("拒绝导出：最终策略包含暖启动程序；"
                       "暖启动只能用于初始化和训练诊断，不能作为最终输出。")
            self._write_selection_status(error=message, candidates=[selected])
            raise RuntimeError(message)
        folder = self.root / "final_solver"
        folder.mkdir()
        runtime_timeout = self.reference_args[3]
        runtime_line_limit = (self.utility.execution_line_limit
                              if self.utility.execution_line_limit is not None
                              else self.utility.work_budget * 10)
        runtime_line_limit_scale = self.utility.execution_line_limit_scale
        write_json(folder / "runtime_config.json", {"timeout": runtime_timeout,
            "line_limit": runtime_line_limit, "line_limit_scale": runtime_line_limit_scale,
            "scope": "per_member",
            "source": ("explicit_execution_line_limit" if self.utility.execution_line_limit is not None
                       else "backward_compatible_work_budget_times_10")})
        if selected.member_ids is not None:
            members = {p.id: p for p in self.archive.programs}
            member_dir = folder / "members"
            member_dir.mkdir()
            for member_id in selected.member_ids:
                (member_dir / f"{member_id}.py").write_text(members[member_id].code + "\n", encoding="utf-8")
            write_json(folder / "members.json", {"member_ids": selected.member_ids,
                "mixture_weights": selected.mixture_weights,
                "strategy_hash": selected.code_hash, "seed": self.seed,
                "policy": ("sample_one_member" if selected.mixture_weights is not None
                           else "run_all_and_choose_lowest_feasible_objective"),
                "timeout_scope": "per_member", "timeout": runtime_timeout,
                "line_limit": runtime_line_limit,
                "line_limit_scale": runtime_line_limit_scale})
        else:
            (folder / "heuristic.py").write_text(selected.code + "\n", encoding="utf-8")
        explanation = selected.explanation
        if selected.member_ids is not None:
            members = {p.id: p for p in self.archive.programs}
            explanation += "\n\n" + "\n\n".join(
                f"[{member_id}] {members[member_id].explanation}" for member_id in selected.member_ids)
        (folder / "explanation.txt").write_text(explanation, encoding="utf-8")
        (folder / "__init__.py").write_text("from .solver import FinalSNDSolver\n", encoding="utf-8")
        if selected.mixture_weights is not None:
            (folder / "solver.py").write_text(
                '"""自动导出的 M-PSRO 混合启发式；每份实例确定性抽取一个成员。"""\n'
                'from pathlib import Path\nfrom multi_agent_snd.deployment import MixedSNDSolver\n\n'
                'class FinalSNDSolver(MixedSNDSolver):\n'
                 f'    def __init__(self, timeout: float = {runtime_timeout!r}, line_limit: int = {runtime_line_limit!r}, line_limit_scale: int | None = {runtime_line_limit_scale!r}):\n'
                '        folder = Path(__file__).with_name("members")\n'
                f'        paths = [folder / name for name in {json.dumps([mid + ".py" for mid in selected.member_ids])}]\n'
                f'        super().__init__(paths, {json.dumps(selected.mixture_weights)}, seed={self.seed},\n'
                f'                         strategy_hash={json.dumps(selected.code_hash)},\n'
                '                         timeout=timeout, line_limit=line_limit, line_limit_scale=line_limit_scale)\n', encoding="utf-8")
        elif selected.member_ids is not None:
            (folder / "solver.py").write_text(
                '"""自动导出的互补启发式集合；逐成员验解并选择最优可行结果。"""\n'
                'from pathlib import Path\nfrom multi_agent_snd.deployment import EnsembleSNDSolver\n\n'
                'class FinalSNDSolver(EnsembleSNDSolver):\n'
                 f'    def __init__(self, timeout: float = {runtime_timeout!r}, line_limit: int = {runtime_line_limit!r}, line_limit_scale: int | None = {runtime_line_limit_scale!r}):\n'
                '        folder = Path(__file__).with_name("members")\n'
                f'        paths = [folder / name for name in {json.dumps([mid + ".py" for mid in selected.member_ids])}]\n'
                '        super().__init__(paths, timeout, line_limit, line_limit_scale)\n', encoding="utf-8")
        else:
            (folder / "solver.py").write_text(
            '"""自动导出的最终启发式类；无需 LLM 即可重复求解。"""\n'
            'from pathlib import Path\nfrom multi_agent_snd.deployment import ProgramSNDSolver\n\n'
            'class FinalSNDSolver(ProgramSNDSolver):\n'
            '    """FinalSNDSolver().solve(instance)，默认沿用训练执行预算。"""\n'
            f'    def __init__(self, timeout: float = {runtime_timeout!r}, line_limit: int = {runtime_line_limit!r}, line_limit_scale: int | None = {runtime_line_limit_scale!r}):\n'
            '        super().__init__(Path(__file__).with_name("heuristic.py"), timeout, line_limit, line_limit_scale)\n', encoding="utf-8")

    def _export_meta(self, strategies: list[Program], probabilities: list[float], *,
                     selected: Program | None = None) -> None:
        """导出剔除暖启动后的部署分布；原始 CCE 仍保存在 result.json。"""
        if (len(strategies) != len(probabilities) or not probabilities
                or any(not np.isfinite(p) or p < 0 for p in probabilities)
                or not np.isclose(sum(probabilities), 1.0)):
            raise ValueError("元策略须为非空策略表及归一化非负概率")
        original_ids = [strategy.id for strategy in strategies]
        original_probabilities = list(probabilities)
        filtered = [(strategy, float(probability))
                    for strategy, probability in zip(strategies, probabilities)
                    if not self._contains_warm_start(strategy)]
        if not filtered:
            message = "拒绝导出元策略：没有非暖启动策略；暖启动不能作为任何最终输出。"
            self._write_selection_status(error=message, candidates=strategies)
            raise RuntimeError(message)
        probability_total = sum(probability for _, probability in filtered)
        strategies = [strategy for strategy, _ in filtered]
        if probability_total > 0:
            probabilities = [probability / probability_total for _, probability in filtered]
            policy_rule = "cce_marginal_conditioned_on_non_warm_start"
        elif selected is not None and selected.id in {strategy.id for strategy in strategies}:
            probabilities = [float(strategy.id == selected.id) for strategy in strategies]
            policy_rule = "selected_non_warm_start_fallback_zero_cce_mass"
            self._report("  非暖启动策略的 CCE 概率总和为 0；元部署仅使用已通过门槛的最终候选，"
                         "此分布不是原始 CCE 边际。")
        else:
            message = "拒绝导出元策略：非暖启动概率为 0，且没有已验证的最终候选可供部署。"
            self._write_selection_status(error=message, candidates=strategies)
            raise RuntimeError(message)
        folder = self.root / "final_solver"
        runtime_timeout = self.reference_args[3]
        runtime_line_limit = (self.utility.execution_line_limit
                              if self.utility.execution_line_limit is not None
                              else self.utility.work_budget * 10)
        runtime_line_limit_scale = self.utility.execution_line_limit_scale
        directory = folder / "meta_members"
        directory.mkdir(exist_ok=True)
        index = {p.id: p for p in self.archive.programs}
        policies = []
        for strategy in strategies:
            if strategy.mixture_weights is not None:
                members = [index[mid] for mid in strategy.member_ids]
            elif strategy.member_ids is not None:
                # 旧 EoH-S 全员择优集合不是概率混合，禁止冒充元策略纯响应。
                continue
            else:
                members = [strategy]
            paths = []
            for member in members:
                path = directory / f"{member.id}.py"
                if not path.exists():
                    path.write_text(member.code + "\n", encoding="utf-8")
                paths.append(f"meta_members/{member.id}.py")
            policies.append({"id": strategy.id, "strategy_hash": strategy.code_hash,
                             "member_paths": paths,
                             "mixture_weights": strategy.mixture_weights})
        if len(policies) != len(strategies):
            raise RuntimeError("M-PSRO 元策略包含不可随机执行的集合策略")
        write_json(folder / "meta_strategy.json", {"policies": policies,
            "probabilities": probabilities, "seed": self.seed,
            "original_strategy_ids": original_ids,
            "original_probabilities": original_probabilities,
            "removed_warm_start_probability": 1.0 - probability_total,
            "deployment_rule": policy_rule,
            "note": "warm-start-free deployment distribution; original CCE is in result.json"})
        (folder / "meta_solver.py").write_text(
            '"""M-PSRO 求解器元策略边际的可复现部署类。"""\n'
            'import json\nfrom pathlib import Path\n'
            'from multi_agent_snd.deployment import MetaSNDSolver\n\n'
            'class FinalMetaSNDSolver(MetaSNDSolver):\n'
            f'    def __init__(self, timeout: float = {runtime_timeout!r}, line_limit: int = {runtime_line_limit!r}, line_limit_scale: int | None = {runtime_line_limit_scale!r}):\n'
            '        folder = Path(__file__).resolve().parent\n'
            '        meta = json.loads((folder / "meta_strategy.json").read_text(encoding="utf-8"))\n'
            '        policies = [{**row, "member_paths": [folder / name for name in row["member_paths"]]}\n'
            '                    for row in meta["policies"]]\n'
            '        super().__init__(policies, meta["probabilities"], seed=meta["seed"],\n'
            '                         timeout=timeout, line_limit=line_limit, line_limit_scale=line_limit_scale)\n', encoding="utf-8")
