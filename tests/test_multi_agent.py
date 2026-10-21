"""多主体回归测试：离线运行、谱系完整性、均衡约束、失败隔离和导出接口。"""
from __future__ import annotations

import copy
import importlib.util
import json
import multiprocessing
import os
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from multi_agent_snd import CCEMetaSolver, MultiAgentSNDSolver, UtilityConfig
from multi_agent_snd.baselines import BASE_SOLVER, OfflineMultiAgentClient
from multi_agent_snd.evaluation import (SNDEvaluator, lower_bound, validate_instance,
                                        validate_perturbation, validate_solution)
from multi_agent_snd.deployment import MixedSNDSolver
from multi_agent_snd.execution import execute, isolated_call, validate_code
from multi_agent_snd.storage import Program, ProgramArchive
from multi_agent_snd.replay import ReplayClient
from multi_agent_snd.clustering import select_representatives
from multi_agent_snd.eoh_s import cpm_select, finite_failure_vectors
from snd_generator import SNDInstance, SNDParameterBounds
from snd_solver import SNDSolver
from snd_solver.solver import BackendUnavailableError


class FailingClient:
    """模块顶层定义以兼容 Windows spawn。"""
    def generate(self, prompt: str, temperature: float = 1, n: int = 1) -> list[str]:
        raise RuntimeError("intentional client failure")


class MixedClient:
    """每次返回合法、重复、非法、缺失解释四个候选。"""
    def generate(self, prompt: str, temperature: float = 1, n: int = 4) -> list[str]:
        valid = OfflineMultiAgentClient().generate(prompt)[0]
        return [valid, valid, "EXPLANATION: rejected\n```python\nimport os\n```",
                "```python\ndef bad(x):\n    return x\n```"]


class TestMetaGame(unittest.TestCase):
    def test_general_sum_three_players_cce_constraints(self) -> None:
        a = [np.array([[.5, -.5], [0, .75]]), np.array([[.2, .4], [.8, .1]])]
        b = [np.array([[0, 1], [2, -.5]]), np.array([[.3, .9], [.1, .4]])]
        result = CCEMetaSolver().solve(a, b)
        profiles, weights = result["profiles"], result["joint"]
        self.assertEqual(len(result["marginals"]), 3)
        self.assertAlmostEqual(sum(weights), 1)
        self.assertLess(result["cce_gap"], 1e-7)

        def utility(profile: list[int], player: int) -> float:
            return sum(x[profile[0], profile[j]] for j, x in enumerate(a, 1)) if player == 0 else b[player-1][profile[0], profile[player]]
        for player in range(3):
            for action in range(2):
                gain = 0
                for p, weight in zip(profiles, weights):
                    changed = list(p)
                    changed[player] = action
                    gain += weight * (utility(changed, player)-utility(p, player))
                self.assertLessEqual(gain, 1e-7)

    def test_weighted_cpm_changes_selection(self) -> None:
        class Item:
            def __init__(self, name: str) -> None:
                self.id = self.code_hash = name
        items = [Item("a"), Item("b"), Item("c")]
        vectors = {"a": [0.0, 4.0], "b": [3.0, 0.0], "c": [1.0, 1.0]}
        unweighted, _ = cpm_select(items, vectors, 1)
        weighted, _ = cpm_select(items, vectors, 1, [1.0, 5.0])
        self.assertEqual(unweighted[0].id, "c")
        self.assertEqual(weighted[0].id, "b")

    def test_cpi_failure_penalty_is_finite_and_worse_than_feasible(self) -> None:
        vectors = finite_failure_vectors({"good": [20.0, float("inf")],
                                          "failed": [float("inf"), float("inf")]}, 1.0)
        self.assertEqual(vectors["good"], [20.0, 1.0])
        self.assertEqual(vectors["failed"], [21.0, 1.0])
        self.assertTrue(all(np.isfinite(value) for row in vectors.values() for value in row))

    def test_six_players_and_profile_limit(self) -> None:
        result = CCEMetaSolver().solve([np.zeros((1, 1))]*6, [np.ones((1, 1))]*6)
        self.assertEqual(len(result["expected_utilities"]), 7)
        with self.assertRaises(ValueError):
            CCEMetaSolver(max_profiles=2).solve([np.ones((2, 2))]*2, [np.ones((2, 2))]*2)

    def test_invalid_tables(self) -> None:
        for table in (np.array(1), np.empty((0, 2)), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                CCEMetaSolver().solve([table], [table])


class TestEoHClustering(unittest.TestCase):
    """验证 EoH 第 3 步的 Top-K 与解释聚类筛选。"""

    def setUp(self) -> None:
        self.candidates = [
            type("Candidate", (), {
                "id": f"p{i}", "code_hash": f"h{i}",
                "explanation": text,
            })()
            for i, text in enumerate((
                "降低运输成本，优先选择费用较低的弧",
                "最短路径搜索，减少需求的加权路径长度",
                "控制容量拥堵，平衡各条弧的流量负载",
                "减少固定开通费用，同时保持网络连通",
                "缩短运行时间并提高求解效率",
            ))
        ]
        self.scores = {candidate.code_hash: float(i) for i, candidate in enumerate(self.candidates)}

    def test_topk_keeps_original_selection(self) -> None:
        selected, diagnostic = select_representatives(
            self.candidates, self.scores, 3, mode="topk"
        )
        self.assertEqual([candidate.id for candidate in selected], ["p4", "p3", "p2"])
        self.assertEqual(diagnostic["cluster_count"], 0)

    def test_each_cluster_keeps_best_candidate(self) -> None:
        for method in ("kmeans", "agglomerative", "keyword"):
            with self.subTest(method=method):
                selected, diagnostic = select_representatives(
                    self.candidates, self.scores, 3, mode="cluster",
                    method=method, min_clusters=2,
                )
                self.assertEqual(len(selected), 3)
                self.assertEqual(diagnostic["cluster_count"], 3)
                self.assertEqual(len(diagnostic["representatives"]), 3)
                self.assertTrue(all(diagnostic["clusters"].values()))

    def test_cluster_selection_is_deterministic(self) -> None:
        first = select_representatives(
            self.candidates, self.scores, 3, mode="cluster",
            method="kmeans", min_clusters=2,
        )
        second = select_representatives(
            self.candidates, self.scores, 3, mode="cluster",
            method="kmeans", min_clusters=2,
        )
        self.assertEqual(first[1], second[1])
        self.assertEqual([item.id for item in first[0]], [item.id for item in second[0]])

    def test_insufficient_candidates_falls_back(self) -> None:
        selected, diagnostic = select_representatives(
            self.candidates[:1], self.scores, 3, mode="cluster",
            method="kmeans", min_clusters=2,
        )
        self.assertEqual([item.id for item in selected], ["p0"])
        self.assertEqual(diagnostic["fallback"], "insufficient_candidates")


class TestEvaluation(unittest.TestCase):
    def setUp(self) -> None:
        self.data = SNDInstance(4, .5, .7, seed=42).to_dict()

    def test_distortion_and_topology(self) -> None:
        changed = copy.deepcopy(self.data)
        self.assertEqual(validate_perturbation(self.data, changed), 0)
        changed["arcs"][0]["fixed_cost"] *= 1.5
        self.assertGreater(validate_perturbation(self.data, changed), 0)
        changed["arcs"][0]["fixed_cost"] *= 3
        with self.assertRaises(ValueError):
            validate_perturbation(self.data, changed)
        changed = copy.deepcopy(self.data)
        changed["arcs"][0]["tail"] = changed["arcs"][0]["head"]
        with self.assertRaises(ValueError):
            validate_perturbation(self.data, changed)

    def test_nonfinite_data_and_solution_rejected(self) -> None:
        for value in (float("nan"), float("inf")):
            changed = copy.deepcopy(self.data)
            changed["arcs"][0]["capacity"] = value
            with self.assertRaises(ValueError):
                validate_instance(changed)
            with self.assertRaises(ValueError):
                validate_solution(self.data, {"arc_open": {"0": value}, "flows": {}})

    def test_utilities_not_zero_sum(self) -> None:
        config = UtilityConfig()
        obs = {"status": "ok", "distortion": .2, "metrics": dict(cost=.2, path=.5, transport=.1, fixed=.4, congestion=.9, work=.05)}
        solver, generator = config.pair(obs, "path")
        self.assertAlmostEqual(generator, .48)
        self.assertNotAlmostEqual(solver + generator, 0)
        self.assertEqual(config.pair({"status": "invalid_instance"}, "path"), (0, -2))
        self.assertEqual(config.pair({"status": "solver_failure", "distortion": 0}, "cost"), (-2, 1))

    def test_temporal_waiting_does_not_count_as_service(self) -> None:
        """等待弧承担跨期库存流，但不增加服务跳数或服务容量利用率。"""
        data = {
            "format": "time-expanded-snd-v1", "nodes": list(range(6)),
            "arcs": [
                {"id": 0, "tail": 0, "head": 3, "capacity": 10,
                 "fixed_cost": 2, "variable_cost": 1},
                {"id": 1, "tail": 3, "head": 5, "capacity": 2,
                 "fixed_cost": 0, "variable_cost": 0},
            ],
            "commodities": [{"id": 0, "origin": 0, "destination": 5, "demand": 2}],
            "time_expansion": {
                "time_points": [0, 1, 2], "physical_nodes": [0, 1],
                "physical_arcs": [{"id": 0, "tail": 0, "head": 1}],
                "node_map": [{"node_id": t * 2 + node, "physical_node": node, "time": t}
                             for t in range(3) for node in range(2)],
                "arc_schedule": [
                    {"arc_id": 0, "kind": "service", "physical_arc_id": 0,
                     "departure_time": 0, "arrival_time": 1},
                    {"arc_id": 1, "kind": "wait", "physical_node": 1,
                     "departure_time": 1, "arrival_time": 2},
                ],
                "commodity_windows": [{"commodity_id": 0, "origin": 0,
                                       "destination": 1, "release_time": 0,
                                       "deadline": 2, "demand": 2}],
            },
        }
        validate_instance(data)
        self.assertEqual(lower_bound(data, None), 2)
        solution = {"arc_open": {"0": 1, "1": 1},
                    "flows": {"0": {"0": 2, "1": 2}}}
        program = Program("p00000", "solver", "solver", "test", "", "test", [], 0, 0, "I1")
        with TemporaryDirectory() as tmp:
            evaluator = SNDEvaluator(data, ProgramArchive(Path(tmp)), UtilityConfig(),
                                     "scipy", 10, 0, 10, 1, 0)
            fake_execution = {"result": solution, "lines": 1, "elapsed": 0.01,
                              "wall_time": 0.01}
            with patch.object(evaluator, "reference", return_value={"value": 4, "kind": "optimal"}), \
                    patch("multi_agent_snd.evaluation.execute", return_value=fake_execution):
                row = evaluator.evaluate(program, data)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["metrics"]["path"], 0)
        self.assertAlmostEqual(row["metrics"]["congestion"], 0.2)

    def test_mixed_payoff_is_member_expectation(self) -> None:
        """混合求解器和混合生成器的收益是纯程序两两收益的双线性加权。"""
        with TemporaryDirectory() as tmp:
            archive = ProgramArchive(Path(tmp))
            solvers = [archive.add(player="solver", role="solver", explanation="base", code=BASE_SOLVER,
                                   parents=[], iteration=0, eoh_iteration=0, operator="base")]
            solvers.append(archive.add(player="solver", role="solver", explanation="variant",
                                       code=BASE_SOLVER.replace("1.0 * arc", "0.5 * arc"),
                                       parents=[], iteration=1, eoh_iteration=0, operator="I1"))
            generators = [archive.add(player="cost", role="generator", explanation="base",
                                      code="def generate_snd(base_instance, seed):\n    return dict(base_instance)",
                                      parents=[], iteration=0, eoh_iteration=0, operator="base")]
            generators.append(archive.add(player="cost", role="generator", explanation="variant",
                                          code="def generate_snd(base_instance, seed):\n    return base_instance",
                                          parents=[], iteration=1, eoh_iteration=0, operator="I1"))
            mixed_solver = archive.add(player="solver", role="solver", explanation="mix", code="",
                                       parents=solvers, iteration=1, eoh_iteration=1,
                                       operator="MPSRO", members=solvers, mixture_weights=[.25, .75])
            mixed_generator = archive.add(player="cost", role="generator", explanation="mix", code="",
                                          parents=generators, iteration=1, eoh_iteration=1,
                                          operator="MPSRO", members=generators, mixture_weights=[.4, .6])
            evaluator = SNDEvaluator(self.data, archive, UtilityConfig(),
                                     "scipy", 10, 0, 10, 1, 42)
            with patch.object(evaluator, "sample", wraps=evaluator.sample):
                generated = evaluator.sample(mixed_generator, 0)
            self.assertIn(generated["selected_generator_member_id"], mixed_generator.member_ids)
            self.assertEqual(generated["status"], "ok")
            values = {(s.id, g.id): (float(2 * i + j), float(10 * i + j))
                      for i, s in enumerate(solvers) for j, g in enumerate(generators)}
            original_pair = SNDEvaluator.pair

            def pair_stub(instance, solver, generator):
                if solver.mixture_weights is None and generator.mixture_weights is None:
                    return values[solver.id, generator.id]
                return original_pair(instance, solver, generator)

            with patch.object(SNDEvaluator, "pair", pair_stub):
                actual = evaluator.pair(mixed_solver, mixed_generator)
            expected = tuple(sum(sw * gw * values[s.id, g.id][k]
                                 for sw, s in zip([.25, .75], solvers)
                                 for gw, g in zip([.4, .6], generators))
                             for k in range(2))
            for observed, target in zip(actual, expected):
                self.assertAlmostEqual(observed, target)
            with self.assertRaises(ValueError):
                archive.add(player="solver", role="solver", explanation="bad", code="",
                            parents=solvers, iteration=1, eoh_iteration=1,
                            operator="MPSRO", members=solvers, mixture_weights=[.2, .2])

    def test_mixed_deployment_returns_one_reproducible_solution(self) -> None:
        """部署不运行全部成员择优，重复求解同一实例会选同一纯程序。"""
        with TemporaryDirectory() as tmp:
            path_a = Path(tmp) / "a.py"
            path_b = Path(tmp) / "b.py"
            path_a.write_text(BASE_SOLVER, encoding="utf-8")
            path_b.write_text(BASE_SOLVER.replace("1.0 * arc", "0.5 * arc"), encoding="utf-8")
            solver = MixedSNDSolver([path_a, path_b], [.5, .5], seed=42)
            first = solver.solve(self.data)
            second = solver.solve(self.data)
            self.assertEqual(first["selected_member"], second["selected_member"])
            self.assertEqual(first["arc_open"], second["arc_open"])
            self.assertTrue(first["constraint_satisfaction"]["feasible"])

    def test_mixture_distribution_reports_expected_feasibility(self) -> None:
        with TemporaryDirectory() as tmp:
            archive = ProgramArchive(Path(tmp))
            members = [archive.add(player="solver", role="solver", explanation=str(i),
                                   code=f"def solve_snd(instance):\n    return {i}\n",
                                   parents=[], iteration=0, eoh_iteration=0, operator="base")
                       for i in range(2)]
            mixture = archive.add(player="solver", role="solver", explanation="mix",
                                  code="", parents=members, iteration=1, eoh_iteration=1,
                                  operator="MPSRO", members=members,
                                  mixture_weights=[0.25, 0.75])
            evaluator = SNDEvaluator(self.data, archive, UtilityConfig(),
                                     "scipy", 10, 0, 10, 1, 42)
            with patch.object(evaluator, "evaluate", side_effect=[
                    {"status": "ok", "gap": 0.2, "objective": 12.0},
                    {"status": "solver_failure", "gap": None}]):
                distribution = evaluator.mixture_distribution(mixture, self.data)
            self.assertAlmostEqual(distribution["expected_success_rate"], 0.25)
            self.assertAlmostEqual(distribution["expected_gap_given_success"], 0.2)
            self.assertIsNone(distribution["expected_gap"])

    def test_baseline_work_is_deterministic(self) -> None:
        first = execute(BASE_SOLVER, "solve_snd", self.data)
        second = execute(BASE_SOLVER, "solve_snd", self.data)
        self.assertEqual(first["lines"], second["lines"])
        self.assertEqual(first["result"], second["result"])
        _, checks = validate_solution(self.data, first["result"])
        self.assertTrue(checks["feasible"])

    def test_timeout_and_large_payload(self) -> None:
        with self.assertRaises(TimeoutError):
            execute("def solve_snd(instance):\n    while True:\n        pass", "solve_snd", {}, timeout=.5, line_limit=10**10)
        result = execute("def solve_snd(instance):\n    return list(range(100000))", "solve_snd", {})
        self.assertEqual(len(result["result"]), 100000)
        self.assertFalse(any(p.is_alive() for p in multiprocessing.active_children()))

    def test_line_limit_is_independent_and_typed(self) -> None:
        from multi_agent_snd.execution import ExecutionError
        with self.assertRaises(ExecutionError) as ctx:
            execute("def solve_snd(instance):\n    x = 0\n    x += 1\n    return x", "solve_snd", {},
                    timeout=5, line_limit=1)
        self.assertEqual(ctx.exception.failure_type, "line_limit")
        self.assertGreaterEqual(ctx.exception.lines, 1)

    def test_source_quality_limits_are_checked_before_execution(self) -> None:
        code = "def solve_snd(instance):\n    return {}\n"
        with self.assertRaisesRegex(ValueError, "源码超过行数上限"):
            validate_code(code, "solve_snd(instance)", max_code_lines=1)
        with self.assertRaisesRegex(ValueError, "源码超过字符上限"):
            validate_code(code, "solve_snd(instance)", max_code_chars=5)
        with self.assertRaisesRegex(ValueError, "AST 复杂度"):
            validate_code(code, "solve_snd(instance)", max_ast_nodes=1)

    def test_scaled_execution_limit_uses_instance_size(self) -> None:
        from multi_agent_snd.evaluation import SNDEvaluator
        with TemporaryDirectory() as tmp:
            evaluator = SNDEvaluator(
                self.data, ProgramArchive(Path(tmp)),
                UtilityConfig(work_budget=10, execution_line_limit=100,
                              execution_line_limit_scale=2),
                "scipy", 10, 0, 10, 1, 42)
            complexity = len(self.data["nodes"]) + len(self.data["arcs"]) + len(self.data["commodities"])
            self.assertEqual(evaluator._execution_line_limit(self.data), min(100, 2 * complexity))

    def test_forbidden_code(self) -> None:
        for code in ("x=1\ndef solve_snd(instance):\n    return {}", "def solve_snd(instance, *args):\n    return {}",
                     "@print\ndef solve_snd(instance):\n    return {}"):
            with self.assertRaises(ValueError):
                validate_code(code, "solve_snd(instance)")

    def test_isolated_client_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "intentional client failure"):
            isolated_call("llm", (FailingClient(), "test", 1, 1, 42), 10)

    def test_commercial_unavailable_falls_back_to_scipy(self) -> None:
        with patch.object(SNDSolver, "_solve_gurobi", side_effect=BackendUnavailableError("no license")):
            reference = SNDSolver(self.data, solver_type="gurobi", mip_gap=0).solve()
        self.assertEqual(reference.solver_used, "scipy")
        self.assertIn("no license", reference.fallback_reason)
        self.assertTrue(reference.constraint_satisfaction["feasible"])


class TestMultiAgentIntegration(unittest.TestCase):
    def test_mpsro_with_original_eoh(self) -> None:
        """M-PSRO 扩池模式也可使用原 EoH 作为纯程序搜索预言机。"""
        with TemporaryDirectory() as tmp:
            solver = MultiAgentSNDSolver(
                SNDInstance(3, .5, .5, seed=7), generator_directions=("cost",),
                outer_iterations=1, eoh_iterations=1, population_size=2,
                n_candidates=2, instances_per_generator=1, output_dir=tmp,
                progress_output=False, make_plots=False, parallel_workers=1,
                test_instance_count=1, response_mode="mpsro",
                mixture_support_size=2, mixture_epsilon=10,
            )
            data = solver.solve("验证 EoH 与 M-PSRO 兼容").to_dict()
            self.assertEqual(data["deployment_kind"], "mixed")
            self.assertTrue(data["constraint_satisfaction"]["feasible"])
            self.assertEqual({p["player"] for p in data["programs"]
                              if p["operator"] == "MPSRO"}, {"solver", "cost"})

    def test_holdout_is_final_only_and_scale_summary_is_written(self) -> None:
        with TemporaryDirectory() as tmp:
            solver = MultiAgentSNDSolver(
                SNDInstance(3, .5, .5, seed=7), generator_directions=("cost",),
                outer_iterations=1, eoh_iterations=1, population_size=2,
                n_candidates=1, instances_per_generator=1, output_dir=tmp,
                progress_output=False, make_plots=False, parallel_workers=1,
                test_instance_count=2, validation_instance_count=1)
            data = solver.solve("检查留出集隔离").to_dict()
            self.assertTrue(all(item.get("test_evaluations") is None
                                for item in data["iterations"]))
            self.assertEqual(len(data["test_evaluations"]), 2)
            self.assertIn("test_summary", data)
            self.assertIn("test_scale_summary", data)

    def test_eohs_mixture_preserves_cpm_members_despite_utility_gap(self) -> None:
        """默认 CPM 混合保留互补成员；utility 模式才按容差删除它。"""
        with TemporaryDirectory() as tmp:
            solver = MultiAgentSNDSolver(SNDInstance(3, .5, .5, seed=7),
                generator_directions=("cost",), output_dir=tmp,
                response_mode="mpsro", eoh_framework="eoh-s",
                mixture_support_size=2, mixture_epsilon=0,
                progress_output=False, make_plots=False)
            solver.root = Path(tmp)
            solver.archive = ProgramArchive(Path(tmp))
            first = solver.archive.add(player="solver", role="solver", explanation="average",
                code="def solve_snd(instance):\n    return instance", parents=[],
                iteration=1, eoh_iteration=1, operator="LS")
            second = solver.archive.add(player="solver", role="solver", explanation="complement",
                code="def solve_snd(instance):\n    return dict(instance)", parents=[],
                iteration=1, eoh_iteration=1, operator="CS")
            def search(*args):
                solver._response_candidates["solver"] = [first, second]
                return first
            with patch.object(solver, "_search_eohs", side_effect=search), \
                    patch.object(solver, "_fitness", side_effect=lambda p, *_: 1.0 if p.id == first.id else 0.0):
                mixed = solver._search("solver", 1, {}, {})
                self.assertEqual(mixed.member_ids, [first.id, second.id])
                solver.eohs_mixture_selection = "utility"
                utility_mixed = solver._search("solver", 1, {}, {})
                self.assertEqual(utility_mixed.member_ids, [first.id])

    def test_export_inherits_nondefault_execution_limits(self) -> None:
        with TemporaryDirectory() as tmp:
            solver = MultiAgentSNDSolver(SNDInstance(3, .5, .5, seed=7),
                generator_directions=("cost",), output_dir=tmp,
                candidate_timeout=9, utility_config=UtilityConfig(work_budget=12345),
                progress_output=False, make_plots=False)
            solver.root = Path(tmp) / "run"
            solver.root.mkdir()
            solver.archive = ProgramArchive(solver.root)
            selected = solver.archive.add(player="solver", role="solver", explanation="variant",
                code=BASE_SOLVER.replace("1.0 * arc", "0.5 * arc"), parents=[],
                iteration=1, eoh_iteration=0, operator="I1")
            solver._export(selected)
            runtime = json.loads((solver.root / "final_solver/runtime_config.json").read_text(encoding="utf-8"))
            source = (solver.root / "final_solver/solver.py").read_text(encoding="utf-8")
            self.assertEqual(runtime["timeout"], 9)
            self.assertEqual(runtime["line_limit"], 123450)
            self.assertIn("timeout: float = 9", source)
            self.assertIn("line_limit: int = 123450", source)

    def test_mpsro_temporal_end_to_end_and_export(self) -> None:
        """时间模式下两方均以混合响应扩池，导出程序与训练的确定性抽样一致。"""
        with TemporaryDirectory() as tmp:
            base = SNDInstance(3, .6, .5, seed=3, time_periods=2)
            solver = MultiAgentSNDSolver(
                base, generator_directions=("cost",), outer_iterations=1,
                eoh_iterations=1, population_size=2, n_candidates=2,
                instances_per_generator=1, output_dir=tmp, progress_output=False,
                make_plots=True, parallel_workers=1, test_instance_count=1,
                candidate_timeout=10, time_limit=10, eoh_framework="eoh-s",
                response_mode="mpsro", mixture_support_size=2, mixture_epsilon=10,
            )
            result = solver.solve("生成互补且可行的时间相关 SND 启发式")
            data = result.to_dict()
            mixtures = [p for p in data["programs"] if p["operator"] == "MPSRO"]
            self.assertEqual({p["player"] for p in mixtures}, {"solver", "cost"})
            self.assertTrue(all(sum(p["mixture_weights"]) == 1 for p in mixtures))
            self.assertTrue(any(len(p["member_ids"]) > 1 for p in mixtures))
            self.assertEqual(data["deployment_kind"], "mixed")
            self.assertTrue(data["constraint_satisfaction"]["feasible"])
            self.assertEqual(data["config"]["eohs_mixture_selection"], "cpm")
            self.assertIsNotNone(data["selected_solver_distribution"])
            self.assertAlmostEqual(data["selected_solver_distribution"]["expected_success_rate"], 1.0)
            self.assertTrue((result.output_dir / "figures/lineage.svg").exists())
            self.assertTrue((result.output_dir / "final_solver/members.json").exists())
            runtime = json.loads((result.output_dir / "final_solver/runtime_config.json").read_text(encoding="utf-8"))
            self.assertEqual(runtime["timeout"], 10)
            self.assertEqual(runtime["line_limit"], 2000000)
            self.assertTrue((result.output_dir / "final_solver/meta_strategy.json").exists())
            spec = importlib.util.spec_from_file_location(
                "mpsro_export", result.output_dir / "final_solver/solver.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            deployed = module.FinalSNDSolver().solve(base)
            self.assertAlmostEqual(deployed["objective"], result.objective)
            self.assertIn(deployed["selected_member"], data["selected_solver"]["member_ids"])
            meta_spec = importlib.util.spec_from_file_location(
                "mpsro_meta_export", result.output_dir / "final_solver/meta_solver.py")
            meta_module = importlib.util.module_from_spec(meta_spec)
            meta_spec.loader.exec_module(meta_module)
            meta_result = meta_module.FinalMetaSNDSolver().solve(base)
            self.assertTrue(meta_result["constraint_satisfaction"]["feasible"])
            self.assertIn(meta_result["selected_meta_policy_id"], data["final_pools"]["solver"])

    def test_temporal_end_to_end_and_export(self) -> None:
        """时间展开实例需贯穿生成器对抗、参考求解、测试和最终部署。"""
        with TemporaryDirectory() as tmp:
            base = SNDInstance(3, .6, .5, seed=3, time_periods=2)
            solver = MultiAgentSNDSolver(
                base, generator_directions=("cost",), outer_iterations=1,
                eoh_iterations=1, population_size=2, n_candidates=1,
                instances_per_generator=1, output_dir=tmp, progress_output=False,
                make_plots=False, parallel_workers=1, test_instance_count=1,
                candidate_timeout=10, time_limit=10,
            )
            result = solver.solve("保持时间窗内交付并降低成本")
            data = result.to_dict()
            self.assertTrue(data["constraint_satisfaction"]["feasible"])
            self.assertEqual(data["config"]["base_instance_format"], "time-expanded-snd-v1")
            self.assertEqual(data["config"]["base_time_periods"], 2)
            self.assertEqual(data["instance_format"], "time-expanded-snd-v1")
            self.assertEqual(data["time_expansion"]["time_points"], [0, 1, 2])
            tests = json.loads((result.output_dir / "test_instances.json").read_text(encoding="utf-8"))
            self.assertEqual(tests[0]["metadata"]["physical_node_count"], 3)
            self.assertEqual(tests[0]["metadata"]["time_periods"], 2)
            spec = importlib.util.spec_from_file_location("temporal_export", result.output_dir / "final_solver/solver.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            deployed = module.FinalSNDSolver().solve(base)
            self.assertTrue(deployed["constraint_satisfaction"]["feasible"])
            self.assertAlmostEqual(deployed["objective"], result.objective)
            self.assertEqual(deployed["time_expansion"]["time_points"], [0, 1, 2])

    def test_temporal_bounds_use_temporal_training_and_test_instances(self) -> None:
        base = SNDInstance(3, .6, .5, seed=3, time_periods=2)
        bounds = SNDParameterBounds(nodes=(3, 4), coverage=(.4, .6),
                                    fixed_cost=(.4, .7), time_periods=(2, 3))
        solver = MultiAgentSNDSolver(base, instance_bounds=bounds,
                                    instances_per_generator=2, test_instance_count=2,
                                    progress_output=False, make_plots=False)
        self.assertEqual(len(solver.training_bases), 2)
        self.assertTrue(all(x["format"] == "time-expanded-snd-v1"
                            for x in solver.training_bases + solver.tests))
        with self.assertRaises(ValueError):
            MultiAgentSNDSolver(base, instance_bounds=SNDParameterBounds(nodes=(3, 4)))

    def test_call_failure_is_visible_redacted_and_archived(self) -> None:
        # 只测试请求错误路径；模拟隔离调用，不花费 API 额度、不执行候选或参考 MIP。
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"LLM_API_KEY": "test-only-key"}), \
                patch("multi_agent_snd.solver.isolated_call", side_effect=RuntimeError(
                    "BadRequestError: supported models: model-a, model-b; test-only-key")) as call:
            solver = MultiAgentSNDSolver(SNDInstance(3, .5, .5, seed=7), llm_client=FailingClient(),
                generator_directions=("cost",), output_dir=tmp, llm_retries=1, make_plots=False)
            solver.client.model = "provider/model-b"
            solver.root = Path(tmp)
            solver.archive = ProgramArchive(solver.root)
            solver.user_prompt = "显示失败原因"
            output = StringIO()
            with redirect_stdout(output):
                candidates = solver._generate("solver", "I1", [], 1, 0,
                    {"cost": []}, {"marginals": [[1.0], []]})
            self.assertEqual(candidates, [])
            self.assertEqual(call.call_count, 2)
            message = output.getvalue()
            self.assertIn("LLM 调用失败", message)
            self.assertIn("model-b", message)
            self.assertIn("重试已耗尽", message)
            self.assertNotIn("test-only-key", message)
            self.assertIn("[REDACTED]", message)
            rows = [json.loads(line) for line in (solver.root / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["model"] == solver.client.model for row in rows))
            self.assertTrue(all("test-only-key" not in row["error"] for row in rows))

    def test_failed_calls_retry_and_record(self) -> None:
        with TemporaryDirectory() as tmp:
            solver = MultiAgentSNDSolver(SNDInstance(3, .5, .5, seed=7), llm_client=FailingClient(),
                generator_directions=("cost",), outer_iterations=1, eoh_iterations=1,
                output_dir=tmp, progress_output=False, llm_retries=1, make_plots=False,
                test_instances=[SNDInstance(3, .5, .5, seed=8)])
            with self.assertRaisesRegex(RuntimeError, "非暖启动"):
                solver.solve("测试失败重试")
            self.assertEqual(len(solver.archive.calls), 8)  # 求解器 I1 + 三个单亲算子重试，失败后不导出基线。
            self.assertTrue(all(c["status"] == "failed" for c in solver.archive.calls))
            self.assertTrue((solver.root / "selection_status.json").exists())
            self.assertFalse((solver.root / "result.json").exists())
            self.assertFalse((solver.root / "final_solver").exists())
            self.assertFalse((Path(tmp) / "latest_run.json").exists())

    def test_replay_and_reproducibility(self) -> None:
        with TemporaryDirectory() as tmp:
            kwargs = dict(generator_directions=("cost",), outer_iterations=1, eoh_iterations=1,
                          output_dir=tmp, progress_output=False, make_plots=False,
                          test_instances=[SNDInstance(3, .5, .5, seed=8)])
            base = SNDInstance(3, .5, .5, seed=7)
            first = MultiAgentSNDSolver(base, **kwargs).solve("重放验证")
            replay = ReplayClient(first.output_dir / "llm_calls.jsonl")
            second_solver = MultiAgentSNDSolver(base, llm_client=replay, **kwargs)
            second = second_solver.solve("重放验证")
            self.assertTrue(all(call["status"] == "ok" for call in second_solver.archive.calls))
            self.assertEqual(first.data["programs"], second.data["programs"])
            self.assertEqual(first.data["meta"], second.data["meta"])
            self.assertEqual(first.objective, second.objective)

    def test_archive_preserves_duplicate_events(self) -> None:
        with TemporaryDirectory() as tmp:
            archive = ProgramArchive(Path(tmp))
            a = archive.add(player="solver", role="solver", explanation="one", code=BASE_SOLVER,
                            parents=[], iteration=0, eoh_iteration=0, operator="base")
            b = archive.add(player="solver", role="solver", explanation="two", code=BASE_SOLVER,
                            parents=[a], iteration=1, eoh_iteration=1, operator="M1")
            self.assertNotEqual(a.id, b.id)
            self.assertEqual(a.code_hash, b.code_hash)
            self.assertEqual(b.parent_ids, [a.id])
            self.assertEqual(len(list((Path(tmp)/"programs").glob("*.py"))), 1)

    def test_end_to_end_outputs_and_export(self) -> None:
        with TemporaryDirectory() as tmp:
            solver = MultiAgentSNDSolver(SNDInstance(3, .5, .5, seed=7), llm_client=MixedClient(),
                generator_directions=("cost", "path"), outer_iterations=1, eoh_iterations=1,
                n_candidates=4, output_dir=tmp, progress_output=False, llm_retries=0,
                test_instances=[SNDInstance(3, .5, .5, seed=8)])
            result = solver.solve("离线验证所有输出。")
            root, data = result.output_dir, result.to_dict()
            self.assertTrue(data["constraint_satisfaction"]["feasible"])
            self.assertEqual(len(data["iterations"]), 2)
            rejected = [p for p in data["programs"] if p["error"]]
            self.assertGreater(len(rejected), 0)
            ids = {p["id"] for p in data["programs"]}
            for p in data["programs"]:
                self.assertTrue(set(p["parent_ids"]) <= ids)
                if p["code_file"]:
                    self.assertTrue((root / p["code_file"]).is_file())
            for filename in ("result.json", "programs.jsonl", "llm_calls.jsonl", "events.jsonl",
                             "lineage.json", "program_catalog.html", "final_solver/solver.py",
                             "figures/lineage.svg", "figures/test_gap.png", "figures/payoff_cost.png"):
                self.assertGreater((root / filename).stat().st_size, 0)
            self.assertNotIn("Infinity", result.to_json())
            self.assertNotIn("NaN", result.to_json())
            spec = importlib.util.spec_from_file_location("exported_solver", root / "final_solver/solver.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            deployed = module.FinalSNDSolver().solve(solver.base)
            self.assertAlmostEqual(deployed["objective"], result.objective)
            self.assertTrue(deployed["constraint_satisfaction"]["feasible"])
            with self.assertRaises(RuntimeError):
                solver.solve("再次执行")

    def test_eohs_ensemble_enters_game_and_deploys(self) -> None:
        with TemporaryDirectory() as tmp:
            solver = MultiAgentSNDSolver(SNDInstance(3, .5, .5, seed=7),
                generator_directions=("cost",), outer_iterations=2, eoh_iterations=1,
                population_size=2, instances_per_generator=1, output_dir=tmp,
                progress_output=False, make_plots=False, parallel_workers=1,
                test_instance_count=1, eoh_framework="eoh-s", cpi_mode="weighted")
            result = solver.solve("验证集合策略")
            data = result.to_dict()
            self.assertEqual(data["deployment_kind"], "ensemble")
            self.assertGreaterEqual(len(data["selected_solver_set"]), 1)
            self.assertGreaterEqual(len(data["iterations"][-1]["solver_tables"][0]), 2)
            self.assertTrue((result.output_dir / "final_solver/members.json").exists())
            spec = importlib.util.spec_from_file_location("eohs_export", result.output_dir / "final_solver/solver.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            deployed = module.FinalSNDSolver().solve(solver.base)
            self.assertTrue(deployed["constraint_satisfaction"]["feasible"])
            self.assertAlmostEqual(deployed["objective"], result.objective)

    def test_invalid_config(self) -> None:
        base = SNDInstance(3, .5, .5, seed=7)
        for directions in ((), ("cost", "cost"), ("unknown",)):
            with self.assertRaises(ValueError):
                MultiAgentSNDSolver(base, generator_directions=directions)


if __name__ == "__main__":
    unittest.main()
