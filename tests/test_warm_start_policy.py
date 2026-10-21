"""Warm starts remain training aids, never deployable solver members.

Most checks stub evaluation/LLM calls so the policy regressions are fast and
independent of network, process startup and solver quality.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from multi_agent_snd import MultiAgentSNDSolver
from multi_agent_snd.baselines import BASE_SOLVER
from multi_agent_snd.storage import ProgramArchive
from snd_generator import SNDInstance


class TestWarmStartPolicy(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.solver = MultiAgentSNDSolver(
            SNDInstance(3, .5, .5, seed=7), generator_directions=("cost",),
            population_size=2, eoh_iterations=1, output_dir=self.temporary.name,
            progress_output=False, make_plots=False, parallel_workers=1,
            test_instance_count=1)
        self.solver.root = Path(self.temporary.name)
        self.solver.archive = ProgramArchive(self.solver.root)
        self.base = self.add(BASE_SOLVER, "base")
        self.first = self.add(BASE_SOLVER.replace("1.0 * arc", "0.5 * arc"))
        self.second = self.add(BASE_SOLVER.replace("1.0 * arc", "1.5 * arc"))
        self.pools = {"solver": [self.base], "cost": []}
        self.meta = {"marginals": [[1.0], []]}

    def add(self, code: str, operator: str = "I1"):
        return self.solver.archive.add(
            player="solver", role="solver", explanation="test", code=code,
            parents=[], iteration=0 if operator == "base" else 1,
            eoh_iteration=0, operator=operator)

    def compound(self, members, *, mixed=False):
        return self.solver.archive.add(
            player="solver", role="solver", explanation="compound", code="",
            parents=members, iteration=1, eoh_iteration=1,
            operator="MPSRO" if mixed else "CPM", members=members,
            mixture_weights=[1 / len(members)] * len(members) if mixed else None)

    def stub_evaluator(self, failed=()):
        self.solver.evaluator = SimpleNamespace(
            evaluate=lambda program, data: {
                "status": "solver_failure" if program.id in failed else "ok",
                "gap": 0.0}, evaluations={})

    def test_relabelled_and_comment_only_copies_are_warm_starts(self):
        clone = self.add(BASE_SOLVER, "LS")
        formatted_clone = self.add("# copied candidate\n" + BASE_SOLVER + "\n", "CS")
        for program in (self.base, clone, formatted_clone):
            self.assertTrue(self.solver._contains_warm_start(program))
        self.assertFalse(self.solver._contains_warm_start(self.first))

    def test_ancestors_do_not_disqualify_a_changed_descendant(self):
        self.first.parent_ids = [self.base.id]
        self.assertFalse(self.solver._contains_warm_start(self.first))

    def test_nested_members_cannot_hide_a_warm_start(self):
        inner = self.compound([self.base, self.first])
        nested = self.compound([self.second, inner])
        self.assertTrue(self.solver._contains_warm_start(nested))

    def test_missing_members_and_cycles_fail_closed(self):
        policy = self.compound([self.first])
        policy.member_ids = ["missing"]
        with self.assertRaisesRegex(ValueError, "成员缺失"):
            self.solver._contains_warm_start(policy)
        policy.member_ids = [policy.id]
        with self.assertRaisesRegex(ValueError, "循环引用"):
            self.solver._contains_warm_start(policy)

    def test_distinct_population_capacity_retires_baseline_permanently(self):
        partial = [self.base, self.first, self.first]
        self.assertIn(self.base, self.solver._training_candidates(partial, "solver"))
        self.assertFalse(getattr(self.solver, "_warm_start_retired", False))
        full = self.solver._training_candidates([self.base, self.first, self.second], "solver")
        self.assertEqual(full, [self.first, self.second])
        self.assertTrue(self.solver._warm_start_retired)
        self.assertEqual(self.solver._training_candidates(partial, "solver"), [self.first])

    def test_eoh_topk_and_cluster_remove_even_a_superior_baseline(self):
        for mode in ("topk", "cluster"):
            with self.subTest(mode=mode):
                self.solver.eoh_selection = mode
                self.solver.cluster_method = "keyword"
                seen_parents = []

                def generate(player, operator, parents, *args):
                    seen_parents.extend(parents)
                    return [self.first, self.second] if operator == "I1" else []

                with patch.object(self.solver, "_generate", side_effect=generate), \
                        patch.object(self.solver, "_fitness", side_effect=lambda p, *_:
                                     10.0 if p.id == self.base.id else 0.0):
                    response = self.solver._search_eoh_original("solver", 1, self.pools, self.meta)
                self.assertFalse(self.solver._contains_warm_start(response))
                self.assertTrue(seen_parents)
                self.assertNotIn(self.base, seen_parents)

    def test_eohs_ensemble_and_mpsro_exclude_superior_baseline(self):
        for response_mode in ("pure", "mpsro"):
            with self.subTest(response_mode=response_mode):
                self.solver.eoh_framework = "eoh-s"
                self.solver.response_mode = response_mode

                def generate(player, operator, parents, *args):
                    return [self.first, self.second] if operator == "I1" else []

                with patch.object(self.solver, "_generate", side_effect=generate), \
                        patch.object(self.solver, "_eohs_scenarios",
                                     return_value=([self.solver.base], [1.0], [{"source": "base"}])), \
                        patch.object(self.solver, "_cpi_vectors", side_effect=lambda programs, *_:
                                     {p.code_hash: [0.0 if p.id == self.base.id else 1.0]
                                      for p in programs}), \
                        patch.object(self.solver, "_fitness", return_value=0.0):
                    response = self.solver._search("solver", 1, self.pools, self.meta)
                self.assertIsNotNone(response.member_ids)
                self.assertNotIn(self.base.id, response.member_ids)
                self.assertFalse(self.solver._contains_warm_start(response))

    def test_mpsro_filters_baseline_even_if_search_returns_it(self):
        self.solver.response_mode = "mpsro"
        self.solver._response_candidates["solver"] = [self.base, self.first]
        with patch.object(self.solver, "_search_eoh_original", return_value=self.base), \
                patch.object(self.solver, "_fitness", return_value=0.0):
            response = self.solver._search("solver", 1, self.pools, self.meta)
        self.assertEqual(response.member_ids, [self.first.id])

    def test_eoh_and_eohs_only_baseline_fail_without_exporting(self):
        for framework in ("eoh", "eoh-s"):
            with self.subTest(framework=framework):
                self.solver.eoh_framework = framework
                with patch.object(self.solver, "_generate", return_value=[]), \
                        patch.object(self.solver, "_fitness", return_value=0.0), \
                        patch.object(self.solver, "_eohs_scenarios",
                                     return_value=([self.solver.base], [1.0], [{"source": "base"}])), \
                        patch.object(self.solver, "_cpi_vectors", side_effect=lambda programs, *_:
                                     {p.code_hash: [0.0] for p in programs}):
                    with self.assertRaisesRegex(RuntimeError, "非暖启动"):
                        self.solver._search("solver", 1, self.pools, self.meta)
                self.assertFalse((self.solver.root / "final_solver").exists())
                self.assertTrue((self.solver.root / "selection_status.json").exists())

    def test_final_selection_ignores_baseline_fitness(self):
        self.stub_evaluator()
        self.pools["solver"].append(self.first)
        with patch.object(self.solver, "_fitness", side_effect=lambda p, *_:
                          10.0 if p.id == self.base.id else 0.0):
            response = self.solver._select(self.pools, self.meta)
        self.assertEqual(response.id, self.first.id)

    def test_final_selection_refuses_warm_start_fallback(self):
        self.stub_evaluator(failed=[self.first.id])
        self.pools["solver"].append(self.first)
        with patch.object(self.solver, "_fitness", return_value=0.0):
            with self.assertRaisesRegex(RuntimeError, "拒绝回退"):
                self.solver._select(self.pools, self.meta)

    def test_final_selection_rejects_baseline_only_and_contaminated_mixtures(self):
        for policy in (self.base, self.compound([self.base, self.first], mixed=True)):
            with self.subTest(policy=policy.id):
                with self.assertRaisesRegex(RuntimeError, "非暖启动"):
                    self.solver._select({"solver": [policy]}, self.meta)

    def test_direct_export_rejects_pure_copies_mixtures_and_ensembles(self):
        for policy in (self.base, self.add(BASE_SOLVER, "LS"),
                       self.compound([self.base, self.first], mixed=True),
                       self.compound([self.second, self.base])):
            with self.subTest(policy=policy.id):
                with self.assertRaisesRegex(RuntimeError, "拒绝导出"):
                    self.solver._export(policy)
                self.assertFalse((self.solver.root / "final_solver").exists())

    def test_meta_export_removes_all_baseline_sources_and_renormalizes(self):
        contaminated = self.compound([self.base, self.first], mixed=True)
        self.solver._export(self.first)
        self.solver._export_meta([self.base, contaminated, self.first, self.second],
                                 [.2, .3, .2, .3])
        folder = self.solver.root / "final_solver"
        payload = json.loads((folder / "meta_strategy.json").read_text(encoding="utf-8"))
        self.assertEqual([p["id"] for p in payload["policies"]], [self.first.id, self.second.id])
        self.assertEqual(payload["probabilities"], [.4, .6])
        self.assertAlmostEqual(payload["removed_warm_start_probability"], .5)
        self.assertFalse((folder / f"meta_members/{self.base.id}.py").exists())

    def test_zero_nonwarm_cce_mass_uses_verified_final_not_baseline(self):
        self.solver._export(self.first)
        self.solver._export_meta([self.base, self.first], [1.0, 0.0], selected=self.first)
        payload = json.loads((self.solver.root / "final_solver/meta_strategy.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["probabilities"], [1.0])
        self.assertEqual(payload["deployment_rule"], "selected_non_warm_start_fallback_zero_cce_mass")

    def test_meta_export_only_baseline_refuses_before_writing(self):
        with self.assertRaisesRegex(RuntimeError, "拒绝导出元策略"):
            self.solver._export_meta([self.base], [1.0])
        self.assertFalse((self.solver.root / "final_solver").exists())


if __name__ == "__main__":
    unittest.main()
