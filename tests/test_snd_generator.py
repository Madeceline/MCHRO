"""实例生成器回归测试：静态兼容、时间语义、可行路径及可复现性。"""

from __future__ import annotations

import json
import copy
import unittest
from collections import deque

from snd_generator import SNDInstance, SNDParameterBounds
from snd_solver import SNDSolver
from multi_agent_snd.evaluation import validate_instance, validate_perturbation, validate_solution


class TestSNDGenerator(unittest.TestCase):
    def test_static_schema_unchanged(self) -> None:
        data = SNDInstance(4, 0.5, 0.7, seed=42).to_dict()
        self.assertEqual(data["format"], "static-snd-v2")
        self.assertNotIn("time_expansion", data)
        self.assertEqual(len(data["nodes"]), 4)
        self.assertEqual(data, SNDInstance(4, 0.5, 0.7, seed=42).to_dict())
        self.assertEqual(SNDParameterBounds(nodes=(4, 4)).to_dict(),
                         {"nodes": [4, 4], "coverage": [0.5, 0.5],
                          "fixed_cost": [0.5, 0.5]})

    def test_temporal_schema_and_reachability(self) -> None:
        for seed in range(12):
            data = SNDInstance(5, 0.45, 0.7, seed=seed,
                               time_periods=4, max_travel_time=3).to_dict()
            temporal = data["time_expansion"]
            node_map = {row["node_id"]: row for row in temporal["node_map"]}
            schedule = {row["arc_id"]: row for row in temporal["arc_schedule"]}
            arcs = {arc["id"]: arc for arc in data["arcs"]}
            self.assertEqual(data["format"], "time-expanded-snd-v1")
            self.assertEqual(temporal["time_points"], [0, 1, 2, 3, 4])
            self.assertEqual(len(data["nodes"]), 25)
            self.assertEqual(set(arcs), set(schedule))
            self.assertEqual(len(data["commodities"]), len(temporal["commodity_windows"]))
            self.assertEqual(data, json.loads(json.dumps(data)))

            total_demand = sum(item["demand"] for item in data["commodities"])
            adjacency: dict[int, list[int]] = {node: [] for node in data["nodes"]}
            for arc_id, arc in arcs.items():
                event = schedule[arc_id]
                tail, head = node_map[arc["tail"]], node_map[arc["head"]]
                self.assertEqual(tail["time"], event["departure_time"])
                self.assertEqual(head["time"], event["arrival_time"])
                self.assertLess(tail["time"], head["time"])
                self.assertGreaterEqual(arc["capacity"], total_demand)
                if event["kind"] == "wait":
                    self.assertEqual(tail["physical_node"], head["physical_node"])
                    self.assertEqual(arc["fixed_cost"], 0)
                    self.assertEqual(arc["variable_cost"], 0)
                    self.assertEqual(head["time"], tail["time"] + 1)
                else:
                    self.assertIn(event["kind"], {"service"})
                adjacency[arc["tail"]].append(arc["head"])

            for commodity, window in zip(data["commodities"], temporal["commodity_windows"]):
                self.assertEqual(commodity["id"], window["commodity_id"])
                self.assertEqual(node_map[commodity["origin"]]["time"], window["release_time"])
                self.assertEqual(node_map[commodity["destination"]]["time"], window["deadline"])
                self.assertLessEqual(window["release_time"], window["deadline"])
                seen = {commodity["origin"]}
                queue = deque([commodity["origin"]])
                while queue:
                    for nxt in adjacency[queue.popleft()]:
                        if nxt not in seen:
                            seen.add(nxt)
                            queue.append(nxt)
                self.assertIn(commodity["destination"], seen)

    def test_solver_consumes_temporal_instance(self) -> None:
        instance = SNDInstance(4, .5, .7, seed=42, time_periods=3,
                               max_travel_time=2)
        result = SNDSolver(instance, time_limit=20, mip_gap=0).solve()
        self.assertTrue(result.constraint_satisfaction["feasible"])
        self.assertIsNotNone(result.objective_value)
        self.assertEqual(result.time_expansion["time_points"], [0, 1, 2, 3])
        value, checks = validate_solution(instance.to_dict(), result.to_dict())
        self.assertTrue(checks["feasible"])
        self.assertAlmostEqual(value, result.objective_value)

    def test_temporal_metadata_tampering_is_rejected(self) -> None:
        base = SNDInstance(4, .5, .7, seed=42, time_periods=3).to_dict()
        wrong = copy.deepcopy(base)
        wrong["time_expansion"]["arc_schedule"][0]["arrival_time"] = 0
        with self.assertRaises(ValueError):
            validate_instance(wrong)
        wrong = copy.deepcopy(base)
        wrong["time_expansion"]["commodity_windows"][0]["deadline"] = 0
        with self.assertRaises(ValueError):
            validate_instance(wrong)
        wrong = copy.deepcopy(base)
        wrong["time_expansion"]["commodity_windows"][0]["demand"] += 1
        with self.assertRaises(ValueError):
            validate_instance(wrong)
        wrong = copy.deepcopy(base)
        wrong["arcs"][0]["fixed_cost"] *= 1.1
        self.assertGreater(validate_perturbation(base, wrong), 0)
        wrong = copy.deepcopy(base)
        wrong["time_expansion"]["arc_schedule"][0]["kind"] = "wait"
        with self.assertRaises(ValueError):
            validate_perturbation(base, wrong)

    def test_temporal_bounds_and_validation(self) -> None:
        bounds = SNDParameterBounds(nodes=(4, 6), coverage=(0.3, 0.7),
                                    time_periods=(2, 4), max_travel_time=2)
        first = bounds.sample(seed=7).to_dict()
        self.assertEqual(first, bounds.sample(seed=7).to_dict())
        self.assertIn(first["metadata"]["time_periods"], (2, 3, 4))
        self.assertEqual(bounds.to_dict()["time_periods"], [2, 4])
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SNDInstance(4, .5, .7, seed=1, time_periods=value)
        with self.assertRaises(ValueError):
            SNDParameterBounds(time_periods=(4, 2))
        with self.assertRaises(ValueError):
            SNDParameterBounds(time_periods=(2, 3), max_travel_time=0)


if __name__ == "__main__":
    unittest.main()
