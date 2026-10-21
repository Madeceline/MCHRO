"""编辑器入口回归：只构建配置，不调用在线模型或执行完整博弈。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
import unittest


DEMO_PATH = Path(__file__).resolve().parents[1] / "examples" / "multi_agent_demo.py"
SPEC = importlib.util.spec_from_file_location("multi_agent_demo", DEMO_PATH)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)


class TestDemoConfig(unittest.TestCase):
    """验证 DemoConfig 直接调用模式的静态和时间相关参数。"""

    def test_default_is_offline(self) -> None:
        config = demo.DemoConfig()
        self.assertEqual(config.mode, "offline")
        self.assertIsNone(demo.build_bounds(config))
        self.assertIsNone(config.time_periods)

    def test_api_is_explicit(self) -> None:
        config = demo.DemoConfig(mode="api", model="openai/model-a")
        self.assertEqual(config.mode, "api")
        self.assertEqual(config.model, "openai/model-a")

    def test_replay_requires_existing_log(self) -> None:
        with self.assertRaises(ValueError):
            demo.build_client(demo.DemoConfig(mode="replay"))

    def test_replay_accepts_log_with_explicit_mode(self) -> None:
        config = demo.DemoConfig(mode="replay", replay_file=Path("calls.jsonl"))
        self.assertEqual(config.replay_file, Path("calls.jsonl"))

    def test_temporal_fixed_horizon(self) -> None:
        config = demo.DemoConfig(time_periods=4, max_travel_time=2)
        self.assertIsNone(demo.build_bounds(config))
        instance = demo.SNDInstance(config.nodes, config.coverage, config.fixed_cost,
                                    seed=config.seed, time_periods=config.time_periods,
                                    max_travel_time=config.max_travel_time)
        self.assertEqual(instance.metadata["time_periods"], 4)

    def test_temporal_range(self) -> None:
        config = demo.DemoConfig(nodes_range=(4, 6), time_periods_range=(2, 4),
                                 max_travel_time=2)
        bounds = demo.build_bounds(config)
        self.assertEqual(bounds.time_periods, (2, 4))
        self.assertEqual(bounds.nodes, (4, 6))
        self.assertEqual(bounds.sample(seed=42).to_dict()["format"], "time-expanded-snd-v1")

    def test_temporal_range_alone_enables_sampling(self) -> None:
        """仅指定时间范围时，也应采样 H，而非意外退回固定静态网络。"""
        config = demo.DemoConfig(time_periods_range=(2, 4))
        bounds = demo.build_bounds(config)
        self.assertIsNotNone(bounds)
        self.assertEqual(bounds.time_periods, (2, 4))
        self.assertEqual(bounds.sample(seed=42).to_dict()["format"], "time-expanded-snd-v1")

    def test_eoh_selection_defaults_to_topk(self) -> None:
        config = demo.DemoConfig()
        self.assertEqual(config.eoh_selection, "topk")
        self.assertEqual(config.cluster_method, "kmeans")
        self.assertEqual(config.min_clusters, 2)

    def test_cluster_selection_options_are_configurable(self) -> None:
        config = demo.DemoConfig(eoh_selection="cluster", cluster_method="agglomerative",
                                 min_clusters=2, population_size=3)
        self.assertEqual(config.eoh_selection, "cluster")
        self.assertEqual(config.cluster_method, "agglomerative")
        self.assertEqual(config.min_clusters, 2)

    def test_invalid_cluster_method_is_rejected_by_solver(self) -> None:
        from multi_agent_snd import MultiAgentSNDSolver
        with self.assertRaises(ValueError):
            MultiAgentSNDSolver(demo.SNDInstance(3, .5, .5, seed=1),
                                cluster_method="unknown")


if __name__ == "__main__":
    unittest.main()
