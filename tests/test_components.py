"""迁移后的共享组件回归：提示词保持不变，解析器和客户端无需旧包。"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from multi_agent_snd import LiteLLMClient, UtilityConfig
from multi_agent_snd.code_parser import CodeParser
from multi_agent_snd.operators import (ALL_OPERATOR_CODES, OPERATOR_PROMPT_FILE,
                                       OPERATOR_TEMPLATES, PROMPT_SECTIONS,
                                       load_operator_prompts, render_parents)
from multi_agent_snd.prompts import build_prompt


class TestSharedComponents(unittest.TestCase):
    def test_all_operators_render_five_sections(self) -> None:
        for player in ("solver", "cost", "path", "work"):
            for operator in ALL_OPERATOR_CODES:
                parents = [SimpleNamespace(id=f"p{i}", explanation="父代思想", code="def f(x):\n    return x")
                           for i in range(OPERATOR_TEMPLATES[operator].parent_count)]
                prompt = build_prompt(player, operator, parents, "保留容量约束", UtilityConfig(), {})
                positions = [prompt.index(f"## {section}") for section in PROMPT_SECTIONS]
                self.assertEqual(positions, sorted(positions))
                self.assertIn(f"PLAYER={player}", prompt)
                signature = "solve_snd(instance)" if player == "solver" else "generate_snd(base_instance, seed)"
                self.assertIn(signature, prompt)

    def test_parent_counts_and_rendering(self) -> None:
        parent = {"id": "p1", "explanation": "原理", "code": "def f(x):\n    return x"}
        self.assertIn("无父代", render_parents("I1", []))
        self.assertIn("ID: p1", render_parents("M1", [parent]))
        for operator in ("E1", "E2", "M1", "M2", "M3"):
            with self.assertRaises(ValueError):
                build_prompt("solver", operator, [], "提示词", UtilityConfig(), {})

    def test_operator_file_is_complete(self) -> None:
        self.assertEqual(OPERATOR_PROMPT_FILE.parent.name, "multi_agent_snd")
        prompts = load_operator_prompts()
        self.assertEqual(set(prompts), set(ALL_OPERATOR_CODES))
        self.assertTrue(all(prompts.values()))

    def test_missing_operator_rejected(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.txt"
            path.write_text("[I1]\nonly initialization", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_operator_prompts(path)

    def test_parser_retains_explanation(self) -> None:
        explanation, code = CodeParser().parse("EXPLANATION: 容量构造\n```python\ndef solve_snd(instance):\n    return {}\n```", "solve_snd(instance)")
        self.assertEqual(explanation, "容量构造")
        self.assertIn("def solve_snd", code)
        with self.assertRaises(ValueError):
            CodeParser().parse("```python\ndef solve_snd(instance):\n    return {}\n```", "solve_snd(instance)")

    def test_client_picklable_and_local_cost_map(self) -> None:
        # 使用不存在的配置路径和临时环境，完全不读取用户密钥、不联网。
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            client = LiteLLMClient(model="openai/gpt-4o-mini", env_file=Path(tmp) / "absent.env", seed=7)
            restored = pickle.loads(pickle.dumps(client))
            self.assertEqual(restored.seed, 7)
            self.assertEqual(restored.model, "openai/gpt-4o-mini")

    def test_client_accepts_generic_model_identifiers(self) -> None:
        # 覆盖短名、API 原始 ID、旧错误写法和正确的 provider/model；不访问 API。
        aliases = {"openai/gpt-4o-mini": "openai/gpt-4o-mini",
                   "custom/model": "custom/model", "  provider/model  ": "provider/model"}
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env = Path(tmp) / "empty.env"
            for name, expected in aliases.items():
                with self.subTest(model=name):
                    self.assertEqual(LiteLLMClient(model=name, env_file=env).model, expected)

    def test_client_model_default_and_env_precedence(self) -> None:
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            env = Path(tmp) / "absent.env"
            self.assertEqual(LiteLLMClient(env_file=env).model, "gpt-4o-mini")
            with patch.dict(os.environ, {"LLM_MODEL": "v4-pro"}):
                self.assertEqual(LiteLLMClient(env_file=env).model, "v4-pro")
                self.assertEqual(LiteLLMClient(model="custom/model", env_file=env).model, "custom/model")
            for value in ("", "   "):
                with self.subTest(model=value), self.assertRaisesRegex(ValueError, "模型名不能为空"):
                    LiteLLMClient(model=value, env_file=env)

    def test_client_passes_actual_model_to_litellm(self) -> None:
        # 模拟 completion，只验证客户端实际提交的参数与当前 LiteLLM 的 provider 解析。
        # 关闭远程价格表，禁止测试依赖 GitHub 或读取用户凭据。
        # 导入 TLS 依赖时保留 Windows 的 SystemRoot 等系统变量。
        with patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"}):
            from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="candidate"))])
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {
            "LLM_API_KEY": "test-only-key", "LLM_BASE_URL": "https://api.example.com",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }, clear=True), patch("litellm.completion", return_value=response) as completion:
            for name in ("openai/model-a", "openai/model-b"):
                with self.subTest(model=name):
                    client = LiteLLMClient(model=name, env_file=Path(tmp) / "absent.env", seed=42)
                    self.assertEqual(client.generate("test", temperature=.7), ["candidate"])
                    kwargs = completion.call_args.kwargs
                    self.assertEqual(kwargs["model"], name)
                    wire_model, provider, _, _ = get_llm_provider(model=kwargs["model"],
                        api_key=kwargs["api_key"], api_base=kwargs["api_base"])
                    self.assertEqual((provider, wire_model), ("openai", name.split("/", 1)[1]))
                    self.assertEqual((kwargs["seed"], kwargs["temperature"], kwargs["n"]), (42, .7, 1))

    def test_archived_prompt_unchanged_after_migration(self) -> None:
        # 交付结果可有可无；有时验证 prompt 语义，允许历史归档中的破折号排版差异。
        latest = Path(__file__).resolve().parents[1] / "runs" / "delivery" / "latest_run.json"
        if not latest.exists():
            self.skipTest("没有交付示例")
        try:
            run = Path(json.loads(latest.read_text(encoding="utf-8"))["output_dir"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            self.skipTest("交付示例仅为本地运行产物，当前不可用")
        if not run.is_absolute():
            run = latest.parent / run
        if not (run / "result.json").exists():
            self.skipTest("交付示例仅为本地运行产物，未随核心代码发布")
        result = json.loads((run / "result.json").read_text(encoding="utf-8"))
        calls = [json.loads(line) for line in (run / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()]
        call = calls[0]  # 第一个 I1 的对手仍是各个暖启动生成器。
        base = [p for p in result["programs"] if p["operator"] == "base" and p["player"] != "solver"]
        opponents = {p["player"]: [{"id": p["id"], "probability": 1.0, "explanation": p["explanation"],
                     "code": (run / p["code_file"]).read_text(encoding="utf-8").rstrip("\n") + "\n"}] for p in base}
        # 归档源文件额外写入一个换行，实际程序源码本身已有一个换行。
        prompt = build_prompt("solver", "I1", [], result["user_prompt"], UtilityConfig(), opponents)
        self.assertEqual(prompt.replace("–", "—"), call["prompt"].replace("–", "—"))


if __name__ == "__main__":
    unittest.main()
