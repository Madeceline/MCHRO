"""重放已归档 LLM 响应，避免依赖远端模型采样的不可控变化。"""
from __future__ import annotations

import json
from pathlib import Path


class ReplayClient:
    """以调用 seed 定位响应，并逐字校验 prompt；不访问网络。"""

    def __init__(self, calls_file: str | Path) -> None:
        self.calls = {row["seed"]: row for row in (
            json.loads(line) for line in Path(calls_file).read_text(encoding="utf-8").splitlines() if line.strip())}
        self.seed = 0

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def generate(self, prompt: str, temperature: float = 1, n: int = 1) -> list[str]:
        call = self.calls.get(self.seed)
        if call is None:
            raise ValueError(f"重放日志中没有 seed={self.seed} 的调用")
        if call["prompt"] != prompt or call["n"] != n or call["temperature"] != temperature:
            raise ValueError("重放配置或程序搜索轨迹与原运行不一致，请使用相同参数、环境和基础实例")
        if call["status"] != "ok":
            raise RuntimeError(call.get("error", "recorded call failure"))
        return call["raw_outputs"]
