"""基于 LiteLLM 的通用 OpenAI-compatible LLM 客户端。"""
from __future__ import annotations

import os
from pathlib import Path

# 使用本地价格表，避免启动时访问 GitHub 超时；不影响模型调用。
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

class LiteLLMClient:
    """从 env 文件读取模型、API Key 和 Base URL 的通用 LLM 客户端。

    推荐配置 ``LLM_MODEL``、``LLM_API_KEY`` 和 ``LLM_BASE_URL``。
    也兼容 ``API_KEY``、``BASE_URL`` 以及 OpenAI SDK 常用变量名。
    模型可以是完整的 LiteLLM 标识（例如 ``openai/gpt-4o-mini``），
    也可以是服务商支持的模型名；不对模型名做服务商专属改写。
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, model: str | None = None, env_file: str | Path | None = None,
                 seed: int = 0, timeout: float = 60.0) -> None:
        self.env_file = Path(env_file) if env_file else (
            Path(__file__).resolve().parents[1] / "llmapi.env"
        )
        self._load_env_file(self.env_file)
        requested_model = model if model is not None else os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        if not isinstance(requested_model, str) or not requested_model.strip():
            raise ValueError("模型名不能为空，请通过 --model 或 LLM_MODEL 指定模型")
        self.model = requested_model.strip()
        self.seed = seed
        self.timeout = timeout

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    @staticmethod
    def _load_env_file(path: Path) -> None:
        if not path.exists():
            return
        with path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    def generate(self, prompt: str, temperature: float = 1.0,
                 n: int = 1) -> list[str]:
        """调用配置的模型；密钥只传给 LiteLLM，不写入日志。"""
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError("未安装 litellm，请执行 `python -m pip install litellm`") from exc
        api_key = (os.getenv("LLM_API_KEY") or os.getenv("API_KEY")
                   or os.getenv("OPENAI_API_KEY"))
        base_url = (os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL")
                    or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"))
        if not api_key:
            raise RuntimeError(f"{self.env_file} 中未找到 API Key")
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "n": n,
            "seed": self.seed,
            "timeout": self.timeout,
            "api_key": api_key,
        }
        if base_url:
            request["api_base"] = base_url
        response = completion(**request)
        return [choice.message.content for choice in response.choices]
