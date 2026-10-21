"""多主体 SND 共演化公共接口；延迟导入避免每个候选子进程加载 SciPy。"""
from __future__ import annotations

__all__ = ["MultiAgentSNDSolver", "MultiAgentResult", "UtilityConfig", "CCEMetaSolver", "OfflineMultiAgentClient", "LiteLLMClient", "EoHSConfig"]


def __getattr__(name: str):
    if name == "LiteLLMClient":
        from .llm_client import LiteLLMClient
        return LiteLLMClient
    if name in ("MultiAgentSNDSolver", "MultiAgentResult"):
        from .solver import MultiAgentSNDSolver, MultiAgentResult
        return {"MultiAgentSNDSolver": MultiAgentSNDSolver, "MultiAgentResult": MultiAgentResult}[name]
    if name == "UtilityConfig":
        from .utilities import UtilityConfig
        return UtilityConfig
    if name == "CCEMetaSolver":
        from .game import CCEMetaSolver
        return CCEMetaSolver
    if name == "OfflineMultiAgentClient":
        from .baselines import OfflineMultiAgentClient
        return OfflineMultiAgentClient
    if name == "EoHSConfig":
        from .eoh_s import EoHSConfig
        return EoHSConfig
    raise AttributeError(name)
