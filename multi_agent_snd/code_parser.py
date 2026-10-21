"""候选源码解析及禁用项检查：仅保留当前多主体执行器使用的接口。"""
from __future__ import annotations

import ast
import re

_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


class CodeParser:
    """从 LLM 响应中分别提取原理说明和启发式程序，并进行 AST 安全校验。"""

    FORBIDDEN_CALLS = {"compile", "eval", "exec", "globals", "input", "locals", "open", "__import__"}
    FORBIDDEN_NAMES = {"os", "pathlib", "shutil", "socket", "subprocess", "sys"}

    def parse(self, raw_output: str, signature: str) -> tuple[str, str]:
        """返回 ``(explanation, code)``；任一部分缺失或不合规都会抛出异常。"""
        explanation = self._extract_explanation(raw_output)
        code = self._extract_code(raw_output)
        tree = ast.parse(code, mode="exec")
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(functions) != 1:
            raise ValueError("代码必须且只能定义一个函数")
        function = functions[0]
        if isinstance(function, ast.AsyncFunctionDef):
            raise ValueError("不支持 async 候选函数")

        expected_name, expected_params = self._parse_signature(signature)
        actual_params = [arg.arg for arg in function.args.args]
        if function.name != expected_name:
            raise ValueError(f"函数名必须是 {expected_name!r}")
        if actual_params != expected_params:
            raise ValueError(f"函数参数必须是 {expected_params!r}")

        # 在执行前拒绝导入、I/O、动态执行和常见系统访问方式。
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ValueError("候选程序禁止 import")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.FORBIDDEN_CALLS:
                    raise ValueError(f"禁止调用 {node.func.id}")
            if isinstance(node, ast.Name) and node.id in self.FORBIDDEN_NAMES:
                raise ValueError(f"禁止使用名称 {node.id}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise ValueError("禁止访问双下划线属性")
        return explanation, code

    @staticmethod
    def _extract_explanation(raw_output: str) -> str:
        # 原理说明必须位于 EXPLANATION: 与代码块之间，避免把代码注释冒充说明。
        match = re.search(
            r"(?:EXPLANATION|原理说明)\s*[:：]\s*(.*?)\s*```",
            raw_output,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not match or not match.group(1).strip():
            raise ValueError("LLM 输出缺少 EXPLANATION 自然语言原理说明")
        return match.group(1).strip()

    @staticmethod
    def _extract_code(raw_output: str) -> str:
        blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", raw_output,
                            flags=re.IGNORECASE | re.DOTALL)
        if len(blocks) != 1 or not blocks[0].strip():
            raise ValueError("LLM 输出必须包含且仅包含一个 Python 代码块")
        return blocks[0].strip()

    @staticmethod
    def _parse_signature(signature: str) -> tuple[str, list[str]]:
        match = re.search(r"(?:def\s+)?([A-Za-z_]\w*)\s*\((.*?)\)", signature or "")
        if not match:
            raise ValueError("solver_signature 无法解析")
        params = [part.split(":", 1)[0].split("=", 1)[0].strip()
                  for part in match.group(2).split(",") if part.strip()]
        return match.group(1), params
