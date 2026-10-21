"""候选进程隔离与可复现工作量统计；不是面向恶意代码的安全容器。"""
from __future__ import annotations

import ast
import math
import multiprocessing as mp
import os
import sys
import threading
import time
from typing import Any

from .code_parser import CodeParser, _SAFE_BUILTINS

_SPAWN_LOCK = threading.Lock()


class ExecutionError(RuntimeError, TimeoutError):
    """候选程序隔离执行失败，同时保留失败类型和已执行工作量。"""

    def __init__(self, message: str, *, lines: int = 0,
                 elapsed: float | None = None, failure_type: str = "runtime") -> None:
        super().__init__(message)
        self.lines = int(lines)
        self.elapsed = elapsed
        self.failure_type = failure_type


def validate_code(code: str, signature: str, *, max_code_lines: int | None = None,
                 max_code_chars: int | None = None,
                 max_ast_nodes: int | None = None) -> None:
    """沿用签名/禁用项检查，再限制顶层语句和函数定义时的副作用。"""
    if max_code_chars is not None and len(code) > max_code_chars:
        raise ValueError(f"候选源码超过字符上限 {max_code_chars}")
    if max_code_lines is not None and len(code.splitlines()) > max_code_lines:
        raise ValueError(f"候选源码超过行数上限 {max_code_lines}")
    CodeParser().parse(f"EXPLANATION: validation\n```python\n{code}\n```", signature)
    tree = ast.parse(code)
    if max_ast_nodes is not None and sum(1 for _ in ast.walk(tree)) > max_ast_nodes:
        raise ValueError(f"候选源码超过 AST 复杂度上限 {max_ast_nodes}")
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("顶层只能有一个普通函数定义")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef)):
            raise ValueError("不允许全局状态、nonlocal 或类定义")
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list or node.args.defaults or any(node.args.kw_defaults):
                raise ValueError("候选函数禁止装饰器和默认参数")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("禁止双下划线名称")
    args = tree.body[0].args
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        raise ValueError("入口函数不能使用额外参数")


def _worker(connection: Any, kind: str, arguments: tuple, line_limit: int) -> None:
    """先发送再退出，父进程先接收再 join，避免大结果堵塞管道。"""
    started = time.perf_counter()
    lines = 0

    def trace(frame: Any, event: str, arg: Any) -> Any:
        nonlocal lines
        if frame.f_code.co_filename == "<candidate>" and event == "line":
            lines += 1
            if lines > line_limit:
                raise RuntimeError("候选超过确定性执行行数预算")
        return trace

    try:
        if kind == "candidate":
            code, name, values = arguments
            namespace = {"__builtins__": _SAFE_BUILTINS}
            sys.settrace(trace)
            exec(compile(code, "<candidate>", "exec", dont_inherit=True), namespace)
            result = namespace[name](*values)
            sys.settrace(None)
        elif kind == "reference_heuristic":
            from .reference import heuristic_reference
            data, config = arguments
            result = heuristic_reference(data, config)
        elif kind == "reference_mip":
            from .reference import mip_reference
            result = mip_reference(*arguments)
        else:
            client, prompt, temperature, n, seed = arguments
            if hasattr(client, "set_seed"):
                client.set_seed(seed)
            result = client.generate(prompt, temperature=temperature, n=n)
        connection.send({"ok": True, "result": result, "lines": lines,
                         "elapsed": time.perf_counter() - started})
    except BaseException as exc:
        sys.settrace(None)
        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                         "lines": lines, "elapsed": time.perf_counter() - started})
    finally:
        connection.close()


def isolated_call(kind: str, arguments: tuple, timeout: float,
                  line_limit: int = 2000000) -> dict[str, Any]:
    """Windows spawn 可用；LLM 客户端须可 pickle，超时会杀死调用进程。"""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout 必须为正有限数")
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(send, kind, arguments, line_limit), daemon=True)
    started = time.perf_counter()
    try:
        # 子解释器在启动时读取哈希种子，保证 set 等结构在同版本环境下可复现。
        # 启动后恢复父进程环境；锁避免本模块并发启动时相互覆盖。
        with _SPAWN_LOCK:
            previous = os.environ.get("PYTHONHASHSEED")
            os.environ["PYTHONHASHSEED"] = "0"
            try:
                process.start()
            finally:
                if previous is None:
                    os.environ.pop("PYTHONHASHSEED", None)
                else:
                    os.environ["PYTHONHASHSEED"] = previous
        send.close()
        if not receive.poll(timeout):
            raise ExecutionError(f"{kind} 调用超过 {timeout} 秒（含进程启动）",
                                 failure_type="timeout")
        payload = receive.recv()
        payload["wall_time"] = time.perf_counter() - started
        if not payload["ok"]:
            error = str(payload.get("error", "candidate execution failed"))
            if "行数预算" in error or "line" in error.lower():
                failure_type = "line_limit"
            elif "超时" in error or "timeout" in error.lower():
                failure_type = "timeout"
            else:
                failure_type = "runtime"
            raise ExecutionError(error, lines=payload.get("lines", 0),
                                 elapsed=payload.get("elapsed"),
                                 failure_type=failure_type)
        return payload
    finally:
        if process.pid is not None:
            process.join(0.2)
            if process.is_alive():
                process.terminate()
                process.join(2)
            process.close()
        receive.close()
        send.close()


def execute(code: str, name: str, *arguments: Any, timeout: float = 10,
            line_limit: int = 2000000, max_code_lines: int | None = None,
            max_code_chars: int | None = None,
            max_ast_nodes: int | None = None) -> dict[str, Any]:
    """执行前再次校验，返回结果、候选行数、纯执行时间和含启动时间。"""
    signature = "solve_snd(instance)" if name == "solve_snd" else "generate_snd(base_instance, seed)"
    validate_code(code, signature, max_code_lines=max_code_lines,
                  max_code_chars=max_code_chars, max_ast_nodes=max_ast_nodes)
    return isolated_call("candidate", (code, name, arguments), timeout, line_limit)
