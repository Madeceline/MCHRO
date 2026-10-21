"""程序谱系及调用日志：每次生成事件立即落盘，源码按 SHA256 去重。"""

from __future__ import annotations

import hashlib
import json
import math
import html
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def clean_json(value: Any) -> Any:
    """严格 JSON 不支持 NaN/Inf；失败值使用 null 并配合 status/error 解释。"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    """以原子替换方式保存完整结果，避免中途退出留下半份 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean_json(data), ensure_ascii=False,
                                    indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass
class Program:
    """一次生成事件是一个节点；重复源码也保留各自父代与轮次。"""

    id: str
    player: str
    role: str
    explanation: str
    code: str
    code_hash: str
    parent_ids: list[str]
    iteration: int
    eoh_iteration: int
    operator: str
    status: str = "parsed"
    error: str | None = None
    member_ids: list[str] | None = None
    # None 表示原 EoH-S 全员择优集合；非空权重表示 M-PSRO 的随机混合响应。
    mixture_weights: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProgramArchive:
    """追加式档案；淘汰、失败、重复候选均有记录，不依赖 solve 正常结束。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        (root / "programs").mkdir(parents=True, exist_ok=True)
        self.programs: list[Program] = []
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self._scope = threading.local()

    @contextmanager
    def scope(self, prefix: str):
        """并发响应使用稳定的主体命名空间，ID 不依赖线程完成顺序。"""
        self._scope.prefix, self._scope.index = prefix, 0
        try:
            yield
        finally:
            del self._scope.prefix
            del self._scope.index

    def append(self, filename: str, data: Any) -> None:
        with self._lock, (self.root / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean_json(data), ensure_ascii=False,
                                    allow_nan=False) + "\n")
            handle.flush()

    def add(self, *, player: str, role: str, explanation: str, code: str,
            parents: list[Program], iteration: int, eoh_iteration: int,
            operator: str, error: str | None = None,
            members: list[Program] | None = None,
            mixture_weights: list[float] | None = None) -> Program:
        with self._lock:
            return self._add(player=player, role=role, explanation=explanation, code=code,
                             parents=parents, iteration=iteration, eoh_iteration=eoh_iteration,
                             operator=operator, error=error, members=members,
                             mixture_weights=mixture_weights)

    def _add(self, *, player, role, explanation, code, parents, iteration,
             eoh_iteration, operator, error, members, mixture_weights) -> Program:
        if mixture_weights is not None:
            if (not members or len(members) != len(mixture_weights)
                    or any(p.player != player or p.member_ids is not None for p in members)
                    or any(not math.isfinite(w) or w <= 0 for w in mixture_weights)
                    or not math.isclose(sum(mixture_weights), 1.0, abs_tol=1e-9)):
                raise ValueError("M-PSRO 混合响应须由同一玩家的纯程序和归一化正权重组成")
        # 集合策略按成员源码哈希定址；单程序保持原有哈希和重放格式。
        identity = (json.dumps({"members": [p.code_hash for p in members],
                                "weights": mixture_weights}) if mixture_weights is not None
                    else json.dumps([p.code_hash for p in members]) if members is not None
                    else code)
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        if hasattr(self._scope, "prefix"):
            program_id = f"p_{self._scope.prefix}_{self._scope.index:05d}"
            self._scope.index += 1
        else:
            program_id = f"p{len(self.programs):05d}"
        program = Program(
            program_id, player, role, explanation, code, digest,
            [parent.id for parent in parents], iteration, eoh_iteration, operator,
            "rejected" if error else "parsed", error,
            [p.id for p in members] if members is not None else None,
            list(mixture_weights) if mixture_weights is not None else None,
        )
        if code:
            target = self.root / "programs" / f"{digest}.py"
            if not target.exists():
                target.write_text(code + "\n", encoding="utf-8")
        self.programs.append(program)
        entry = program.to_dict()
        entry.pop("code")
        entry["code_file"] = f"programs/{digest}.py" if code else None
        self.append("programs.jsonl", entry)
        return program

    def catalog(self) -> list[dict[str, Any]]:
        """结果 JSON 保留谱系元数据，代码只在 programs 目录保存一份。"""
        rows = []
        for program in self.programs:
            row = program.to_dict()
            row.pop("code")
            row["code_file"] = f"programs/{program.code_hash}.py" if program.code else None
            rows.append(row)
        return rows

    def write_browser(self, selected_id: str) -> None:
        """浏览器直接查看所有源码、解释和父代，不必在大 JSON 里搜索转义代码。"""
        parts = ['<!doctype html><meta charset="utf-8"><title>Program archive</title>',
                 '<style>body{font-family:system-ui;max-width:1100px;margin:2em auto}pre{white-space:pre-wrap;background:#f4f4f4;padding:1em}details{margin:1em 0;border:1px solid #ccc;padding:1em}</style>',
                 '<h1>Program archive / 全部程序档案</h1><p>使用浏览器 Ctrl+F 搜索 ID、算子或解释；展开节点查看源码。重复源码也是独立生成事件。</p>']
        for p in self.programs:
            title = f"{p.id} | {p.player} | T{p.iteration}/R{p.eoh_iteration} | {p.operator}"
            if p.id == selected_id:
                title += " | DEPLOYED"
            parts.append(f'<details id="{p.id}"><summary>{html.escape(title)}</summary>')
            parents = ' '.join(f'<a href="#{parent}">{parent}</a>' for parent in p.parent_ids) or 'None'
            parts.append(f'<p>Parents: {parents}</p><p>{html.escape(p.explanation)}</p>')
            if p.error:
                parts.append(f'<p>REJECTED: {html.escape(p.error)}</p>')
            parts.append(f'<pre>{html.escape(p.code)}</pre></details>')
        (self.root / "program_catalog.html").write_text('\n'.join(parts), encoding="utf-8")
