"""训练性能工具：有序、无嵌套线程池的映射，以及线程安全阶段计时。"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import threading
import time


_map_context = threading.local()


def ordered_map(function, items, workers: int):
    """保持结果顺序；已在线程池任务内时串行，避免 workers² 个工作线程。"""
    items = list(items)
    if workers == 1 or len(items) < 2 or getattr(_map_context, "active", False):
        return [function(item) for item in items]

    def run(item):
        _map_context.active = True
        try:
            return function(item)
        finally:
            _map_context.active = False

    with ThreadPoolExecutor(max_workers=min(workers, len(items)),
                            thread_name_prefix="snd-eval") as pool:
        return list(pool.map(run, items))


class TimingRecorder:
    """累计的是调用耗时；并发/嵌套阶段可重叠，不能相加当作总墙钟。"""

    def __init__(self, archive):
        self.archive = archive
        self._lock = threading.Lock()
        self._stages = defaultdict(lambda: {"calls": 0, "seconds": 0.0, "failures": 0})
        self._counters = defaultdict(int)

    def count(self, name: str):
        with self._lock:
            self._counters[name] += 1

    @contextmanager
    def measure(self, stage: str, **details):
        started = time.perf_counter()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            seconds = time.perf_counter() - started
            with self._lock:
                row = self._stages[stage]
                row["calls"] += 1
                row["seconds"] += seconds
                row["failures"] += int(failed)
            self.archive.append("timings.jsonl", {"stage": stage, "seconds": seconds,
                                                    "failed": failed, **details})

    def summary(self):
        with self._lock:
            return {"scope": "overlapping_call_durations_not_additive_wall_time",
                    "stages": {k: dict(v) for k, v in sorted(self._stages.items())},
                    "counters": dict(sorted(self._counters.items()))}
