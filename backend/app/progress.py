"""In-memory live progress for benchmark runs and strength sweeps."""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SttProgress:
    stt_id: str
    label: str
    completed: int = 0
    total: int = 0

    @property
    def pct(self) -> float:
        if self.total <= 0:
            return 0.0
        return round(100.0 * self.completed / self.total, 1)


@dataclass
class TrialProgress:
    engine: str
    strength: float
    label: str
    completed: int = 0
    total: int = 0

    @property
    def pct(self) -> float:
        if self.total <= 0:
            return 0.0
        return round(100.0 * self.completed / self.total, 1)


@dataclass
class TaskProgress:
    task_id: str
    task_type: str  # "run" | "sweep"
    status: str = "running"
    jobs_total: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0
    jobs_cached: int = 0
    calls_total: int = 0
    calls_touched: int = 0
    calls_completed: int = 0
    stt_stats: list[SttProgress] = field(default_factory=list)
    trial_stats: list[TrialProgress] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None

    _stt_index: dict[str, SttProgress] = field(default_factory=dict, repr=False)
    _trial_index: dict[str, TrialProgress] = field(default_factory=dict, repr=False)
    _call_totals: dict[str, int] = field(default_factory=dict, repr=False)
    _call_done: dict[str, int] = field(default_factory=dict, repr=False)
    _calls_touched_set: set[str] = field(default_factory=set, repr=False)
    _calls_completed_set: set[str] = field(default_factory=set, repr=False)

    @property
    def progress_pct(self) -> float:
        if self.jobs_total <= 0:
            return 0.0
        return round(100.0 * self.jobs_completed / self.jobs_total, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress_pct": self.progress_pct,
            "jobs_total": self.jobs_total,
            "jobs_completed": self.jobs_completed,
            "jobs_failed": self.jobs_failed,
            "jobs_cached": self.jobs_cached,
            "calls_total": self.calls_total,
            "calls_touched": self.calls_touched,
            "calls_completed": self.calls_completed,
            "stt_stats": [
                {
                    "stt_id": s.stt_id,
                    "label": s.label,
                    "completed": s.completed,
                    "total": s.total,
                    "pct": s.pct,
                }
                for s in self.stt_stats
            ],
            "trial_stats": [
                {
                    "engine": t.engine,
                    "strength": t.strength,
                    "label": t.label,
                    "completed": t.completed,
                    "total": t.total,
                    "pct": t.pct,
                }
                for t in self.trial_stats
            ],
            "result": self.result,
            "error": self.error,
        }


def _stt_key(job: dict[str, Any]) -> str:
    return str(job.get("stt_id") or f"{job.get('stt_provider')}/{job.get('stt_model')}")


def _stt_label(job: dict[str, Any]) -> str:
    stt_id = job.get("stt_id")
    if stt_id:
        return str(stt_id)
    return f"{job.get('stt_provider', '?')}/{job.get('stt_model', '?')}"


def _trial_key(job: dict[str, Any]) -> str | None:
    engine = job.get("engine")
    if engine is None:
        return None
    return f"{engine}:{job.get('strength')}"


class ProgressStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, TaskProgress] = {}

    def create(self, task_id: str, task_type: str, jobs: list[dict[str, Any]]) -> TaskProgress:
        stt_totals: dict[str, int] = defaultdict(int)
        stt_labels: dict[str, str] = {}
        trial_totals: dict[str, int] = defaultdict(int)
        trial_meta: dict[str, tuple[str, float]] = {}
        call_totals: dict[str, int] = defaultdict(int)

        for job in jobs:
            sk = _stt_key(job)
            stt_totals[sk] += 1
            stt_labels[sk] = _stt_label(job)
            call_totals[str(job["call_id"])] += 1
            tk = _trial_key(job)
            if tk:
                trial_totals[tk] += 1
                trial_meta[tk] = (str(job["engine"]), float(job["strength"]))

        task = TaskProgress(
            task_id=task_id,
            task_type=task_type,
            jobs_total=len(jobs),
            calls_total=len(call_totals),
            stt_stats=[
                SttProgress(stt_id=sid, label=stt_labels[sid], total=stt_totals[sid])
                for sid in sorted(stt_totals)
            ],
            trial_stats=[
                TrialProgress(
                    engine=trial_meta[tk][0],
                    strength=trial_meta[tk][1],
                    label=f"{trial_meta[tk][0]} @ {trial_meta[tk][1]}",
                    total=trial_totals[tk],
                )
                for tk in sorted(trial_totals, key=lambda k: (trial_meta[k][0], trial_meta[k][1]))
            ],
        )
        task._stt_index = {s.stt_id: s for s in task.stt_stats}
        task._trial_index = {t.label: t for t in task.trial_stats}
        for t in task.trial_stats:
            task._trial_index[f"{t.engine}:{t.strength}"] = t
        task._call_totals = dict(call_totals)

        with self._lock:
            self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> TaskProgress | None:
        with self._lock:
            return self._tasks.get(task_id)

    def record_job(self, task_id: str, job: dict[str, Any], result: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != "running":
                return

            task.jobs_completed += 1
            ok = isinstance(result, dict) and result.get("ok")
            if isinstance(result, Exception) or not ok:
                task.jobs_failed += 1
            if isinstance(result, dict) and result.get("cached"):
                task.jobs_cached += 1

            call_id = str(job["call_id"])
            if call_id not in task._calls_touched_set:
                task._calls_touched_set.add(call_id)
                task.calls_touched += 1
            task._call_done[call_id] = task._call_done.get(call_id, 0) + 1
            if (
                call_id not in task._calls_completed_set
                and task._call_done[call_id] >= task._call_totals.get(call_id, 1)
            ):
                task._calls_completed_set.add(call_id)
                task.calls_completed += 1

            sk = _stt_key(job)
            if sk in task._stt_index:
                task._stt_index[sk].completed += 1

            tk = _trial_key(job)
            if tk and tk in task._trial_index:
                task._trial_index[tk].completed += 1

    def finish(
        self,
        task_id: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return
            task.status = status
            task.result = result
            task.error = error
            if task.jobs_total > 0 and task.jobs_completed >= task.jobs_total:
                task.jobs_completed = task.jobs_total

    def pop(self, task_id: str) -> None:
        with self._lock:
            self._tasks.pop(task_id, None)


progress_store = ProgressStore()
