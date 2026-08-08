from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    id: int
    name: str
    description: str | None = None
    call_count: int
    source_path: str | None = None
    created_at: datetime | None = None


class ImportDatasetRequest(BaseModel):
    name: str
    description: str | None = None


class CreateRunRequest(BaseModel):
    dataset_id: int
    name: str
    tier: str = "tier_a"
    call_ids: list[str] | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class RunSummary(BaseModel):
    id: int
    dataset_id: int
    name: str
    tier: str
    status: str
    progress: float
    total_calls: int
    completed_calls: int
    failed_calls: int
    jobs_total: int | None = None
    progress_detail: dict[str, Any] | None = None
    summary_json: dict | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class TaskProgressSummary(BaseModel):
    task_id: str
    task_type: str
    status: str
    progress_pct: float
    jobs_total: int
    jobs_completed: int
    jobs_failed: int
    jobs_cached: int
    calls_total: int
    calls_touched: int
    calls_completed: int
    stt_stats: list[dict[str, Any]] = Field(default_factory=list)
    trial_stats: list[dict[str, Any]] = Field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None


class StrengthSweepStartResponse(BaseModel):
    sweep_id: str
    status: str
    dataset_id: int
    dataset_name: str
    total_calls: int
    turn_align: str
    scoring: str
    execution_mode: str
    jobs_total: int
    stt_configs: list[dict[str, Any]] = Field(default_factory=list)
    trials_count: int = 0


class CallResultSummary(BaseModel):
    id: int
    call_log_id: str
    status: str
    report_json: dict | None = None
    error_message: str | None = None


class StrengthSweepRequest(BaseModel):
    dataset_id: int
    dtln_strengths: list[float] = Field(default_factory=list)
    hush_strengths: list[float] = Field(default_factory=list)
    hecttor_strengths: list[float] = Field(default_factory=list)
    hecttor_models: list[str] = Field(default_factory=lambda: ["coda-1.0"])
    max_calls: int | None = None
    turn_align: str = "forced"
    scoring: str = "itn+oiwer"
    execution_mode: str | None = None
    workers: int | None = None
    stt_provider: str | None = None
    stt_model: str | None = None
    stt_language: str | None = None
    stt_preset_ids: list[str] = Field(default_factory=list)
