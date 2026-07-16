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
    summary_json: dict | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CallResultSummary(BaseModel):
    id: int
    call_log_id: str
    status: str
    report_json: dict | None = None
    error_message: str | None = None
