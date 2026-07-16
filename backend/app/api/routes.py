from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_session
from app.models import BenchmarkRun, CallRecord, CallResult
from app.schemas import (
    CallResultSummary,
    CreateRunRequest,
    DatasetSummary,
    ImportDatasetRequest,
    RunSummary,
)
from app.services import dataset_service, run_service

router = APIRouter(prefix="/api/v1", tags=["api"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/datasets", response_model=list[DatasetSummary])
async def list_datasets(session: AsyncSession = Depends(get_session)) -> list[DatasetSummary]:
    datasets = await dataset_service.list_datasets(session)
    return [
        DatasetSummary(
            id=d.id,
            name=d.name,
            description=d.description,
            call_count=d.call_count,
            source_path=d.source_path,
            created_at=d.created_at,
        )
        for d in datasets
    ]


@router.post("/datasets/import", response_model=DatasetSummary)
async def import_dataset(
    name: str = Form(...),
    description: str | None = Form(None),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> DatasetSummary:
    dest = settings.uploads_dir / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    dataset = await dataset_service.import_manifest_file(
        session,
        name=name,
        file_path=dest,
        description=description,
    )
    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        call_count=dataset.call_count,
        source_path=dataset.source_path,
        created_at=dataset.created_at,
    )


@router.post("/datasets/import-path", response_model=DatasetSummary)
async def import_dataset_path(
    payload: ImportDatasetRequest,
    path: str = Form(...),
    session: AsyncSession = Depends(get_session),
) -> DatasetSummary:
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {path}")

    dataset = await dataset_service.import_manifest_file(
        session,
        name=payload.name,
        file_path=file_path,
        description=payload.description,
    )
    return DatasetSummary(
        id=dataset.id,
        name=dataset.name,
        description=dataset.description,
        call_count=dataset.call_count,
        source_path=dataset.source_path,
        created_at=dataset.created_at,
    )


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(session: AsyncSession = Depends(get_session)) -> list[RunSummary]:
    result = await session.scalars(
        select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc())
    )
    runs = result.all()
    return [
        RunSummary(
            id=r.id,
            dataset_id=r.dataset_id,
            name=r.name,
            tier=r.tier,
            status=r.status,
            progress=r.progress,
            total_calls=r.total_calls,
            completed_calls=r.completed_calls,
            failed_calls=r.failed_calls,
            summary_json=r.summary_json,
            error_message=r.error_message,
            created_at=r.created_at,
            updated_at=r.updated_at,
        )
        for r in runs
    ]


@router.post("/runs", response_model=RunSummary)
async def create_run(
    payload: CreateRunRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> RunSummary:
    try:
        run = await run_service.create_run(
            session,
            dataset_id=payload.dataset_id,
            name=payload.name,
            tier=payload.tier,
            config=payload.config,
            call_ids=payload.call_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    background_tasks.add_task(run_service._process_run, run.id)
    return RunSummary(
        id=run.id,
        dataset_id=run.dataset_id,
        name=run.name,
        tier=run.tier,
        status=run.status,
        progress=run.progress,
        total_calls=run.total_calls,
        completed_calls=run.completed_calls,
        failed_calls=run.failed_calls,
        summary_json=run.summary_json,
        error_message=run.error_message,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@router.get("/runs/{run_id}", response_model=RunSummary)
async def get_run(
    run_id: int, session: AsyncSession = Depends(get_session)
) -> RunSummary:
    run = await session.get(BenchmarkRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunSummary(
        id=run.id,
        dataset_id=run.dataset_id,
        name=run.name,
        tier=run.tier,
        status=run.status,
        progress=run.progress,
        total_calls=run.total_calls,
        completed_calls=run.completed_calls,
        failed_calls=run.failed_calls,
        summary_json=run.summary_json,
        error_message=run.error_message,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


@router.get("/runs/{run_id}/results", response_model=list[CallResultSummary])
async def get_run_results(
    run_id: int, session: AsyncSession = Depends(get_session)
) -> list[CallResultSummary]:
    rows = await session.execute(
        select(CallResult, CallRecord)
        .join(CallRecord, CallResult.call_id == CallRecord.id)
        .where(CallResult.run_id == run_id)
    )
    return [
        CallResultSummary(
            id=result.id,
            call_log_id=call.call_log_id,
            status=result.status,
            report_json=result.report_json,
            error_message=result.error_message,
        )
        for result, call in rows.all()
    ]
