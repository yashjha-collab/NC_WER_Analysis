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


@router.get("/diagnostics")
async def diagnostics() -> dict:
    """Probe Tier A NC engines on this machine (root, hush lib, hecttor wheel/key)."""
    import os

    from worker.diagnostics import probe_tier_a

    # Mirror keys the benchmark subprocess receives from settings.
    env_backup = {
        "HECTTOR_API_KEY": os.environ.get("HECTTOR_API_KEY"),
        "LIVEKIT_WORKER_ROOT": os.environ.get("LIVEKIT_WORKER_ROOT"),
    }
    try:
        if settings.hecttor_api_key:
            os.environ["HECTTOR_API_KEY"] = settings.hecttor_api_key
        if settings.livekit_worker_root:
            os.environ["LIVEKIT_WORKER_ROOT"] = settings.livekit_worker_root
        return probe_tier_a(settings.livekit_worker_root or None)
    finally:
        for key, value in env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


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
    from app.config import settings

    if payload.tier == "tier_a":
        if not (settings.livekit_worker_root or "").strip():
            raise HTTPException(
                status_code=400,
                detail=(
                    "LIVEKIT_WORKER_ROOT is not set in .env. "
                    "Hush/Hecttor/DTLN need the livekit-agent-worker checkout path "
                    "(e.g. /Users/.../Desktop/livekit-agent-worker). "
                    "Without it those engines fail with ModuleNotFoundError: services."
                ),
            )

        # Fail fast with the real import/runtime error (root set ≠ engines ready).
        import os

        from worker.diagnostics import probe_tier_a

        if settings.hecttor_api_key:
            os.environ["HECTTOR_API_KEY"] = settings.hecttor_api_key
        os.environ["LIVEKIT_WORKER_ROOT"] = settings.livekit_worker_root
        probe = probe_tier_a(settings.livekit_worker_root)
        if not probe.get("ok"):
            raise HTTPException(
                status_code=400,
                detail={
                    "message": probe.get("message"),
                    "failed_engines": probe.get("failed_engines"),
                    "hints": probe.get("hints"),
                    "engines": {
                        name: {
                            "ok": result.get("ok"),
                            "error": result.get("error"),
                        }
                        for name, result in (probe.get("engines") or {}).items()
                    },
                },
            )

    # Warn early if dataset has no human_url — composite audio inflates WER.
    if payload.tier in ("tier_a", "tier_b"):
        calls = await session.scalars(
            select(CallRecord).where(CallRecord.dataset_id == payload.dataset_id)
        )
        call_list = list(calls.all())
        missing_human = sum(1 for c in call_list if not (getattr(c, "human_url", None) or ""))
        if call_list and missing_human == len(call_list):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Dataset calls have no human_url. Re-import a manifest that includes "
                    "signed human_url fields — composite recording.ogg produces useless WER."
                ),
            )

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
