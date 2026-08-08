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
    RunSummary,
    StrengthSweepRequest,
    StrengthSweepStartResponse,
    TaskProgressSummary,
)
from app.services import dataset_service, run_service


def _run_summary(run: BenchmarkRun) -> RunSummary:
    cfg = run.config_json or {}
    detail = cfg.get("progress_detail")
    from app.progress import progress_store

    live = progress_store.get(str(run.id))
    if live and run.status == "running":
        detail = live.to_dict()
    return RunSummary(
        id=run.id,
        dataset_id=run.dataset_id,
        name=run.name,
        tier=run.tier,
        status=run.status,
        progress=live.progress_pct if live and run.status == "running" else run.progress,
        total_calls=run.total_calls,
        completed_calls=live.jobs_completed if live and run.status == "running" else run.completed_calls,
        failed_calls=live.jobs_failed if live and run.status == "running" else run.failed_calls,
        jobs_total=cfg.get("jobs_total") or (live.jobs_total if live else None),
        progress_detail=detail,
        summary_json=run.summary_json,
        error_message=run.error_message,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )

router = APIRouter(prefix="/api/v1", tags=["api"])


@router.get("/stt-presets")
async def list_stt_presets() -> list[dict]:
    from worker.stt_presets import STT_PRESETS

    return STT_PRESETS


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
    name: str = Form(...),
    path: str = Form(...),
    description: str | None = Form(None),
    session: AsyncSession = Depends(get_session),
) -> DatasetSummary:
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {path}")

    dataset = await dataset_service.import_manifest_file(
        session,
        name=name,
        file_path=file_path,
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


@router.get("/runs", response_model=list[RunSummary])
async def list_runs(session: AsyncSession = Depends(get_session)) -> list[RunSummary]:
    result = await session.scalars(
        select(BenchmarkRun).order_by(BenchmarkRun.created_at.desc())
    )
    runs = result.all()
    return [
        _run_summary(r)
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
    return _run_summary(run)


@router.get("/runs/{run_id}", response_model=RunSummary)
async def get_run(
    run_id: int, session: AsyncSession = Depends(get_session)
) -> RunSummary:
    run = await session.get(BenchmarkRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return _run_summary(run)


@router.post("/strength-sweep", response_model=StrengthSweepStartResponse)
async def strength_sweep(
    payload: StrengthSweepRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> StrengthSweepStartResponse:
    if not (payload.dtln_strengths or payload.hush_strengths or payload.hecttor_strengths):
        raise HTTPException(status_code=400, detail="Select at least one engine/strength")

    if settings.livekit_worker_root:
        import os
        os.environ["LIVEKIT_WORKER_ROOT"] = settings.livekit_worker_root
        if settings.hecttor_api_key:
            os.environ["HECTTOR_API_KEY"] = settings.hecttor_api_key

    try:
        plan = await run_service.start_strength_sweep(
            session,
            dataset_id=payload.dataset_id,
            dtln_strengths=payload.dtln_strengths,
            hush_strengths=payload.hush_strengths,
            hecttor_strengths=payload.hecttor_strengths,
            hecttor_models=payload.hecttor_models,
            max_calls=payload.max_calls,
            turn_align=payload.turn_align,
            workers=payload.workers,
            stt_provider=payload.stt_provider,
            stt_model=payload.stt_model,
            stt_language=payload.stt_language,
            stt_preset_ids=payload.stt_preset_ids or None,
            scoring=payload.scoring,
            execution_mode=payload.execution_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    sweep_id = plan["sweep_id"]
    background_tasks.add_task(
        run_service._process_strength_sweep,
        sweep_id,
        dataset_id=payload.dataset_id,
        dtln_strengths=payload.dtln_strengths,
        hush_strengths=payload.hush_strengths,
        hecttor_strengths=payload.hecttor_strengths,
        hecttor_models=payload.hecttor_models,
        max_calls=payload.max_calls,
        turn_align=payload.turn_align,
        workers=payload.workers,
        stt_provider=payload.stt_provider,
        stt_model=payload.stt_model,
        stt_language=payload.stt_language,
        stt_preset_ids=payload.stt_preset_ids or None,
        scoring=payload.scoring,
        execution_mode=payload.execution_mode,
    )
    return StrengthSweepStartResponse(**plan)


@router.get("/strength-sweep/{sweep_id}/progress", response_model=TaskProgressSummary)
async def strength_sweep_progress(sweep_id: str) -> TaskProgressSummary:
    from app.progress import progress_store

    snap = progress_store.get(sweep_id)
    if not snap:
        raise HTTPException(status_code=404, detail="Sweep not found or expired")
    return TaskProgressSummary(**snap.to_dict())


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
