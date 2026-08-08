from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import BenchmarkRun, CallRecord, CallResult, Dataset
from app.progress import progress_store
from worker.parallel import amap_parallel
from worker.parallel_jobs import run_strength_job, run_tier_job
from worker.wer import aggregate_engine_ranking

REPO_ROOT = Path(__file__).resolve().parents[3]


def _tier_skip_engines(tier: str) -> str:
    if tier == "tier_a":
        return "bvc"
    if tier == "tier_c":
        return "none,dtln,hush,hecttor,bvc"
    return "bvc"


def _common_job_env() -> dict[str, Any]:
    env = {
        "hecttor_api_key": settings.hecttor_api_key,
        "deepgram_api_key": settings.deepgram_api_key,
        "sarvam_api_key": settings.sarvam_api_key,
        "livekit_worker_root": settings.livekit_worker_root,
        "cache_dir": str(settings.cache_dir),
        "cpu_threads": settings.worker_cpu_threads,
    }
    return env


def _apply_stt_env() -> None:
    """Mirror STT/cloud credentials into os.environ for worker subprocesses."""
    if settings.livekit_worker_root:
        os.environ["LIVEKIT_WORKER_ROOT"] = settings.livekit_worker_root
    if settings.hecttor_api_key:
        os.environ["HECTTOR_API_KEY"] = settings.hecttor_api_key
    if settings.deepgram_api_key:
        os.environ["DEEPGRAM_API_KEY"] = settings.deepgram_api_key
    if settings.sarvam_api_key:
        os.environ["SARVAM_API_KEY"] = settings.sarvam_api_key
    if settings.google_service_account_json:
        os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"] = settings.google_service_account_json
    elif os.getenv("GCS_SERVICE_ACCOUNT_JSON"):
        os.environ.setdefault(
            "GOOGLE_SERVICE_ACCOUNT_JSON", os.environ["GCS_SERVICE_ACCOUNT_JSON"]
        )
    if settings.google_cloud_project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = settings.google_cloud_project
    if settings.google_stt_location:
        os.environ["GOOGLE_STT_LOCATION"] = settings.google_stt_location


def _call_urls(call: CallRecord) -> dict[str, str | None]:
    return {
        "human_url": getattr(call, "human_url", None),
        "public_url": call.public_url,
        "recording_url": call.recording_url,
    }


def _set_parallel_env() -> None:
    os.environ.setdefault(
        "STT_ASYNC_CONCURRENCY", str(settings.stt_async_concurrency)
    )
    _apply_stt_env()


def _resolve_stt_job_configs(config: dict[str, Any]) -> list[dict[str, str]]:
    from worker.stt_presets import resolve_stt_configs

    return resolve_stt_configs(
        preset_ids=config.get("stt_preset_ids"),
        stt_configs=config.get("stt_configs"),
        stt_provider=config.get("stt_provider"),
        stt_model=config.get("stt_model"),
        stt_language=config.get("stt_language"),
        default_provider=settings.default_stt_provider,
        default_model=settings.default_stt_model,
        default_language=settings.default_stt_language,
    )


async def _process_run(run_id: int) -> None:
    """Process a benchmark run with multiprocessing across calls."""
    from app.database import SessionLocal

    _set_parallel_env()

    async with SessionLocal() as session:
        run = await session.get(BenchmarkRun, run_id)
        if not run:
            return

        # Honor the call selection made in create_run (scoped via call_ids):
        # only process CallRecords that have a CallResult row for this run.
        # When no scope was given, create_run seeds a CallResult per dataset
        # call, so this stays equivalent to "all calls".
        selected_ids = list(
            (
                await session.scalars(
                    select(CallResult.call_id).where(
                        CallResult.run_id == run.id
                    )
                )
            ).all()
        )
        call_query = select(CallRecord).where(
            CallRecord.dataset_id == run.dataset_id
        )
        if selected_ids:
            call_query = call_query.where(CallRecord.id.in_(selected_ids))
        calls = await session.scalars(call_query)
        call_list = list(calls.all())
        run.total_calls = len(call_list)
        run.status = "running"
        run.updated_at = datetime.now(timezone.utc)
        await session.commit()

        run_dir = settings.runs_dir / f"run_{run.id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = run.config_json or {}
        stt_configs = _resolve_stt_job_configs(config)

        jobs: list[dict[str, Any]] = []
        for stt in stt_configs:
            for call in call_list:
                result = await session.scalar(
                    select(CallResult).where(
                        CallResult.run_id == run.id,
                        CallResult.call_id == call.id,
                    )
                )
                if not result:
                    result = CallResult(
                        run_id=run.id, call_id=call.id, status="running"
                    )
                    session.add(result)
                else:
                    result.status = "running"
                    result.error_message = None
                jobs.append(
                    {
                        "call_id": call.call_log_id,
                        "call_db_id": call.id,
                        "transcript_json": call.transcript_json,
                        "run_dir": str(run_dir),
                        "tier": run.tier,
                        "skip_engines": _tier_skip_engines(run.tier),
                        "segment_mode": config.get("segment_mode", "turn"),
                        "turn_align": config.get(
                            "turn_align", settings.default_turn_align
                        ),
                        "scoring": config.get(
                            "scoring", settings.default_scoring
                        ),
                        "stt_provider": stt["stt_provider"],
                        "stt_model": stt["stt_model"],
                        "stt_language": stt["stt_language"],
                        "stt_id": stt["stt_id"],
                        "nc_strength": config.get(
                            "nc_strength", settings.default_nc_strength
                        ),
                        "hush_strength": config.get(
                            "hush_strength", settings.default_hush_strength
                        ),
                        **_call_urls(call),
                        **_common_job_env(),
                    }
                )
        await session.commit()

        execution_mode = str(
            config.get("execution_mode", settings.default_execution_mode)
        ).lower()
        if execution_mode == "local":
            # Single-process, sequential across calls (asyncio STT overlap still
            # happens inside each call subprocess). Give the lone worker more
            # CPU threads since it isn't competing with sibling processes.
            workers = 1
            cpu_threads = max(
                settings.worker_cpu_threads,
                os.cpu_count() or settings.worker_cpu_threads,
            )
            for job in jobs:
                job["cpu_threads"] = cpu_threads
        else:
            execution_mode = "server"
            workers = int(
                config.get("workers") or settings.worker_processes or 8
            )
            cpu_threads = settings.worker_cpu_threads

        task_id = str(run.id)
        progress_store.create(task_id, "run", jobs)

        async def _flush_run_progress() -> None:
            snap = progress_store.get(task_id)
            if not snap:
                return
            run.progress = snap.progress_pct
            run.completed_calls = snap.jobs_completed
            run.failed_calls = snap.jobs_failed
            cfg = dict(run.config_json or {})
            cfg["progress_detail"] = snap.to_dict()
            cfg["jobs_total"] = snap.jobs_total
            run.config_json = cfg
            run.updated_at = datetime.now(timezone.utc)
            await session.commit()

        async def _progress_poller() -> None:
            try:
                while True:
                    snap = progress_store.get(task_id)
                    if not snap or snap.status != "running":
                        break
                    await _flush_run_progress()
                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                pass

        def _on_job_complete(
            _done: int, _total: int, job: dict[str, Any], result: Any
        ) -> None:
            progress_store.record_job(task_id, job, result)

        poller = asyncio.create_task(_progress_poller())
        try:
            raw_results = await amap_parallel(
                run_tier_job,
                jobs,
                max_workers=workers,
                cpu_threads=cpu_threads,
                on_complete=_on_job_complete,
            )
        finally:
            poller.cancel()
            await asyncio.gather(poller, return_exceptions=True)
            await _flush_run_progress()

        completed = 0
        failed = 0
        reports: list[dict] = []
        by_job_key = {
            (j["call_id"], j["stt_provider"], j["stt_model"]): j for j in jobs
        }

        for item in raw_results:
            if isinstance(item, Exception):
                failed += 1
                continue
            if not isinstance(item, dict):
                failed += 1
                continue

            call_id = item.get("call_id")
            stt_provider = item.get("stt_provider")
            stt_model = item.get("stt_model")
            job = (
                by_job_key.get((call_id, stt_provider, stt_model))
                if call_id
                else None
            )
            call_db_id = job.get("call_db_id") if job else None
            result = None
            if call_db_id is not None:
                result = await session.scalar(
                    select(CallResult).where(
                        CallResult.run_id == run.id,
                        CallResult.call_id == call_db_id,
                    )
                )

            if item.get("ok") and item.get("report") is not None:
                if result:
                    result.status = "completed"
                    result.report_json = item["report"]
                    result.report_path = item.get("output")
                    result.error_message = None
                reports.append(item["report"])
                completed += 1
            else:
                if result:
                    result.status = "failed"
                    result.error_message = item.get("error") or "unknown error"
                failed += 1

        run.completed_calls = completed
        run.failed_calls = failed
        run.progress = 100.0
        run.updated_at = datetime.now(timezone.utc)
        cfg = dict(run.config_json or {})
        cfg["jobs_total"] = len(jobs)

        if reports:
            from worker.nc_engines import expected_labels_for_tier

            summary = aggregate_engine_ranking(
                reports,
                expected_labels=expected_labels_for_tier(run.tier),
            )
            summary["calls"] = len(reports)
            summary["failed_calls"] = failed
            summary["tier"] = run.tier
            summary["execution_mode"] = execution_mode
            summary["workers"] = workers
            summary["stt_configs"] = stt_configs
            run.summary_json = summary
            summary_path = run_dir / "summary.json"
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        run.status = "completed" if failed == 0 else "completed_with_errors"
        if completed == 0:
            run.status = "failed"
            run.error_message = "All calls failed"
        final_status = run.status
        snap = progress_store.get(task_id)
        if snap:
            progress_store.finish(task_id, status=final_status)
            cfg["progress_detail"] = progress_store.get(task_id).to_dict()  # type: ignore[union-attr]
        run.config_json = cfg
        await session.commit()


async def create_run(
    session: AsyncSession,
    *,
    dataset_id: int,
    name: str,
    tier: str,
    config: dict | None,
    call_ids: list[str] | None = None,
) -> BenchmarkRun:
    dataset = await session.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")

    run = BenchmarkRun(
        dataset_id=dataset_id,
        name=name,
        tier=tier,
        status="pending",
        config_json=config or {},
    )
    session.add(run)
    await session.flush()

    query = select(CallRecord).where(CallRecord.dataset_id == dataset_id)
    if call_ids:
        query = query.where(CallRecord.call_log_id.in_(call_ids))
    calls = await session.scalars(query)
    for call in calls.all():
        session.add(
            CallResult(run_id=run.id, call_id=call.id, status="pending")
        )

    run.total_calls = len(call_ids) if call_ids else dataset.call_count
    await session.commit()
    await session.refresh(run)
    return run


def schedule_run(run_id: int) -> None:
    asyncio.create_task(_process_run(run_id))


def _aggregate_strength_results(
    *,
    engine: str,
    strength: float,
    call_reports: list[dict],
    stt_provider: str | None = None,
    stt_model: str | None = None,
    stt_id: str | None = None,
) -> list[dict]:
    if not call_reports:
        return [
            {
                "engine": engine,
                "strength": strength,
                "stt_provider": stt_provider,
                "stt_model": stt_model,
                "stt_id": stt_id,
                "calls": 0,
                "word_weighted_wer_pct": None,
                "turn_avg_wer_pct": None,
                "error": "All calls failed",
            }
        ]

    engine_keys_seen: set[str] = set()
    for report in call_reports:
        engine_keys_seen.update(report.get("engines", {}).keys())

    relevant_keys = [k for k in engine_keys_seen if k.startswith(engine)] or [
        engine
    ]
    out: list[dict] = []
    for eng_key in sorted(relevant_keys):
        total_ref = 0
        total_edits = 0
        turn_wer_sum = 0.0
        turn_count = 0
        total_sub = 0
        total_del = 0
        total_ins = 0
        calls_scored = 0

        for report in call_reports:
            eng_data = report.get("engines", {}).get(eng_key, {})
            if not eng_data:
                continue
            calls_scored += 1
            edit_totals = eng_data.get("edit_totals", {})
            ref_words = edit_totals.get("ref_words", 0)
            s = edit_totals.get("substitutions", 0)
            d = edit_totals.get("deletions", 0)
            i = edit_totals.get("insertions", 0)
            total_ref += ref_words
            total_edits += s + d + i
            total_sub += s
            total_del += d
            total_ins += i
            for t in eng_data.get("turns", []):
                w = t.get("wer")
                if w is not None:
                    turn_wer_sum += float(w)
                    turn_count += 1

        out.append(
            {
                "engine": eng_key,
                "strength": strength,
                "stt_provider": stt_provider,
                "stt_model": stt_model,
                "stt_id": stt_id,
                "calls": calls_scored,
                "word_weighted_wer_pct": round(
                    total_edits / max(1, total_ref) * 100, 1
                ),
                "turn_avg_wer_pct": round(
                    turn_wer_sum / max(1, turn_count) * 100, 1
                ),
                "substitutions": total_sub,
                "deletions": total_del,
                "insertions": total_ins,
                "ref_words": total_ref,
            }
        )
    return out


async def start_strength_sweep(
    session: AsyncSession,
    *,
    dataset_id: int,
    dtln_strengths: list[float],
    hush_strengths: list[float],
    hecttor_strengths: list[float],
    hecttor_models: list[str],
    max_calls: int | None,
    turn_align: str,
    workers: int | None = None,
    stt_provider: str | None = None,
    stt_model: str | None = None,
    stt_language: str | None = None,
    stt_preset_ids: list[str] | None = None,
    scoring: str | None = None,
    execution_mode: str | None = None,
) -> dict:
    """Validate sweep inputs and return sweep_id + job plan (work runs in background)."""
    dataset = await session.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")

    query = select(CallRecord).where(CallRecord.dataset_id == dataset_id)
    calls_result = await session.scalars(query)
    call_list = list(calls_result.all())
    if max_calls and max_calls < len(call_list):
        call_list = call_list[:max_calls]

    trials: list[dict] = []
    if dtln_strengths:
        for s in dtln_strengths:
            trials.append({"engine": "dtln", "strength": s})
    if hush_strengths:
        for s in hush_strengths:
            trials.append({"engine": "hush", "strength": s})
    if hecttor_strengths:
        for s in hecttor_strengths:
            trials.append({"engine": "hecttor", "strength": s})

    stt_configs = _resolve_stt_job_configs(
        {
            "stt_preset_ids": stt_preset_ids,
            "stt_provider": stt_provider,
            "stt_model": stt_model,
            "stt_language": stt_language,
        }
    )
    jobs_total = len(call_list) * len(trials) * len(stt_configs)
    sweep_id = uuid.uuid4().hex

    stub_jobs = [
        {
            "call_id": call.call_log_id,
            "stt_id": stt["stt_id"],
            "stt_provider": stt["stt_provider"],
            "stt_model": stt["stt_model"],
            "engine": trial["engine"],
            "strength": trial["strength"],
        }
        for stt in stt_configs
        for trial in trials
        for call in call_list
    ]
    progress_store.create(sweep_id, "sweep", stub_jobs)

    return {
        "sweep_id": sweep_id,
        "status": "running",
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "total_calls": len(call_list),
        "turn_align": turn_align,
        "scoring": scoring or settings.default_scoring,
        "execution_mode": execution_mode or settings.default_execution_mode,
        "jobs_total": jobs_total,
        "stt_configs": stt_configs,
        "trials_count": len(trials),
    }


async def _process_strength_sweep(
    sweep_id: str,
    *,
    dataset_id: int,
    dtln_strengths: list[float],
    hush_strengths: list[float],
    hecttor_strengths: list[float],
    hecttor_models: list[str],
    max_calls: int | None,
    turn_align: str,
    workers: int | None = None,
    stt_provider: str | None = None,
    stt_model: str | None = None,
    stt_language: str | None = None,
    stt_preset_ids: list[str] | None = None,
    scoring: str | None = None,
    execution_mode: str | None = None,
) -> None:
    """Background strength sweep with live progress updates."""
    from app.database import SessionLocal

    _set_parallel_env()

    async with SessionLocal() as session:
        try:
            result = await run_strength_sweep(
                session,
                dataset_id=dataset_id,
                dtln_strengths=dtln_strengths,
                hush_strengths=hush_strengths,
                hecttor_strengths=hecttor_strengths,
                hecttor_models=hecttor_models,
                max_calls=max_calls,
                turn_align=turn_align,
                workers=workers,
                stt_provider=stt_provider,
                stt_model=stt_model,
                stt_language=stt_language,
                stt_preset_ids=stt_preset_ids,
                scoring=scoring,
                execution_mode=execution_mode,
                sweep_id=sweep_id,
            )
            progress_store.finish(sweep_id, status="completed", result=result)
        except Exception as exc:
            progress_store.finish(sweep_id, status="failed", error=str(exc))


async def run_strength_sweep(
    session: AsyncSession,
    *,
    dataset_id: int,
    dtln_strengths: list[float],
    hush_strengths: list[float],
    hecttor_strengths: list[float],
    hecttor_models: list[str],
    max_calls: int | None,
    turn_align: str,
    workers: int | None = None,
    stt_provider: str | None = None,
    stt_model: str | None = None,
    stt_language: str | None = None,
    stt_preset_ids: list[str] | None = None,
    scoring: str | None = None,
    execution_mode: str | None = None,
    sweep_id: str | None = None,
) -> dict:
    """Strength sweep with multiprocessing fan-out across call×engine×strength."""
    _set_parallel_env()

    dataset = await session.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")

    query = select(CallRecord).where(CallRecord.dataset_id == dataset_id)
    calls_result = await session.scalars(query)
    call_list = list(calls_result.all())
    if max_calls and max_calls < len(call_list):
        call_list = call_list[:max_calls]

    run_dir = (
        settings.runs_dir
        / f"sweep_{dataset_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    trials: list[dict] = []
    if dtln_strengths:
        for s in dtln_strengths:
            trials.append({"engine": "dtln", "strength": s})
    if hush_strengths:
        for s in hush_strengths:
            trials.append({"engine": "hush", "strength": s})
    if hecttor_strengths:
        for s in hecttor_strengths:
            trials.append({"engine": "hecttor", "strength": s})

    stt_configs = _resolve_stt_job_configs(
        {
            "stt_preset_ids": stt_preset_ids,
            "stt_provider": stt_provider,
            "stt_model": stt_model,
            "stt_language": stt_language,
        }
    )
    scoring_mode = scoring or settings.default_scoring
    mode = str(execution_mode or settings.default_execution_mode).lower()
    if mode == "local":
        max_workers = 1
        sweep_cpu_threads = max(
            settings.worker_cpu_threads,
            os.cpu_count() or settings.worker_cpu_threads,
        )
    else:
        mode = "server"
        max_workers = max(1, int(workers or settings.worker_processes or 8))
        sweep_cpu_threads = settings.worker_cpu_threads

    jobs: list[dict[str, Any]] = []
    for stt in stt_configs:
        for trial in trials:
            for call in call_list:
                jobs.append(
                    {
                        "call_id": call.call_log_id,
                        "transcript_json": call.transcript_json,
                        "run_dir": str(run_dir),
                        "engine": trial["engine"],
                        "strength": trial["strength"],
                        "turn_align": turn_align,
                        "scoring": scoring_mode,
                        "stt_provider": stt["stt_provider"],
                        "stt_model": stt["stt_model"],
                        "stt_language": stt["stt_language"],
                        "stt_id": stt["stt_id"],
                        "reuse_cache": True,
                        **_call_urls(call),
                        **_common_job_env(),
                    }
                )

    if mode == "local":
        for job in jobs:
            job["cpu_threads"] = sweep_cpu_threads

    task_id = sweep_id or uuid.uuid4().hex
    if not progress_store.get(task_id):
        progress_store.create(task_id, "sweep", jobs)

    def _on_job_complete(
        _done: int, _total: int, job: dict[str, Any], result: Any
    ) -> None:
        progress_store.record_job(task_id, job, result)

    raw_results = await amap_parallel(
        run_strength_job,
        jobs,
        max_workers=max_workers,
        cpu_threads=sweep_cpu_threads,
        on_complete=_on_job_complete,
    )

    grouped: dict[tuple[str, float, str, str], list[dict]] = defaultdict(list)
    errors = 0
    cached = 0
    for item in raw_results:
        if isinstance(item, Exception):
            errors += 1
            continue
        if not isinstance(item, dict):
            errors += 1
            continue
        if item.get("cached"):
            cached += 1
        if item.get("ok") and item.get("report"):
            key = (
                item["engine"],
                float(item["strength"]),
                item.get("stt_provider", ""),
                item.get("stt_model", ""),
            )
            grouped[key].append(item["report"])
        else:
            errors += 1

    results: list[dict] = []
    for stt in stt_configs:
        for trial in trials:
            engine = trial["engine"]
            strength = float(trial["strength"])
            key = (
                engine,
                strength,
                stt["stt_provider"],
                stt["stt_model"],
            )
            results.extend(
                _aggregate_strength_results(
                    engine=engine,
                    strength=strength,
                    call_reports=grouped.get(key, []),
                    stt_provider=stt["stt_provider"],
                    stt_model=stt["stt_model"],
                    stt_id=stt["stt_id"],
                )
            )

    return {
        "sweep_id": task_id,
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "total_calls": len(call_list),
        "turn_align": turn_align,
        "scoring": scoring_mode,
        "execution_mode": mode,
        "workers": max_workers,
        "cpu_threads_per_worker": settings.worker_cpu_threads,
        "jobs_total": len(jobs),
        "jobs_failed": errors,
        "jobs_cached": cached,
        "stt_configs": stt_configs,
        "results": results,
    }
