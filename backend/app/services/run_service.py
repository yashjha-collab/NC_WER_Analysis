from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import BenchmarkRun, CallRecord, CallResult, Dataset
from worker.merge_reports import merge_run_directory
from worker.wer import aggregate_engine_ranking

REPO_ROOT = Path(__file__).resolve().parents[3]


def _tier_skip_engines(tier: str) -> str:
    if tier == "tier_a":
        return "sanas,bvc"
    if tier == "tier_b":
        return "none,dtln,hush,hecttor,bvc"
    if tier == "tier_c":
        return "none,dtln,hush,hecttor,sanas,bvc"
    return "bvc"


def _run_benchmark_subprocess(
    *,
    call: CallRecord,
    run: BenchmarkRun,
    run_dir: Path,
    tier: str,
) -> Path:
    config = run.config_json or {}
    transcript_path = run_dir / f"{call.call_log_id}_transcript.json"
    transcript_path.write_text(
        json.dumps(call.transcript_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    suffix = {
        "tier_a": "tier_a",
        "tier_b": "tier_b",
        "tier_c": "tier_c",
    }.get(tier, "wer")
    output = run_dir / f"{call.call_log_id}_{suffix}_wer_report.json"

    cmd = [
        sys.executable,
        str(REPO_ROOT / "worker" / "benchmark_nc_wer.py"),
        "--recording",
        str(run_dir / "placeholder.ogg"),
        "--transcript",
        str(transcript_path),
        "--call-id",
        call.call_log_id,
        "--auto-call-start-ts",
        "--segment-mode",
        config.get("segment_mode", "turn"),
        "--turn-align",
        config.get("turn_align", settings.default_turn_align),
        "--all-models",
        "--skip-engines",
        _tier_skip_engines(tier),
        "--stt-provider",
        config.get("stt_provider", settings.default_stt_provider),
        "--stt-model",
        config.get("stt_model", settings.default_stt_model),
        "--stt-language",
        config.get("stt_language", settings.default_stt_language),
        "--nc-strength",
        str(config.get("nc_strength", settings.default_nc_strength)),
        "--output",
        str(output),
        "--cache-dir",
        str(settings.cache_dir),
    ]

    if getattr(call, "human_url", None):
        cmd.extend(["--human-url", call.human_url])
    if call.public_url:
        cmd.extend(["--public-url", call.public_url])
    if call.recording_url:
        cmd.extend(["--recording-url", call.recording_url])
    if settings.livekit_worker_root:
        cmd.extend(["--livekit-worker-root", settings.livekit_worker_root])

    env = {**os.environ}
    if settings.hecttor_api_key:
        env["HECTTOR_API_KEY"] = settings.hecttor_api_key
    if settings.cartesia_api_key:
        env["CARTESIA_API_KEY"] = settings.cartesia_api_key
    if settings.deepgram_api_key:
        env["DEEPGRAM_API_KEY"] = settings.deepgram_api_key
    if settings.sanas_endpoint:
        env["SANAS_ENDPOINT"] = settings.sanas_endpoint
    if settings.sanas_account_id:
        env["SANAS_ACCOUNT_ID"] = settings.sanas_account_id
    if settings.sanas_account_secret:
        env["SANAS_ACCOUNT_SECRET"] = settings.sanas_account_secret
    env["SANAS_SECURE_MEDIA"] = "true" if settings.sanas_secure_media else "false"

    if settings.livekit_worker_root:
        env["LIVEKIT_WORKER_ROOT"] = settings.livekit_worker_root
        worker_src = str(Path(settings.livekit_worker_root) / "src")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{worker_src}{os.pathsep}{existing}" if existing else worker_src
        )

    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env)
    return output


async def _process_run(run_id: int) -> None:
    from app.database import SessionLocal

    async with SessionLocal() as session:
        run = await session.get(BenchmarkRun, run_id)
        if not run:
            return

        calls = await session.scalars(
            select(CallRecord).where(CallRecord.dataset_id == run.dataset_id)
        )
        call_list = list(calls.all())
        run.total_calls = len(call_list)
        run.status = "running"
        run.updated_at = datetime.now(timezone.utc)
        await session.commit()

        run_dir = settings.runs_dir / f"run_{run.id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        completed = 0
        failed = 0
        reports: list[dict] = []

        for idx, call in enumerate(call_list, start=1):
            result = await session.scalar(
                select(CallResult).where(
                    CallResult.run_id == run.id,
                    CallResult.call_id == call.id,
                )
            )
            if not result:
                result = CallResult(run_id=run.id, call_id=call.id, status="running")
                session.add(result)
                await session.commit()

            try:
                report_path = await asyncio.to_thread(
                    _run_benchmark_subprocess,
                    call=call,
                    run=run,
                    run_dir=run_dir,
                    tier=run.tier,
                )
                report = json.loads(report_path.read_text(encoding="utf-8"))
                result.status = "completed"
                result.report_json = report
                result.report_path = str(report_path)
                result.error_message = None
                reports.append(report)
                completed += 1
            except Exception as exc:  # noqa: BLE001
                result.status = "failed"
                result.error_message = str(exc)
                failed += 1

            run.completed_calls = completed
            run.failed_calls = failed
            run.progress = round(100.0 * idx / max(1, len(call_list)), 1)
            run.updated_at = datetime.now(timezone.utc)
            await session.commit()

        if reports:
            from worker.nc_engines import expected_labels_for_tier

            summary = aggregate_engine_ranking(
                reports,
                expected_labels=expected_labels_for_tier(run.tier),
            )
            summary["calls"] = len(reports)
            summary["failed_calls"] = failed
            summary["tier"] = run.tier
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
        run.updated_at = datetime.now(timezone.utc)
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
