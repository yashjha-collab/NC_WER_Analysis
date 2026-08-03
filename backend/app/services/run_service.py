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
        return "bvc"
    if tier == "tier_c":
        return "none,dtln,hush,hecttor,bvc"
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
        "--hush-strength",
        str(config.get("hush_strength", settings.default_hush_strength)),
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
    if settings.deepgram_api_key:
        env["DEEPGRAM_API_KEY"] = settings.deepgram_api_key
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


def _run_single_engine_strength(
    *,
    call: CallRecord,
    run_dir: Path,
    engine: str,
    model: str | None,
    strength: float,
    turn_align: str,
) -> dict | None:
    config: dict = {
        "turn_align": turn_align,
        "segment_mode": "turn",
        "stt_provider": settings.default_stt_provider,
        "stt_model": settings.default_stt_model,
        "stt_language": settings.default_stt_language,
    }

    transcript_path = run_dir / f"{call.call_log_id}_transcript.json"
    if not transcript_path.exists():
        transcript_path.write_text(
            json.dumps(call.transcript_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    label = f"{engine}/{model}" if model else engine
    safe_label = label.replace("/", "_")
    output = run_dir / f"{call.call_log_id}_{safe_label}_s{strength}.json"

    skip_set = {"none", "dtln", "hush", "hecttor", "bvc"} - {engine}

    cmd = [
        sys.executable,
        str(REPO_ROOT / "worker" / "benchmark_nc_wer.py"),
        "--recording", str(run_dir / "placeholder.ogg"),
        "--transcript", str(transcript_path),
        "--call-id", call.call_log_id,
        "--auto-call-start-ts",
        "--segment-mode", "turn",
        "--turn-align", turn_align,
        "--engines", engine,
        "--skip-engines", ",".join(skip_set),
        "--stt-provider", config["stt_provider"],
        "--stt-model", config["stt_model"],
        "--stt-language", config["stt_language"],
        "--nc-strength", str(strength),
        "--hush-strength", str(strength),
        "--output", str(output),
        "--cache-dir", str(settings.cache_dir),
    ]

    if engine == "hecttor":
        cmd.append("--all-models")

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
    if settings.deepgram_api_key:
        env["DEEPGRAM_API_KEY"] = settings.deepgram_api_key
    if settings.livekit_worker_root:
        env["LIVEKIT_WORKER_ROOT"] = settings.livekit_worker_root
        worker_src = str(Path(settings.livekit_worker_root) / "src")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{worker_src}{os.pathsep}{existing}" if existing else worker_src
        )

    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT), check=True, env=env, capture_output=True)
        report = json.loads(output.read_text(encoding="utf-8"))
        return report
    except Exception:
        return None


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
) -> dict:
    dataset = await session.get(Dataset, dataset_id)
    if not dataset:
        raise ValueError("Dataset not found")

    query = select(CallRecord).where(CallRecord.dataset_id == dataset_id)
    calls_result = await session.scalars(query)
    call_list = list(calls_result.all())
    if max_calls and max_calls < len(call_list):
        call_list = call_list[:max_calls]

    run_dir = settings.runs_dir / f"sweep_{dataset_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
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

    results: list[dict] = []

    for trial in trials:
        engine = trial["engine"]
        strength = trial["strength"]

        call_reports: list[dict] = []
        for call in call_list:
            report = await asyncio.to_thread(
                _run_single_engine_strength,
                call=call,
                run_dir=run_dir,
                engine=engine,
                model=None,
                strength=strength,
                turn_align=turn_align,
            )
            if report:
                call_reports.append(report)

        if not call_reports:
            results.append({
                "engine": engine,
                "strength": strength,
                "calls": 0,
                "word_weighted_wer_pct": None,
                "turn_avg_wer_pct": None,
                "error": "All calls failed",
            })
            continue

        engine_keys_seen: set[str] = set()
        for report in call_reports:
            engine_keys_seen.update(report.get("engines", {}).keys())

        relevant_keys = [
            k for k in engine_keys_seen if k.startswith(engine)
        ] or [engine]

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

            ww_wer = round(total_edits / max(1, total_ref) * 100, 1)
            ta_wer = round(turn_wer_sum / max(1, turn_count) * 100, 1)

            results.append({
                "engine": eng_key,
                "strength": strength,
                "calls": calls_scored,
                "word_weighted_wer_pct": ww_wer,
                "turn_avg_wer_pct": ta_wer,
                "substitutions": total_sub,
                "deletions": total_del,
                "insertions": total_ins,
                "ref_words": total_ref,
            })

    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.name,
        "total_calls": len(call_list),
        "turn_align": turn_align,
        "stt": {
            "provider": settings.default_stt_provider,
            "model": settings.default_stt_model,
            "language": settings.default_stt_language,
        },
        "results": results,
    }
