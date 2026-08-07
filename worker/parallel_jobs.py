"""Picklable job entrypoints for ProcessPoolExecutor.

Each job runs in its own process (multiprocessing) and shells out to
``benchmark_nc_wer.py`` so NC engine imports stay isolated. The orchestrator
fans these jobs out with asyncio + ProcessPoolExecutor.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from worker.parallel import configure_worker_threads
from worker.stt_presets import stt_slug

REPO_ROOT = Path(__file__).resolve().parents[1]


def _worker_pythonpath(worker_root: str) -> str:
    path = Path(worker_root)
    parts = [str(path / "src")]
    for pattern in (
        str(path / ".venv" / "lib" / "python*" / "site-packages"),
        str(path / ".venv" / "lib64" / "python*" / "site-packages"),
    ):
        parts.extend(glob.glob(pattern))
    return os.pathsep.join(parts)


def _build_env(job: dict[str, Any]) -> dict[str, str]:
    env = {**os.environ}
    if job.get("hecttor_api_key"):
        env["HECTTOR_API_KEY"] = job["hecttor_api_key"]
    if job.get("deepgram_api_key"):
        env["DEEPGRAM_API_KEY"] = job["deepgram_api_key"]
    if job.get("sarvam_api_key"):
        env["SARVAM_API_KEY"] = job["sarvam_api_key"]
    worker_root = job.get("livekit_worker_root") or ""
    if worker_root:
        env["LIVEKIT_WORKER_ROOT"] = worker_root
        worker_paths = _worker_pythonpath(worker_root)
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{worker_paths}{os.pathsep}{existing}" if existing else worker_paths
        )
    return env


def run_strength_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one call × engine × strength benchmark job."""
    configure_worker_threads(int(job.get("cpu_threads", 2)))

    call_id = job["call_id"]
    engine = job["engine"]
    strength = float(job["strength"])
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = run_dir / f"{call_id}_transcript.json"
    if not transcript_path.exists():
        transcript_path.write_text(
            json.dumps(job["transcript_json"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    scoring = job.get("scoring", "itn+oiwer")
    stt_provider = job.get("stt_provider", "deepgram")
    stt_model = job.get("stt_model", "nova-2")
    stt_language = job.get("stt_language", "hi")
    safe_stt = stt_slug(stt_provider, stt_model)
    label = engine
    safe_label = label.replace("/", "_")
    safe_scoring = scoring.replace("+", "_")
    output = run_dir / f"{call_id}_{safe_label}_s{strength}_{safe_scoring}_{safe_stt}.json"

    if output.exists() and job.get("reuse_cache", True):
        try:
            report = json.loads(output.read_text(encoding="utf-8"))
            return {
                "ok": True,
                "cached": True,
                "call_id": call_id,
                "engine": engine,
                "strength": strength,
                "stt_provider": stt_provider,
                "stt_model": stt_model,
                "report": report,
                "output": str(output),
            }
        except Exception:
            pass

    skip_set = {"none", "dtln", "hush", "hecttor", "bvc"} - {engine}
    turn_align = job.get("turn_align", "forced")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "worker" / "benchmark_nc_wer.py"),
        "--recording",
        str(run_dir / "placeholder.ogg"),
        "--transcript",
        str(transcript_path),
        "--call-id",
        call_id,
        "--auto-call-start-ts",
        "--segment-mode",
        "turn",
        "--turn-align",
        turn_align,
        "--scoring",
        scoring,
        "--engines",
        engine,
        "--skip-engines",
        ",".join(sorted(skip_set)),
        "--stt-provider",
        stt_provider,
        "--stt-model",
        stt_model,
        "--stt-language",
        stt_language,
        "--nc-strength",
        str(strength),
        "--hush-strength",
        str(strength),
        "--output",
        str(output),
        "--cache-dir",
        job.get("cache_dir", str(REPO_ROOT / "data" / "cache")),
    ]
    if engine == "hecttor":
        cmd.append("--all-models")
    if job.get("human_url"):
        cmd.extend(["--human-url", job["human_url"]])
    if job.get("public_url"):
        cmd.extend(["--public-url", job["public_url"]])
    if job.get("recording_url"):
        cmd.extend(["--recording-url", job["recording_url"]])
    if job.get("livekit_worker_root"):
        cmd.extend(["--livekit-worker-root", job["livekit_worker_root"]])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            check=False,
            env=_build_env(job),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not output.exists():
            err = (proc.stderr or proc.stdout or "benchmark failed").strip()
            return {
                "ok": False,
                "cached": False,
                "call_id": call_id,
                "engine": engine,
                "strength": strength,
                "stt_provider": stt_provider,
                "stt_model": stt_model,
                "error": err[-2000:],
                "output": str(output),
            }
        report = json.loads(output.read_text(encoding="utf-8"))
        return {
            "ok": True,
            "cached": False,
            "call_id": call_id,
            "engine": engine,
            "strength": strength,
            "stt_provider": stt_provider,
            "stt_model": stt_model,
            "report": report,
            "output": str(output),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "cached": False,
            "call_id": call_id,
            "engine": engine,
            "strength": strength,
            "stt_provider": stt_provider,
            "stt_model": stt_model,
            "error": str(exc),
            "output": str(output),
        }


def run_tier_job(job: dict[str, Any]) -> dict[str, Any]:
    """Run one full tier-A/C benchmark for a single call."""
    configure_worker_threads(int(job.get("cpu_threads", 2)))

    call_id = job["call_id"]
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    tier = job.get("tier", "tier_a")

    transcript_path = run_dir / f"{call_id}_transcript.json"
    transcript_path.write_text(
        json.dumps(job["transcript_json"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    suffix = {"tier_a": "tier_a", "tier_b": "tier_b", "tier_c": "tier_c"}.get(
        tier, "wer"
    )
    stt_provider = job.get("stt_provider", "deepgram")
    stt_model = job.get("stt_model", "nova-2")
    stt_language = job.get("stt_language", "hi")
    safe_stt = stt_slug(stt_provider, stt_model)
    output = run_dir / f"{call_id}_{suffix}_wer_report_{safe_stt}.json"

    skip = job.get("skip_engines", "bvc")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "worker" / "benchmark_nc_wer.py"),
        "--recording",
        str(run_dir / "placeholder.ogg"),
        "--transcript",
        str(transcript_path),
        "--call-id",
        call_id,
        "--auto-call-start-ts",
        "--segment-mode",
        job.get("segment_mode", "turn"),
        "--turn-align",
        job.get("turn_align", "forced"),
        "--scoring",
        job.get("scoring", "itn+oiwer"),
        "--all-models",
        "--skip-engines",
        skip,
        "--stt-provider",
        stt_provider,
        "--stt-model",
        stt_model,
        "--stt-language",
        stt_language,
        "--nc-strength",
        str(job.get("nc_strength", 0.5)),
        "--hush-strength",
        str(job.get("hush_strength", 0.35)),
        "--output",
        str(output),
        "--cache-dir",
        job.get("cache_dir", str(REPO_ROOT / "data" / "cache")),
    ]
    if job.get("human_url"):
        cmd.extend(["--human-url", job["human_url"]])
    if job.get("public_url"):
        cmd.extend(["--public-url", job["public_url"]])
    if job.get("recording_url"):
        cmd.extend(["--recording-url", job["recording_url"]])
    if job.get("livekit_worker_root"):
        cmd.extend(["--livekit-worker-root", job["livekit_worker_root"]])

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            check=False,
            env=_build_env(job),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or not output.exists():
            err = (proc.stderr or proc.stdout or "benchmark failed").strip()
            return {
                "ok": False,
                "call_id": call_id,
                "stt_provider": stt_provider,
                "stt_model": stt_model,
                "error": err[-2000:],
                "output": str(output),
            }
        report = json.loads(output.read_text(encoding="utf-8"))
        return {
            "ok": True,
            "call_id": call_id,
            "stt_provider": stt_provider,
            "stt_model": stt_model,
            "report": report,
            "output": str(output),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "call_id": call_id,
            "stt_provider": stt_provider,
            "stt_model": stt_model,
            "error": str(exc),
            "output": str(output),
        }
