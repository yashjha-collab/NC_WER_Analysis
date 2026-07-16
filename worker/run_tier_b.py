#!/usr/bin/env python3
"""Execute tier B (Sanas-only) for a benchmark run stored in SQLite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.models import BenchmarkRun, CallRecord, CallResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_url = os.getenv("NC_WER_DATABASE_URL", "sqlite+aiosqlite:///./data/nc_wer.db").replace(
        "aiosqlite", "pysqlite"
    )
    engine = create_engine(db_url)
    run_dir = REPO_ROOT / "data" / "runs" / f"run_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with Session(engine) as session:
        run = session.get(BenchmarkRun, args.run_id)
        if not run:
            raise SystemExit(f"Run {args.run_id} not found")
        calls = session.scalars(
            select(CallRecord).where(CallRecord.dataset_id == run.dataset_id)
        ).all()

        for call in calls:
            transcript_path = run_dir / f"{call.call_log_id}_transcript.json"
            transcript_path.write_text(
                json.dumps(call.transcript_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output = run_dir / f"{call.call_log_id}_tier_b_wer_report.json"
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
                "--turn-align",
                "vad",
                "--all-models",
                "--skip-engines",
                "none,dtln,hush,hecttor,bvc",
                "--output",
                str(output),
                "--cache-dir",
                str(REPO_ROOT / "data" / "cache"),
            ]
            if call.public_url:
                cmd.extend(["--public-url", call.public_url])
            if call.recording_url:
                cmd.extend(["--recording-url", call.recording_url])
            if os.getenv("LIVEKIT_WORKER_ROOT"):
                cmd.extend(["--livekit-worker-root", os.environ["LIVEKIT_WORKER_ROOT"]])
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
