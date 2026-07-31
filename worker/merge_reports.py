from __future__ import annotations

import json
from pathlib import Path


def merge_call_reports(
    call_id: str,
    *,
    tier_a: Path | None = None,
    tier_c: Path | None = None,
) -> dict:
    def load(path: Path | None) -> dict:
        if path and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"engines": {}, "skipped_engines": {}}

    main = load(tier_a)
    bvc = load(tier_c)

    engines = dict(main.get("engines") or {})
    for label, data in (bvc.get("engines") or {}).items():
        engines[label] = data

    skipped = {
        **(main.get("skipped_engines") or {}),
        **(bvc.get("skipped_engines") or {}),
    }
    return {
        "call_id": call_id,
        "stt": main.get("stt") or bvc.get("stt"),
        "engines": engines,
        "skipped_engines": skipped,
        "reference_turns": main.get("reference_turns")
        or bvc.get("reference_turns"),
    }


def merge_run_directory(run_dir: Path) -> dict:
    per_call: dict[str, dict] = {}
    for tier_a in run_dir.glob("*_tier_a_wer_report.json"):
        call_id = tier_a.name.replace("_tier_a_wer_report.json", "")
        tier_c = run_dir / f"{call_id}_tier_c_wer_report.json"
        per_call[call_id] = merge_call_reports(
            call_id, tier_a=tier_a, tier_c=tier_c if tier_c.exists() else None
        )

    from worker.wer import aggregate_engine_ranking

    ranking = aggregate_engine_ranking(list(per_call.values()))
    return {
        "metric": "WER (lower is better)",
        "calls": len(per_call),
        "per_call": per_call,
        **ranking,
    }
