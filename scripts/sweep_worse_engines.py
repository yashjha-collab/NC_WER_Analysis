#!/usr/bin/env python3
"""Sweep methods to reduce WER for DTLN / Hush / crest-1.0 vs none.

Reuses one forced-alignment pass, then trials:
  - strength grid
  - per-slice NC vs full-file-then-slice
  - edge trim before STT (bleed reduction)

Output: data/runs/6a58b524_worse_engine_sweep.json
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from worker.audio_utils import (  # noqa: E402
    assign_forced_alignment_segments,
    load_audio_pcm,
    load_user_turns,
    resample_pcm,
    slice_pcm,
)
from worker.nc_engines import apply_frame_processor, build_nc_processor  # noqa: E402
from worker.stt import transcribe_pcm  # noqa: E402
from worker.wer import explain_wer, normalize_text  # noqa: E402

STT_SR = 16_000
CALL_ID = "6a58b524a69d400f958c05e1"
RECORDING = ROOT / "data/cache" / CALL_ID / "human.ogg"
TRANSCRIPT = ROOT / "data/uploads" / f"{CALL_ID}_transcript.json"
OUT = ROOT / "data/runs" / "6a58b524_worse_engine_sweep.json"


@dataclass
class Trial:
    engine: str
    model: str | None
    strength: float
    apply_mode: str  # per_slice | full_then_slice
    edge_trim_ms: int
    label: str


def build_trials() -> list[Trial]:
    trials: list[Trial] = []
    # Baseline none once
    trials.append(
        Trial("none", None, 0.0, "per_slice", 0, "none@baseline")
    )

    # DTLN strength + modes
    for s in (0.05, 0.1, 0.25, 0.5):
        for mode in ("per_slice", "full_then_slice"):
            trials.append(
                Trial("dtln", None, s, mode, 0, f"dtln@str{s}_{mode}")
            )
    # DTLN best-strength candidates with edge trim
    for s in (0.05, 0.1):
        trials.append(
            Trial("dtln", None, s, "per_slice", 150, f"dtln@str{s}_per_slice_trim150")
        )

    # Hush
    for s in (0.25, 0.35, 0.5):
        for mode in ("per_slice", "full_then_slice"):
            trials.append(
                Trial("hush", None, s, mode, 0, f"hush@str{s}_{mode}")
            )
    for s in (0.25, 0.35):
        trials.append(
            Trial("hush", None, s, "per_slice", 150, f"hush@str{s}_per_slice_trim150")
        )

    # crest-1.0
    for s in (0.25, 0.35, 0.5):
        for mode in ("per_slice", "full_then_slice"):
            trials.append(
                Trial(
                    "hecttor",
                    "crest-1.0",
                    s,
                    mode,
                    0,
                    f"crest-1.0@str{s}_{mode}",
                )
            )
    for s in (0.25, 0.35):
        trials.append(
            Trial(
                "hecttor",
                "crest-1.0",
                s,
                "per_slice",
                150,
                f"crest-1.0@str{s}_per_slice_trim150",
            )
        )

    # Reference: crest-2 at default (known good) for context
    trials.append(
        Trial("hecttor", "crest-2.0", 0.5, "per_slice", 0, "crest-2.0@str0.5_per_slice")
    )
    return trials


def score_trial(
    *,
    trial: Trial,
    pcm: np.ndarray,
    sample_rate: int,
    turns,
    worker_root: str,
    stt_provider: str,
    stt_model: str,
    stt_language: str,
) -> dict:
    processor = None
    if trial.engine != "none":
        processor = build_nc_processor(
            trial.engine,
            trial.model,
            strength=trial.strength,
            worker_root=worker_root,
        )

    working = pcm
    if processor and trial.apply_mode == "full_then_slice":
        working = apply_frame_processor(pcm, sample_rate, processor)

    turn_rows = []
    total_s = total_d = total_i = total_ref = 0
    scored = 0
    excluded = 0
    seen_ts: set[float] = set()
    empty_hyp = 0

    for turn in turns:
        if turn.start_s is None or turn.end_s is None:
            continue
        ts_key = round(float(turn.created_at), 3)
        duplicate = ts_key in seen_ts
        seen_ts.add(ts_key)

        start_s, end_s = turn.start_s, turn.end_s
        if trial.edge_trim_ms > 0:
            trim = trial.edge_trim_ms / 1000.0
            # Only trim if window stays >= 0.25s
            if (end_s - start_s) - 2 * trim >= 0.25:
                start_s += trim
                end_s -= trim

        segment = slice_pcm(working, sample_rate, start_s, end_s)
        if len(segment) < int(0.2 * sample_rate) or duplicate:
            excluded += 1
            turn_rows.append(
                {
                    "turn": turn.turn,
                    "excluded_from_avg": True,
                    "exclude_reason": "duplicate_timestamp" if duplicate else "too_short",
                    "reference": turn.reference,
                    "hypothesis": "",
                    "wer_pct": None,
                }
            )
            continue

        if processor and trial.apply_mode == "per_slice":
            segment = apply_frame_processor(segment, sample_rate, processor)

        stt_pcm = resample_pcm(segment, sample_rate, STT_SR)
        hyp = transcribe_pcm(
            stt_pcm,
            STT_SR,
            provider=stt_provider,
            model=stt_model,
            language=stt_language,
        )
        if not (hyp or "").strip():
            empty_hyp += 1
        detail = explain_wer(turn.reference, hyp)
        n_ref = int(detail["n_ref"])
        n_hyp = int(detail["n_hyp"])
        pathological = n_ref > 0 and n_hyp > max(8, n_ref * 4)
        row = {
            "turn": turn.turn,
            "reference": turn.reference,
            "hypothesis": hyp,
            "wer": detail["wer"] if not pathological else None,
            "wer_pct": detail["wer_pct"] if not pathological else None,
            "start_s": start_s,
            "end_s": end_s,
            "excluded_from_avg": pathological,
            "exclude_reason": "pathological_hypothesis" if pathological else None,
            "edits": {
                "S": detail["substitutions"],
                "D": detail["deletions"],
                "I": detail["insertions"],
                "n_ref": n_ref,
                "n_hyp": n_hyp,
            },
        }
        turn_rows.append(row)
        if pathological:
            excluded += 1
            continue
        total_s += detail["substitutions"]
        total_d += detail["deletions"]
        total_i += detail["insertions"]
        total_ref += detail["n_ref"]
        scored += 1

    micro = (total_s + total_d + total_i) / total_ref if total_ref else 1.0
    return {
        "label": trial.label,
        "engine": trial.engine,
        "model": trial.model,
        "strength": trial.strength,
        "apply_mode": trial.apply_mode,
        "edge_trim_ms": trial.edge_trim_ms,
        "avg_wer": round(micro, 4),
        "avg_wer_pct": round(micro * 100, 1),
        "scored_turns": scored,
        "excluded_turns": excluded,
        "empty_hyp_turns": empty_hyp,
        "edit_totals": {
            "S": total_s,
            "D": total_d,
            "I": total_i,
            "ref_words": total_ref,
        },
        "turns": turn_rows,
    }


def main() -> None:
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

    worker_root = os.getenv("LIVEKIT_WORKER_ROOT", "")
    if not worker_root:
        raise SystemExit("LIVEKIT_WORKER_ROOT not set")

    transcript = json.loads(TRANSCRIPT.read_text(encoding="utf-8"))
    turns = load_user_turns(transcript)
    pcm, sample_rate = load_audio_pcm(RECORDING)
    print(f"Aligning {len(turns)} turns (forced MMS)...", flush=True)
    turns = assign_forced_alignment_segments(pcm, sample_rate, turns)

    trials = build_trials()
    print(f"Running {len(trials)} trials...", flush=True)
    results = []
    none_wer = None
    for i, trial in enumerate(trials, 1):
        print(f"[{i}/{len(trials)}] {trial.label}", flush=True)
        row = score_trial(
            trial=trial,
            pcm=pcm,
            sample_rate=sample_rate,
            turns=turns,
            worker_root=worker_root,
            stt_provider="deepgram",
            stt_model="nova-2",
            stt_language="hi",
        )
        if trial.engine == "none":
            none_wer = row["avg_wer_pct"]
        row["delta_vs_none_pp"] = (
            round(row["avg_wer_pct"] - none_wer, 1) if none_wer is not None else None
        )
        print(
            f"  → WER={row['avg_wer_pct']}%  "
            f"Δnone={row['delta_vs_none_pp']}  "
            f"empty={row['empty_hyp_turns']}  "
            f"SDI={row['edit_totals']}",
            flush=True,
        )
        results.append(row)

    # Ranking within each engine family
    by_family: dict[str, list] = {}
    for r in results:
        key = r["model"] or r["engine"]
        by_family.setdefault(key, []).append(r)

    best = {}
    for fam, rows in by_family.items():
        ranked = sorted(rows, key=lambda x: (x["avg_wer_pct"], x["empty_hyp_turns"]))
        best[fam] = {
            "best_label": ranked[0]["label"],
            "best_wer_pct": ranked[0]["avg_wer_pct"],
            "delta_vs_none_pp": ranked[0]["delta_vs_none_pp"],
            "beats_none": ranked[0]["avg_wer_pct"] < (none_wer or 999),
            "top3": [
                {
                    "label": x["label"],
                    "wer_pct": x["avg_wer_pct"],
                    "delta_vs_none_pp": x["delta_vs_none_pp"],
                    "empty_hyp_turns": x["empty_hyp_turns"],
                    "edit_totals": x["edit_totals"],
                }
                for x in ranked[:3]
            ],
        }

    report = {
        "call_id": CALL_ID,
        "none_wer_pct": none_wer,
        "stt": {"provider": "deepgram", "model": "nova-2", "language": "hi"},
        "turn_align": "forced",
        "n_trials": len(results),
        "best_by_family": best,
        "trials": results,
        "method_notes": {
            "per_slice": "NC each turn window independently (current default)",
            "full_then_slice": "NC full call then slice turns",
            "trim150": "Trim 150ms from both edges after alignment (bleed cut)",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}", flush=True)
    print("\n=== BEST BY FAMILY ===", flush=True)
    for fam, info in best.items():
        print(
            f"{fam:12} → {info['best_label']:40} "
            f"WER={info['best_wer_pct']}% Δnone={info['delta_vs_none_pp']} "
            f"beats_none={info['beats_none']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
