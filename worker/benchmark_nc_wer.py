#!/usr/bin/env python3
"""NC + STT + WER benchmark for golden transcripts and public recording URLs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from worker.audio_utils import (  # noqa: E402
    assign_forced_alignment_segments,
    assign_timestamp_segments,
    ensure_audio_for_call,
    infer_call_start_ts,
    load_audio_pcm,
    load_user_turns,
    refine_segments_with_vad,
    resample_pcm,
    slice_pcm,
    tighten_with_silero_vad,
    tighten_with_silero_vad_v5,
    write_transcript_file,
)
from worker.nc_engines import (  # noqa: E402
    apply_nc,
    build_nc_processor,
    list_engine_variants,
    native_sample_rate,
    skipped_engines,
)
from worker.stt import transcribe_pcm  # noqa: E402
from worker.wer import (  # noqa: E402
    explain_oiwer,
    explain_wer,
    load_noun_files,
    normalize_text,
    normalize_text_itn,
)

STT_SAMPLE_RATE = 16_000

# Per-engine defaults — Hush 0.35 from WER sweep (0.5 over-suppresses).
HUSH_DEFAULT_STRENGTH = 0.35


def resolve_nc_strength(engine: str, args: argparse.Namespace) -> float:
    if engine == "hush":
        return float(getattr(args, "hush_strength", HUSH_DEFAULT_STRENGTH))
    return float(args.nc_strength)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NC engines with WER")
    parser.add_argument("--recording", required=True, help="Path to human/user audio")
    parser.add_argument("--user-recording", default="", help="Isolated user track path")
    parser.add_argument("--transcript", required=True, help="Golden transcript JSON")
    parser.add_argument("--call-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--call-start-ts", type=float, default=None)
    parser.add_argument("--auto-call-start-ts", action="store_true")
    parser.add_argument(
        "--segment-mode",
        choices=("turn", "full"),
        default="turn",
    )
    parser.add_argument(
        "--turn-align",
        choices=("timestamp", "vad", "forced"),
        default="vad",
    )
    parser.add_argument("--require-user-track", action="store_true")
    parser.add_argument("--no-vad-tighten", action="store_true",
                        help="Skip Silero VAD tightening after forced alignment")
    parser.add_argument("--vad-version", default="v4", choices=["v4", "v5"],
                        help="Silero VAD tightening version (v4=shrink-only, v5=gap-split)")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--engines", default="")
    parser.add_argument("--skip-engines", default="bvc")
    parser.add_argument("--stt-provider", default="deepgram")
    parser.add_argument("--stt-model", default="nova-2")
    parser.add_argument("--stt-language", default="hi")
    parser.add_argument("--scoring", default="wer",
                        choices=["wer", "itn", "oiwer", "itn+oiwer"],
                        help="Scoring mode: wer=plain, itn=ITN-normalized, oiwer=lattice-based, itn+oiwer=both")
    parser.add_argument("--noun-files", nargs="*", default=[],
                        help="JSON files with noun/variation maps (e.g. golden set misinterpretations)")
    parser.add_argument("--min-segment-s", type=float, default=0.2,
                        help="Segments shorter than this are excluded pre-STT (default 0.2s)")
    parser.add_argument("--short-empty-s", type=float, default=0.8,
                        help="If segment < this AND STT returns empty, exclude instead of scoring 100%% WER")
    parser.add_argument("--nc-strength", type=float, default=0.5)
    parser.add_argument(
        "--hush-strength",
        type=float,
        default=HUSH_DEFAULT_STRENGTH,
        help="Hush-specific strength (default 0.35; 0.5 over-suppresses speech)",
    )
    parser.add_argument("--livekit-worker-root", default="")
    parser.add_argument("--public-url", default="")
    parser.add_argument("--human-url", default="")
    parser.add_argument("--recording-url", default="")
    parser.add_argument("--cache-dir", default="")
    return parser.parse_args()


def resolve_recording(args: argparse.Namespace) -> Path:
    user_path = Path(args.user_recording or args.recording)
    if user_path.exists():
        return user_path

    if args.human_url or args.public_url or args.recording_url:
        cache_dir = Path(args.cache_dir or REPO_ROOT / "data" / "cache")
        return ensure_audio_for_call(
            call_log_id=args.call_id,
            human_url=args.human_url or None,
            public_url=args.public_url or None,
            recording_url=args.recording_url or None,
            cache_dir=cache_dir,
        )

    raise FileNotFoundError(f"Recording not found: {user_path}")


def score_engine(
    *,
    label: str,
    engine: str,
    model: str | None,
    pcm: np.ndarray,
    sample_rate: int,
    turns,
    args: argparse.Namespace,
) -> dict:
    strength = resolve_nc_strength(engine, args)
    processor = None
    if engine != "none":
        processor = build_nc_processor(
            engine,
            model,
            strength=strength,
            worker_root=args.livekit_worker_root or None,
        )

    turn_rows: list[dict] = []
    wer_values: list[float] = []
    exact_matches = 0
    total_s = total_d = total_i = total_ref = 0
    scored_turns = 0
    excluded_turns = 0
    seen_ts: set[float] = set()

    if args.segment_mode == "full":
        processed, nc_sr = apply_nc(pcm, sample_rate, engine, processor)
        stt_pcm = resample_pcm(processed, nc_sr, STT_SAMPLE_RATE)
        hypothesis = transcribe_pcm(
            stt_pcm,
            STT_SAMPLE_RATE,
            provider=args.stt_provider,
            model=args.stt_model,
            language=args.stt_language,
        )
        reference = " ".join(t.reference for t in turns)
        detail = explain_wer(reference, hypothesis)
        turn_rows.append(
            {
                "turn": 1,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": detail["wer"],
                "wer_pct": detail["wer_pct"],
                "wer_detail": detail,
            }
        )
        wer_values.append(detail["wer"])
        total_s += detail["substitutions"]
        total_d += detail["deletions"]
        total_i += detail["insertions"]
        total_ref += detail["n_ref"]
        scored_turns = 1
        if normalize_text(reference) == normalize_text(hypothesis):
            exact_matches = 1
    else:
        for turn in turns:
            if turn.start_s is None or turn.end_s is None:
                continue

            # Golden often has two user lines with identical createdAt (manual
            # edits). The second cannot own a real audio span — exclude it.
            ts_key = round(float(turn.created_at), 3)
            duplicate_ts = ts_key in seen_ts
            seen_ts.add(ts_key)

            segment = slice_pcm(pcm, sample_rate, turn.start_s, turn.end_s)
            seg_dur_s = len(segment) / sample_rate
            min_seg = getattr(args, "min_segment_s", 0.5)
            if seg_dur_s < min_seg:
                detail = explain_wer(turn.reference, "")
                detail["reasons"] = [
                    f"Audio slice too short "
                    f"({seg_dur_s:.2f}s < {min_seg}s) "
                    f"→ excluded from WER average"
                ] + detail["reasons"]
                detail["primary_reason"] = detail["reasons"][0]
                turn_rows.append(
                    {
                        "turn": turn.turn,
                        "reference": turn.reference,
                        "hypothesis": "",
                        "wer": detail["wer"],
                        "wer_pct": detail["wer_pct"],
                        "start_s": turn.start_s,
                        "end_s": turn.end_s,
                        "wer_detail": detail,
                        "excluded_from_avg": True,
                        "exclude_reason": "too_short",
                    }
                )
                excluded_turns += 1
                continue

            if duplicate_ts:
                detail = explain_wer(turn.reference, "")
                detail["reasons"] = [
                    "Duplicate createdAt with an earlier user turn "
                    "→ no unique audio window; excluded from WER average"
                ] + detail["reasons"]
                detail["primary_reason"] = detail["reasons"][0]
                turn_rows.append(
                    {
                        "turn": turn.turn,
                        "reference": turn.reference,
                        "hypothesis": "",
                        "wer": None,
                        "wer_pct": None,
                        "start_s": turn.start_s,
                        "end_s": turn.end_s,
                        "wer_detail": detail,
                        "excluded_from_avg": True,
                        "exclude_reason": "duplicate_timestamp",
                    }
                )
                excluded_turns += 1
                continue

            processed, nc_sr = apply_nc(segment, sample_rate, engine, processor)
            stt_pcm = resample_pcm(processed, nc_sr, STT_SAMPLE_RATE)
            hypothesis = transcribe_pcm(
                stt_pcm,
                STT_SAMPLE_RATE,
                provider=args.stt_provider,
                model=args.stt_model,
                language=args.stt_language,
            )

            short_empty_thresh = getattr(args, "short_empty_s", 0.8)
            if not hypothesis.strip() and seg_dur_s < short_empty_thresh:
                detail = explain_wer(turn.reference, "")
                detail["reasons"] = [
                    f"Short segment ({seg_dur_s:.2f}s < {short_empty_thresh}s) "
                    f"with empty STT → excluded from WER average"
                ] + detail["reasons"]
                detail["primary_reason"] = detail["reasons"][0]
                turn_rows.append(
                    {
                        "turn": turn.turn,
                        "reference": turn.reference,
                        "hypothesis": "",
                        "wer": detail["wer"],
                        "wer_pct": detail["wer_pct"],
                        "start_s": turn.start_s,
                        "end_s": turn.end_s,
                        "wer_detail": detail,
                        "excluded_from_avg": True,
                        "exclude_reason": "short_empty",
                    }
                )
                excluded_turns += 1
                continue

            scoring = getattr(args, "scoring", "wer")
            if scoring == "itn+oiwer":
                ref_itn = normalize_text_itn(turn.reference)
                hyp_itn = normalize_text_itn(hypothesis)
                detail = explain_oiwer(ref_itn, hyp_itn)
            elif scoring == "oiwer":
                detail = explain_oiwer(turn.reference, hypothesis)
            elif scoring == "itn":
                ref_itn = normalize_text_itn(turn.reference)
                hyp_itn = normalize_text_itn(hypothesis)
                detail = explain_wer(ref_itn, hyp_itn)
            else:
                detail = explain_wer(turn.reference, hypothesis)
            n_ref = int(detail["n_ref"])
            n_hyp = int(detail["n_hyp"])
            pathological = n_ref > 0 and n_hyp > max(8, n_ref * 4)
            if pathological:
                detail["reasons"] = [
                    f"Pathological hypothesis ({n_hyp} hyp vs {n_ref} ref) "
                    f"— likely wrong window or STT hallucination loop; "
                    f"excluded from WER average"
                ] + detail["reasons"]
                detail["primary_reason"] = detail["reasons"][0]

            row = {
                "turn": turn.turn,
                "reference": turn.reference,
                "hypothesis": hypothesis,
                "wer": detail["wer"],
                "wer_pct": detail["wer_pct"],
                "start_s": turn.start_s,
                "end_s": turn.end_s,
                "wer_detail": detail,
                "excluded_from_avg": pathological,
                "exclude_reason": "pathological_hypothesis" if pathological else None,
            }
            turn_rows.append(row)
            if pathological:
                excluded_turns += 1
                continue

            if normalize_text(turn.reference) == normalize_text(hypothesis):
                exact_matches += 1
            wer_values.append(detail["wer"])
            total_s += detail["substitutions"]
            total_d += detail["deletions"]
            total_i += detail["insertions"]
            total_ref += detail["n_ref"]
            scored_turns += 1

    micro_wer = (
        (total_s + total_d + total_i) / total_ref if total_ref > 0 else None
    )
    # Prefer micro (word-weighted) as the primary call metric.
    if micro_wer is not None:
        avg_wer = round(micro_wer, 4)
    elif wer_values:
        avg_wer = round(sum(wer_values) / len(wer_values), 4)
    else:
        avg_wer = 1.0
    turn_avg = (
        round(sum(wer_values) / len(wer_values), 4) if wer_values else None
    )
    engine_formula = (
        f"Σ(S+D+I)/ΣN = ({total_s}+{total_d}+{total_i})/{total_ref}"
        if total_ref
        else "no scored reference words"
    )
    top_reasons = _top_wer_reasons(turn_rows)
    return {
        "engine": engine,
        "nc_model": model,
        "nc_strength": strength,
        "file_sample_rate": sample_rate,
        "nc_sample_rate": (
            native_sample_rate(engine) or sample_rate
            if engine != "none"
            else sample_rate
        ),
        "avg_wer": avg_wer,
        "avg_wer_pct": round(avg_wer * 100, 1),
        "turn_avg_wer": turn_avg,
        "turn_avg_wer_pct": round(turn_avg * 100, 1) if turn_avg is not None else None,
        "exact_match_rate_pct": round(
            100.0 * exact_matches / max(1, scored_turns), 1
        ),
        "scored_turns": scored_turns,
        "excluded_turns": excluded_turns,
        "nonempty_hypothesis_turns": sum(
            1 for t in turn_rows if (t.get("hypothesis") or "").strip()
        ),
        "edit_totals": {
            "substitutions": total_s,
            "deletions": total_d,
            "insertions": total_i,
            "ref_words": total_ref,
            "formula": engine_formula,
        },
        "high_wer_reasons": top_reasons,
        "primary_reason": top_reasons[0] if top_reasons else None,
        "turns": turn_rows,
    }


def _top_wer_reasons(turn_rows: list[dict], limit: int = 5) -> list[str]:
    """Aggregate high-WER reasons; scored-turn issues first, exclusions last."""
    scored: list[tuple[float, str]] = []
    excluded_notes: list[str] = []
    for row in turn_rows:
        detail = row.get("wer_detail") or {}
        if row.get("excluded_from_avg"):
            reason = row.get("exclude_reason") or "excluded"
            primary = detail.get("primary_reason")
            if not primary:
                reasons = detail.get("reasons") or []
                primary = reasons[0] if reasons else None
            excluded_notes.append(
                f"Turn {row.get('turn')} excluded ({reason})"
                + (f": {primary}" if primary else "")
            )
            continue
        raw_wer = row.get("wer")
        if raw_wer is None:
            continue
        wer = float(raw_wer)
        if wer < 0.35:
            continue
        for reason in detail.get("reasons") or []:
            scored.append((wer, reason))
        if wer >= 0.8 and detail.get("formula"):
            scored.append(
                (
                    wer,
                    f"Turn {row.get('turn')}: {detail['formula']}",
                )
            )

    scored.sort(key=lambda x: x[0], reverse=True)
    seen: set[str] = set()
    out: list[str] = []
    for _, reason in scored:
        if reason in seen:
            continue
        seen.add(reason)
        out.append(reason)
        if len(out) >= limit:
            return out

    for note in excluded_notes:
        if note in seen:
            continue
        seen.add(note)
        out.append(note)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

    if args.noun_files:
        loaded = load_noun_files(args.noun_files)
        print(f"Loaded {len(loaded)} nouns from {len(args.noun_files)} file(s)")

    transcript_path = Path(args.transcript)
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    turns = load_user_turns(transcript)
    if not turns:
        raise SystemExit("No user turns found in transcript")

    recording_path = resolve_recording(args)
    pcm, sample_rate = load_audio_pcm(recording_path)
    audio_duration_s = len(pcm) / sample_rate

    call_start_ts = args.call_start_ts
    if call_start_ts is None and args.auto_call_start_ts:
        call_start_ts = infer_call_start_ts(transcript)
    if call_start_ts is None:
        call_start_ts = infer_call_start_ts(transcript)

    if args.turn_align == "forced":
        turns = assign_forced_alignment_segments(pcm, sample_rate, turns)
        if not args.no_vad_tighten:
            if args.vad_version == "v5":
                turns = tighten_with_silero_vad_v5(pcm, sample_rate, turns)
            else:
                turns = tighten_with_silero_vad(pcm, sample_rate, turns)
    else:
        turns = assign_timestamp_segments(
            turns,
            call_start_ts,
            audio_duration_s,
            transcript=transcript,
        )
        if args.turn_align == "vad":
            turns = refine_segments_with_vad(pcm, sample_rate, turns)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()] or None
    skip = {e.strip() for e in args.skip_engines.split(",") if e.strip()}
    variants = list_engine_variants(
        all_models=args.all_models,
        engines=engines,
        skip_engines=skip,
    )

    source_marker = recording_path.parent / "source_url.txt"
    audio_source_url = (
        source_marker.read_text(encoding="utf-8").strip()
        if source_marker.exists()
        else None
    )

    report = {
        "call_id": args.call_id,
        "recording": str(recording_path),
        "audio_source_url": audio_source_url,
        "audio_is_human_track": bool(
            audio_source_url and "/human.ogg" in audio_source_url
        ),
        "segment_mode": args.segment_mode,
        "turn_align": args.turn_align,
        "user_utterance_count": len(turns),
        "stt": {
            "provider": args.stt_provider,
            "model": args.stt_model,
            "language": args.stt_language,
        },
        "engines": {},
        "skipped_engines": skipped_engines(skip),
        "reference_turns": [
            {
                "turn": t.turn,
                "reference": t.reference,
                "created_at": t.created_at,
                "start_s": t.start_s,
                "end_s": t.end_s,
            }
            for t in turns
        ],
    }

    none_wer: float | None = None
    for label, engine, model in variants:
        if engine == "bvc":
            continue
        try:
            result = score_engine(
                label=label,
                engine=engine,
                model=model,
                pcm=pcm,
                sample_rate=sample_rate,
                turns=turns,
                args=args,
            )
            if label == "none":
                none_wer = result["avg_wer"]
            elif none_wer is not None:
                result["delta_wer_vs_none"] = round(result["avg_wer"] - none_wer, 4)
            report["engines"][label] = result
            reason = result.get("primary_reason") or ""
            suffix = f" — {reason}" if reason and result["avg_wer"] >= 0.35 else ""
            print(
                f"  {label:24s} WER={result['avg_wer_pct']:5.1f}%{suffix}",
                flush=True,
            )
            edits = result.get("edit_totals") or {}
            if edits and result["avg_wer"] >= 0.35:
                print(
                    f"  {'':24s} Calculation: {edits.get('formula')} "
                    f"(S={edits.get('substitutions')} "
                    f"D={edits.get('deletions')} "
                    f"I={edits.get('insertions')})",
                    flush=True,
                )
            for extra in (result.get("high_wer_reasons") or [])[1:3]:
                print(f"  {'':24s} Reason: {extra}", flush=True)
        except Exception as exc:  # noqa: BLE001
            report["skipped_engines"][label] = str(exc)
            print(f"  {label:24s} SKIP: {exc}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote report → {output_path}", flush=True)


if __name__ == "__main__":
    main()
