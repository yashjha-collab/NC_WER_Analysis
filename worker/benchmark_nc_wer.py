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
    assign_timestamp_segments,
    ensure_audio_for_call,
    infer_call_start_ts,
    load_audio_pcm,
    load_user_turns,
    refine_segments_with_vad,
    resample_pcm,
    slice_pcm,
    write_transcript_file,
)
from worker.nc_engines import (  # noqa: E402
    apply_frame_processor,
    build_nc_processor,
    list_engine_variants,
    skipped_engines,
)
from worker.stt import transcribe_pcm  # noqa: E402
from worker.wer import normalize_text, word_error_rate  # noqa: E402

STT_SAMPLE_RATE = 16_000


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
        choices=("timestamp", "vad"),
        default="vad",
    )
    parser.add_argument("--require-user-track", action="store_true")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--engines", default="")
    parser.add_argument("--skip-engines", default="bvc")
    parser.add_argument("--stt-provider", default="cartesia")
    parser.add_argument("--stt-model", default="ink-whisper")
    parser.add_argument("--stt-language", default="hi")
    parser.add_argument("--nc-strength", type=float, default=0.5)
    parser.add_argument("--livekit-worker-root", default="")
    parser.add_argument("--public-url", default="")
    parser.add_argument("--recording-url", default="")
    parser.add_argument("--cache-dir", default="")
    return parser.parse_args()


def resolve_recording(args: argparse.Namespace) -> Path:
    user_path = Path(args.user_recording or args.recording)
    if user_path.exists():
        return user_path

    if args.public_url or args.recording_url:
        cache_dir = Path(args.cache_dir or REPO_ROOT / "data" / "cache")
        return ensure_audio_for_call(
            call_log_id=args.call_id,
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
    processor = None
    if engine != "none":
        processor = build_nc_processor(
            engine,
            model,
            strength=args.nc_strength,
            worker_root=args.livekit_worker_root or None,
        )

    turn_rows: list[dict] = []
    wer_values: list[float] = []
    exact_matches = 0

    if args.segment_mode == "full":
        processed = (
            apply_frame_processor(pcm, sample_rate, processor)
            if processor
            else pcm
        )
        stt_pcm = resample_pcm(processed, sample_rate, STT_SAMPLE_RATE)
        hypothesis = transcribe_pcm(
            stt_pcm,
            STT_SAMPLE_RATE,
            provider=args.stt_provider,
            model=args.stt_model,
            language=args.stt_language,
        )
        reference = " ".join(t.reference for t in turns)
        wer = word_error_rate(reference, hypothesis)
        turn_rows.append(
            {
                "turn": 1,
                "reference": reference,
                "hypothesis": hypothesis,
                "wer": wer,
                "wer_pct": round(wer * 100, 1),
            }
        )
        wer_values.append(wer)
        if normalize_text(reference) == normalize_text(hypothesis):
            exact_matches = 1
    else:
        for turn in turns:
            if turn.start_s is None or turn.end_s is None:
                continue
            segment = slice_pcm(pcm, sample_rate, turn.start_s, turn.end_s)
            if len(segment) < int(0.2 * sample_rate):
                turn_rows.append(
                    {
                        "turn": turn.turn,
                        "reference": turn.reference,
                        "hypothesis": "",
                        "wer": 1.0,
                        "wer_pct": 100.0,
                        "start_s": turn.start_s,
                        "end_s": turn.end_s,
                    }
                )
                wer_values.append(1.0)
                continue

            processed = (
                apply_frame_processor(segment, sample_rate, processor)
                if processor
                else segment
            )
            stt_pcm = resample_pcm(processed, sample_rate, STT_SAMPLE_RATE)
            hypothesis = transcribe_pcm(
                stt_pcm,
                STT_SAMPLE_RATE,
                provider=args.stt_provider,
                model=args.stt_model,
                language=args.stt_language,
            )
            wer = word_error_rate(turn.reference, hypothesis)
            if normalize_text(turn.reference) == normalize_text(hypothesis):
                exact_matches += 1
            wer_values.append(wer)
            turn_rows.append(
                {
                    "turn": turn.turn,
                    "reference": turn.reference,
                    "hypothesis": hypothesis,
                    "wer": wer,
                    "wer_pct": round(wer * 100, 1),
                    "start_s": turn.start_s,
                    "end_s": turn.end_s,
                }
            )

    avg_wer = round(sum(wer_values) / len(wer_values), 4) if wer_values else 1.0
    return {
        "engine": engine,
        "nc_model": model,
        "avg_wer": avg_wer,
        "avg_wer_pct": round(avg_wer * 100, 1),
        "exact_match_rate_pct": round(
            100.0 * exact_matches / max(1, len(turn_rows)), 1
        ),
        "nonempty_hypothesis_turns": sum(
            1 for t in turn_rows if (t.get("hypothesis") or "").strip()
        ),
        "turns": turn_rows,
    }


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()

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

    turns = assign_timestamp_segments(turns, call_start_ts, audio_duration_s)
    if args.turn_align == "vad":
        turns = refine_segments_with_vad(pcm, sample_rate, turns)

    engines = [e.strip() for e in args.engines.split(",") if e.strip()] or None
    skip = {e.strip() for e in args.skip_engines.split(",") if e.strip()}
    variants = list_engine_variants(
        all_models=args.all_models,
        engines=engines,
        skip_engines=skip,
    )

    report = {
        "call_id": args.call_id,
        "recording": str(recording_path),
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
            {"turn": t.turn, "reference": t.reference, "created_at": t.created_at}
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
            print(f"  {label:24s} WER={result['avg_wer_pct']:5.1f}%", flush=True)
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
