from __future__ import annotations

import re
from dataclasses import dataclass


HECTTOR_MODELS = (
    "coda-1.0",
    "coda-vi-1.0",
    "crest-1.0",
    "crest-2.0",
    "mist-1.0",
)

SANAS_MODELS = (
    "AGENTIC_VI_G_NC",
    "AGENTIC_ST_NC",
)

TIER_A_VARIANTS = (
    "none",
    "dtln",
    "hush",
    *(f"hecttor/{m}" for m in HECTTOR_MODELS),
)

TIER_B_VARIANTS = tuple(f"sanas/{m}" for m in SANAS_MODELS)


@dataclass
class UserTurn:
    turn: int
    reference: str
    created_at: float
    start_s: float | None = None
    end_s: float | None = None


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    return [w for w in normalize_text(text).split() if w]


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = tokenize(reference)
    hyp_words = tokenize(hypothesis)
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / n


def weighted_wer(turns: list[dict]) -> tuple[float, int]:
    total_edits = 0
    total_ref_words = 0
    for turn in turns:
        ref_words = tokenize(turn.get("reference", ""))
        if not ref_words:
            continue
        wer = turn.get("wer", 1.0)
        total_edits += int(round(wer * len(ref_words)))
        total_ref_words += len(ref_words)
    if total_ref_words == 0:
        return 1.0, 0
    return total_edits / total_ref_words, total_ref_words


def aggregate_engine_ranking(
    reports: list[dict],
    *,
    expected_labels: list[str] | None = None,
) -> dict:
    combined_turn_avg: dict[str, list[float]] = {}
    combined_weighted: dict[str, list[tuple[float, int]]] = {}
    skipped_acc: dict[str, int] = {}

    for report in reports:
        for label, reason in (report.get("skipped_engines") or {}).items():
            skipped_acc[label] = skipped_acc.get(label, 0) + 1
        for label, eng in (report.get("engines") or {}).items():
            if eng.get("avg_wer") is None:
                continue
            turns = eng.get("turns") or []
            w_wer, ref_words = weighted_wer(turns)
            combined_turn_avg.setdefault(label, []).append(eng["avg_wer"])
            combined_weighted.setdefault(label, []).append((w_wer, ref_words))

    def aggregate_weighted(pairs: list[tuple[float, int]]) -> float:
        num = sum(w * n for w, n in pairs)
        den = sum(n for _, n in pairs)
        return round(num / den, 4) if den else 1.0

    combined: dict[str, dict] = {}
    for label, values in combined_turn_avg.items():
        turn_avg = round(sum(values) / len(values), 4)
        word_w = aggregate_weighted(combined_weighted[label])
        combined[label] = {
            "turn_avg_wer": turn_avg,
            "turn_avg_wer_pct": round(turn_avg * 100, 1),
            "word_weighted_wer": word_w,
            "word_weighted_wer_pct": round(word_w * 100, 1),
            "calls": len(values),
            "status": "ok",
        }

    ranking = sorted(
        ((k, v) for k, v in combined.items() if v.get("status") == "ok"),
        key=lambda x: x[1]["word_weighted_wer"],
    )

    expected = list(expected_labels or [])
    for label in expected:
        if label not in combined:
            combined[label] = {
                "turn_avg_wer": None,
                "turn_avg_wer_pct": None,
                "word_weighted_wer": None,
                "word_weighted_wer_pct": None,
                "calls": 0,
                "status": "missing",
                "skip_count": skipped_acc.get(label, 0),
            }

    # Preserve expected order for UI, then any extras by WER.
    ordered_labels = [k for k, _ in ranking]
    for label in expected:
        if label not in ordered_labels:
            ordered_labels.append(label)
    for label in combined:
        if label not in ordered_labels:
            ordered_labels.append(label)

    return {
        "combined": {k: combined[k] for k in ordered_labels if k in combined},
        "ranking_by_word_weighted_wer": [k for k, _ in ranking],
        "expected_engines": expected,
        "skipped_engine_counts": skipped_acc,
    }
