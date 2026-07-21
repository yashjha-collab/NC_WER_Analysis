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
    return float(explain_wer(reference, hypothesis)["wer"])


def explain_wer(reference: str, hypothesis: str) -> dict:
    """Compute WER with edit counts, alignment, and human-readable reasons.

    WER = (S + D + I) / N where N = reference word count.
    """
    ref_words = tokenize(reference)
    hyp_words = tokenize(hypothesis)
    n, m = len(ref_words), len(hyp_words)

    if n == 0:
        wer = 0.0 if m == 0 else 1.0
        ops = (
            [{"op": "i", "ref": None, "hyp": w} for w in hyp_words]
            if m
            else []
        )
        return _finalize_wer_explanation(
            reference=reference,
            hypothesis=hypothesis,
            ref_words=ref_words,
            hyp_words=hyp_words,
            wer=wer,
            substitutions=0,
            deletions=0,
            insertions=m,
            correct=0,
            ops=ops,
        )

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[str]] = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
        if i:
            back[i][0] = "d"
    for j in range(m + 1):
        dp[0][j] = j
        if j:
            back[0][j] = "i"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = ref_words[i - 1] == hyp_words[j - 1]
            candidates = (
                (dp[i - 1][j] + 1, "d"),
                (dp[i][j - 1] + 1, "i"),
                (dp[i - 1][j - 1] + (0 if match else 1), "c" if match else "s"),
            )
            best_cost, best_op = min(candidates, key=lambda x: x[0])
            dp[i][j] = best_cost
            back[i][j] = best_op

    ops: list[dict] = []
    i, j = n, m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "c" or op == "s":
            ops.append(
                {
                    "op": op,
                    "ref": ref_words[i - 1],
                    "hyp": hyp_words[j - 1],
                }
            )
            i -= 1
            j -= 1
        elif op == "d":
            ops.append({"op": "d", "ref": ref_words[i - 1], "hyp": None})
            i -= 1
        elif op == "i":
            ops.append({"op": "i", "ref": None, "hyp": hyp_words[j - 1]})
            j -= 1
        else:
            break
    ops.reverse()

    substitutions = sum(1 for o in ops if o["op"] == "s")
    deletions = sum(1 for o in ops if o["op"] == "d")
    insertions = sum(1 for o in ops if o["op"] == "i")
    correct = sum(1 for o in ops if o["op"] == "c")
    wer = dp[n][m] / n

    return _finalize_wer_explanation(
        reference=reference,
        hypothesis=hypothesis,
        ref_words=ref_words,
        hyp_words=hyp_words,
        wer=wer,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        correct=correct,
        ops=ops,
    )


def _finalize_wer_explanation(
    *,
    reference: str,
    hypothesis: str,
    ref_words: list[str],
    hyp_words: list[str],
    wer: float,
    substitutions: int,
    deletions: int,
    insertions: int,
    correct: int,
    ops: list[dict],
) -> dict:
    n = len(ref_words)
    m = len(hyp_words)
    edits = substitutions + deletions + insertions
    formula = (
        f"WER = (S+D+I)/N = ({substitutions}+{deletions}+{insertions})/{n or 0}"
        f" = {edits}/{n or 0} = {wer:.4f} ({round(wer * 100, 1)}%)"
    )

    sub_pairs = [
        f"{o['ref']}→{o['hyp']}" for o in ops if o["op"] == "s"
    ][:12]
    deleted = [str(o["ref"]) for o in ops if o["op"] == "d"][:12]
    inserted = [str(o["hyp"]) for o in ops if o["op"] == "i"][:12]

    reasons = _wer_reasons(
        wer=wer,
        n_ref=n,
        n_hyp=m,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        hypothesis=hypothesis,
        sub_pairs=sub_pairs,
        deleted=deleted,
        inserted=inserted,
    )

    return {
        "wer": wer,
        "wer_pct": round(wer * 100, 1),
        "n_ref": n,
        "n_hyp": m,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "correct": correct,
        "edits": edits,
        "formula": formula,
        "ops": ops,
        "substitution_pairs": sub_pairs,
        "deleted_words": deleted,
        "inserted_words": inserted,
        "reasons": reasons,
        "primary_reason": reasons[0] if reasons else "Exact match after normalization",
        "reference_normalized": " ".join(ref_words),
        "hypothesis_normalized": " ".join(hyp_words),
        "reference_raw": reference,
        "hypothesis_raw": hypothesis,
    }


def _wer_reasons(
    *,
    wer: float,
    n_ref: int,
    n_hyp: int,
    substitutions: int,
    deletions: int,
    insertions: int,
    hypothesis: str,
    sub_pairs: list[str],
    deleted: list[str],
    inserted: list[str],
) -> list[str]:
    """Return human-readable reasons in 'Situation → explanation' form."""
    reasons: list[str] = []
    hyp_stripped = (hypothesis or "").strip()

    if n_ref == 0 and n_hyp > 0:
        reasons.append(
            "Empty reference → "
            f"hypothesis has {n_hyp} words (insertions={insertions}) → WER forced to 100%"
        )
        return reasons

    if n_ref > 0 and not hyp_stripped:
        reasons.append(
            "Empty STT → "
            f"Empty hypothesis vs {n_ref} reference words "
            f"(all deletions) → WER=100%"
        )
        return reasons

    if n_ref > 0 and n_hyp > max(3, int(n_ref * 2.5)):
        reasons.append(
            "Hyp much longer → "
            f"Hypothesis much longer than reference "
            f"({n_hyp} hyp words vs {n_ref} ref) — likely wrong audio "
            f"window, composite track, or STT hallucination"
        )

    if n_ref > 0 and n_hyp > 0 and insertions > n_ref:
        reasons.append(
            "WER > 100% → "
            f"Insertions ({insertions}) exceed reference length ({n_ref})"
        )

    # Digit/word mismatch pattern: single digit-ish ref vs spelled-out hyp.
    if (
        n_ref == 1
        and n_hyp >= 2
        and ref_looks_numeric(sub_pairs, deleted)
        and any(not w.isdigit() for w in inserted + [p.split("→")[-1] for p in sub_pairs])
    ):
        sample = ", ".join(sub_pairs[:3] + [f"+{w}" for w in inserted[:3]])
        reasons.append(
            f"Digit vs words → number form mismatch ({sample})"
        )

    dominant = max(
        (
            (substitutions, "substitutions"),
            (deletions, "deletions"),
            (insertions, "insertions"),
        ),
        key=lambda x: x[0],
    )
    if dominant[0] > 0:
        if dominant[1] == "substitutions" and sub_pairs:
            sample = ", ".join(sub_pairs[:5])
            reasons.append(
                f"Mostly substitutions → S={substitutions}: {sample}"
            )
        elif dominant[1] == "deletions" and deleted:
            sample = ", ".join(deleted[:5])
            reasons.append(
                f"Mostly deletions → D={deletions}: missing [{sample}]"
            )
        elif dominant[1] == "insertions" and inserted:
            sample = ", ".join(inserted[:5])
            reasons.append(
                f"Mostly insertions → I={insertions}: extra [{sample}]"
            )
        else:
            reasons.append(
                f"Edit mix → S={substitutions} D={deletions} I={insertions}"
            )

    if wer >= 1.0 and n_ref > 0:
        reasons.append(
            "WER ≥ 100% → "
            f"edits ({substitutions + deletions + insertions}) "
            f"≥ reference words ({n_ref})"
        )
    elif wer >= 0.5 and len(reasons) < 2:
        reasons.append(
            "High error rate → "
            f"S={substitutions} D={deletions} I={insertions} "
            f"on {n_ref} reference words"
        )

    if not reasons and wer == 0.0:
        reasons.append("Exact match → no edits after normalization")
    elif not reasons:
        reasons.append(
            f"Normal mismatch → S={substitutions} D={deletions} I={insertions} "
            f"on {n_ref} ref words"
        )

    return reasons


def ref_looks_numeric(sub_pairs: list[str], deleted: list[str]) -> bool:
    tokens = deleted[:]
    for pair in sub_pairs:
        left = pair.split("→", 1)[0]
        tokens.append(left)
    return any(t.replace(".", "", 1).isdigit() for t in tokens if t)


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
    skipped_reasons: dict[str, str] = {}
    high_wer_reasons: dict[str, list[str]] = {}
    intentional_skips = {"bvc", "sanas"}

    for report in reports:
        for label, reason in (report.get("skipped_engines") or {}).items():
            # Tier-level skips (bvc/sanas on Tier A) are expected — keep separately.
            if label in intentional_skips and not label.startswith("hecttor/") and not label.startswith("sanas/"):
                # "sanas" base key from skipped_engines() helper; real variants are sanas/MODEL
                if label == "bvc" or label == "sanas":
                    skipped_reasons.setdefault(label, str(reason))
                    continue
            skipped_acc[label] = skipped_acc.get(label, 0) + 1
            skipped_reasons.setdefault(label, str(reason)[:500])
        for label, eng in (report.get("engines") or {}).items():
            if eng.get("avg_wer") is None:
                continue
            turns = eng.get("turns") or []
            w_wer, ref_words = weighted_wer(turns)
            combined_turn_avg.setdefault(label, []).append(eng["avg_wer"])
            combined_weighted.setdefault(label, []).append((w_wer, ref_words))
            for reason in eng.get("high_wer_reasons") or []:
                bucket = high_wer_reasons.setdefault(label, [])
                if reason not in bucket and len(bucket) < 5:
                    bucket.append(str(reason)[:400])

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
            "high_wer_reasons": high_wer_reasons.get(label, []),
            "primary_reason": (high_wer_reasons.get(label) or [None])[0],
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
                "skip_reason": skipped_reasons.get(label),
            }

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
        "skipped_engine_reasons": skipped_reasons,
    }
