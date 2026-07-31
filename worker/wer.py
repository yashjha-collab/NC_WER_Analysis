from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from num2words import num2words


# ---------------------------------------------------------------------------
# Cross-script (Latin ↔ Devanagari) consonant-skeleton fuzzy matching
# ---------------------------------------------------------------------------
_DEV_CONSONANT_MAP: dict[str, str] = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "f", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क़": "q", "ख़": "kh", "ग़": "gh", "ज़": "z", "फ़": "f",
    "ड़": "d", "ढ़": "dh", "ङ": "ng", "ञ": "n",
}

_CROSSSCRIPT_THRESHOLD = 0.35


def _is_devanagari_token(tok: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in tok)


def _consonant_skeleton(token: str) -> str:
    if _is_devanagari_token(token):
        parts: list[str] = []
        for ch in token:
            if ch in _DEV_CONSONANT_MAP:
                parts.append(_DEV_CONSONANT_MAP[ch])
            elif ch == "\u0902":  # anusvara
                parts.append("n")
        return "".join(parts)
    return re.sub(r"[aeiou]", "", token.lower())


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1,
                          prev[j - 1] + (0 if ca == cb else 1))
        prev = curr
    return prev[-1]


def _crossscript_match(a: str, b: str) -> bool:
    """Return True if two tokens are likely the same word across scripts."""
    a_dev = _is_devanagari_token(a)
    b_dev = _is_devanagari_token(b)
    if a_dev == b_dev:
        return False
    sa = _consonant_skeleton(a)
    sb = _consonant_skeleton(b)
    if not sa or not sb:
        return False
    dist = _edit_distance(sa, sb)
    norm = dist / max(len(sa), len(sb))
    return norm <= _CROSSSCRIPT_THRESHOLD


HECTTOR_MODELS = (
    "coda-1.0",
    "coda-vi-1.0",
    "crest-1.0",
    "crest-2.0",
    "mist-1.0",
)

TIER_A_VARIANTS = (
    "none",
    "dtln",
    "hush",
    *(f"hecttor/{m}" for m in HECTTOR_MODELS),
)


@dataclass
class UserTurn:
    turn: int
    reference: str
    created_at: float
    start_s: float | None = None
    end_s: float | None = None


# ---------------------------------------------------------------------------
# Custom domain synonyms — each set of strings is treated as equivalent.
# Add your own nouns, brand names, Hindi↔English equivalents here.
# ---------------------------------------------------------------------------
CUSTOM_SYNONYMS: list[set[str]] = [
    # Hindi confirmations / fillers
    {"हाँ", "हां", "han", "haan", "haa"},
    {"जी", "ji", "jee", "g"},
    {"नहीं", "nahi", "nahin", "nah"},
    {"ठीक", "theek", "thik", "theek"},
    {"अच्छा", "achha", "acha", "accha"},
    # Common business / IndiaMART nouns
    {"bakery", "बेकरी", "bekri"},
    {"mixer", "मिक्सर", "mixar"},
    {"machine", "मशीन", "masheen", "mashin"},
    {"electric", "इलेक्ट्रिक", "electrik", "electrical"},
    {"toast", "टोस्ट", "tost", "tos"},
    {"bread", "ब्रेड", "bred"},
    {"cashback", "कैशबैक", "cash back"},
    {"message", "मैसेज", "msg", "mesaj"},
    {"doctor", "डॉक्टर", "dr", "daktar"},
    {"hotel", "होटल", "hotal"},
    {"laptop", "लैपटॉप", "labtop"},
    {"whatsapp", "व्हाट्सएप", "watsapp", "whats app"},
    {"google pay", "गूगल पे", "gpay", "g pay"},
    # Units
    {"ml", "एमएल", "एम एल", "milliliter", "millilitre"},
    {"kg", "किलो", "किलोग्राम", "kilogram", "kilograms", "kilo"},
    {"km", "किलोमीटर", "kilometer", "kilometres"},
    {"meter", "मीटर", "m"},
    # Currency
    {"rupees", "रुपये", "rs", "inr", "₹"},
    {"dollar", "डॉलर", "usd"},
    # Misc
    {"ok", "ओके", "okay", "okey"},
    {"hello", "हेलो", "helo"},
    {"sir", "सर", "sar"},
    {"madam", "मैडम", "maam", "ma am", "mam"},
    {"please", "प्लीज", "plz", "pls"},
    {"thank you", "धन्यवाद", "thanks", "thankyou", "shukriya", "शुक्रिया"},
    {"hundred", "सौ", "sau"},
    {"thousand", "हज़ार", "हजार", "hazar", "hazaar"},
    {"lakh", "लाख", "lac", "lack"},
    {"crore", "करोड़", "karod"},
    {"piece", "पीस", "pieces", "pcs"},
    {"pack", "पैक", "packet"},
    {"price", "प्राइस", "दाम", "कीमत", "rate", "रेट"},
    {"size", "साइज", "साइज़"},
    {"quality", "क्वालिटी", "qualiti"},
    {"delivery", "डिलीवरी", "deliveri"},
    {"order", "ऑर्डर", "आर्डर"},
    {"payment", "पेमेंट", "भुगतान"},
    {"available", "अवेलेबल", "उपलब्ध"},
    {"confirm", "कन्फर्म", "controm", "confrim"},
    {"winter", "विंटर", "vintar"},
]

_NOUN_VARIATIONS: dict[str, list[str]] = {}


def load_noun_files(paths: list[str | Path]) -> dict[str, list[str]]:
    """Load noun misinterpretation JSON files and return a merged variation dict.

    Each JSON has categories (person_names, places, brands_products, etc.)
    mapping canonical_noun → [variation1, variation2, ...].
    Returns a dict mapping lowercase(canonical) → [all lowercase variations including canonical].
    """
    global _NOUN_VARIATIONS
    merged: dict[str, list[str]] = {}
    for p in paths:
        data = json.loads(Path(p).read_text())
        for key, entries in data.items():
            if key == "_meta":
                continue
            for canonical, variations in entries.items():
                low_canonical = canonical.lower()
                all_forms = {low_canonical}
                for v in variations:
                    all_forms.add(v.lower())
                if low_canonical in merged:
                    merged[low_canonical] = list(set(merged[low_canonical]) | all_forms)
                else:
                    merged[low_canonical] = sorted(all_forms)
    _NOUN_VARIATIONS = merged
    return merged


def get_noun_variations() -> dict[str, list[str]]:
    return _NOUN_VARIATIONS


_SYNONYM_MAP: dict[str, str] | None = None


def _build_synonym_map() -> dict[str, str]:
    global _SYNONYM_MAP
    if _SYNONYM_MAP is not None:
        return _SYNONYM_MAP
    mapping: dict[str, str] = {}
    for group in CUSTOM_SYNONYMS:
        canonical = sorted(group)[0]
        for word in group:
            mapping[word.lower()] = canonical
    _SYNONYM_MAP = mapping
    return _SYNONYM_MAP


# ---------------------------------------------------------------------------
# ITN: digits ↔ English words normalization
# ---------------------------------------------------------------------------
_WORD_TO_NUM: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "lakh": 100_000, "lac": 100_000, "crore": 10_000_000,
    "million": 1_000_000, "billion": 1_000_000_000,
}

_HINDI_NUM_WORDS: dict[str, int] = {
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4,
    "पांच": 5, "पाँच": 5, "छह": 6, "सात": 7, "आठ": 8, "नौ": 9,
    "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14,
    "पंद्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
    "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50,
    "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
    "सौ": 100, "हज़ार": 1000, "हजार": 1000,
    "लाख": 100_000, "करोड़": 10_000_000,
    "पच्चीस": 25, "पचहत्तर": 75,
}


def _compose_number(tokens: list[str]) -> tuple[int | None, int]:
    """Try to compose a multi-word number from consecutive tokens.

    Returns (value, count_consumed). If the first token isn't a number word,
    returns (None, 0).
    """
    all_nums = {**_WORD_TO_NUM, **_HINDI_NUM_WORDS}
    if not tokens or tokens[0] not in all_nums:
        return None, 0

    total = 0
    current = 0
    consumed = 0

    for tok in tokens:
        if tok not in all_nums:
            break
        val = all_nums[tok]
        consumed += 1
        if val == 100:
            current = max(current, 1) * 100
        elif val >= 1000:
            current = max(current, 1) * val
            total += current
            current = 0
        else:
            current += val

    total += current
    return total, consumed


def _digits_to_words(token: str) -> str | None:
    try:
        n = int(token)
        if abs(n) > 999_999_999:
            return None
        return num2words(n, lang="en").replace(",", "").replace("-", " ")
    except (ValueError, NotImplementedError):
        return None


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\u0900-\u097F]", " ", text)
    return " ".join(text.split())


def normalize_text_itn(text: str) -> str:
    """normalize_text + compose multi-word numbers to digits + synonym canonicalization."""
    text = normalize_text(text)
    syn_map = _build_synonym_map()
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = syn_map.get(tokens[i], tokens[i])
        if tok.isdigit():
            out.append(tok)
            i += 1
            continue
        val, consumed = _compose_number([syn_map.get(t, t) for t in tokens[i:]])
        if val is not None and consumed > 0:
            out.append(str(val))
            i += consumed
        else:
            out.append(tok)
            i += 1
    return " ".join(out)


def tokenize(text: str) -> list[str]:
    return [w for w in normalize_text(text).split() if w]


def tokenize_itn(text: str) -> list[str]:
    return [w for w in normalize_text_itn(text).split() if w]


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
            match = (ref_words[i - 1] == hyp_words[j - 1]
                     or _crossscript_match(ref_words[i - 1], hyp_words[j - 1]))
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


# ---------------------------------------------------------------------------
# OIWER: Lattice-based WER (orthographically-informed)
# ---------------------------------------------------------------------------
def _generate_word_variations(word: str) -> list[str]:
    """Generate acceptable variations for a single word using ITN + synonyms + loaded nouns."""
    variations: set[str] = set()
    low = normalize_text(word)
    variations.add(low)

    syn_map = _build_synonym_map()
    canonical = syn_map.get(low, low)
    variations.add(canonical)
    for group in CUSTOM_SYNONYMS:
        lower_group = {w.lower() for w in group}
        if low in lower_group or canonical in lower_group:
            variations.update(lower_group)

    if low.isdigit():
        words_form = _digits_to_words(low)
        if words_form:
            variations.add(words_form.replace(" and ", " "))
            variations.add(words_form)
    all_nums = {**_WORD_TO_NUM, **_HINDI_NUM_WORDS}
    if low in all_nums:
        variations.add(str(all_nums[low]))

    noun_vars = _NOUN_VARIATIONS
    if low in noun_vars:
        variations.update(noun_vars[low])
    else:
        for canon, forms in noun_vars.items():
            if low in forms:
                variations.add(canon)
                variations.update(forms)
                break

    return sorted(variations)


def build_variation_lattice(
    reference: str,
    extra_variations: Optional[dict[str, list[str]]] = None,
) -> list[list[str]]:
    """Build a per-word variation lattice from a reference string.

    Handles multi-word number phrases by composing them into a single lattice
    slot with the digit form as a variation.
    """
    tokens = tokenize(reference)
    lattice: list[list[str]] = []
    all_nums = {**_WORD_TO_NUM, **_HINDI_NUM_WORDS}
    i = 0
    while i < len(tokens):
        val, consumed = _compose_number(tokens[i:])
        if val is not None and consumed > 1:
            phrase = " ".join(tokens[i:i + consumed])
            variations = [phrase, str(val)]
            words_form = _digits_to_words(str(val))
            if words_form:
                normed = words_form.replace(" and ", " ")
                if normed != phrase:
                    variations.append(normed)
            lattice.append(sorted(set(variations)))
            i += consumed
        else:
            tok = tokens[i]
            variations = _generate_word_variations(tok)
            if extra_variations:
                low = tok.lower()
                if low in extra_variations:
                    for v in extra_variations[low]:
                        normed = normalize_text(v)
                        if normed and normed not in variations:
                            variations.append(normed)
            lattice.append(variations)
            i += 1
    return lattice


def explain_oiwer(reference: str, hypothesis: str,
                  extra_variations: Optional[dict[str, list[str]]] = None) -> dict:
    """Compute OIWER — WER scored against a variation lattice.

    For each reference word, generates acceptable variations (via ITN + synonyms +
    optional extra_variations). The DP alignment can match the hypothesis word
    against any variation at each position.
    """
    lattice = build_variation_lattice(reference, extra_variations)
    hyp_words = tokenize(hypothesis)
    n = len(lattice)
    m = len(hyp_words)

    if n == 0:
        wer = 0.0 if m == 0 else 1.0
        ops = [{"op": "i", "ref": None, "hyp": w} for w in hyp_words] if m else []
        return _finalize_wer_explanation(
            reference=reference, hypothesis=hypothesis,
            ref_words=[lat[0] for lat in lattice] if lattice else [],
            hyp_words=hyp_words, wer=wer,
            substitutions=0, deletions=0, insertions=m, correct=0, ops=ops,
        )

    ref_words = [lat[0] for lat in lattice]

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[str, int]]] = [
        [("", -1)] * (m + 1) for _ in range(n + 1)
    ]
    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = ("d", -1)
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = ("i", -1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_var_idx = -1
            is_match = False
            for vi, var in enumerate(lattice[i - 1]):
                if var == hyp_words[j - 1]:
                    is_match = True
                    best_var_idx = vi
                    break
            if not is_match:
                for vi, var in enumerate(lattice[i - 1]):
                    if var.replace(" ", "") == hyp_words[j - 1].replace(" ", ""):
                        is_match = True
                        best_var_idx = vi
                        break
            if not is_match:
                for vi, var in enumerate(lattice[i - 1]):
                    if _crossscript_match(var, hyp_words[j - 1]):
                        is_match = True
                        best_var_idx = vi
                        break

            sub_cost = dp[i - 1][j - 1] + (0 if is_match else 1)
            del_cost = dp[i - 1][j] + 1
            ins_cost = dp[i][j - 1] + 1

            if sub_cost <= del_cost and sub_cost <= ins_cost:
                dp[i][j] = sub_cost
                back[i][j] = ("c" if is_match else "s", best_var_idx)
            elif del_cost <= ins_cost:
                dp[i][j] = del_cost
                back[i][j] = ("d", -1)
            else:
                dp[i][j] = ins_cost
                back[i][j] = ("i", -1)

    ops: list[dict] = []
    i, j = n, m
    while i > 0 or j > 0:
        op, var_idx = back[i][j]
        if op in ("c", "s"):
            matched_var = lattice[i - 1][var_idx] if var_idx >= 0 else lattice[i - 1][0]
            ops.append({"op": op, "ref": ref_words[i - 1], "hyp": hyp_words[j - 1],
                         "matched_variation": matched_var if op == "c" else None})
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
    wer = dp[n][m] / n if n > 0 else (0.0 if m == 0 else 1.0)

    return _finalize_wer_explanation(
        reference=reference, hypothesis=hypothesis,
        ref_words=ref_words, hyp_words=hyp_words, wer=wer,
        substitutions=substitutions, deletions=deletions,
        insertions=insertions, correct=correct, ops=ops,
    )


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
        if turn.get("excluded_from_avg"):
            continue
        raw_wer = turn.get("wer")
        if raw_wer is None:
            continue
        ref_words = tokenize(turn.get("reference", ""))
        if not ref_words:
            continue
        wer = float(raw_wer)
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
    intentional_skips = {"bvc"}

    for report in reports:
        for label, reason in (report.get("skipped_engines") or {}).items():
            if label in intentional_skips and not label.startswith("hecttor/"):
                if label == "bvc":
                    skipped_reasons.setdefault(label, str(reason))
                    continue
            skipped_acc[label] = skipped_acc.get(label, 0) + 1
            skipped_reasons.setdefault(label, str(reason)[:500])
        for label, eng in (report.get("engines") or {}).items():
            if eng.get("avg_wer") is None:
                continue
            turns = eng.get("turns") or []
            edits = eng.get("edit_totals") or {}
            if edits.get("ref_words"):
                ref_words = int(edits["ref_words"])
                total_edits = (
                    int(edits.get("substitutions") or 0)
                    + int(edits.get("deletions") or 0)
                    + int(edits.get("insertions") or 0)
                )
                w_wer = (total_edits / ref_words) if ref_words else 1.0
            else:
                w_wer, ref_words = weighted_wer(turns)
            combined_turn_avg.setdefault(label, []).append(eng["avg_wer"])
            combined_weighted.setdefault(label, []).append((w_wer, ref_words))
            # Prefer scored-turn reasons over exclusion notices for the banner.
            scored_reasons = [
                r
                for r in (eng.get("high_wer_reasons") or [])
                if "excluded (" not in str(r).lower()
            ]
            fallback_reasons = eng.get("high_wer_reasons") or []
            for reason in scored_reasons or fallback_reasons:
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
