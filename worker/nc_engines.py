from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from worker.wer import HECTTOR_MODELS, SANAS_MODELS, TIER_A_VARIANTS, TIER_B_VARIANTS


def _setup_livekit_worker(worker_root: str | None) -> Path | None:
    root = worker_root or os.getenv("LIVEKIT_WORKER_ROOT", "")
    if not root:
        return None
    path = Path(root).resolve()
    src = path / "src"
    if not src.is_dir():
        return None
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    # Prefer the worker Poetry/.venv site-packages so hecttor_sdk / dtln / hush resolve.
    for pattern in (
        str(path / ".venv" / "lib" / "python*" / "site-packages"),
        str(path / ".venv" / "lib64" / "python*" / "site-packages"),
    ):
        for site in glob.glob(pattern):
            if site not in sys.path:
                sys.path.insert(0, site)

    return path


def apply_frame_processor(
    pcm: np.ndarray, sample_rate: int, processor: Any
) -> np.ndarray:
    from livekit import rtc

    chunk_ms = 20
    chunk_samples = max(1, int(sample_rate * chunk_ms / 1000))
    out_chunks: list[np.ndarray] = []

    for offset in range(0, len(pcm), chunk_samples):
        chunk = pcm[offset : offset + chunk_samples]
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))

        mono_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
        frame = rtc.AudioFrame(
            data=mono_int16.tobytes(),
            sample_rate=sample_rate,
            num_channels=1,
            samples_per_channel=len(mono_int16),
        )
        processed = processor._process(frame)
        processed_samples = (
            np.frombuffer(processed.data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        out_chunks.append(processed_samples)

    if not out_chunks:
        return pcm
    return np.concatenate(out_chunks)[: len(pcm)]


def build_nc_processor(
    engine: str,
    model: str | None,
    *,
    strength: float,
    worker_root: str | None = None,
) -> Any | None:
    root = _setup_livekit_worker(worker_root)
    engine = engine.lower()

    if engine == "none":
        return None

    if engine == "dtln":
        from livekit.plugins import dtln

        return dtln.noise_suppression(strength=strength)

    if engine == "hush":
        from services.hush_suppressor import HushSuppressor

        return HushSuppressor(strength=strength)

    if engine == "hecttor":
        from services.hecttor_enhancer import HecttorEnhancer

        api_key = os.getenv("HECTTOR_API_KEY", "")
        if not api_key:
            raise RuntimeError("HECTTOR_API_KEY is not set")
        try:
            import hecttor_sdk  # noqa: F401
        except ImportError as exc:
            hint = ""
            if root:
                hint = (
                    f" Install the macOS/Linux wheel from {root / 'wheels'} "
                    "into this venv (see LOCAL_SETUP.md)."
                )
            raise RuntimeError(
                f"hecttor_sdk not importable.{hint} Original: {exc}"
            ) from exc
        return HecttorEnhancer(
            api_key=api_key,
            model_name=model or "coda-1.0",
            enhancer_weight=strength,
        )

    if engine == "sanas":
        from services.sanas_enhancer import SanasEnhancer

        return SanasEnhancer(
            model_name=model or SANAS_MODELS[0],
            strength=strength,
        )

    raise ValueError(f"Unknown NC engine: {engine}")


def list_engine_variants(
    *,
    all_models: bool,
    engines: list[str] | None,
    skip_engines: set[str],
) -> list[tuple[str, str, str | None]]:
    requested = set(engines or ["none", "dtln", "hush", "hecttor", "sanas"])
    requested -= skip_engines
    variants: list[tuple[str, str, str | None]] = []

    if "none" in requested:
        variants.append(("none", "none", None))
    if "dtln" in requested:
        variants.append(("dtln", "dtln", None))
    if "hush" in requested:
        variants.append(("hush", "hush", None))
    if "hecttor" in requested:
        models = HECTTOR_MODELS if all_models else (HECTTOR_MODELS[0],)
        for model in models:
            variants.append((f"hecttor/{model}", "hecttor", model))
    if "sanas" in requested:
        models = SANAS_MODELS if all_models else (SANAS_MODELS[0],)
        for model in models:
            variants.append((f"sanas/{model}", "sanas", model))
    if "bvc" in requested:
        variants.append(("bvc", "bvc", None))

    return variants


def expected_labels_for_tier(tier: str) -> list[str]:
    if tier == "tier_b":
        return list(TIER_B_VARIANTS)
    return list(TIER_A_VARIANTS)


def skipped_engines(skip_engines: set[str]) -> dict[str, str]:
    skips: dict[str, str] = {}
    if "bvc" in skip_engines:
        skips["bvc"] = (
            "BVC/Krisp runs on live LiveKit audio tracks only — "
            "cannot process offline recordings in this benchmark"
        )
    if "sanas" in skip_engines:
        skips["sanas"] = (
            "Sanas skipped for this tier — run on Linux x86_64 (Docker tier B)"
        )
    return skips
