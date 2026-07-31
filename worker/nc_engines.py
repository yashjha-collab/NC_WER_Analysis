from __future__ import annotations

import glob
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from worker.audio_utils import resample_pcm
from worker.wer import HECTTOR_MODELS, TIER_A_VARIANTS

# Model-native rates. Feeding file SR (often 48 kHz) through DTLN/Hush
# triggers their internal 48↔16 resampler; offline 20 ms chunks then passthrough
# and destroy STT. Run NC at native rate instead.
ENGINE_NATIVE_SAMPLE_RATE: dict[str, int] = {
    "dtln": 16_000,
    "hush": 16_000,
    "hecttor": 48_000,
}


def native_sample_rate(engine: str) -> int | None:
    """Return the rate the engine's model expects, or None if passthrough."""
    return ENGINE_NATIVE_SAMPLE_RATE.get(engine.lower())


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


def apply_nc(
    pcm: np.ndarray,
    file_sample_rate: int,
    engine: str,
    processor: Any | None,
) -> tuple[np.ndarray, int]:
    """Run NC at the engine's native sample rate.

    Returns ``(processed_pcm, nc_sample_rate)``. For ``none`` / no processor,
    returns the input unchanged at ``file_sample_rate``.
    """
    if processor is None or engine.lower() == "none":
        return pcm, file_sample_rate

    native = native_sample_rate(engine)
    if native is None or native == file_sample_rate:
        nc_pcm = pcm
        nc_sr = file_sample_rate
    else:
        nc_pcm = resample_pcm(pcm, file_sample_rate, native)
        nc_sr = native

    return apply_frame_processor(nc_pcm, nc_sr, processor), nc_sr


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

    if engine != "none" and root is None:
        raise RuntimeError(
            "LIVEKIT_WORKER_ROOT is not set or invalid. "
            "Set it in .env to your livekit-agent-worker absolute path "
            "(needs src/services for hush/hecttor and livekit plugins for dtln)."
        )
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

    raise ValueError(f"Unknown NC engine: {engine}")


def list_engine_variants(
    *,
    all_models: bool,
    engines: list[str] | None,
    skip_engines: set[str],
) -> list[tuple[str, str, str | None]]:
    requested = set(engines or ["none", "dtln", "hush", "hecttor"])
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
    if "bvc" in requested:
        variants.append(("bvc", "bvc", None))

    return variants


def expected_labels_for_tier(tier: str) -> list[str]:
    return list(TIER_A_VARIANTS)


def skipped_engines(skip_engines: set[str]) -> dict[str, str]:
    skips: dict[str, str] = {}
    if "bvc" in skip_engines:
        skips["bvc"] = (
            "BVC/Krisp runs on live LiveKit audio tracks only — "
            "cannot process offline recordings in this benchmark"
        )
    return skips
