"""Disk cache paths for staged WER pipeline artifacts."""

from __future__ import annotations

from pathlib import Path


def audio_dir(cache_dir: Path, call_id: str) -> Path:
    return Path(cache_dir) / "audio" / call_id


def fa_windows_path(cache_dir: Path, call_id: str) -> Path:
    return Path(cache_dir) / "fa" / call_id / "windows.json"


def nc_audio_path(
    cache_dir: Path,
    call_id: str,
    engine: str,
    strength: float,
) -> Path:
    safe_engine = engine.replace("/", "_")
    return (
        Path(cache_dir)
        / "nc"
        / call_id
        / safe_engine
        / f"s{strength:.2f}.wav"
    )


def stt_hyp_path(
    cache_dir: Path,
    call_id: str,
    engine: str,
    strength: float,
    provider: str,
    model: str,
    language: str,
) -> Path:
    safe_engine = engine.replace("/", "_")
    stt_key = f"{provider}_{model}_{language}".replace("/", "_")
    return (
        Path(cache_dir)
        / "stt"
        / call_id
        / safe_engine
        / f"s{strength:.2f}"
        / f"{stt_key}.json"
    )


def wer_result_path(
    cache_dir: Path,
    call_id: str,
    engine: str,
    strength: float,
    provider: str,
    model: str,
    language: str,
) -> Path:
    safe_engine = engine.replace("/", "_")
    stt_key = f"{provider}_{model}_{language}".replace("/", "_")
    return (
        Path(cache_dir)
        / "wer"
        / call_id
        / safe_engine
        / f"s{strength:.2f}"
        / f"{stt_key}.json"
    )


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
