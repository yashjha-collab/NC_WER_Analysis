from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import httpx

from worker.audio_utils import pcm_to_wav_bytes


def transcribe_pcm(
    pcm,
    sample_rate: int,
    *,
    provider: str,
    model: str,
    language: str,
) -> str:
    if len(pcm) < int(0.15 * sample_rate):
        return ""

    # Skip near-silence — STT often hallucinates long garbage on quiet slices.
    peak = float(max(abs(float(x)) for x in pcm)) if len(pcm) else 0.0
    if peak < 0.01:
        return ""

    wav_bytes = pcm_to_wav_bytes(pcm, sample_rate)
    provider = provider.lower()

    if provider == "deepgram":
        return _deepgram_transcribe_sync(wav_bytes, model=model, language=language)
    if provider == "google":
        from worker.google_stt import transcribe_wav_bytes

        return transcribe_wav_bytes(wav_bytes, model=model, language=language)
    if provider == "sarvam":
        return _sarvam_transcribe_sync(wav_bytes, model=model, language=language)
    raise ValueError(f"Unsupported STT provider: {provider}")


async def transcribe_pcm_async(
    pcm,
    sample_rate: int,
    *,
    provider: str,
    model: str,
    language: str,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Async STT path for overlapping network waits (asyncio)."""
    if len(pcm) < int(0.15 * sample_rate):
        return ""

    peak = float(max(abs(float(x)) for x in pcm)) if len(pcm) else 0.0
    if peak < 0.01:
        return ""

    wav_bytes = pcm_to_wav_bytes(pcm, sample_rate)
    provider = provider.lower()

    if provider == "deepgram":
        return await _deepgram_transcribe_async(
            wav_bytes, model=model, language=language, client=client
        )
    if provider in ("google", "sarvam"):
        return await asyncio.to_thread(
            transcribe_pcm,
            pcm,
            sample_rate,
            provider=provider,
            model=model,
            language=language,
        )
    raise ValueError(f"Unsupported STT provider: {provider}")


def _sarvam_transcribe_sync(
    wav_bytes: bytes, *, model: str, language: str
) -> str:
    del model, language  # Sarvam model/mode come from SARVAM_* env vars today.
    from worker.sarvam_stt import transcribe_audio_file

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = Path(tmp.name)
        segments, _ = transcribe_audio_file(tmp_path)
        return " ".join(
            str(seg.get("content", "")).strip()
            for seg in segments
            if str(seg.get("content", "")).strip()
        ).strip()
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _deepgram_transcribe_sync(
    wav_bytes: bytes, *, model: str, language: str
) -> str:
    api_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")

    params = {
        "model": model,
        "language": language,
        "punctuate": "true",
        "smart_format": "true",
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers=headers,
            content=wav_bytes,
        )
        response.raise_for_status()
        data = response.json()
    return (
        data.get("results", {})
        .get("channels", [{}])[0]
        .get("alternatives", [{}])[0]
        .get("transcript", "")
        .strip()
    )


async def _deepgram_transcribe_async(
    wav_bytes: bytes,
    *,
    model: str,
    language: str,
    client: httpx.AsyncClient | None = None,
) -> str:
    api_key = os.getenv("DEEPGRAM_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set")

    params = {
        "model": model,
        "language": language,
        "punctuate": "true",
        "smart_format": "true",
    }
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }

    async def _post(c: httpx.AsyncClient) -> str:
        response = await c.post(
            "https://api.deepgram.com/v1/listen",
            params=params,
            headers=headers,
            content=wav_bytes,
        )
        response.raise_for_status()
        data = response.json()
        return (
            data.get("results", {})
            .get("channels", [{}])[0]
            .get("alternatives", [{}])[0]
            .get("transcript", "")
            .strip()
        )

    if client is not None:
        return await _post(client)

    async with httpx.AsyncClient(timeout=120.0) as owned:
        return await _post(owned)
