from __future__ import annotations

import os
from pathlib import Path

import httpx

from worker.audio_utils import pcm_to_wav_bytes


async def transcribe_pcm_async(
    pcm,
    sample_rate: int,
    *,
    provider: str,
    model: str,
    language: str,
) -> str:
    wav_bytes = pcm_to_wav_bytes(pcm, sample_rate)
    provider = provider.lower()

    if provider == "deepgram":
        return await _deepgram_transcribe(wav_bytes, model=model, language=language)
    if provider == "cartesia":
        return await _cartesia_transcribe(wav_bytes, model=model, language=language)
    raise ValueError(f"Unsupported STT provider: {provider}")


def transcribe_pcm(
    pcm,
    sample_rate: int,
    *,
    provider: str,
    model: str,
    language: str,
) -> str:
    wav_bytes = pcm_to_wav_bytes(pcm, sample_rate)
    provider = provider.lower()

    if provider == "deepgram":
        return _deepgram_transcribe_sync(wav_bytes, model=model, language=language)
    if provider == "cartesia":
        return _cartesia_transcribe_sync(wav_bytes, model=model, language=language)
    raise ValueError(f"Unsupported STT provider: {provider}")


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


async def _deepgram_transcribe(
    wav_bytes: bytes, *, model: str, language: str
) -> str:
    return _deepgram_transcribe_sync(wav_bytes, model=model, language=language)


def _cartesia_transcribe_sync(
    wav_bytes: bytes, *, model: str, language: str
) -> str:
    api_key = os.getenv("CARTESIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("CARTESIA_API_KEY is not set")

    headers = {
        "X-API-Key": api_key,
        "Cartesia-Version": "2024-06-10",
    }
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {
        "model": model,
        "language": language,
    }
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            "https://api.cartesia.ai/stt",
            headers=headers,
            data=data,
            files=files,
        )
        response.raise_for_status()
        payload = response.json()
    if isinstance(payload, dict):
        return (payload.get("text") or payload.get("transcript") or "").strip()
    return str(payload).strip()


async def _cartesia_transcribe(
    wav_bytes: bytes, *, model: str, language: str
) -> str:
    return _cartesia_transcribe_sync(wav_bytes, model=model, language=language)
