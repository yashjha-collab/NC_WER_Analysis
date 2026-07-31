from __future__ import annotations

import os

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


