from __future__ import annotations

import os
from typing import Any

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


def _extract_transcript_text(payload: Any) -> str:
    """Pull transcript text from Cartesia (or similar) JSON — never stringify the blob."""
    if payload is None:
        return ""
    if isinstance(payload, str):
        # Reject accidental JSON dumps used as "transcript".
        stripped = payload.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return ""
        return stripped
    if isinstance(payload, list):
        parts = [_extract_transcript_text(item) for item in payload]
        return " ".join(p for p in parts if p).strip()
    if not isinstance(payload, dict):
        return ""

    for key in ("text", "transcript", "transcription"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    words = payload.get("words")
    if isinstance(words, list) and words:
        joined = " ".join(
            str(w.get("word") or w.get("text") or "").strip()
            for w in words
            if isinstance(w, dict)
        ).strip()
        if joined:
            return joined

    # Nested shapes
    for key in ("data", "result", "results"):
        if key in payload:
            nested = _extract_transcript_text(payload[key])
            if nested:
                return nested
    return ""


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
    text = _extract_transcript_text(payload)
    if not text:
        # Second try: OpenAI-compatible endpoint used by some Cartesia setups.
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                "https://api.cartesia.ai/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
            )
            if response.is_success:
                text = _extract_transcript_text(response.json())
    return text
