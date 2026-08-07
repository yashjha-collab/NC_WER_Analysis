"""Canonical STT presets for benchmark sweeps."""

from __future__ import annotations

from typing import Any, TypedDict


class SttConfig(TypedDict):
    id: str
    label: str
    provider: str
    model: str
    language: str


STT_PRESETS: list[SttConfig] = [
    {
        "id": "deepgram-nova-2",
        "label": "Deepgram Nova-2",
        "provider": "deepgram",
        "model": "nova-2",
        "language": "hi",
    },
    {
        "id": "deepgram-nova-3",
        "label": "Deepgram Nova-3",
        "provider": "deepgram",
        "model": "nova-3",
        "language": "hi",
    },
    {
        "id": "google-chirp-3",
        "label": "Google Chirp 3",
        "provider": "google",
        "model": "chirp_3",
        "language": "hi-IN",
    },
    {
        "id": "sarvam-saaras-v3",
        "label": "Sarvam Saaras v3",
        "provider": "sarvam",
        "model": "saaras:v3",
        "language": "hi-IN",
    },
]

_PRESET_BY_ID = {p["id"]: p for p in STT_PRESETS}


def stt_slug(provider: str, model: str) -> str:
    raw = f"{provider}_{model}"
    return raw.replace("/", "_").replace(":", "_").replace("+", "_")


def resolve_stt_configs(
    *,
    preset_ids: list[str] | None = None,
    stt_configs: list[dict[str, Any]] | None = None,
    stt_provider: str | None = None,
    stt_model: str | None = None,
    stt_language: str | None = None,
    default_provider: str = "deepgram",
    default_model: str = "nova-2",
    default_language: str = "hi",
) -> list[dict[str, str]]:
    """Resolve one or more STT configs for job fan-out."""
    if stt_configs:
        out: list[dict[str, str]] = []
        for raw in stt_configs:
            provider = str(raw.get("provider", default_provider)).lower()
            model = str(raw.get("model", default_model))
            language = str(raw.get("language", default_language))
            out.append(
                {
                    "stt_id": str(raw.get("id") or stt_slug(provider, model)),
                    "stt_provider": provider,
                    "stt_model": model,
                    "stt_language": language,
                }
            )
        return out

    if preset_ids:
        resolved: list[dict[str, str]] = []
        for pid in preset_ids:
            preset = _PRESET_BY_ID.get(pid)
            if not preset:
                raise ValueError(f"Unknown STT preset: {pid}")
            resolved.append(
                {
                    "stt_id": preset["id"],
                    "stt_provider": preset["provider"],
                    "stt_model": preset["model"],
                    "stt_language": preset["language"],
                }
            )
        return resolved

    provider = stt_provider or default_provider
    model = stt_model or default_model
    language = stt_language or default_language
    return [
        {
            "stt_id": stt_slug(provider, model),
            "stt_provider": provider,
            "stt_model": model,
            "stt_language": language,
        }
    ]
