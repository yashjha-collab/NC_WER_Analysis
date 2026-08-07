"""Google Cloud Speech-to-Text (Chirp) helpers for offline WER benchmarks."""

from __future__ import annotations

import base64
import json
import os

import httpx


def _service_account_info() -> dict:
    raw = (
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        or os.getenv("GCS_SERVICE_ACCOUNT_JSON")
        or ""
    ).strip()
    if not raw:
        raise RuntimeError(
            "GOOGLE_SERVICE_ACCOUNT_JSON or GCS_SERVICE_ACCOUNT_JSON is not set"
        )
    return json.loads(raw)


def _access_token_and_project() -> tuple[str, str]:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account
    except ImportError as exc:
        raise RuntimeError(
            "google-auth is required for Google STT — pip install google-auth"
        ) from exc

    info = _service_account_info()
    project = info.get("project_id") or os.getenv("GOOGLE_CLOUD_PROJECT", "")
    if not project:
        raise RuntimeError("Google project_id missing in service account JSON")

    creds = service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    creds.refresh(Request())
    token = creds.token
    if not token:
        raise RuntimeError("Failed to obtain Google access token")
    return token, project


def _language_code(language: str) -> str:
    lang = (language or "hi").strip()
    if "-" in lang:
        return lang
    return f"{lang}-IN"


def transcribe_wav_bytes(
    wav_bytes: bytes,
    *,
    model: str,
    language: str,
) -> str:
    """Transcribe WAV bytes with Google Speech-to-Text v2 (Chirp family)."""
    token, project = _access_token_and_project()
    location = os.getenv("GOOGLE_STT_LOCATION", "us")
    lang = _language_code(language)
    url = (
        f"https://{location}-speech.googleapis.com/v2/projects/{project}"
        f"/locations/{location}/recognizers/_:recognize"
    )
    body = {
        "config": {
            "autoDecodingConfig": {},
            "languageCodes": [lang],
            "model": model,
        },
        "content": base64.b64encode(wav_bytes).decode("ascii"),
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=180.0) as client:
        response = client.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

    parts: list[str] = []
    for result in data.get("results") or []:
        for alt in result.get("alternatives") or []:
            text = str(alt.get("transcript", "")).strip()
            if text:
                parts.append(text)
    return " ".join(parts).strip()
