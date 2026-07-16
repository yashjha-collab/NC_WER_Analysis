from __future__ import annotations

import io
import json
import subprocess
import tempfile
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from worker.wer import UserTurn


def is_signed_url(url: str) -> bool:
    return "X-Goog-Signature=" in url or "X-Amz-Signature=" in url or "?" in url


def human_url_from_recording_url(url: str) -> str | None:
    """Rewrite unsigned .../recording.ogg → .../human.ogg.

    Never rewrite signed URLs — the signature is path-bound and will 403.
    """
    if "/recording.ogg" not in url:
        return None
    if is_signed_url(url):
        return None
    return url.replace("/recording.ogg", "/human.ogg")


def download_audio(url: str, dest: Path, timeout: float = 120.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)
    return dest


def _decode_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = Path(tmp.name)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                "48000",
                str(wav_path),
            ],
            check=True,
            capture_output=True,
        )
        pcm, sample_rate = sf.read(wav_path, dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        return pcm.astype(np.float32), int(sample_rate)
    finally:
        wav_path.unlink(missing_ok=True)


def load_audio_pcm(path: Path) -> tuple[np.ndarray, int]:
    try:
        pcm, sample_rate = sf.read(path, dtype="float32")
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        return pcm.astype(np.float32), int(sample_rate)
    except Exception:
        return _decode_with_ffmpeg(path)


def resample_pcm(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return pcm
    duration = len(pcm) / src_rate
    target_len = max(1, int(duration * dst_rate))
    x_old = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, pcm).astype(np.float32)


def slice_pcm(
    pcm: np.ndarray, sample_rate: int, start_s: float, end_s: float
) -> np.ndarray:
    start_idx = max(0, int(start_s * sample_rate))
    end_idx = min(len(pcm), int(end_s * sample_rate))
    if end_idx <= start_idx:
        return np.zeros(0, dtype=np.float32)
    return pcm[start_idx:end_idx]


def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, pcm, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def load_user_turns(transcript: dict) -> list[UserTurn]:
    messages = transcript.get("messages") or []
    turns: list[UserTurn] = []
    idx = 0
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        raw_ts = msg.get("createdAt")
        if raw_ts is None or str(raw_ts).strip() == "":
            # Manual golden lines without timestamps cannot be sliced reliably.
            continue
        idx += 1
        turns.append(
            UserTurn(
                turn=idx,
                reference=content,
                created_at=float(raw_ts),
            )
        )
    return turns


def infer_call_start_ts(transcript: dict) -> float:
    messages = transcript.get("messages") or []
    timestamps = [
        float(m.get("createdAt"))
        for m in messages
        if m.get("createdAt") is not None
    ]
    return min(timestamps) if timestamps else 0.0


def assign_timestamp_segments(
    turns: list[UserTurn], call_start_ts: float, audio_duration_s: float
) -> list[UserTurn]:
    if not turns:
        return turns

    padded: list[UserTurn] = []
    for i, turn in enumerate(turns):
        start_s = max(0.0, turn.created_at - call_start_ts - 0.35)
        if i + 1 < len(turns):
            next_start = max(0.0, turns[i + 1].created_at - call_start_ts - 0.1)
            end_s = min(audio_duration_s, next_start)
        else:
            end_s = min(audio_duration_s, start_s + 8.0)
        if end_s <= start_s:
            end_s = min(audio_duration_s, start_s + 2.0)
        padded.append(
            UserTurn(
                turn=turn.turn,
                reference=turn.reference,
                created_at=turn.created_at,
                start_s=start_s,
                end_s=end_s,
            )
        )
    return padded


def refine_segments_with_vad(
    pcm: np.ndarray, sample_rate: int, turns: list[UserTurn]
) -> list[UserTurn]:
    try:
        import webrtcvad
    except ImportError:
        return turns

    vad = webrtcvad.Vad(2)
    frame_ms = 30
    frame_len = int(sample_rate * frame_ms / 1000)

    def speech_bounds(start_s: float, end_s: float) -> tuple[float, float]:
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        region = pcm[start_idx:end_idx]
        if len(region) < frame_len:
            return start_s, end_s

        speech_frames: list[int] = []
        pcm16 = (np.clip(region, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        for offset in range(0, len(region) - frame_len + 1, frame_len):
            frame = pcm16[offset * 2 : (offset + frame_len) * 2]
            if len(frame) < frame_len * 2:
                continue
            if vad.is_speech(frame, sample_rate):
                speech_frames.append(offset)

        if not speech_frames:
            return start_s, end_s

        first = speech_frames[0] / sample_rate
        last = (speech_frames[-1] + frame_len) / sample_rate
        pad = 0.15
        return max(start_s, start_s + first - pad), min(end_s, start_s + last + pad)

    refined: list[UserTurn] = []
    for turn in turns:
        if turn.start_s is None or turn.end_s is None:
            refined.append(turn)
            continue
        rs, re = speech_bounds(turn.start_s, turn.end_s)
        refined.append(
            UserTurn(
                turn=turn.turn,
                reference=turn.reference,
                created_at=turn.created_at,
                start_s=rs,
                end_s=re,
            )
        )
    return refined


def ensure_audio_for_call(
    *,
    call_log_id: str,
    public_url: str | None,
    recording_url: str | None,
    cache_dir: Path,
    human_url: str | None = None,
) -> Path:
    cached = cache_dir / call_log_id / "human.ogg"
    if cached.exists() and cached.stat().st_size > 0:
        return cached

    # Prefer an explicitly signed human track. Never fall back to composite
    # recording.ogg when human_url exists. Do not rewrite signed URLs.
    candidates: list[str] = []

    def add(url: str | None) -> None:
        if url and url not in candidates:
            candidates.append(url)

    add(human_url)

    for url in (public_url, recording_url):
        if url and "/human.ogg" in url:
            add(url)

    for url in (public_url, recording_url):
        add(human_url_from_recording_url(url) if url else None)

    if not candidates:
        raise RuntimeError(
            f"No human_url for {call_log_id}. "
            "Composite recording.ogg is not used for WER — add a signed human_url."
        )

    last_error: Exception | None = None
    for url in candidates:
        try:
            return download_audio(url, cached)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    raise RuntimeError(
        f"Could not download human audio for {call_log_id}. "
        f"Provide a valid signed human_url. Last error: {last_error}"
    )


def write_transcript_file(transcript: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
