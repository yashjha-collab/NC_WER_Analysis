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
    timestamps: list[float] = []
    for m in messages:
        raw = m.get("createdAt")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            timestamps.append(float(raw))
        except (TypeError, ValueError):
            continue
    return min(timestamps) if timestamps else 0.0


def _message_timeline(
    transcript: dict, call_start_ts: float
) -> list[tuple[float, int, str]]:
    """Sorted (offset_s, original_index, role) for every timestamped message."""
    rows: list[tuple[float, int, str]] = []
    for idx, m in enumerate(transcript.get("messages") or []):
        raw = m.get("createdAt")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            offset = max(0.0, float(raw) - call_start_ts)
        except (TypeError, ValueError):
            continue
        role = str(m.get("role") or "")
        rows.append((offset, idx, role))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def _message_offsets_s(transcript: dict, call_start_ts: float) -> list[float]:
    """Wall-clock offsets for every message with a valid timestamp."""
    return [off for off, _, _ in _message_timeline(transcript, call_start_ts)]


def assign_timestamp_segments(
    turns: list[UserTurn],
    call_start_ts: float,
    audio_duration_s: float,
    *,
    transcript: dict | None = None,
    max_turn_s: float = 6.0,
) -> list[UserTurn]:
    """Slice user turns with end-anchored windows from message timestamps.

    Console ``createdAt`` is typically when the utterance was committed (near
    speech *end*), and the agent often replies within 20–50ms. The old logic
    used ``createdAt - 0.25`` as start and ignored any next message within
    +50ms — so almost every turn fell through to a full ``max_turn_s`` (6s)
    *forward* window → STT dumped long hypotheses → WER hundreds of percent.

    New rules:
    - ``end_s`` ≈ this message offset (small pad), clipped by the next message
      even if it is only a few ms later
    - ``start_s`` looks back from ``createdAt``, bounded by the previous
      message and a word-count-based lookback (not a blind 6s of silence)
    - Same-timestamp user lines get non-overlapping sub-slices by message order
    """
    if not turns:
        return turns

    timeline = _message_timeline(transcript or {}, call_start_ts)
    unused_user_idxs = [
        i for i, (_off, _idx, role) in enumerate(timeline) if role == "user"
    ]

    padded: list[UserTurn] = []
    for turn in turns:
        turn_off = max(0.0, turn.created_at - call_start_ts)
        n_words = max(1, len((turn.reference or "").split()))
        # ~0.45s/word with floor/ceiling — short refs must not pull in seconds
        # of leading silence (Cartesia/Deepgram hallucinate on quiet audio).
        lookback_s = min(max_turn_s, max(0.9, 0.45 * n_words + 0.35))

        match_i: int | None = None
        for ui, ti in enumerate(unused_user_idxs):
            off, _, _ = timeline[ti]
            if abs(off - turn_off) < 1e-3:
                match_i = ti
                unused_user_idxs.pop(ui)
                break
        if match_i is None and unused_user_idxs:
            match_i = unused_user_idxs.pop(0)

        if match_i is None:
            prev_off = max(0.0, turn_off - lookback_s)
            next_off = min(audio_duration_s, turn_off + 0.25)
        else:
            prev_off = timeline[match_i - 1][0] if match_i > 0 else 0.0
            if match_i + 1 < len(timeline):
                next_off = timeline[match_i + 1][0]
            else:
                next_off = min(audio_duration_s, turn_off + 0.5)
            turn_off = timeline[match_i][0]

        end_pad = 0.12
        start_pad = 0.02
        end_s = min(audio_duration_s, turn_off + end_pad)
        if next_off > turn_off + 1e-4:
            end_s = min(end_s, max(turn_off, next_off - 0.02))

        start_s = max(0.0, turn_off - lookback_s)
        if match_i is not None and match_i > 0:
            start_s = max(start_s, prev_off + start_pad)

        # Same-ts / agent reply in <20ms can collapse the window — keep a
        # short minimum centered on the commit time so STT still gets audio.
        min_dur = min(0.45, lookback_s)
        if end_s - start_s < min_dur:
            mid = turn_off
            start_s = max(0.0, mid - min_dur * 0.85)
            end_s = min(audio_duration_s, start_s + min_dur)
            if next_off > turn_off + 1e-4:
                end_s = min(end_s, max(start_s + 0.15, next_off - 0.01))
            if match_i is not None and match_i > 0:
                start_s = max(start_s, prev_off + start_pad)
            if end_s <= start_s:
                end_s = min(audio_duration_s, start_s + min_dur)

        end_s = min(end_s, start_s + max_turn_s, audio_duration_s)
        if end_s <= start_s:
            end_s = min(audio_duration_s, start_s + min_dur)

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

    # webrtcvad only accepts 8/16/32/48 kHz — normalize to 16 kHz for detection.
    vad_rate = 16_000
    pcm_vad = resample_pcm(pcm, sample_rate, vad_rate) if sample_rate != vad_rate else pcm

    vad = webrtcvad.Vad(2)
    frame_ms = 30
    frame_len = int(vad_rate * frame_ms / 1000)

    def speech_bounds(start_s: float, end_s: float) -> tuple[float, float]:
        start_idx = int(start_s * vad_rate)
        end_idx = int(end_s * vad_rate)
        region = pcm_vad[start_idx:end_idx]
        if len(region) < frame_len:
            return start_s, end_s

        speech_frames: list[int] = []
        pcm16 = (np.clip(region, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        for offset in range(0, len(region) - frame_len + 1, frame_len):
            frame = pcm16[offset * 2 : (offset + frame_len) * 2]
            if len(frame) < frame_len * 2:
                continue
            if vad.is_speech(frame, vad_rate):
                speech_frames.append(offset)

        if not speech_frames:
            return start_s, end_s

        first = speech_frames[0] / vad_rate
        last = (speech_frames[-1] + frame_len) / vad_rate
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
    call_dir = cache_dir / call_log_id
    cached = call_dir / "human.ogg"
    marker = call_dir / "source_url.txt"

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

    preferred = candidates[0]
    if (
        cached.exists()
        and cached.stat().st_size > 0
        and marker.exists()
        and marker.read_text(encoding="utf-8").strip() == preferred
    ):
        return cached

    # Stale cache from an older composite download — force re-fetch.
    if cached.exists():
        cached.unlink(missing_ok=True)

    last_error: Exception | None = None
    for url in candidates:
        try:
            path = download_audio(url, cached)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(url, encoding="utf-8")
            return path
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            cached.unlink(missing_ok=True)
            continue

    raise RuntimeError(
        f"Could not download human audio for {call_log_id}. "
        f"Provide a valid signed human_url. Last error: {last_error}"
    )


def write_transcript_file(transcript: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
