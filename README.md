# NC WER Analysis

End-to-end platform for benchmarking noise-cancellation (NC) engines against golden transcripts using public recording URLs.

Repository: [NC_WER_Analysis](https://github.com/yashjha-collab/NC_WER_Analysis)

## Architecture

```
Golden manifest JSON (callLogId → messages + public_url)
        ↓
   FastAPI backend (jobs, datasets, reports)
        ↓
   worker/benchmark_nc_wer.py
        ├── download/cache human.ogg from public URL
        ├── slice user turns (timestamp + VAD)
        ├── apply NC engines via livekit-agent-worker FrameProcessors
        ├── re-transcribe (Cartesia / Deepgram)
        └── WER vs golden user text
        ↓
   React dashboard (rankings, per-call drill-down)
```

### Execution tiers

| Tier | Engines | Where |
|------|---------|-------|
| **A** | `none`, `dtln`, `hush`, `hecttor/*` | Mac / local backend |
| **B** | `sanas/*` | Linux amd64 Docker |
| **C** | `bvc` | Out of scope (LiveKit Cloud replay only) |

## Quick start

### 1. Configure

```bash
cp .env.example .env
# Set at minimum:
#   CARTESIA_API_KEY or DEEPGRAM_API_KEY
#   LIVEKIT_WORKER_ROOT=/path/to/livekit-agent-worker
#   HECTTOR_API_KEY (for hecttor models)
```

### 2. Install

```bash
make setup
```

### 3. Run backend + frontend

```bash
# terminal 1
make backend

# terminal 2
make frontend
```

Open http://localhost:5173

### 4. Import dataset

Use the UI **Import from path** with your manifest, e.g.:

- `/Users/yash.jha.ext/Desktop/Muthoot_final_with_public_urls.json` (582 calls)
- `/Users/yash.jha.ext/Desktop/indiamart_final_with_public_urls.json` (19 calls)

Expected manifest fields per call:

```json
{
  "callLogId": "...",
  "number": 1,
  "messages": [{ "role": "user", "content": "...", "createdAt": "..." }],
  "public_url": "https://.../human.ogg?...",
  "recordingUrl": "https://.../recording.ogg"
}
```

The worker prefers `human.ogg` (derived from `recording.ogg` URL when needed).

### 5. Start a benchmark

1. Select dataset in the UI
2. Choose tier (`tier_a` for local engines)
3. Click **Start run**

Reports are written to `data/runs/run_{id}/` and aggregated in the dashboard.

## CLI (direct worker)

```bash
python worker/benchmark_nc_wer.py \
  --recording data/cache/CALL_ID/human.ogg \
  --transcript /path/to/transcript.json \
  --call-id CALL_ID \
  --auto-call-start-ts \
  --turn-align vad \
  --all-models \
  --skip-engines sanas,bvc \
  --stt-provider cartesia --stt-model ink-whisper --stt-language hi \
  --output data/runs/CALL_ID_wer_report.json
```

With public URL (auto-download):

```bash
python worker/benchmark_nc_wer.py \
  --recording placeholder.ogg \
  --public-url "https://..." \
  --transcript transcript.json \
  --call-id CALL_ID \
  --auto-call-start-ts --turn-align vad \
  --all-models --skip-engines sanas,bvc \
  --output report.json
```

## Sanas (Tier B)

```bash
# After creating a run in the UI, note run_id
make docker-sanas RUN_ID=1
```

Requires Docker Desktop, Linux Sanas wheel in `livekit-agent-worker/sdk/sanas/`, and `SANAS_*` credentials.

## Project layout

```
NC_WER_Analysis/
├── backend/          # FastAPI + SQLite job store
├── frontend/         # React + Vite dashboard
├── worker/           # NC + STT + WER pipeline
├── docker/           # docker-compose + images
├── scripts/          # Sanas Docker helper
└── data/             # cache, runs, uploads (gitignored)
```

## Dependencies on livekit-agent-worker

NC engines (dtln, hush, hecttor, sanas) reuse production `FrameProcessor` classes from `livekit-agent-worker`. Set `LIVEKIT_WORKER_ROOT` in `.env`.

Without it, only the `none` baseline + STT + WER path works.

## Notes

- Use isolated **human** audio, not composite `recording.ogg`, or WER will be meaningless.
- Lock STT provider/model/language across all engines.
- Always include `none` baseline to validate golden set alignment.
- BVC is excluded from this repo (requires LiveKit Cloud room replay).
