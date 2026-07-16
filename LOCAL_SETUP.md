# Local setup guide — NC WER Analysis (including Sanas)

Complete guide to run the full noise-cancellation WER benchmark stack on a Mac (or Linux), including **Sanas via Docker**.

Repo: https://github.com/yashjha-collab/NC_WER_Analysis

---

## What you are running

```
Manifest JSON (golden transcripts + public recording URLs)
        ↓
FastAPI backend + React UI
        ↓
worker/benchmark_nc_wer.py
        ├── download/cache human.ogg
        ├── slice user turns (timestamp + VAD)
        ├── apply NC engines
        ├── re-STT (Cartesia / Deepgram)
        └── WER vs golden text
```

### Engine tiers

| Tier | Engines | Where it runs |
|------|---------|----------------|
| **A** | `none`, `dtln`, `hush`, `hecttor/*` (5 models) | Your Mac / Linux host |
| **B** | `sanas/AGENTIC_VI_G_NC`, `sanas/AGENTIC_ST_NC` | **Docker `linux/amd64` only** |
| **C** | `bvc` | **Out of scope** (LiveKit Cloud room replay) |

You need **two checkouts**:

1. **NC_WER_Analysis** — API, UI, WER worker
2. **livekit-agent-worker** — production NC `FrameProcessor` code + Sanas/Hecttor wheels + Hush assets

---

## Prerequisites

| Tool | Version / notes |
|------|------------------|
| macOS or Linux | Apple Silicon Mac is fine for Tier A |
| Python | 3.12+ (3.13 preferred; worker NC wheels are often `cp313`) |
| Node.js | 20+ |
| ffmpeg | `brew install ffmpeg` |
| Git | — |
| Docker Desktop | **Required for Sanas (Tier B)** |
| Disk | Space for cached `.ogg` files under `data/cache/` |

Optional but recommended:

- Poetry (inside `livekit-agent-worker` for installing its deps)

---

## Step 0 — Clone repos

```bash
# NC WER Analysis
git clone https://github.com/yashjha-collab/NC_WER_Analysis.git
cd NC_WER_Analysis

# livekit-agent-worker (same machine, sibling folder is fine)
# Use your org’s clone URL if private
git clone <your-livekit-agent-worker-url> ../livekit-agent-worker
```

Example layout:

```text
~/Desktop/
  NC_WER_Analysis/
  livekit-agent-worker/
```

---

## Step 1 — Prepare livekit-agent-worker (NC engines)

NC_WER_Analysis does **not** vendor DTLN/Hush/Hecttor/Sanas. It imports them from the worker.

```bash
cd /path/to/livekit-agent-worker

# Install Python deps (project uses Poetry)
make setup    # or: poetry install

# Hush assets (native lib + ONNX) — required for hush engine
make download-hush   # or: ./scripts/download_hush_assets.sh

# Confirm Sanas Linux wheel exists (for Tier B)
ls sdk/sanas/sanas_remote_sdk-*-linux_x86_64.whl

# Confirm Hecttor wheels exist (for hecttor models)
ls wheels/hecttor_sdk-*.whl
```

Keep a `.env` in `livekit-agent-worker` if that repo’s imports read env vars (Hecttor/Sanas). For NC_WER_Analysis, secrets also live in **its own** `.env` (next step).

---

## Step 2 — Configure NC_WER_Analysis `.env`

```bash
cd /path/to/NC_WER_Analysis
cp .env.example .env
```

Edit `.env` to at least:

```bash
# App
NC_WER_HOST=0.0.0.0
NC_WER_PORT=8080
NC_WER_DATA_DIR=./data
NC_WER_DATABASE_URL=sqlite+aiosqlite:///./data/nc_wer.db
NC_WER_CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

# REQUIRED for NC engines (absolute path)
LIVEKIT_WORKER_ROOT=/absolute/path/to/livekit-agent-worker

# STT — at least one
CARTESIA_API_KEY=...
DEEPGRAM_API_KEY=...

# Hecttor (Tier A)
HECTTOR_API_KEY=...

# Sanas (Tier B)
SANAS_ENDPOINT=...
SANAS_ACCOUNT_ID=...
SANAS_ACCOUNT_SECRET=...
SANAS_SECURE_MEDIA=true

# Defaults (safe to keep)
DEFAULT_STT_PROVIDER=cartesia
DEFAULT_STT_MODEL=ink-whisper
DEFAULT_STT_LANGUAGE=hi
DEFAULT_TURN_ALIGN=vad
DEFAULT_NC_STRENGTH=0.5
```

**Do not commit `.env`.** It is gitignored.

### Why `LIVEKIT_WORKER_ROOT`?

At runtime the worker adds `{LIVEKIT_WORKER_ROOT}/src` to `PYTHONPATH` and imports production classes (`HushSuppressor`, `HecttorEnhancer`, `SanasEnhancer`, DTLN plugin). Without it, only the `none` baseline works.

---

## Step 3 — Install NC_WER_Analysis

```bash
cd /path/to/NC_WER_Analysis
make setup
```

This creates `.venv`, installs the backend, installs frontend npm deps, and creates `data/` folders.

Verify ffmpeg:

```bash
ffmpeg -version
```

---

## Step 4 — Start the app (backend + frontend)

Two terminals:

```bash
# Terminal 1 — API (http://localhost:8080)
cd /path/to/NC_WER_Analysis
make backend
```

```bash
# Terminal 2 — UI (http://localhost:5173)
cd /path/to/NC_WER_Analysis
make frontend
```

Open: **http://localhost:5173**

Health check:

```bash
curl http://127.0.0.1:8080/api/v1/health
# {"status":"ok"}
```

---

## Step 5 — Import a dataset

Your manifest must be JSON keyed by call id (or a list of call objects) with fields like:

```json
{
  "6a4f15ba3a66b11ea2881f94": {
    "callLogId": "6a4f15ba3a66b11ea2881f94",
    "number": 1,
    "messages": [
      { "role": "user", "content": "हां जी.", "createdAt": "1782...", "type": "message" }
    ],
    "public_url": "https://storage.googleapis.com/.../recording.ogg?X-Goog-Signature=...",
    "recordingUrl": "https://storage.googleapis.com/.../recording.ogg"
  }
}
```

Notes:

- Prefer **human-isolated** audio. The worker rewrites `.../recording.ogg` → `.../human.ogg` when possible.
- Signed `public_url`s expire — refresh if downloads start failing.
- Golden text lives in `messages` where `role == "user"`.

### Via UI

1. Open http://localhost:5173  
2. **Import from path** → absolute path to your JSON + dataset name  
3. Confirm call count appears under Datasets  

### Via API

```bash
curl -s -X POST http://127.0.0.1:8080/api/v1/datasets/import-path \
  -F name=muthoot \
  -F path=/absolute/path/to/Muthoot_final_with_public_urls.json \
  -F description="Muthoot golden set"
```

Or upload a file:

```bash
curl -s -X POST http://127.0.0.1:8080/api/v1/datasets/import \
  -F name=muthoot \
  -F file=@/absolute/path/to/Muthoot_final_with_public_urls.json
```

---

## Step 6 — Run Tier A (local engines)

Engines: `none`, `dtln`, `hush`, `hecttor/*`  
Skipped: `sanas`, `bvc`

### Via UI

1. Select dataset  
2. Choose **Tier A — none, dtln, hush, hecttor**  
3. **Start run**  
4. Watch progress + ranking on the dashboard  

Reports land in:

```text
data/runs/run_{id}/
  {call_id}_transcript.json
  {call_id}_tier_a_wer_report.json
  summary.json
```

Cached audio:

```text
data/cache/{call_id}/human.ogg
```

### Via CLI (single call)

```bash
cd /path/to/NC_WER_Analysis
source .venv/bin/activate

python worker/benchmark_nc_wer.py \
  --recording placeholder.ogg \
  --public-url "https://.../human.ogg?..." \
  --transcript /path/to/transcript.json \
  --call-id YOUR_CALL_ID \
  --auto-call-start-ts \
  --turn-align vad \
  --all-models \
  --skip-engines sanas,bvc \
  --stt-provider cartesia \
  --stt-model ink-whisper \
  --stt-language hi \
  --livekit-worker-root "$LIVEKIT_WORKER_ROOT" \
  --cache-dir ./data/cache \
  --output ./data/runs/YOUR_CALL_ID_tier_a_wer_report.json
```

### Pilot tip

For a first validation, import the full set but run only a few calls (or use a small IndiaMART manifest). Confirm `none` WER looks sane before burning Hecttor/Sanas minutes on hundreds of calls.

---

## Step 7 — Run Tier B (Sanas) with Docker

Sanas has a **Linux x86_64 Python wheel only**. On a Mac you **must** use Docker with `--platform linux/amd64`.

### 7.1 Start Docker Desktop

1. Open **Docker.app** and wait until it is running.  
2. Confirm:

```bash
docker info >/dev/null && echo "Docker OK"
```

### 7.2 Fix file sharing (common failure on Mac)

Previous Sanas runs failed with mount errors like:

```text
error while creating mount source path ... operation not permitted
```

Fix:

1. Docker Desktop → **Settings → Resources → File sharing**  
2. Allow at least:
   - `/path/to/NC_WER_Analysis`
   - `/path/to/livekit-agent-worker`
3. **Apply & restart** Docker  

Verify mounts:

```bash
docker run --rm --platform linux/amd64 \
  -v /path/to/NC_WER_Analysis:/app \
  -v /path/to/livekit-agent-worker:/livekit-worker:ro \
  alpine ls /app /livekit-worker/sdk/sanas
```

You should see the repo and Sanas wheel directory listed.

### 7.3 Confirm `.env` has Sanas + STT keys

In `NC_WER_Analysis/.env`:

- `SANAS_ENDPOINT`
- `SANAS_ACCOUNT_ID`
- `SANAS_ACCOUNT_SECRET`
- `CARTESIA_API_KEY` or `DEEPGRAM_API_KEY`
- `LIVEKIT_WORKER_ROOT` (host path; script also mounts it)

### 7.4 Create a Tier B run in the UI (or reuse a run id)

Option A — create a **Tier B** run from the UI (same dataset). Note the **run id** from the Runs list.

Option B — if you already have Tier A reports under `data/runs/run_N/`, you can run Sanas against the **same run id** so reports land next to Tier A files:

```text
data/runs/run_N/{call_id}_tier_b_wer_report.json
```

### 7.5 Run the Sanas Docker script

```bash
cd /path/to/NC_WER_Analysis

# Export worker root if not already in the shell
export LIVEKIT_WORKER_ROOT=/absolute/path/to/livekit-agent-worker

# Replace 1 with your run id
make docker-sanas RUN_ID=1
```

Equivalent:

```bash
./scripts/docker_sanas_benchmark.sh 1
```

What this does:

1. Starts `python:3.13-slim` as **`linux/amd64`**
2. Mounts `NC_WER_Analysis` → `/app`
3. Mounts `livekit-agent-worker` → `/livekit-worker` (read-only)
4. Loads secrets from `.env`
5. Installs ffmpeg + backend deps + Sanas Linux wheel
6. Runs `worker/run_tier_b.py <RUN_ID>` which benchmarks **only Sanas** for each call in that run’s dataset

**First run is slow** (apt + pip + Poetry-less install). Later runs are still heavy on Apple Silicon because amd64 is emulated (QEMU).

Expect:

```text
data/runs/run_{id}/{call_id}_tier_b_wer_report.json
```

### 7.6 Manual one-call Sanas Docker (debug)

Useful if the batch script fails:

```bash
REPO=/path/to/NC_WER_Analysis
WORKER=/path/to/livekit-agent-worker
CALL_ID=your_call_id

docker run --rm --platform linux/amd64 \
  -v "$REPO:/app" \
  -v "$WORKER:/livekit-worker:ro" \
  --env-file "$REPO/.env" \
  -w /app \
  python:3.13-slim \
  bash -lc "
    set -euo pipefail
    apt-get update -qq
    apt-get install -y -qq ffmpeg build-essential curl git >/dev/null
    pip install -q -e backend/
    pip install -q /livekit-worker/sdk/sanas/sanas_remote_sdk-*.whl
    export LIVEKIT_WORKER_ROOT=/livekit-worker
    export PYTHONPATH=/livekit-worker/src

    python worker/benchmark_nc_wer.py \
      --recording placeholder.ogg \
      --public-url 'https://...your-signed-url...' \
      --transcript data/runs/run_1/${CALL_ID}_transcript.json \
      --call-id ${CALL_ID} \
      --auto-call-start-ts \
      --turn-align vad \
      --all-models \
      --skip-engines none,dtln,hush,hecttor,bvc \
      --stt-provider cartesia --stt-model ink-whisper --stt-language hi \
      --cache-dir /app/data/cache \
      --livekit-worker-root /livekit-worker \
      --output data/runs/run_1/${CALL_ID}_tier_b_wer_report.json
  "
```

---

## Step 8 — Merge Tier A + Tier B results

Per-call reports can be merged with the worker helper:

```bash
cd /path/to/NC_WER_Analysis
source .venv/bin/activate

python - <<'PY'
from pathlib import Path
from worker.merge_reports import merge_run_directory
import json

run_dir = Path("data/runs/run_1")  # change run id
summary = merge_run_directory(run_dir)
out = run_dir / "merged_summary.json"
out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("Wrote", out)
print("Ranking:", summary.get("ranking_by_word_weighted_wer"))
PY
```

Primary metric: **word-weighted WER** (lower is better). Always keep `none` in the ranking as the baseline.

The UI shows the summary attached to the run after Tier A completes; after Sanas, re-merge or refresh as needed.

---

## Optional — full stack via docker compose

```bash
cd /path/to/NC_WER_Analysis
# Ensure .env exists with secrets + LIVEKIT_WORKER_ROOT
cd docker
docker compose up --build
```

- Backend: http://localhost:8080  
- Frontend: http://localhost:5173  

Sanas still uses the **dedicated** `scripts/docker_sanas_benchmark.sh` (profile / platform `linux/amd64`), not the default compose app containers.

---

## Checklist (copy/paste)

### Once

- [ ] Clone `NC_WER_Analysis` + `livekit-agent-worker`
- [ ] `poetry install` (or `make setup`) in livekit-agent-worker
- [ ] Hush assets downloaded
- [ ] Sanas `.whl` present under `sdk/sanas/`
- [ ] `ffmpeg` installed
- [ ] `NC_WER_Analysis/.env` filled (`LIVEKIT_WORKER_ROOT`, STT, Hecttor, Sanas)
- [ ] `make setup` in NC_WER_Analysis

### Every session

- [ ] `make backend` + `make frontend`
- [ ] Import / select dataset
- [ ] Start **Tier A** run
- [ ] For Sanas: Docker running + file sharing OK
- [ ] `make docker-sanas RUN_ID=<id>`
- [ ] Merge / review rankings

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Only `none` works; others SKIP | Missing / wrong `LIVEKIT_WORKER_ROOT` | Absolute path to worker; ensure `src/` exists |
| Import errors for hush/hecttor | Worker deps or assets missing | `poetry install`, `make download-hush`, check wheels |
| Audio download fails | Expired signed URL | Regenerate `public_url`s |
| WER ~100% for all engines | Using composite `recording.ogg` | Prefer `human.ogg` track |
| Sanas Docker: mount not permitted | Docker file sharing | Allow repo + worker paths; restart Docker |
| Sanas wheel install fails | Wrong platform / no wheel | Must use `--platform linux/amd64`; confirm `sdk/sanas/*.whl` |
| Sanas init fails | Missing `SANAS_*` or network | Fill `.env`; allow outbound to Sanas endpoint |
| STT empty / errors | Missing Cartesia/Deepgram key | Set key in `.env` |
| Backend won’t start | venv / Python | Re-run `make setup`; use project `.venv` |
| Frontend can’t reach API | CORS / proxy | Use http://localhost:5173 (Vite proxies `/api`) |

---

## Important rules (don’t skip)

1. **Golden transcript = reference.** NC output is re-STT’d and compared to golden — do not compare golden to the old live call transcript for NC ranking.  
2. **Lock STT** (`provider` / `model` / `language`) across all engines including `none`.  
3. **Always run `none`** so ΔWER vs baseline is meaningful.  
4. **Never use mixed room recordings** for this WER.  
5. **BVC is not supported** in this repo’s offline path.  
6. **Rotate secrets** if they were shared in chat or committed by mistake.

---

## Quick command reference

```bash
# Setup
cp .env.example .env   # then edit
make setup

# Run app
make backend
make frontend

# Sanas (after a run exists)
export LIVEKIT_WORKER_ROOT=/absolute/path/to/livekit-agent-worker
make docker-sanas RUN_ID=1

# Health
curl http://127.0.0.1:8080/api/v1/health
```

UI: http://localhost:5173  
API docs (if enabled): http://localhost:8080/docs
