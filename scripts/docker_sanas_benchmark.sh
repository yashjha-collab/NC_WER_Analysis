#!/usr/bin/env bash
# Run Tier B (Sanas-only) benchmarks inside Linux amd64 Docker.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LIVEKIT_WORKER_ROOT="${LIVEKIT_WORKER_ROOT:-/Users/yash.jha.ext/Desktop/livekit-agent-worker}"
RUN_ID="${1:-}"

docker run --rm --platform linux/amd64 \
  -v "$REPO:/app" \
  -v "$LIVEKIT_WORKER_ROOT:/livekit-worker:ro" \
  --env-file "$REPO/.env" \
  -w /app \
  python:3.13-slim \
  bash -lc '
    set -euo pipefail
    apt-get update -qq
    apt-get install -y -qq ffmpeg build-essential curl git >/dev/null
    pip install -q -e backend/
    pip install -q /livekit-worker/sdk/sanas/sanas_remote_sdk-*.whl
    export LIVEKIT_WORKER_ROOT=/livekit-worker
    export PYTHONPATH=/livekit-worker/src
    python worker/run_tier_b.py '"${RUN_ID}"'
  '
