.PHONY: setup backend frontend dev install install-nc-wheels download-hush test lint

setup: install install-nc-wheels download-hush
	cp -n .env.example .env || true
	mkdir -p data/cache data/runs data/uploads

install:
	python3.13 -m venv .venv || python3 -m venv .venv
	.venv/bin/python -c "import sys; assert sys.version_info[:2] == (3, 13), f'Need Python 3.13 for NC wheels, got {sys.version}'" || \
		(echo "ERROR: Create the venv with Python 3.13 (hecttor/sanas wheels are cp313). Example: python3.13 -m venv .venv" && exit 1)
	.venv/bin/pip install -U pip
	.venv/bin/pip install -e backend/
	cd frontend && npm install

# Install vendor NC wheels from livekit-agent-worker (hecttor). Requires CPython 3.13.
install-nc-wheels:
	@ROOT="$$LIVEKIT_WORKER_ROOT"; \
	if [ -z "$$ROOT" ] && [ -f .env ]; then ROOT=$$(grep -E '^LIVEKIT_WORKER_ROOT=' .env | cut -d= -f2-); fi; \
	if [ -z "$$ROOT" ]; then \
		echo "LIVEKIT_WORKER_ROOT not set — skip hecttor wheel install"; \
		exit 0; \
	fi; \
	if [ ! -d "$$ROOT/wheels" ]; then \
		echo "ERROR: $$ROOT/wheels not found — check LIVEKIT_WORKER_ROOT"; \
		exit 1; \
	fi; \
	echo "Installing hecttor from $$ROOT/wheels ($$(.venv/bin/python -V), $$(uname -sm)) ..."; \
	.venv/bin/pip install --no-index --find-links "$$ROOT/wheels" hecttor_sdk \
		|| { echo "ERROR: no hecttor wheel compatible with $$(.venv/bin/python -V) on $$(uname -sm) under $$ROOT/wheels"; exit 1; }

# Fetch Hush native lib + ONNX model from worker checkout (once per machine).
# Requires network; on Ubuntu pulls libweya_nc.so, on macOS libweya_nc.dylib.
# Optional mirrors: HUSH_LIB_URL, HUSH_MODEL_URL (for restricted VMs).
# Optional tokens: GITHUB_TOKEN, HF_TOKEN (forwarded to worker download script).
download-hush:
	@ROOT="$$LIVEKIT_WORKER_ROOT"; \
	if [ -z "$$ROOT" ] && [ -f .env ]; then ROOT=$$(grep -E '^LIVEKIT_WORKER_ROOT=' .env | cut -d= -f2-); fi; \
	if [ -z "$$ROOT" ]; then \
		echo "LIVEKIT_WORKER_ROOT not set — skip Hush asset download"; \
		exit 0; \
	fi; \
	if [ "$$(uname -s)" = "Darwin" ]; then LIB=libweya_nc.dylib; else LIB=libweya_nc.so; fi; \
	LIB_PATH="$$ROOT/sdk/hush/lib/$$LIB"; \
	MODEL_PATH="$$ROOT/sdk/hush/models/onnx/advanced_dfnet16k_model_best_onnx.tar.gz"; \
	MIN_LIB_BYTES=1000000; \
	MIN_MODEL_BYTES=1000000; \
	verify_hush_assets() { \
		if [ ! -s "$$LIB_PATH" ] || [ "$$(wc -c < "$$LIB_PATH" | tr -d ' ')" -lt "$$MIN_LIB_BYTES" ]; then \
			echo "ERROR: $$LIB_PATH missing or too small — likely a failed download (HTML/error page?)"; \
			return 1; \
		fi; \
		if [ ! -s "$$MODEL_PATH" ] || [ "$$(wc -c < "$$MODEL_PATH" | tr -d ' ')" -lt "$$MIN_MODEL_BYTES" ]; then \
			echo "ERROR: $$MODEL_PATH missing or too small — model download failed"; \
			return 1; \
		fi; \
		echo "Hush assets OK: $$LIB_PATH ($$(wc -c < "$$LIB_PATH" | tr -d ' ') bytes), $$MODEL_PATH ($$(wc -c < "$$MODEL_PATH" | tr -d ' ') bytes)"; \
	}; \
	if [ -f "$$LIB_PATH" ] && [ -f "$$MODEL_PATH" ]; then \
		verify_hush_assets || exit 1; \
		exit 0; \
	fi; \
	mkdir -p "$$ROOT/sdk/hush/lib" "$$ROOT/sdk/hush/models/onnx"; \
	if [ -n "$$HUSH_LIB_URL" ]; then \
		echo "Fetching Hush lib from mirror $$HUSH_LIB_URL ..."; \
		curl -fsSL -o "$$LIB_PATH" "$$HUSH_LIB_URL" || { echo "ERROR: mirror lib download failed"; exit 1; }; \
		if [ -n "$$HUSH_MODEL_URL" ]; then \
			echo "Fetching Hush model from mirror $$HUSH_MODEL_URL ..."; \
			curl -fsSL -o "$$MODEL_PATH" "$$HUSH_MODEL_URL" || { echo "ERROR: mirror model download failed"; exit 1; }; \
		fi; \
	else \
		if [ ! -f "$$ROOT/scripts/download_hush_assets.sh" ]; then \
			echo "ERROR: missing $$ROOT/scripts/download_hush_assets.sh — check LIVEKIT_WORKER_ROOT"; \
			exit 1; \
		fi; \
		echo "Downloading Hush assets from $$ROOT (GitHub + Hugging Face)..."; \
		if ! (cd "$$ROOT" && poetry run python -c "import huggingface_hub" 2>/dev/null); then \
			echo "Installing worker Poetry deps (needed for Hugging Face model download)..."; \
			(cd "$$ROOT" && poetry install --no-interaction); \
		fi; \
		chmod +x "$$ROOT/scripts/download_hush_assets.sh"; \
		GITHUB_TOKEN="$$GITHUB_TOKEN" HF_TOKEN="$$HF_TOKEN" $(MAKE) -C "$$ROOT" download-hush \
			|| { echo "ERROR: Hush download failed (network/token?)"; exit 1; }; \
	fi; \
	verify_hush_assets || exit 1

backend:
	.venv/bin/python -m app.main

frontend:
	cd frontend && npm run dev

dev:
	@echo "Start backend: make backend"
	@echo "Start frontend: make frontend"

test:
	cd backend && python3 -m pytest -q || true

lint:
	cd backend && python3 -m ruff check app worker || true

docker-up:
	cd docker && docker compose up --build

docker-sanas:
	chmod +x scripts/docker_sanas_benchmark.sh
	./scripts/docker_sanas_benchmark.sh $(RUN_ID)

import-muthoot:
	curl -s -X POST http://127.0.0.1:8080/api/v1/datasets/import-path \
		-F name=muthoot \
		-F path=/Users/yash.jha.ext/Desktop/Muthoot_final_with_public_urls.json \
		-F description="Muthoot golden set with public URLs"
