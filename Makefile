.PHONY: setup backend frontend dev install install-nc-wheels test lint

setup: install install-nc-wheels
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
	if [ -n "$$ROOT" ]; then \
		echo "Installing NC wheels from $$ROOT/wheels ($$(.venv/bin/python -V)) ..."; \
		WHL=$$(ls "$$ROOT"/wheels/hecttor_sdk-*-macosx_*_arm64.whl 2>/dev/null | head -1); \
		if [ -z "$$WHL" ]; then WHL=$$(ls "$$ROOT"/wheels/hecttor_sdk-*-manylinux*_x86_64.whl 2>/dev/null | head -1); fi; \
		if [ -n "$$WHL" ]; then \
		  .venv/bin/pip install -q "$$WHL" && echo "Installed $$WHL"; \
		else \
		  echo "WARN: no hecttor wheel found under $$ROOT/wheels"; \
		fi; \
	else \
		echo "LIVEKIT_WORKER_ROOT not set — skip hecttor wheel install"; \
	fi

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
