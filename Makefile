.PHONY: setup backend frontend dev install test lint

setup: install
	cp -n .env.example .env || true
	mkdir -p data/cache data/runs data/uploads

install:
	python3 -m venv .venv
	.venv/bin/pip install -e backend/
	cd frontend && npm install

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
