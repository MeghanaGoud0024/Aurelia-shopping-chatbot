# Aurelia AI Shopping Assistant
#
# Every target assumes a virtualenv at .venv. `make setup` creates it.

PY := .venv/bin/python
PIP := .venv/bin/pip
PORT ?= 8000

.PHONY: help setup seed reseed run dev test smoke clean check

help:					## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[1m%-10s\033[0m %s\n", $$1, $$2}'

setup:					## Create the virtualenv and install dependencies
	python3 -m venv .venv
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -r requirements-dev.txt
	@echo "Environment ready. Next: cp .env.example .env, add your key, then 'make run'."

seed:					## Create the schema and load the synthetic dataset
	$(PY) scripts/seed_db.py

reseed:					## Wipe and regenerate the dataset
	$(PY) scripts/seed_db.py --force

run:					## Start the application on $(PORT)
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port $(PORT)

dev:					## Start with auto-reload
	$(PY) -m uvicorn app.main:app --host 127.0.0.1 --port $(PORT) --reload

test:					## Run the test suite
	$(PY) -m pytest

smoke:					## Exercise the documented example questions end to end
	$(PY) scripts/smoke_test.py

check: test smoke			## Run everything

clean:					## Remove the local database and caches
	rm -rf data/*.db data/*.db-wal data/*.db-shm .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
