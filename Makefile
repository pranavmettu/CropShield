.PHONY: install fetch-data build-features train evaluate app test clean lint

# ── Setup ─────────────────────────────────────────────────────────────────────
PYTHON ?= python3
PIP    ?= pip3

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

# ── Data pipeline ─────────────────────────────────────────────────────────────
fetch-data:
	$(PYTHON) scripts/01_fetch_data.py

build-features:
	$(PYTHON) scripts/02_build_features.py

# ── Modeling ──────────────────────────────────────────────────────────────────
train:
	$(PYTHON) scripts/03_train_model.py

evaluate:
	$(PYTHON) scripts/04_evaluate_model.py

# ── Application ───────────────────────────────────────────────────────────────
app:
	streamlit run app/streamlit_app.py

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src/cropshield --cov-report=html

# ── Quality ───────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/ scripts/

format:
	ruff format src/ tests/ scripts/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage

clean-data:
	@echo "This will delete all data files. Are you sure? (Ctrl+C to cancel)"
	@sleep 3
	rm -f data/raw/*.csv data/interim/*.csv data/processed/*.csv

# ── Full pipeline (for CI / reproducibility check) ────────────────────────────
pipeline: fetch-data build-features train evaluate

all: install pipeline test
