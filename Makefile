.PHONY: setup data train score pipeline api app test lint format clean help

PYTHON ?= python

help:
	@echo "ChurnGuard - Command Reference"
	@echo "-----------------------------------"
	@echo "make setup     - Install project dependencies"
	@echo "make data      - Ingest, synthesize fallback, and clean transaction data"
	@echo "make train     - Train Random Forest, optimize threshold, and compute SHAP"
	@echo "make pipeline  - Run full data generation, RFM, training, and database seeding"
	@echo "make api       - Launch FastAPI uvicorn backend on port 8000"
	@echo "make app       - Launch Streamlit application on port 8501"
	@echo "make test      - Run pytest test suite"
	@echo "make lint      - Run ruff and black code checks"
	@echo "make format    - Auto-format code with black and ruff"
	@echo "make clean     - Remove caches and temporary files"

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.churn.data_ingestion
	$(PYTHON) -m src.churn.preprocessing

train:
	$(PYTHON) -m src.churn.pipeline train

score:
	$(PYTHON) -m src.churn.pipeline score

pipeline:
	$(PYTHON) -m src.churn.pipeline run-all

api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

app:
	$(PYTHON) -m streamlit run app/Home.py --server.port 8501

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m black .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
