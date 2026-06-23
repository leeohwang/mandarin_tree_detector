# =============================================================================
# Grove — local convenience targets (Mac, no GPU). The GPU pipeline runs in the
# Kaggle notebook, not here. See OPERATOR_GUIDE.md.
# =============================================================================
.DEFAULT_GOAL := help
VENV := .venv
PY   := $(VENV)/bin/python
GROVE := $(VENV)/bin/grove
CONFIG ?= config.yaml

.PHONY: help setup review export test clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: ## One-command local setup (venv + GPU-free [review] install)
	./setup.sh

review: ## Launch the local review UI (browser) on the dataset in $(CONFIG)
	$(GROVE) review --config $(CONFIG)

export: ## Re-export YOLO + COCO from the current (corrected) annotation state
	$(GROVE) export --config $(CONFIG)

test: ## Run the GPU-free test suite (converters, tiling, ingest, export, review)
	$(PY) -m pytest -q

clean: ## Remove caches and generated working data (keeps your raw images)
	rm -rf .pytest_cache **/__pycache__ grove/**/__pycache__ data/work data/dataset
