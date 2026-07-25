.PHONY: help install dataset train run test cov lint fmt sim docker clean
export PYTHONPATH := .

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install dev dependencies
	poetry install

dataset:  ## Regenerate the synthetic dataset
	python scripts/generate_dataset.py

train:  ## Train the classifier and print metrics
	python scripts/train_classifier.py --save

run:  ## Start the API with reload
	uvicorn ticketiq.main:app --reload

test:  ## Run the test suite
	pytest

cov:  ## Run tests with a coverage report
	pytest --cov --cov-report=term-missing --cov-report=html

lint:  ## Lint and type-check
	ruff check ticketiq tests scripts && black --check ticketiq tests scripts && mypy ticketiq

fmt:  ## Auto-format
	ruff check --fix ticketiq tests scripts && black ticketiq tests scripts

sim:  ## Run the bandit learning experiment
	python scripts/bandit_simulation.py --rounds 20000 --plot

docker:  ## Build and run in Docker
	docker compose up --build

clean:  ## Remove caches and generated state
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	rm -f data/*.db data/*.db-wal data/*.db-shm data/classifier.json data/bandit.json
	find . -type d -name __pycache__ -exec rm -rf {} +
