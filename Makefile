DOCS_DIR := ./docs

# Load environment variables from the `.env` file if it exists.
ifneq (,$(wildcard .env))
    include .env
endif

.PHONY: lint
lint:
	ruff check --exit-zero .
	ruff format --check .

.PHONY: format
format:
	ruff check --fix .
	ruff format .

.PHONY: typecheck
typecheck:
	pyright --project pyproject.toml .

.PHONY: pre-commit
pre-commit:
	pre-commit run --all-files

.PHONY: test
test:
	pytest -v .
