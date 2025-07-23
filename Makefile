.PHONY: help install install-dev clean lint format test train

help:
	@echo "ModelGate -- targets"
	@echo "  install      install runtime deps"
	@echo "  install-dev  install dev deps"
	@echo "  test         pytest"
	@echo "  train        train week-0 baseline"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check modelgate tests
	mypy modelgate

format:
	black modelgate tests
	ruff check --fix modelgate tests

test:
	pytest

train:
	python -m modelgate.train --week 0

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
