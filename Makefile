.PHONY: install test lint check universe data backtest dry-run

install:
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ".[dev]"

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

check:
	.venv/bin/quantbot check

universe:
	.venv/bin/quantbot universe

data:
	.venv/bin/quantbot download

backtest:
	.venv/bin/quantbot backtest

dry-run:
	.venv/bin/quantbot live --dry-run
