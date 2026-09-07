SHELL := /bin/bash

.PHONY: setup play arena gauntlet zip gate

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

# Is this version stronger than the last one? GAMES=800 for a change worth a few Elo.
gauntlet:
	uv run python tools/gauntlet.py --a . --b snapshots/stage5 --games $(or $(GAMES),400)

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
