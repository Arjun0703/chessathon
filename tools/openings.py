"""Generate a set of balanced start positions for local testing.

    uv run python tools/openings.py --count 120 --out tools/openings.epd

Rated games start from curated neutral positions rather than the standard start, so measuring
from move one both under-samples the middlegame and lets a change that only helps the opening
look better than it is. This walks a few plies out of the book by picking randomly among the
engine's own top moves, then keeps the position only if a shallow search calls it roughly
level, which is as close to "curated and neutral" as we can get without the real set.
"""

import argparse
import contextlib
import io
import random
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
with contextlib.redirect_stdout(io.StringIO()):  # the warm-up search prints its iterations
    import agent


def _score(board: chess.Board, think_ms: int) -> int:
    """Centipawns from the side to move's point of view, via a real short search."""
    with contextlib.redirect_stdout(io.StringIO()):
        agent.get_move(board.fen(), think_ms)
    return int(agent.CTRL[agent.C_ROOT_SCORE])


def _candidates(board: chess.Board, think_ms: int, top: int) -> list[chess.Move]:
    """The `top` moves the engine likes best here, by a shallow search of each reply."""
    scored: list[tuple[int, chess.Move]] = []
    for move in board.legal_moves:
        board.push(move)
        if board.is_game_over():
            board.pop()
            continue
        scored.append((-_score(board, think_ms), move))
        board.pop()
    scored.sort(key=lambda pair: -pair[0])
    return [move for _, move in scored[:top]]


def generate(count: int, plies: int, balance: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    seen: set[str] = set()
    out: list[str] = []
    attempts = 0
    while len(out) < count and attempts < count * 20:
        attempts += 1
        board = chess.Board()
        agent.new_game()
        for _ in range(plies):
            moves = _candidates(board, 25, top=4)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over() or board.is_check():
            continue
        epd = board.epd()
        if epd in seen:
            continue
        if abs(_score(board, 250)) > balance:
            continue
        seen.add(epd)
        out.append(board.fen())
        print(f"{len(out)}/{count}  {board.fen()}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=120)
    parser.add_argument("--plies", type=int, default=8)
    parser.add_argument("--balance", type=int, default=60, help="max |score| in centipawns")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--out", type=Path, default=Path("tools/openings.epd"))
    arguments = parser.parse_args()

    fens = generate(arguments.count, arguments.plies, arguments.balance, arguments.seed)
    arguments.out.write_text("\n".join(fens) + "\n", encoding="utf-8")
    print(f"wrote {len(fens)} positions to {arguments.out}")


if __name__ == "__main__":
    main()
