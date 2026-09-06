"""Correctness checks for the engine in agent.py, independent of any search result.

    uv run python tools/check_engine.py

Perft counts on the standard reference positions, a random-playout cross-check of legal
moves and hashes against python-chess, and conversion of won endgames to mate.
"""

import random
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import agent

PERFT_CASES: list[tuple[str, list[int]]] = [
    (chess.STARTING_FEN, [20, 400, 8902, 197281]),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        [48, 2039, 97862, 4085603],
    ),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238, 674624]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467, 422333]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486, 62379, 2103487]),
    (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        [46, 2079, 89890, 3894594],
    ),
]
WON_ENDINGS = [
    ("KQ v K", "8/8/8/4k3/8/8/8/4K2Q w - - 0 1"),
    ("KR v K", "8/8/8/4k3/8/8/8/R3K3 w - - 0 1"),
    ("KBB v K", "8/8/8/4k3/8/8/8/2B1KB2 w - - 0 1"),
    ("KP v K", "4k3/8/4K3/4P3/8/8/8/8 b - - 0 1"),
]


def check_perft() -> bool:
    stack = np.zeros((agent.MAX_PLY, agent.POS_LEN), dtype=np.int64)
    moves = np.zeros((agent.MAX_PLY, agent.MAX_MOVES), dtype=np.int64)
    ok = True
    started = time.monotonic()
    nodes = 0
    for fen, expected in PERFT_CASES:
        stack[0] = agent.position_from_board(chess.Board(fen), agent.tables)
        for depth, want in enumerate(expected, start=1):
            got = int(agent.perft(stack, 0, agent.tables, moves, depth))
            nodes += got
            if got != want:
                ok = False
                print(f"perft mismatch: {fen} depth {depth}: {got} != {want}")
    rate = nodes / (time.monotonic() - started) / 1e6
    print(f"perft: {'ok' if ok else 'FAILED'} ({nodes} nodes, {rate:.1f} M/s)")
    return ok


def check_against_python_chess(games: int) -> bool:
    stack = np.zeros((agent.MAX_PLY, agent.POS_LEN), dtype=np.int64)
    moves = np.zeros((agent.MAX_PLY, agent.MAX_MOVES), dtype=np.int64)
    rng = random.Random(7)
    positions = 0
    bad = 0
    for _ in range(games):
        board = chess.Board()
        for _ply in range(rng.randint(10, 160)):
            if board.is_game_over():
                break
            stack[0] = agent.position_from_board(board, agent.tables)
            count = int(agent.gen_moves(stack[0], agent.tables, moves[0], 0))
            ours: set[str] = set()
            for i in range(count):
                move = int(moves[0, i])
                agent.make_move(stack, 0, agent.tables, move)
                if agent.left_king_in_check(stack, 0, agent.tables, int(stack[0, agent.SIDE])):
                    continue
                uci = agent.move_to_uci(move)
                ours.add(uci)
                after = board.copy()
                after.push_uci(uci)
                expected = int(agent.position_from_board(after, agent.tables)[agent.HASH])
                if int(stack[1, agent.HASH]) != expected:
                    bad += 1
                    print(f"hash mismatch after {uci} in {board.fen()}")
            theirs = {move.uci() for move in board.legal_moves}
            if ours != theirs:
                bad += 1
                print(f"move mismatch in {board.fen()}: {ours ^ theirs}")
            positions += 1
            board.push(rng.choice(list(board.legal_moves)))
    print(f"cross-check: {'ok' if bad == 0 else 'FAILED'} ({positions} positions, {bad} bad)")
    return bad == 0


def check_conversions() -> bool:
    ok = True
    for name, fen in WON_ENDINGS:
        board = chess.Board(fen)
        agent.new_game()
        plies = 0
        while not board.is_game_over(claim_draw=True) and plies < 160:
            board.push_uci(agent.get_move(board.fen(), 20000))
            plies += 1
        outcome = board.outcome(claim_draw=True)
        won = outcome is not None and outcome.winner is not None
        ok = ok and won
        verdict = outcome.termination.name if outcome else "unfinished"
        print(f"{name}: {'ok' if won else 'FAILED'} ({verdict} after {plies} plies)")
    return ok


def main() -> None:
    results = [check_perft(), check_against_python_chess(games=100), check_conversions()]
    if not all(results):
        raise SystemExit("engine checks failed")


if __name__ == "__main__":
    main()
