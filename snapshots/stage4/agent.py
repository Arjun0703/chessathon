"""AI Chessathon submission.

Negamax search with alpha-beta pruning, iterative deepening, a transposition table, killer-move
ordering and a quiescence search, over python-chess. The evaluation is material plus tapered
piece-square tables, jitted with numba. Board state and move legality belong to python-chess,
so every move returned is legal by construction.

The process lives for one game, so module state (the position history and transposition table)
survives between our moves and resets when the next game starts.
"""

import time
import traceback
from collections.abc import Callable
from typing import Any, cast

import chess
import numpy as np
from numba import njit

PIECE_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}

MATE = 1_000_000
MAX_DEPTH = 64
INCREMENT_MS = 500.0
SAFETY_MS = 120.0
NODES_PER_CLOCK_CHECK = 1024
TT_MAX_ENTRIES = 1_500_000  # tuple entries stay well inside the 2 GB budget
QS_DELTA_MARGIN = 200

EXACT, LOWER, UPPER = 0, 1, 2

# ---------------------------------------------------------------------------------------------
# Evaluation: material plus piece-square tables, tapered between middlegame and endgame by the
# non-pawn material left on the board. The tables are Tomasz Michniewski's published
# "Simplified Evaluation Function" (chessprogramming.org) - educational constants, hand-entered,
# not extracted from any engine. They are written from White's view with a8 first, so white
# pieces index with the rank flipped (sq ^ 56) and black pieces index directly.
# ---------------------------------------------------------------------------------------------

_PAWN_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_KNIGHT_TABLE = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_BISHOP_TABLE = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_ROOK_TABLE = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]
_QUEEN_TABLE = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
_KING_MID_TABLE = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]
_KING_END_TABLE = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -50, -50,
]


def _pst(raw_by_piece: dict[int, list[int]]) -> np.ndarray:
    # Row 0 unused so tables index directly by python-chess piece type (PAWN=1 .. KING=6).
    # The raw tables list a8 first; store them a1-first so white indexes with sq ^ 56.
    table = np.zeros((7, 64), dtype=np.int64)
    for piece, raw in raw_by_piece.items():
        for square in range(64):
            table[piece, square] = raw[square]
    return table


PST_MID = _pst(
    {
        chess.PAWN: _PAWN_TABLE,
        chess.KNIGHT: _KNIGHT_TABLE,
        chess.BISHOP: _BISHOP_TABLE,
        chess.ROOK: _ROOK_TABLE,
        chess.QUEEN: _QUEEN_TABLE,
        chess.KING: _KING_MID_TABLE,
    }
)
PST_END = _pst(
    {
        chess.PAWN: _PAWN_TABLE,
        chess.KNIGHT: _KNIGHT_TABLE,
        chess.BISHOP: _BISHOP_TABLE,
        chess.ROOK: _ROOK_TABLE,
        chess.QUEEN: _QUEEN_TABLE,
        chess.KING: _KING_END_TABLE,
    }
)
MATERIAL = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.int64)
PHASE_WEIGHT = np.array([0, 0, 1, 1, 2, 4, 0], dtype=np.int64)  # knights/bishops 1, rooks 2, Q 4
PHASE_TOTAL = 24

_jit = cast(Callable[[str], Callable[[Callable[..., Any]], Callable[..., Any]]], njit)


@_jit(
    "int64(uint64, uint64, uint64, uint64, uint64, uint64, uint64, uint64, int64)"
)
def _eval_core(
    pawns: int,
    knights: int,
    bishops: int,
    rooks: int,
    queens: int,
    kings: int,
    white: int,
    black: int,
    white_to_move: int,
) -> int:
    one = np.uint64(1)
    occupied = white | black
    material = 0
    mid = 0
    end = 0
    phase = 0
    for square in range(64):
        shift = np.uint64(square)
        if (occupied >> shift) & one == 0:
            continue
        if (pawns >> shift) & one:
            piece = 1
        elif (knights >> shift) & one:
            piece = 2
        elif (bishops >> shift) & one:
            piece = 3
        elif (rooks >> shift) & one:
            piece = 4
        elif (queens >> shift) & one:
            piece = 5
        else:
            piece = 6
        phase += PHASE_WEIGHT[piece]
        if (white >> shift) & one:
            material += MATERIAL[piece]
            mid += PST_MID[piece, square ^ 56]
            end += PST_END[piece, square ^ 56]
        else:
            material -= MATERIAL[piece]
            mid -= PST_MID[piece, square]
            end -= PST_END[piece, square]
    if phase > PHASE_TOTAL:
        phase = PHASE_TOTAL
    positional = (mid * phase + end * (PHASE_TOTAL - phase)) // PHASE_TOTAL
    score = material + positional
    return score if white_to_move else -score


def _evaluate(board: chess.Board) -> int:
    return int(
        _eval_core(
            np.uint64(board.pawns),
            np.uint64(board.knights),
            np.uint64(board.bishops),
            np.uint64(board.rooks),
            np.uint64(board.queens),
            np.uint64(board.kings),
            np.uint64(board.occupied_co[chess.WHITE]),
            np.uint64(board.occupied_co[chess.BLACK]),
            np.int64(1 if board.turn else 0),
        )
    )


# ---------------------------------------------------------------------------------------------
# Game state that survives between our moves. The platform hands us only a FEN, which carries
# no history, so repetitions across the real game are invisible unless we remember every
# position we reach: the referee claims threefold automatically, and without this a winning
# engine can shuffle its way into a draw. The transposition table caches search results; a
# position reached again (same or later move, any move order) reuses the stored score or at
# least its best move for ordering.
# ---------------------------------------------------------------------------------------------

_seen: dict[object, int] = {}
_tt: dict[object, tuple[int, int, int, chess.Move | None]] = {}


class SearchTimeout(Exception):
    """Raised inside the search when the move budget is spent."""


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    moves = list(board.legal_moves)
    if not moves:
        return "0000"
    fallback = moves[0]
    try:
        return _choose(board, time_left_ms, fallback).uci()
    except Exception:
        # A crash forfeits the game; the fallback move only risks losing it.
        traceback.print_exc()
        return fallback.uci()


def _choose(board: chess.Board, time_left_ms: int, fallback: chess.Move) -> chess.Move:
    _record(board)
    start = time.monotonic()
    budget_s = _budget_ms(time_left_ms) / 1000.0
    searcher = Searcher(deadline=start + budget_s)
    best = fallback
    # SearchTimeout unwinds through pushed moves that never get popped, so the search runs on
    # a scratch board and the real one stays at the root position.
    scratch = board.copy()
    try:
        for depth in range(1, MAX_DEPTH + 1):
            score, best = searcher.search_root(scratch, depth, first=best)
            print(f"depth {depth} score {score} move {best.uci()} nodes {searcher.nodes}")
            if score >= MATE - 2 * MAX_DEPTH:
                break
            if time.monotonic() - start > 0.4 * budget_s:
                break  # the next depth costs several times this one; it will not finish
    except SearchTimeout:
        pass
    board.push(best)
    _record(board)
    board.pop()
    return best


def _budget_ms(time_left_ms: int) -> float:
    # A slice of the remaining clock plus most of the increment, capped so a long game can
    # never drain the tank, minus a hard margin: the referee measures wall time and the
    # watchdog grace is not ours to spend.
    budget = min(time_left_ms / 25.0 + 0.8 * INCREMENT_MS, time_left_ms / 4.0)
    return max(budget - SAFETY_MS, 10.0)


def _record(board: chess.Board) -> None:
    key = board._transposition_key()
    _seen[key] = _seen.get(key, 0) + 1


class Searcher:
    def __init__(self, deadline: float) -> None:
        self.deadline = deadline
        self.nodes = 0
        self.killers: list[list[chess.Move]] = [[] for _ in range(2 * MAX_DEPTH)]

    def search_root(
        self, board: chess.Board, depth: int, first: chess.Move
    ) -> tuple[int, chess.Move]:
        best_score = -2 * MATE
        best_move: chess.Move | None = None
        alpha = -2 * MATE
        for move in self._ordered(board, ply=0, tt_move=first):
            board.push(move)
            score = -self.negamax(board, depth - 1, -2 * MATE, -alpha, ply=1)
            board.pop()
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
        assert best_move is not None
        return best_score, best_move

    def negamax(self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
        self._check_clock()
        if _is_search_draw(board):
            return 0
        if depth == 0 and ply < 2 * MAX_DEPTH - 1 and board.is_check():
            depth = 1  # never stand pat while in check; repetition scoring bounds the recursion

        alpha_orig = alpha
        key = board._transposition_key()
        entry = _tt.get(key)
        tt_move: chess.Move | None = None
        if entry is not None:
            entry_depth, flag, score, tt_move = entry
            if entry_depth >= depth:
                if flag == LOWER:
                    alpha = max(alpha, score)
                elif flag == UPPER:
                    beta = min(beta, score)
                if flag == EXACT or alpha >= beta:
                    return score

        if depth == 0:
            return self.quiesce(board, alpha, beta, ply)

        moves = self._ordered(board, ply, tt_move)
        if not moves:
            return -(MATE - ply) if board.is_check() else 0

        best = -2 * MATE
        best_move: chess.Move | None = None
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best, best_move = score, move
            alpha = max(alpha, best)
            if alpha >= beta:
                # The opponent already has a better option; this line is refuted. A quiet
                # refutation tends to refute siblings too, so remember it for ordering.
                if not board.is_capture(move):
                    killers = self.killers[ply]
                    if move not in killers:
                        killers.insert(0, move)
                        del killers[2:]
                break

        # Mate scores are relative to the root of this search, so they would be wrong replayed
        # from another; keep them out of the table and lose nothing that matters.
        if abs(best) < MATE - 4 * MAX_DEPTH:
            if len(_tt) >= TT_MAX_ENTRIES:
                _tt.clear()
            flag = LOWER if best >= beta else (UPPER if best <= alpha_orig else EXACT)
            _tt[key] = (depth, flag, best, best_move)
        return best

    def quiesce(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self._check_clock()
        stand = _evaluate(board)
        if stand >= beta:
            return stand
        best = stand
        alpha = max(alpha, stand)
        captures = sorted(
            board.generate_legal_captures(),
            key=lambda move: _order_key(board, move),
            reverse=True,
        )
        for move in captures:
            victim = board.piece_type_at(move.to_square) or chess.PAWN
            if move.promotion is None and stand + PIECE_VALUE[victim] + QS_DELTA_MARGIN < alpha:
                continue  # delta pruning: even winning this piece cannot raise alpha
            board.push(move)
            score = -self.quiesce(board, -beta, -alpha, ply + 1)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    def _ordered(
        self, board: chess.Board, ply: int, tt_move: chess.Move | None
    ) -> list[chess.Move]:
        killers = self.killers[ply] if ply < len(self.killers) else []

        def order(move: chess.Move) -> int:
            if move == tt_move:
                return 1_000_000_000
            key = _order_key(board, move)
            if key == 0 and move in killers:
                return 90_000
            return key

        return sorted(board.legal_moves, key=order, reverse=True)

    def _check_clock(self) -> None:
        self.nodes += 1
        if self.nodes % NODES_PER_CLOCK_CHECK == 0 and time.monotonic() >= self.deadline:
            raise SearchTimeout


def _is_search_draw(board: chess.Board) -> bool:
    if board.halfmove_clock >= 100 or board.is_insufficient_material():
        return True
    if _seen.get(board._transposition_key(), 0) >= 2:
        return True  # one more visit and the referee claims threefold
    return board.halfmove_clock >= 4 and board.is_repetition(2)


def _order_key(board: chess.Board, move: chess.Move) -> int:
    # Captures above quiet moves, ordered by most valuable victim, least valuable attacker:
    # winning a queen with a pawn is worth trying long before the reverse.
    key = 0
    if board.is_capture(move):
        victim = board.piece_type_at(move.to_square) or chess.PAWN  # empty square: en passant
        attacker = board.piece_type_at(move.from_square) or chess.PAWN
        key = 100_000 + 10 * PIECE_VALUE[victim] - PIECE_VALUE.get(attacker, 0)
    if move.promotion is not None:
        key += PIECE_VALUE.get(move.promotion, 0)
    return key


# The signature on _eval_core makes numba compile it at import, inside the 60 second init
# budget; this call verifies the compiled path before the clock ever starts.
assert _evaluate(chess.Board()) == 0
