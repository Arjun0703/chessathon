"""AI Chessathon submission.

Negamax search with alpha-beta pruning and iterative deepening over python-chess, with a
material evaluation (piece-square tables arrive in the next stage). Board state and move
legality belong to python-chess, so every move returned is legal by construction.

The process lives for one game, so module state (the position history below, later the
transposition table) survives between our moves and resets when the next game starts.
"""

import time
import traceback

import chess

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

# The platform hands us only a FEN, which carries no history, so repetitions across the real
# game are invisible unless we remember every position we reach. The referee claims threefold
# automatically; without this a winning engine can shuffle its way into a draw.
_seen: dict[object, int] = {}


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
    try:
        for depth in range(1, MAX_DEPTH + 1):
            score, best = searcher.search_root(board, depth, first=best)
            print(f"depth {depth} score {score} move {best.uci()} nodes {searcher.nodes}")
            if score >= MATE - MAX_DEPTH:
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

    def search_root(
        self, board: chess.Board, depth: int, first: chess.Move
    ) -> tuple[int, chess.Move]:
        best_score = -2 * MATE
        best_move: chess.Move | None = None
        alpha = -2 * MATE
        for move in _ordered_moves(board, first):
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
        if depth == 0:
            if any(board.legal_moves):
                return _evaluate(board)
            return -(MATE - ply) if board.is_check() else 0
        moves = _ordered_moves(board)
        if not moves:
            return -(MATE - ply) if board.is_check() else 0
        best = -2 * MATE
        for move in moves:
            board.push(move)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            best = max(best, score)
            alpha = max(alpha, best)
            if alpha >= beta:
                break  # the opponent already has a better option; this line is refuted
        return best

    def _check_clock(self) -> None:
        self.nodes += 1
        if self.nodes % NODES_PER_CLOCK_CHECK == 0 and time.monotonic() >= self.deadline:
            raise SearchTimeout


def _is_search_draw(board: chess.Board) -> bool:
    if board.halfmove_clock >= 100 or board.is_insufficient_material():
        return True
    if _seen.get(board._transposition_key(), 0) >= 1:
        return True  # the game has been here before; score it as the draw it can become
    return board.halfmove_clock >= 4 and board.is_repetition(2)


def _evaluate(board: chess.Board) -> int:
    mover = board.turn
    score = 0
    for piece, value in PIECE_VALUE.items():
        score += value * (
            chess.popcount(board.pieces_mask(piece, mover))
            - chess.popcount(board.pieces_mask(piece, not mover))
        )
    return score


def _ordered_moves(board: chess.Board, first: chess.Move | None = None) -> list[chess.Move]:
    moves = sorted(board.legal_moves, key=lambda move: _order_key(board, move), reverse=True)
    if first is not None and first in moves:
        moves.remove(first)
        moves.insert(0, first)
    return moves


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
