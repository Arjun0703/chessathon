"""AI Chessathon submission.

A bitboard chess engine compiled with numba: magic-bitboard move generation, copy-make, a
tapered material-plus-piece-square evaluation, and a principal-variation search with a
transposition table, killer and history ordering, null-move pruning, late-move reductions and a
quiescence search. python-chess parses the FEN and double-checks the legality of the move we
return, so an engine bug can cost a game but never forfeit it.

Everything below runs single-threaded on one core. Every jitted function is compiled at import,
inside the init budget, so the clock never pays for compilation.
"""

import math
import threading
import time
import traceback
from collections.abc import Callable
from typing import cast

import chess
import numpy as np
from llvmlite import ir
from numba import float64, int64, njit, objmode
from numba.extending import intrinsic

# ---------------------------------------------------------------------------------------------
# Bit helpers. Bitboards are int64: a1 is bit 0, h8 is bit 63, north is +8. Signed integers
# make ">>" an arithmetic shift, so every logical right shift goes through lshr64 below and a
# bitboard is never shifted right by hand.
# ---------------------------------------------------------------------------------------------


def i64(x: int) -> int:
    """Wrap an unsigned 64-bit constant to the signed value with the same bits."""
    x &= (1 << 64) - 1
    return x - (1 << 64) if x >= (1 << 63) else x


@intrinsic
def _popcount(typingctx, x):  # type: ignore[no-untyped-def]
    def codegen(context, builder, sig, args):  # type: ignore[no-untyped-def]
        fn = builder.module.declare_intrinsic("llvm.ctpop", [ir.IntType(64)])
        return builder.call(fn, [args[0]])

    return int64(int64), codegen


@intrinsic
def _lsb(typingctx, x):  # type: ignore[no-untyped-def]
    """Index of the lowest set bit; 64 for an empty board."""

    def codegen(context, builder, sig, args):  # type: ignore[no-untyped-def]
        i64_t, i1_t = ir.IntType(64), ir.IntType(1)
        fnty = ir.FunctionType(i64_t, [i64_t, i1_t])
        fn = builder.module.declare_intrinsic("llvm.cttz", [i64_t], fnty)
        return builder.call(fn, [args[0], ir.Constant(i1_t, 0)])

    return int64(int64), codegen


@intrinsic
def _lshr64(typingctx, x, n):  # type: ignore[no-untyped-def]
    """Logical right shift of a 64-bit value."""

    def codegen(context, builder, sig, args):  # type: ignore[no-untyped-def]
        return builder.lshr(args[0], args[1])

    return int64(int64, int64), codegen


# The intrinsics are numba objects; these names give the type checker their real shape.
popcount = cast(Callable[[int], int], _popcount)
lsb = cast(Callable[[int], int], _lsb)
lshr64 = cast(Callable[[int, int], int], _lshr64)


# ---------------------------------------------------------------------------------------------
# Pieces, colours and the position layout. A position is one int64 array so that copy-make is a
# single memcpy: bitboards first, then scalars, then a 64-square mailbox of piece codes.
# ---------------------------------------------------------------------------------------------

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
EMPTY = 12  # mailbox code for no piece; piece codes are colour * 6 + piece

OCC_W, OCC_B, OCC_ALL = 12, 13, 14
SIDE, EP, CASTLE, HALF, HASH = 15, 16, 17, 18, 19
MAIL = 20
POS_LEN = MAIL + 64

CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ = 1, 2, 4, 8

# Move encoding: from | to << 6 | promotion << 12 | kind << 15 | (victim + 1) << 17.
# Promotion is the piece code (KNIGHT..QUEEN) or 0. Kind: 0 normal, 1 en passant, 2 castling,
# 3 double pawn push. Victim is the captured piece type, so ordering never touches the board.
KIND_NORMAL, KIND_EP, KIND_CASTLE, KIND_DOUBLE = 0, 1, 2, 3
MAX_MOVES = 256

A1, B1, C1, D1, E1, F1, G1, H1 = range(8)
A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)
# Magic multipliers for the slider attack tables, found by scratch search with our own
# finder (tools/find_magics.py in the repo): random sparse 64-bit candidates, kept when the
# mapping from every relevant occupancy to its attack set has no destructive collision.
_ROOK_MAGICS = [
    0x1080001040008021, 0x8140002000401000, 0x0100102001004008, 0x0100080410010020,
    0x0200082004020010, 0x0200212402005008, 0x0400411082100408, 0x4200020021008044,
    0x8016800440022082, 0x22C8401008A001C0, 0x0002808020009000, 0x4002000840122200,
    0x8010808044004800, 0x0882000201040810, 0x4080808001000200, 0x8002000889260444,
    0x0080024000200440, 0x18C0006000281000, 0x1021010010200441, 0x0002828008001002,
    0x0160808004000800, 0x0704004002010040, 0x2008010100020004, 0x0220020000804401,
    0x0000400180006080, 0x8240210500400080, 0x0A02008200104020, 0x2010040240080040,
    0x0844300500080100, 0x0500020080040080, 0x0001011400421008, 0x0000410A00004084,
    0x2000400184800120, 0x0210002000404001, 0x2000801008802001, 0x3002000822004012,
    0x2000040181800800, 0x0004020080800400, 0x0009000409001200, 0x0018008052000421,
    0x8800802040108000, 0x0040002000808040, 0x40E0100020008080, 0x90101200400A0020,
    0x0240080100110004, 0x0500020004008080, 0x8224081001840002, 0x0004042080420001,
    0x8120400080002080, 0x0600308042010200, 0x0800811000200180, 0x004A100058008280,
    0x8246210040801002, 0x1000800200040080, 0x100008100AA91400, 0x00021100A0440200,
    0x020281C200241102, 0x008A044082110222, 0x01C2908904A00041, 0x0200100100042009,
    0x1852010804201002, 0x0001000400080201, 0x04001A4090080104, 0x4002002084004902,
]
_BISHOP_MAGICS = [
    0x0010040088084700, 0x0120210208811804, 0x802102040040080C, 0x008C410020001080,
    0x1004042000251000, 0x7001012090808484, 0x0010440404400800, 0x4800A02C10043014,
    0x000C408218090100, 0x0800081A00821212, 0x2000080A48420440, 0x8101040428810006,
    0x84040404203000B0, 0x0204010108408080, 0x2020010450040430, 0x0010820084012920,
    0x0C10044004484080, 0xA024701010022248, 0x0040808101010100, 0x0003000824050120,
    0x0081023820080000, 0x0000818808042200, 0x0684080201010810, 0x0805100201008208,
    0x1008402023021200, 0x8402201088018400, 0x2008080004002820, 0x0004080200220040,
    0x2009004004044000, 0x2310210005808C83, 0x2401142000420800, 0x1451010C00440080,
    0x0284108402400400, 0x4880820820208820, 0x4002802082040800, 0x0008020081080080,
    0x0828020400841100, 0x8430100020074400, 0x0011540400009200, 0x0004040241908060,
    0x48008420094A2010, 0x08C0580410927400, 0xF086092088029008, 0x0020084010430202,
    0x0104202410101100, 0x0040088808800140, 0x4010840108404C00, 0x0104015043080200,
    0x1D02080402080120, 0x0001040084041020, 0x1000060221040000, 0x0000000222880020,
    0x40908928502C1000, 0x2700102021010201, 0x0020200182008001, 0x0008020420420048,
    0x000100289008080C, 0x000C802201100940, 0x0000220022011000, 0x00C00002020A0208,
    0x0004800828030400, 0x910270281015C201, 0x0000102001010A05, 0x00514A18044A8200,
]

# ---------------------------------------------------------------------------------------------
# Lookup tables. Everything the jitted code looks up lives in one flat int64 array, tables, built
# once at import and passed as an argument (numba would otherwise freeze each table into the
# compiled code, and the slider tables are 800 KB). Offsets below name the regions.
# ---------------------------------------------------------------------------------------------

T_KNIGHT = 0  # 64: knight attacks from each square
T_KING = 64  # 64: king attacks
T_PAWN = 128  # 2 * 64: pawn attacks, white then black
T_MAGIC = 256  # 64 * 8: rook mask, magic, shift, table offset; then the same for the bishop
T_CASTLE_KEEP = 768  # 64: castling rights that survive a move touching this square
T_ZOB = 832  # 12 * 64 piece keys, 16 castling keys, 8 en passant file keys, 1 side key
T_ZOB_CASTLE = T_ZOB + 768
T_ZOB_EP = T_ZOB_CASTLE + 16
T_ZOB_SIDE = T_ZOB_EP + 8
T_PASSED = T_ZOB_SIDE + 1  # 2 * 64: squares an enemy pawn must not hold for a pawn to be passed
T_FRONT = T_PASSED + 128  # 2 * 64: squares ahead on the same file
T_ADJACENT = T_FRONT + 128  # 8: the files next to each file
T_FILE = T_ADJACENT + 8  # 8: file masks
T_SHIELD = T_FILE + 8  # 2 * 64: the three squares in front of a king on this square
T_LMR = T_SHIELD + 128  # 64 * 64: late move reduction by depth and move number
T_ATTACKS = T_LMR + 4096  # rook attack tables for all squares, then bishop tables

_ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _ray_attacks(square: int, dirs: tuple[tuple[int, int], ...], occupied: int) -> int:
    attacks = 0
    rank, file = divmod(square, 8)
    for d_rank, d_file in dirs:
        r, f = rank + d_rank, file + d_file
        while 0 <= r < 8 and 0 <= f < 8:
            attacks |= 1 << (r * 8 + f)
            if occupied >> (r * 8 + f) & 1:
                break
            r, f = r + d_rank, f + d_file
    return attacks


def _relevant_mask(square: int, dirs: tuple[tuple[int, int], ...]) -> int:
    # Blockers on the board edge never change an attack set, so they stay out of the index.
    mask = 0
    rank, file = divmod(square, 8)
    for d_rank, d_file in dirs:
        r, f = rank + d_rank, file + d_file
        while 0 <= r + d_rank < 8 and 0 <= f + d_file < 8:
            mask |= 1 << (r * 8 + f)
            r, f = r + d_rank, f + d_file
    return mask


def _step_attacks(square: int, steps: tuple[tuple[int, int], ...]) -> int:
    attacks = 0
    rank, file = divmod(square, 8)
    for d_rank, d_file in steps:
        r, f = rank + d_rank, file + d_file
        if 0 <= r < 8 and 0 <= f < 8:
            attacks |= 1 << (r * 8 + f)
    return attacks


def _build_tables() -> np.ndarray:
    rook_bits = [bin(_relevant_mask(sq, _ROOK_DIRS)).count("1") for sq in range(64)]
    bishop_bits = [bin(_relevant_mask(sq, _BISHOP_DIRS)).count("1") for sq in range(64)]
    total = T_ATTACKS + sum(1 << b for b in rook_bits) + sum(1 << b for b in bishop_bits)
    table = np.zeros(total, dtype=np.int64)

    knight_steps = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
    king_steps = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
    for sq in range(64):
        table[T_KNIGHT + sq] = i64(_step_attacks(sq, knight_steps))
        table[T_KING + sq] = i64(_step_attacks(sq, king_steps))
        table[T_PAWN + sq] = i64(_step_attacks(sq, ((1, -1), (1, 1))))
        table[T_PAWN + 64 + sq] = i64(_step_attacks(sq, ((-1, -1), (-1, 1))))
        table[T_CASTLE_KEEP + sq] = 15
    table[T_CASTLE_KEEP + E1] = CASTLE_BK | CASTLE_BQ
    table[T_CASTLE_KEEP + H1] = 15 & ~CASTLE_WK
    table[T_CASTLE_KEEP + A1] = 15 & ~CASTLE_WQ
    table[T_CASTLE_KEEP + E8] = CASTLE_WK | CASTLE_WQ
    table[T_CASTLE_KEEP + H8] = 15 & ~CASTLE_BK
    table[T_CASTLE_KEEP + A8] = 15 & ~CASTLE_BQ

    files = [i64(0x0101010101010101 << f) for f in range(8)]
    for f in range(8):
        table[T_FILE + f] = files[f]
        table[T_ADJACENT + f] = (files[f - 1] if f > 0 else 0) | (files[f + 1] if f < 7 else 0)
    for sq in range(64):
        rank, file = divmod(sq, 8)
        for colour, step in ((WHITE, 1), (BLACK, -1)):
            front = 0
            passed = 0
            r = rank + step
            while 0 <= r < 8:
                front |= 1 << (r * 8 + file)
                for f in (file - 1, file, file + 1):
                    if 0 <= f < 8:
                        passed |= 1 << (r * 8 + f)
                r += step
            table[T_FRONT + colour * 64 + sq] = i64(front)
            table[T_PASSED + colour * 64 + sq] = i64(passed)
            shield = 0
            if 0 <= rank + step < 8:
                for f in (file - 1, file, file + 1):
                    if 0 <= f < 8:
                        shield |= 1 << ((rank + step) * 8 + f)
            table[T_SHIELD + colour * 64 + sq] = i64(shield)
    for depth in range(1, 64):
        for moves in range(1, 64):
            table[T_LMR + depth * 64 + moves] = int(0.5 + math.log(depth) * math.log(moves) / 2.0)

    offset = T_ATTACKS
    for slider, (dirs, magics, bits) in enumerate(
        ((_ROOK_DIRS, _ROOK_MAGICS, rook_bits), (_BISHOP_DIRS, _BISHOP_MAGICS, bishop_bits))
    ):
        for sq in range(64):
            mask = _relevant_mask(sq, dirs)
            base = T_MAGIC + sq * 8 + slider * 4
            table[base] = i64(mask)
            table[base + 1] = i64(magics[sq])
            table[base + 2] = 64 - bits[sq]
            table[base + 3] = offset
            subset = 0
            while True:
                index = ((subset * magics[sq]) & ((1 << 64) - 1)) >> (64 - bits[sq])
                table[offset + index] = i64(_ray_attacks(sq, dirs, subset))
                subset = (subset - mask) & mask
                if subset == 0:
                    break
            offset += 1 << bits[sq]

    # Zobrist keys. Any fixed seed works: hashes only have to agree within one process.
    keys = np.random.default_rng(20240904).integers(0, 2**64, size=793, dtype=np.uint64)
    table[T_ZOB : T_ZOB + 793] = keys.view(np.int64)
    return table


@njit(int64(int64[::1], int64, int64))
def rook_attacks(tables: np.ndarray, square: int, occupied: int) -> int:
    base = T_MAGIC + square * 8
    index = lshr64((occupied & tables[base]) * tables[base + 1], tables[base + 2])
    return int(tables[tables[base + 3] + index])


@njit(int64(int64[::1], int64, int64))
def bishop_attacks(tables: np.ndarray, square: int, occupied: int) -> int:
    base = T_MAGIC + square * 8 + 4
    index = lshr64((occupied & tables[base]) * tables[base + 1], tables[base + 2])
    return int(tables[tables[base + 3] + index])

# ---------------------------------------------------------------------------------------------
# Position setup, attack detection, move generation and copy-make. Moves are generated
# pseudo-legally; the search makes each one and discards it if its own king is left in check.
# ---------------------------------------------------------------------------------------------


@njit(int64(int64[::1], int64[::1]))
def compute_hash(pos: np.ndarray, tables: np.ndarray) -> int:
    h = 0
    for square in range(64):
        piece = pos[MAIL + square]
        if piece != EMPTY:
            h ^= tables[T_ZOB + piece * 64 + square]
    h ^= tables[T_ZOB_CASTLE + pos[CASTLE]]
    if pos[EP] >= 0:
        h ^= tables[T_ZOB_EP + (pos[EP] & 7)]
    if pos[SIDE] == BLACK:
        h ^= tables[T_ZOB_SIDE]
    return int(h)


def position_from_board(board: chess.Board, tables: np.ndarray) -> np.ndarray:
    pos = np.zeros(POS_LEN, dtype=np.int64)
    pos[MAIL : MAIL + 64] = EMPTY
    for square, occupant in board.piece_map().items():
        code = (0 if occupant.color == chess.WHITE else 6) + occupant.piece_type - 1
        pos[code] |= i64(1 << square)
        pos[MAIL + square] = code
    for colour in (WHITE, BLACK):
        occupancy = 0
        for piece in range(6):
            occupancy |= int(pos[colour * 6 + piece])
        pos[OCC_W + colour] = occupancy
    pos[OCC_ALL] = pos[OCC_W] | pos[OCC_B]
    pos[SIDE] = WHITE if board.turn else BLACK
    pos[EP] = -1
    if board.ep_square is not None and board.has_pseudo_legal_en_passant():
        pos[EP] = board.ep_square
    rights = 0
    if board.castling_rights & chess.BB_H1:
        rights |= CASTLE_WK
    if board.castling_rights & chess.BB_A1:
        rights |= CASTLE_WQ
    if board.castling_rights & chess.BB_H8:
        rights |= CASTLE_BK
    if board.castling_rights & chess.BB_A8:
        rights |= CASTLE_BQ
    pos[CASTLE] = rights
    pos[HALF] = board.halfmove_clock
    pos[HASH] = compute_hash(pos, tables)
    return pos


@njit(int64(int64[::1], int64[::1], int64, int64))
def attacked(pos: np.ndarray, tables: np.ndarray, square: int, by: int) -> int:
    base = by * 6
    if tables[T_PAWN + (1 - by) * 64 + square] & pos[base + PAWN]:
        return 1
    if tables[T_KNIGHT + square] & pos[base + KNIGHT]:
        return 1
    if tables[T_KING + square] & pos[base + KING]:
        return 1
    occupied = pos[OCC_ALL]
    if bishop_attacks(tables, square, occupied) & (pos[base + BISHOP] | pos[base + QUEEN]):
        return 1
    if rook_attacks(tables, square, occupied) & (pos[base + ROOK] | pos[base + QUEEN]):
        return 1
    return 0


@njit(int64(int64[::1], int64[::1]))
def in_check(pos: np.ndarray, tables: np.ndarray) -> int:
    side = pos[SIDE]
    return attacked(pos, tables, lsb(pos[side * 6 + KING]), 1 - side)


@njit(int64(int64, int64, int64, int64, int64))
def encode(frm: int, to: int, promotion: int, kind: int, victim: int) -> int:
    return frm | (to << 6) | (promotion << 12) | (kind << 15) | ((victim + 1) << 17)


@njit(int64(int64[::1], int64[::1], int64[::1], int64))
def gen_moves(pos: np.ndarray, tables: np.ndarray, out: np.ndarray, captures_only: int) -> int:
    """Write pseudo-legal moves into out and return how many. With captures_only, only
    captures and queen promotions (the moves quiescence looks at)."""
    side = pos[SIDE]
    them = 1 - side
    base = side * 6
    own = pos[OCC_W + side]
    enemy = pos[OCC_W + them]
    occupied = pos[OCC_ALL]
    targets = enemy if captures_only else ~own
    n = 0

    forward = 8 if side == WHITE else -8
    last_rank = 7 if side == WHITE else 0
    start_rank = 1 if side == WHITE else 6
    ep = pos[EP]
    pawns = pos[base + PAWN]
    while pawns:
        frm = lsb(pawns)
        pawns &= pawns - 1
        to = frm + forward
        promotes = (to >> 3) == last_rank
        attacks = tables[T_PAWN + side * 64 + frm]
        captures = attacks & enemy
        while captures:
            target = lsb(captures)
            captures &= captures - 1
            victim = pos[MAIL + target] - them * 6
            if promotes:
                for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                    out[n] = encode(frm, target, promotion, KIND_NORMAL, victim)
                    n += 1
            else:
                out[n] = encode(frm, target, 0, KIND_NORMAL, victim)
                n += 1
        if ep >= 0 and (attacks >> ep) & 1:
            out[n] = encode(frm, ep, 0, KIND_EP, PAWN)
            n += 1
        if (occupied >> to) & 1:
            continue
        if promotes:
            out[n] = encode(frm, to, QUEEN, KIND_NORMAL, -1)
            n += 1
            if not captures_only:
                for promotion in (ROOK, BISHOP, KNIGHT):
                    out[n] = encode(frm, to, promotion, KIND_NORMAL, -1)
                    n += 1
        elif not captures_only:
            out[n] = encode(frm, to, 0, KIND_NORMAL, -1)
            n += 1
            if (frm >> 3) == start_rank and not (occupied >> (to + forward)) & 1:
                out[n] = encode(frm, to + forward, 0, KIND_DOUBLE, -1)
                n += 1

    for piece in (KNIGHT, BISHOP, ROOK, QUEEN, KING):
        pieces = pos[base + piece]
        while pieces:
            frm = lsb(pieces)
            pieces &= pieces - 1
            if piece == KNIGHT:
                attacks = tables[T_KNIGHT + frm]
            elif piece == BISHOP:
                attacks = bishop_attacks(tables, frm, occupied)
            elif piece == ROOK:
                attacks = rook_attacks(tables, frm, occupied)
            elif piece == QUEEN:
                attacks = bishop_attacks(tables, frm, occupied)
                attacks |= rook_attacks(tables, frm, occupied)
            else:
                attacks = tables[T_KING + frm]
            attacks &= targets
            while attacks:
                to = lsb(attacks)
                attacks &= attacks - 1
                victim = pos[MAIL + to]
                victim = -1 if victim == EMPTY else victim - them * 6
                out[n] = encode(frm, to, 0, KIND_NORMAL, victim)
                n += 1

    if captures_only:
        return n
    rights = pos[CASTLE]
    if side == WHITE:
        if (
            rights & CASTLE_WK
            and not occupied & ((1 << F1) | (1 << G1))
            and not attacked(pos, tables, E1, BLACK)
            and not attacked(pos, tables, F1, BLACK)
        ):
            out[n] = encode(E1, G1, 0, KIND_CASTLE, -1)
            n += 1
        if (
            rights & CASTLE_WQ
            and not occupied & ((1 << B1) | (1 << C1) | (1 << D1))
            and not attacked(pos, tables, E1, BLACK)
            and not attacked(pos, tables, D1, BLACK)
        ):
            out[n] = encode(E1, C1, 0, KIND_CASTLE, -1)
            n += 1
    else:
        if (
            rights & CASTLE_BK
            and not occupied & ((1 << F8) | (1 << G8))
            and not attacked(pos, tables, E8, WHITE)
            and not attacked(pos, tables, F8, WHITE)
        ):
            out[n] = encode(E8, G8, 0, KIND_CASTLE, -1)
            n += 1
        if (
            rights & CASTLE_BQ
            and not occupied & ((1 << B8) | (1 << C8) | (1 << D8))
            and not attacked(pos, tables, E8, WHITE)
            and not attacked(pos, tables, D8, WHITE)
        ):
            out[n] = encode(E8, C8, 0, KIND_CASTLE, -1)
            n += 1
    return n


@njit((int64[:, ::1], int64, int64[::1], int64))
def make_move(stack: np.ndarray, ply: int, tables: np.ndarray, move: int) -> None:
    """Play move from stack[ply] into stack[ply + 1]; stack[ply] is untouched, so no unmake."""
    src = stack[ply]
    dst = stack[ply + 1]
    dst[:] = src
    frm = move & 63
    to = (move >> 6) & 63
    promotion = (move >> 12) & 7
    kind = (move >> 15) & 3
    side = dst[SIDE]
    them = 1 - side
    piece = dst[MAIL + frm]
    victim = dst[MAIL + to]
    h = dst[HASH]
    halfmove = dst[HALF] + 1
    from_bb = 1 << frm
    to_bb = 1 << to

    if victim != EMPTY:
        dst[victim] ^= to_bb
        h ^= tables[T_ZOB + victim * 64 + to]
        halfmove = 0
    dst[piece] ^= from_bb | to_bb
    dst[MAIL + frm] = EMPTY
    dst[MAIL + to] = piece
    h ^= tables[T_ZOB + piece * 64 + frm] ^ tables[T_ZOB + piece * 64 + to]

    if piece == side * 6 + PAWN:
        halfmove = 0
        if kind == KIND_EP:
            captured_square = to - (8 if side == WHITE else -8)
            captured = them * 6 + PAWN
            dst[captured] ^= 1 << captured_square
            dst[MAIL + captured_square] = EMPTY
            h ^= tables[T_ZOB + captured * 64 + captured_square]
        elif promotion:
            promoted = side * 6 + promotion
            dst[piece] ^= to_bb
            dst[promoted] |= to_bb
            dst[MAIL + to] = promoted
            h ^= tables[T_ZOB + piece * 64 + to] ^ tables[T_ZOB + promoted * 64 + to]
    elif kind == KIND_CASTLE:
        rook = side * 6 + ROOK
        if to == G1:
            rook_from, rook_to = H1, F1
        elif to == C1:
            rook_from, rook_to = A1, D1
        elif to == G8:
            rook_from, rook_to = H8, F8
        else:
            rook_from, rook_to = A8, D8
        dst[rook] ^= (1 << rook_from) | (1 << rook_to)
        dst[MAIL + rook_from] = EMPTY
        dst[MAIL + rook_to] = rook
        h ^= tables[T_ZOB + rook * 64 + rook_from] ^ tables[T_ZOB + rook * 64 + rook_to]

    if dst[EP] >= 0:
        h ^= tables[T_ZOB_EP + (dst[EP] & 7)]
    new_ep = -1
    if kind == KIND_DOUBLE:
        passed = (frm + to) >> 1
        # Only record the square when an enemy pawn could actually capture there, which is
        # what python-chess prints in its FENs, so repeated positions hash the same.
        if tables[T_PAWN + side * 64 + passed] & dst[them * 6 + PAWN]:
            new_ep = passed
            h ^= tables[T_ZOB_EP + (passed & 7)]
    dst[EP] = new_ep

    old_rights = dst[CASTLE]
    rights = old_rights & tables[T_CASTLE_KEEP + frm] & tables[T_CASTLE_KEEP + to]
    if rights != old_rights:
        h ^= tables[T_ZOB_CASTLE + old_rights] ^ tables[T_ZOB_CASTLE + rights]
    dst[CASTLE] = rights

    white = dst[0] | dst[1] | dst[2] | dst[3] | dst[4] | dst[5]
    black = dst[6] | dst[7] | dst[8] | dst[9] | dst[10] | dst[11]
    dst[OCC_W] = white
    dst[OCC_B] = black
    dst[OCC_ALL] = white | black
    dst[HALF] = halfmove
    dst[SIDE] = them
    dst[HASH] = h ^ tables[T_ZOB_SIDE]


@njit((int64[:, ::1], int64, int64[::1]))
def make_null(stack: np.ndarray, ply: int, tables: np.ndarray) -> None:
    dst = stack[ply + 1]
    dst[:] = stack[ply]
    h = dst[HASH] ^ tables[T_ZOB_SIDE]
    if dst[EP] >= 0:
        h ^= tables[T_ZOB_EP + (dst[EP] & 7)]
        dst[EP] = -1
    dst[SIDE] = 1 - dst[SIDE]
    dst[HALF] += 1
    dst[HASH] = h


@njit(int64(int64[:, ::1], int64, int64[::1], int64))
def left_king_in_check(stack: np.ndarray, ply: int, tables: np.ndarray, side: int) -> int:
    """After make_move from stack[ply], did side (the mover) leave its own king attacked?"""
    return attacked(stack[ply + 1], tables, lsb(stack[ply + 1, side * 6 + KING]), 1 - side)


@njit(int64(int64[:, ::1], int64, int64[::1], int64[:, ::1], int64))
def perft(stack: np.ndarray, ply: int, tables: np.ndarray, moves: np.ndarray, depth: int) -> int:
    count = gen_moves(stack[ply], tables, moves[ply], 0)
    side = stack[ply, SIDE]
    total = 0
    for i in range(count):
        make_move(stack, ply, tables, moves[ply, i])
        if left_king_in_check(stack, ply, tables, side):
            continue
        total += perft(stack, ply + 1, tables, moves, depth - 1) if depth > 1 else 1
    return total


SQUARE_NAMES = [f"{file}{rank}" for rank in "12345678" for file in "abcdefgh"]
PROMOTION_CHARS = {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}


def move_to_uci(move: int) -> str:
    text = SQUARE_NAMES[move & 63] + SQUARE_NAMES[(move >> 6) & 63]
    promotion = (move >> 12) & 7
    return text + PROMOTION_CHARS[promotion] if promotion else text

MAX_PLY = 128
tables = _build_tables()

# ---------------------------------------------------------------------------------------------
# Evaluation. Material and piece-square tables tapered between middlegame and endgame by the
# non-pawn material left, plus pawn structure, mobility, king safety and a mop-up term for won
# endings. The piece-square tables are hand-entered educational constants (Michniewski's
# "Simplified Evaluation Function" on chessprogramming.org) with our own endgame variants; the
# other weights are ours. Scores are centipawns from the side to move's point of view.
# ---------------------------------------------------------------------------------------------

MG_VALUE = np.array([100, 320, 330, 500, 900, 0], dtype=np.int64)
EG_VALUE = np.array([110, 310, 320, 520, 920, 0], dtype=np.int64)
PHASE_WEIGHT = np.array([0, 1, 1, 2, 4, 0], dtype=np.int64)
PHASE_TOTAL = 24

# Tables are written from White's view with a8 first, as they are usually printed. _pst
# flips them so that a white piece indexes by its square directly and a black piece by
# square ^ 56.
_PAWN_MG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
    5, 5, 10, 25, 25, 10, 5, 5,
    0, 0, 0, 20, 20, 0, 0, 0,
    5, -5, -10, 0, 0, -10, -5, 5,
    5, 10, 10, -20, -20, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_PAWN_EG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    80, 80, 80, 80, 80, 80, 80, 80,
    50, 50, 50, 50, 50, 50, 50, 50,
    30, 30, 30, 30, 30, 30, 30, 30,
    15, 15, 15, 15, 15, 15, 15, 15,
    5, 5, 5, 5, 5, 5, 5, 5,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_KNIGHT = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 10, 15, 15, 10, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50,
]
_BISHOP = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -20, -10, -10, -10, -10, -10, -10, -20,
]
_ROOK_MG = [
    0, 0, 0, 0, 0, 0, 0, 0,
    5, 10, 10, 10, 10, 10, 10, 5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    0, 0, 0, 5, 5, 0, 0, 0,
]
_ROOK_EG = [0] * 64
_QUEEN_MG = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
_QUEEN_EG = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -10, 5, 10, 10, 10, 10, 5, -10,
    -5, 5, 10, 15, 15, 10, 5, -5,
    -5, 5, 10, 15, 15, 10, 5, -5,
    -10, 5, 10, 10, 10, 10, 5, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20,
]
_KING_MG = [
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -20, -30, -30, -40, -40, -30, -30, -20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, 20, 0, 0, 0, 0, 20, 20,
    20, 30, 10, 0, 0, 10, 30, 20,
]
_KING_EG = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10, 0, 0, -10, -20, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 30, 40, 40, 30, -10, -30,
    -30, -10, 20, 30, 30, 20, -10, -30,
    -30, -30, 0, 0, 0, 0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50,
]


def _pst(tables: list[list[int]]) -> np.ndarray:
    out = np.zeros((6, 64), dtype=np.int64)
    for piece, raw in enumerate(tables):
        for square in range(64):
            out[piece, square ^ 56] = raw[square]
    return out


PST_MG = _pst([_PAWN_MG, _KNIGHT, _BISHOP, _ROOK_MG, _QUEEN_MG, _KING_MG])
PST_EG = _pst([_PAWN_EG, _KNIGHT, _BISHOP, _ROOK_EG, _QUEEN_EG, _KING_EG])

# Passed pawn bonus by rank from the pawn's own side (rank 0 and 7 never hold a pawn).
PASSED_MG = np.array([0, 5, 10, 20, 35, 60, 100, 0], dtype=np.int64)
PASSED_EG = np.array([0, 10, 20, 35, 60, 100, 160, 0], dtype=np.int64)
ISOLATED_MG, ISOLATED_EG = 12, 16
DOUBLED_MG, DOUBLED_EG = 10, 20
BISHOP_PAIR_MG, BISHOP_PAIR_EG = 30, 45
ROOK_OPEN_MG, ROOK_OPEN_EG = 25, 10
ROOK_SEMI_MG, ROOK_SEMI_EG = 12, 5
TEMPO = 10
# Mobility per available square, centred on a typical count so a normal piece scores zero.
MOBILITY_MG = np.array([0, 4, 3, 2, 1, 0], dtype=np.int64)
MOBILITY_EG = np.array([0, 4, 3, 4, 2, 0], dtype=np.int64)
MOBILITY_CENTRE = np.array([0, 4, 7, 7, 14, 0], dtype=np.int64)
# King safety: attack units per piece type reaching the zone around the king, turned into a
# middlegame penalty by a quadratic that flattens out.
ATTACK_UNITS = np.array([0, 2, 2, 3, 5, 0], dtype=np.int64)
SHIELD_BONUS = 12
SAFETY_MAX = 500

# Chebyshev distance between squares and distance from the centre, for the mop-up term.
_DIST = np.zeros((64, 64), dtype=np.int64)
_CENTRE_DIST = np.zeros(64, dtype=np.int64)
for _a in range(64):
    _CENTRE_DIST[_a] = max(abs((_a & 7) - 3.5), abs((_a >> 3) - 3.5)) * 2 - 1
    for _b in range(64):
        _DIST[_a, _b] = max(abs((_a & 7) - (_b & 7)), abs((_a >> 3) - (_b >> 3)))
DIST = _DIST
CENTRE_DIST = _CENTRE_DIST


@njit(int64(int64[::1], int64[::1]))
def evaluate(pos: np.ndarray, tables: np.ndarray) -> int:
    mg = 0
    eg = 0
    phase = 0
    occupied = pos[OCC_ALL]
    king_square = (lsb(pos[KING]), lsb(pos[6 + KING]))
    pawn_attacks = (0, 0)
    pawns_w = pos[PAWN]
    pawns_b = pos[6 + PAWN]
    # Squares each side's pawns attack, for mobility and king safety.
    att_w = 0
    bb = pawns_w
    while bb:
        sq = lsb(bb)
        bb &= bb - 1
        att_w |= tables[T_PAWN + sq]
    att_b = 0
    bb = pawns_b
    while bb:
        sq = lsb(bb)
        bb &= bb - 1
        att_b |= tables[T_PAWN + 64 + sq]
    pawn_attacks = (att_w, att_b)

    for colour in range(2):
        sign = 1 if colour == WHITE else -1
        base = colour * 6
        them = 1 - colour
        own = pos[OCC_W + colour]
        own_pawns = pos[base + PAWN]
        enemy_pawns = pos[them * 6 + PAWN]
        safe = ~own & ~pawn_attacks[them]
        enemy_king = king_square[them]
        zone = tables[T_KING + enemy_king] | (1 << enemy_king)
        attack_units = 0
        cmg = 0
        ceg = 0

        for piece in range(6):
            bb = pos[base + piece]
            if piece == BISHOP and popcount(bb) >= 2:
                cmg += BISHOP_PAIR_MG
                ceg += BISHOP_PAIR_EG
            while bb:
                sq = lsb(bb)
                bb &= bb - 1
                rel = sq if colour == WHITE else sq ^ 56
                cmg += MG_VALUE[piece] + PST_MG[piece, rel]
                ceg += EG_VALUE[piece] + PST_EG[piece, rel]
                phase += PHASE_WEIGHT[piece]
                if piece == PAWN:
                    file = sq & 7
                    if tables[T_PASSED + colour * 64 + sq] & enemy_pawns == 0:
                        rank = rel >> 3
                        cmg += PASSED_MG[rank]
                        ceg += PASSED_EG[rank]
                    if tables[T_ADJACENT + file] & own_pawns == 0:
                        cmg -= ISOLATED_MG
                        ceg -= ISOLATED_EG
                    if tables[T_FRONT + colour * 64 + sq] & own_pawns:
                        cmg -= DOUBLED_MG
                        ceg -= DOUBLED_EG
                    continue
                if piece == KING:
                    continue
                if piece == KNIGHT:
                    attacks = tables[T_KNIGHT + sq]
                elif piece == BISHOP:
                    attacks = bishop_attacks(tables, sq, occupied)
                elif piece == ROOK:
                    attacks = rook_attacks(tables, sq, occupied)
                    file_mask = tables[T_FILE + (sq & 7)]
                    if file_mask & own_pawns == 0:
                        if file_mask & enemy_pawns == 0:
                            cmg += ROOK_OPEN_MG
                            ceg += ROOK_OPEN_EG
                        else:
                            cmg += ROOK_SEMI_MG
                            ceg += ROOK_SEMI_EG
                else:
                    attacks = bishop_attacks(tables, sq, occupied)
                    attacks |= rook_attacks(tables, sq, occupied)
                mobility = popcount(attacks & safe) - MOBILITY_CENTRE[piece]
                cmg += MOBILITY_MG[piece] * mobility
                ceg += MOBILITY_EG[piece] * mobility
                attack_units += ATTACK_UNITS[piece] * popcount(attacks & zone)

        # Our king: pawn shield in front of it counts for the middlegame.
        ksq = king_square[colour]
        shield = tables[T_SHIELD + colour * 64 + ksq] & own_pawns
        cmg += SHIELD_BONUS * popcount(shield)
        # The enemy king: pressure from our pieces, as a penalty on their side.
        penalty = attack_units * attack_units * 2 // 3
        if penalty > SAFETY_MAX:
            penalty = SAFETY_MAX
        cmg += penalty

        mg += sign * cmg
        eg += sign * ceg

    if phase > PHASE_TOTAL:
        phase = PHASE_TOTAL
    score = (mg * phase + eg * (PHASE_TOTAL - phase)) // PHASE_TOTAL

    # Mop-up: with no pawns left and a decisive edge, drive the enemy king to a corner.
    if pawns_w == 0 and pawns_b == 0:
        winner = WHITE if score > 0 else BLACK
        margin = score if score > 0 else -score
        if margin >= 400:
            loser_king = king_square[1 - winner]
            winner_king = king_square[winner]
            mop = 10 * CENTRE_DIST[loser_king] + 4 * (14 - DIST[winner_king, loser_king])
            score += mop if winner == WHITE else -mop
        elif margin < 200:
            score //= 8  # a lone minor piece cannot win

    score += TEMPO if pos[SIDE] == WHITE else -TEMPO
    return int(score if pos[SIDE] == WHITE else -score)

# ---------------------------------------------------------------------------------------------
# Search. Principal variation search with iterative deepening and aspiration windows at the
# root. Inside: transposition table, check extension, reverse futility, null move, futility
# and late-move pruning, late move reductions, and a capture-only quiescence that also
# resolves checks. Move scores ride in the high bits of the move word so ordering needs no
# second array. Mate scores count plies from the root: MATE - ply.
# ---------------------------------------------------------------------------------------------

MATE = 30000
MATE_BOUND = MATE - MAX_PLY
INF = 32000
MOVE_MASK = (1 << 21) - 1
SCORE_SHIFT = 24
TT_BITS = 22
TT_SIZE = 1 << TT_BITS  # entries of four int64: key, move, depth | flag << 8 | age << 10, score
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
MAX_DEPTH = 64

# Indices into the control array shared between the search and the Python driver.
C_STOP, C_NODES, C_AGE, C_ROOT_BEST, C_ROOT_SCORE, C_HISTORY_LEN, C_SELDEPTH, C_ROOT_SIDE = (
    range(8)
)
CONTEMPT = 10  # a draw is worth this much less than an even position to the side searching
NODES_PER_CLOCK_CHECK = 2048

ORDER_TT = 1 << 30
ORDER_CAPTURE = 1 << 29
ORDER_KILLER = 1 << 28
HISTORY_MAX = 1 << 26

FUTILITY_MARGIN = 120
REVERSE_FUTILITY_MARGIN = 100
QS_DELTA = 200

ARR1 = int64[::1]
ARR2 = int64[:, ::1]
FLOAT1 = float64[::1]
# (stack, ply, tables, moves, tt, history, killers, ctrl, clock, game_hashes)
SEARCH_STATE = (ARR2, int64, ARR1, ARR2, ARR2, ARR1, ARR2, ARR1, FLOAT1, ARR1)


@njit((ARR2, int64, int64, int64))
def pick_move(moves: np.ndarray, ply: int, i: int, n: int) -> None:
    """Move the best-scored remaining move into slot i (one step of selection sort)."""
    best = i
    for j in range(i + 1, n):
        if moves[ply, j] > moves[ply, best]:
            best = j
    if best != i:
        moves[ply, i], moves[ply, best] = moves[ply, best], moves[ply, i]


@njit((ARR1, ARR2, int64, int64, int64, ARR1, ARR2))
def score_moves(
    pos: np.ndarray, moves: np.ndarray, ply: int, n: int, tt_move: int, history: np.ndarray,
    killers: np.ndarray,
) -> None:
    side = pos[SIDE]
    for i in range(n):
        move = moves[ply, i] & MOVE_MASK
        victim = ((move >> 17) & 15) - 1
        promotion = (move >> 12) & 7
        if move == tt_move:
            score = ORDER_TT
        elif victim >= 0:
            attacker = pos[MAIL + (move & 63)] - side * 6
            score = ORDER_CAPTURE + 16 * (victim + 1) - attacker
            if promotion == QUEEN:
                score += 64
        elif promotion == QUEEN:
            score = ORDER_CAPTURE + 48
        elif promotion:
            score = 0
        elif move == killers[ply, 0]:
            score = ORDER_KILLER + 2
        elif move == killers[ply, 1]:
            score = ORDER_KILLER + 1
        else:
            score = history[side * 4096 + (move & 4095)]
        moves[ply, i] = move | (score << SCORE_SHIFT)


@njit(int64(ARR2, int64, ARR1, int64))
def is_repetition(stack: np.ndarray, ply: int, game_hashes: np.ndarray, history_len: int) -> int:
    """Has this position occurred before, in the search path or earlier in the game? One
    earlier occurrence is enough: the side that can repeat can also make it a third time."""
    h = stack[ply, HASH]
    reach = stack[ply, HALF]  # nothing before the last irreversible move can match
    i = ply - 2
    while i >= 0 and i >= ply - reach:
        if stack[i, HASH] == h:
            return 1
        i -= 2
    back = 1
    while back <= history_len and ply + back <= reach:
        if game_hashes[history_len - back] == h:
            return 1
        back += 1
    return 0


@njit(int64(ARR1))
def insufficient_material(pos: np.ndarray) -> int:
    if pos[PAWN] | pos[6 + PAWN] | pos[ROOK] | pos[6 + ROOK] | pos[QUEEN] | pos[6 + QUEEN]:
        return 0
    minors = popcount(pos[KNIGHT] | pos[BISHOP] | pos[6 + KNIGHT] | pos[6 + BISHOP])
    return 1 if minors <= 1 else 0


@njit(int64(ARR1))
def has_non_pawn_material(pos: np.ndarray) -> int:
    base = pos[SIDE] * 6
    pieces = pos[base + KNIGHT] | pos[base + BISHOP] | pos[base + ROOK] | pos[base + QUEEN]
    return 1 if pieces else 0


@njit((ARR2, int64, int64, int64, int64, int64, int64))
def tt_store(
    tt: np.ndarray, key: int, move: int, depth: int, flag: int, score: int, age: int
) -> None:
    index = key & (TT_SIZE - 1)
    old_key = tt[index, 0]
    old_info = tt[index, 2]
    if old_key == key:
        if move == 0:
            move = tt[index, 1]
        if depth < (old_info & 255) and (old_info >> 10) == age and flag != TT_EXACT:
            return
    tt[index, 0] = key
    tt[index, 1] = move
    tt[index, 2] = depth | (flag << 8) | (age << 10)
    tt[index, 3] = score


@njit(int64(*SEARCH_STATE, int64, int64))
def quiesce(
    stack: np.ndarray, ply: int, tables: np.ndarray, moves: np.ndarray, tt: np.ndarray,
    history: np.ndarray, killers: np.ndarray, ctrl: np.ndarray, clock: np.ndarray,
    game_hashes: np.ndarray, alpha: int, beta: int,
) -> int:
    ctrl[C_NODES] += 1
    if ctrl[C_STOP]:
        return 0
    pos = stack[ply]
    side = pos[SIDE]
    checked = in_check(pos, tables)
    if checked:
        best = -INF
        n = gen_moves(pos, tables, moves[ply], 0)
    else:
        best = evaluate(pos, tables)
        if best >= beta or ply >= MAX_PLY - 2:
            return best
        if best > alpha:
            alpha = best
        n = gen_moves(pos, tables, moves[ply], 1)
    score_moves(pos, moves, ply, n, 0, history, killers)
    legal = 0
    for i in range(n):
        pick_move(moves, ply, i, n)
        move = moves[ply, i] & MOVE_MASK
        if not checked:
            victim = ((move >> 17) & 15) - 1
            promotion = (move >> 12) & 7
            if promotion == 0 and victim >= 0 and best + MG_VALUE[victim] + QS_DELTA < alpha:
                continue  # even winning this piece cleanly cannot raise alpha
        make_move(stack, ply, tables, move)
        if left_king_in_check(stack, ply, tables, side):
            continue
        legal += 1
        score = -quiesce(
            stack, ply + 1, tables, moves, tt, history, killers, ctrl, clock, game_hashes,
            -beta, -alpha,
        )
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    if checked and legal == 0:
        return -MATE + ply
    return best


@njit(int64(*SEARCH_STATE, int64, int64, int64, int64))
def negamax(
    stack: np.ndarray, ply: int, tables: np.ndarray, moves: np.ndarray, tt: np.ndarray,
    history: np.ndarray, killers: np.ndarray, ctrl: np.ndarray, clock: np.ndarray,
    game_hashes: np.ndarray, depth: int, alpha: int, beta: int, allow_null: int,
) -> int:
    ctrl[C_NODES] += 1
    if ctrl[C_NODES] & (NODES_PER_CLOCK_CHECK - 1) == 0:
        with objmode(now="float64"):
            now = time.monotonic()
        if now >= clock[0]:
            ctrl[C_STOP] = 1
    if ctrl[C_STOP]:
        return 0
    pos = stack[ply]
    side = pos[SIDE]
    pv = beta - alpha > 1

    if ply > 0 and (
        pos[HALF] >= 100
        or insufficient_material(pos)
        or is_repetition(stack, ply, game_hashes, ctrl[C_HISTORY_LEN])
    ):
        return -CONTEMPT if side == ctrl[C_ROOT_SIDE] else CONTEMPT
    if ply > 0:
        # Mate distance pruning: a mate found here can be no shorter than ply.
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    checked = in_check(pos, tables)
    if checked:
        depth += 1
    if depth <= 0:
        return quiesce(
            stack, ply, tables, moves, tt, history, killers, ctrl, clock, game_hashes, alpha, beta
        )
    if ply >= MAX_PLY - 2:
        return evaluate(pos, tables)
    if ply > ctrl[C_SELDEPTH]:
        ctrl[C_SELDEPTH] = ply

    key = pos[HASH]
    index = key & (TT_SIZE - 1)
    tt_move = 0
    if tt[index, 0] == key:
        tt_move = tt[index, 1]
        info = tt[index, 2]
        if not pv and (info & 255) >= depth:
            tt_score = tt[index, 3]
            if tt_score > MATE_BOUND:
                tt_score -= ply
            elif tt_score < -MATE_BOUND:
                tt_score += ply
            flag = (info >> 8) & 3
            if (
                flag == TT_EXACT
                or (flag == TT_LOWER and tt_score >= beta)
                or (flag == TT_UPPER and tt_score <= alpha)
            ):
                return int(tt_score)

    static_eval = evaluate(pos, tables)
    if not pv and not checked and abs(beta) < MATE_BOUND:
        if depth <= 6 and static_eval - REVERSE_FUTILITY_MARGIN * depth >= beta:
            return static_eval
        if allow_null and depth >= 3 and static_eval >= beta and has_non_pawn_material(pos):
            make_null(stack, ply, tables)
            reduction = 3 + depth // 6
            score = -negamax(
                stack, ply + 1, tables, moves, tt, history, killers, ctrl, clock, game_hashes,
                depth - 1 - reduction, -beta, -beta + 1, 0,
            )
            if ctrl[C_STOP]:
                return 0
            if score >= beta and score < MATE_BOUND:
                return score

    n = gen_moves(pos, tables, moves[ply], 0)
    score_moves(pos, moves, ply, n, tt_move, history, killers)
    futile = not pv and not checked and depth <= 3 and (
        static_eval + FUTILITY_MARGIN * depth + 50 <= alpha
    )
    alpha_orig = alpha
    best = -INF
    best_move = 0
    legal = 0
    quiets = 0
    for i in range(n):
        pick_move(moves, ply, i, n)
        move = moves[ply, i] & MOVE_MASK
        quiet = ((move >> 17) & 15) == 0 and ((move >> 12) & 7) == 0
        make_move(stack, ply, tables, move)
        if left_king_in_check(stack, ply, tables, side):
            continue
        legal += 1
        gives_check = in_check(stack[ply + 1], tables)
        if quiet:
            quiets += 1
            if legal > 1 and not gives_check and not pv:
                if futile:
                    continue
                if depth <= 3 and quiets > 3 + 2 * depth * depth:
                    continue  # late move pruning: a quiet move this far down the list
        new_depth = depth - 1
        if legal == 1:
            score = -negamax(
                stack, ply + 1, tables, moves, tt, history, killers, ctrl, clock, game_hashes,
                new_depth, -beta, -alpha, 1,
            )
        else:
            reduction = 0
            if quiet and depth >= 3 and legal > 3 and not checked and not gives_check:
                reduction = tables[T_LMR + min(depth, 63) * 64 + min(legal, 63)]
                if pv:
                    reduction -= 1
                if reduction < 0:
                    reduction = 0
            score = -negamax(
                stack, ply + 1, tables, moves, tt, history, killers, ctrl, clock, game_hashes,
                new_depth - reduction, -alpha - 1, -alpha, 1,
            )
            if score > alpha and reduction > 0:
                score = -negamax(
                    stack, ply + 1, tables, moves, tt, history, killers, ctrl, clock, game_hashes,
                    new_depth, -alpha - 1, -alpha, 1,
                )
            if score > alpha and score < beta:
                score = -negamax(
                    stack, ply + 1, tables, moves, tt, history, killers, ctrl, clock, game_hashes,
                    new_depth, -beta, -alpha, 1,
                )
        if ctrl[C_STOP]:
            return 0
        if score > best:
            best = score
            best_move = move
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    if quiet:
                        if killers[ply, 0] != move:
                            killers[ply, 1] = killers[ply, 0]
                            killers[ply, 0] = move
                        slot = side * 4096 + (move & 4095)
                        history[slot] += depth * depth
                        if history[slot] > HISTORY_MAX:
                            for j in range(8192):
                                history[j] >>= 1
                    break

    if legal == 0:
        if checked:
            return -MATE + ply
        return -CONTEMPT if side == ctrl[C_ROOT_SIDE] else CONTEMPT
    if best == -INF:
        best = alpha_orig  # every legal move was pruned: a fail low, not a lost position
    flag = TT_LOWER if best >= beta else (TT_EXACT if best > alpha_orig else TT_UPPER)
    stored = best
    if stored > MATE_BOUND:
        stored += ply
    elif stored < -MATE_BOUND:
        stored -= ply
    tt_store(tt, key, best_move, depth, flag, stored, ctrl[C_AGE])
    return best


@njit(int64(*SEARCH_STATE, int64, int64, int64), nogil=True)
def search_root(
    stack: np.ndarray, ply: int, tables: np.ndarray, moves: np.ndarray, tt: np.ndarray,
    history: np.ndarray, killers: np.ndarray, ctrl: np.ndarray, clock: np.ndarray,
    game_hashes: np.ndarray, depth: int, alpha: int, beta: int,
) -> int:
    """One iteration at the root. The best move so far is published in ctrl[C_ROOT_BEST] as
    soon as a move is fully searched and beats alpha, so a timed-out iteration still
    contributes when its first moves completed."""
    pos = stack[0]
    side = pos[SIDE]
    n = gen_moves(pos, tables, moves[0], 0)
    score_moves(pos, moves, 0, n, ctrl[C_ROOT_BEST], history, killers)
    best = -INF
    best_move = 0
    legal = 0
    for i in range(n):
        pick_move(moves, 0, i, n)
        move = moves[0, i] & MOVE_MASK
        make_move(stack, 0, tables, move)
        if left_king_in_check(stack, 0, tables, side):
            continue
        legal += 1
        if legal == 1:
            score = -negamax(
                stack, 1, tables, moves, tt, history, killers, ctrl, clock, game_hashes,
                depth - 1, -beta, -alpha, 1,
            )
        else:
            score = -negamax(
                stack, 1, tables, moves, tt, history, killers, ctrl, clock, game_hashes,
                depth - 1, -alpha - 1, -alpha, 1,
            )
            if score > alpha and score < beta:
                score = -negamax(
                    stack, 1, tables, moves, tt, history, killers, ctrl, clock, game_hashes,
                    depth - 1, -beta, -alpha, 1,
                )
        if ctrl[C_STOP]:
            break
        if score > best:
            best = score
            best_move = move
            if score > alpha:
                alpha = score
                ctrl[C_ROOT_BEST] = move
                ctrl[C_ROOT_SCORE] = score
                if alpha >= beta:
                    break
    if not ctrl[C_STOP] and best_move != 0 and best > -INF:
        flag = TT_LOWER if best >= beta else (TT_EXACT if alpha > -INF else TT_UPPER)
        tt_store(tt, pos[HASH], best_move, depth, flag, best, ctrl[C_AGE])
    return best

# ---------------------------------------------------------------------------------------------
# Driver. Module state lives for one game: the search arrays, the transposition table, and the
# hashes of every position the game has passed through, since the FEN we are handed carries no
# history and the referee claims threefold repetition on its own.
# ---------------------------------------------------------------------------------------------

INCREMENT_MS = 500.0
SAFETY_MS = 150.0
MOVES_HORIZON = 24  # spend about this fraction of the remaining clock per move

stack = np.zeros((MAX_PLY + 2, POS_LEN), dtype=np.int64)
moves = np.zeros((MAX_PLY + 2, MAX_MOVES), dtype=np.int64)
TT = np.zeros((TT_SIZE, 4), dtype=np.int64)
HISTORY = np.zeros(2 * 4096, dtype=np.int64)
KILLERS = np.zeros((MAX_PLY + 2, 2), dtype=np.int64)
CTRL = np.zeros(16, dtype=np.int64)
CLOCK = np.zeros(2, dtype=np.float64)
GAME_HASHES = np.zeros(1024, dtype=np.int64)
_game: list[int] = []

# Pondering. The process keeps its core while the opponent thinks, so after answering we keep
# searching the position we just handed them, on a thread with its own stacks, sharing only
# the transposition table and history. The next get_move stops it before searching.
PONDER_STACK = np.zeros((MAX_PLY + 2, POS_LEN), dtype=np.int64)
PONDER_MOVES = np.zeros((MAX_PLY + 2, MAX_MOVES), dtype=np.int64)
PONDER_KILLERS = np.zeros((MAX_PLY + 2, 2), dtype=np.int64)
PONDER_CTRL = np.zeros(16, dtype=np.int64)
PONDER_CLOCK = np.zeros(2, dtype=np.float64)
PONDER_HASHES = np.zeros(1024, dtype=np.int64)
_ponder_thread: threading.Thread | None = None
PONDER = True  # search on the opponent's time
PONDER_JOIN_S = 1.0


def _ponder() -> None:
    """Thread body: deepen on PONDER_STACK[0] until PONDER_CTRL[C_STOP] is set. Everything it
    needs was prepared by _start_ponder, so a stop flag raised before the thread gets going
    is never wiped by the thread itself."""
    score = 0
    for depth in range(1, MAX_DEPTH + 1):
        window = 30 if depth >= 5 else INF
        alpha, beta = max(score - window, -INF), min(score + window, INF)
        while not PONDER_CTRL[C_STOP]:
            value = search_root(
                PONDER_STACK, 0, tables, PONDER_MOVES, TT, HISTORY, PONDER_KILLERS,
                PONDER_CTRL, PONDER_CLOCK, PONDER_HASHES, depth, alpha, beta,
            )
            if value <= -INF:
                return  # the opponent has no legal move: nothing to think about
            if value <= alpha:
                alpha = max(alpha - window, -INF)
            elif value >= beta:
                beta = min(beta + window, INF)
            else:
                score = value
                break
            window *= 2
        if PONDER_CTRL[C_STOP] or abs(score) >= MATE_BOUND:
            break


def _start_ponder() -> None:
    global _ponder_thread
    if not PONDER:
        return
    if _ponder_thread is not None and _ponder_thread.is_alive():
        return  # never two at once; the old one is still winding down
    if in_check(stack[1], tables) and gen_moves(stack[1], tables, moves[1], 0) == 0:
        return
    history_len = min(len(_game), len(GAME_HASHES))
    PONDER_HASHES[:] = 0
    PONDER_HASHES[:history_len] = _game[len(_game) - history_len :]
    PONDER_STACK[0] = stack[1]
    PONDER_CTRL[:] = 0
    PONDER_CTRL[C_AGE] = CTRL[C_AGE]
    PONDER_CTRL[C_HISTORY_LEN] = history_len
    PONDER_CTRL[C_ROOT_SIDE] = int(stack[1, SIDE])
    PONDER_CLOCK[0] = float("inf")
    PONDER_KILLERS[:] = 0
    _ponder_thread = threading.Thread(target=_ponder, daemon=True)
    _ponder_thread.start()


def _stop_ponder() -> None:
    global _ponder_thread
    if _ponder_thread is None:
        return
    PONDER_CTRL[C_STOP] = 1
    _ponder_thread.join(PONDER_JOIN_S)
    if _ponder_thread.is_alive():
        print("ponder thread did not stop in time; searching alongside it")
    else:
        _ponder_thread = None


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    try:
        uci = _think(board, time_left_ms)
        if chess.Move.from_uci(uci) in legal:
            return uci
        print(f"engine returned illegal move {uci} in {fen}")
    except Exception:
        traceback.print_exc()  # a crash forfeits the game; a fallback move only risks it
    return _fallback(board, legal).uci()


def _fallback(board: chess.Board, legal: list[chess.Move]) -> chess.Move:
    """Best capture by victim value, else the first legal move: an emergency, not a plan."""
    def gain(move: chess.Move) -> int:
        victim = board.piece_type_at(move.to_square)
        return int(MG_VALUE[victim - 1]) if victim else 0

    return max(legal, key=gain)


def _budget_ms(time_left_ms: int) -> float:
    budget = min(time_left_ms / MOVES_HORIZON + 0.8 * INCREMENT_MS, time_left_ms / 4.0)
    return max(budget - SAFETY_MS, 10.0)


def _think(board: chess.Board, time_left_ms: int) -> str:
    start = time.monotonic()
    _stop_ponder()
    budget_s = _budget_ms(time_left_ms) / 1000.0
    stack[0] = position_from_board(board, tables)
    root_hash = int(stack[0, HASH])
    history_len = min(len(_game), len(GAME_HASHES))
    GAME_HASHES[:history_len] = _game[len(_game) - history_len :]
    CTRL[C_STOP] = 0
    CTRL[C_NODES] = 0
    CTRL[C_AGE] = (CTRL[C_AGE] + 1) & 255
    CTRL[C_ROOT_BEST] = 0
    CTRL[C_ROOT_SCORE] = 0
    CTRL[C_HISTORY_LEN] = history_len
    CTRL[C_SELDEPTH] = 0
    CTRL[C_ROOT_SIDE] = int(stack[0, SIDE])
    CLOCK[0] = start + budget_s
    KILLERS[:] = 0
    HISTORY[:] >>= 2

    best = 0
    score = 0
    for depth in range(1, MAX_DEPTH + 1):
        window = 30 if depth >= 5 else INF
        alpha, beta = max(score - window, -INF), min(score + window, INF)
        while True:
            value = search_root(
                stack, 0, tables, moves, TT, HISTORY, KILLERS, CTRL, CLOCK, GAME_HASHES,
                depth, alpha, beta,
            )
            if CTRL[C_STOP]:
                break
            if value <= alpha:
                alpha = max(alpha - window, -INF)
            elif value >= beta:
                beta = min(beta + window, INF)
            else:
                score = value
                break
            window *= 2
        if CTRL[C_ROOT_BEST]:
            best = int(CTRL[C_ROOT_BEST])
        if CTRL[C_STOP]:
            break
        elapsed = time.monotonic() - start
        print(
            f"depth {depth}/{int(CTRL[C_SELDEPTH])} score {score} move {move_to_uci(best)} "
            f"nodes {int(CTRL[C_NODES])} time {elapsed:.2f}s"
        )
        if abs(score) >= MATE_BOUND or elapsed > 0.45 * budget_s:
            break

    if best == 0:  # never happens with a legal position, but never return nothing
        n = gen_moves(stack[0], tables, moves[0], 0)
        best = int(moves[0, 0] & MOVE_MASK) if n else 0
    _game.append(root_hash)
    make_move(stack, 0, tables, best)
    _game.append(int(stack[1, HASH]))
    _start_ponder()
    return move_to_uci(best)


# Compile everything now, inside the init budget, by searching one position; the assert
# checks the compiled path returns a real move before the clock ever starts.
def new_game() -> None:
    """Forget the previous game. The platform starts a fresh process per game, so this only
    matters for local tests that play several games in one process."""
    _stop_ponder()
    _game.clear()
    TT[:] = 0
    HISTORY[:] = 0


_warm = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
assert chess.Move.from_uci(_think(_warm, 2000)) in _warm.legal_moves
new_game()
