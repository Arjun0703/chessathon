"""Find the magic multipliers used by the slider attack tables in agent.py.

    uv run python tools/find_magics.py

For every square, try random sparse 64-bit numbers until one maps each relevant occupancy
subset to its attack set without a destructive collision. Prints the two lists to paste into
agent.py. Not needed at runtime; kept so anyone can regenerate or verify the constants.
"""

from collections.abc import Iterator

import numpy as np

Directions = tuple[tuple[int, int], ...]
ROOK: Directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP: Directions = ((1, 1), (1, -1), (-1, 1), (-1, -1))
MASK64 = (1 << 64) - 1


def ray_attacks(square: int, dirs: Directions, occupied: int) -> int:
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


def relevant_mask(square: int, dirs: Directions) -> int:
    mask = 0
    rank, file = divmod(square, 8)
    for d_rank, d_file in dirs:
        r, f = rank + d_rank, file + d_file
        while 0 <= r + d_rank < 8 and 0 <= f + d_file < 8:
            mask |= 1 << (r * 8 + f)
            r, f = r + d_rank, f + d_file
    return mask


def subsets(mask: int) -> Iterator[int]:
    subset = 0
    while True:
        yield subset
        subset = (subset - mask) & mask
        if subset == 0:
            return


def works(magic: int, shift: int, occupancies: list[int], attacks: list[int]) -> bool:
    table: dict[int, int] = {}
    for occupancy, attack in zip(occupancies, attacks, strict=True):
        index = ((occupancy * magic) & MASK64) >> shift
        if table.setdefault(index, attack) != attack:
            return False
    return True


def find(square: int, dirs: Directions, rng: np.random.Generator) -> int:
    mask = relevant_mask(square, dirs)
    shift = 64 - bin(mask).count("1")
    occupancies = list(subsets(mask))
    attacks = [ray_attacks(square, dirs, occupancy) for occupancy in occupancies]
    while True:
        candidate = 1
        for _ in range(3):  # sparse candidates converge much faster
            candidate &= int(rng.integers(0, 2**64, dtype=np.uint64))
        if bin(((mask * candidate) & MASK64) >> 56).count("1") < 6:
            continue
        if works(candidate, shift, occupancies, attacks):
            return candidate


def main() -> None:
    rng = np.random.default_rng(12345)
    for name, dirs in (("_ROOK_MAGICS", ROOK), ("_BISHOP_MAGICS", BISHOP)):
        magics = [find(square, dirs, rng) for square in range(64)]
        print(f"{name} = [")
        for row in range(0, 64, 4):
            print("    " + ", ".join(f"0x{magic:016X}" for magic in magics[row : row + 4]) + ",")
        print("]")


if __name__ == "__main__":
    main()
