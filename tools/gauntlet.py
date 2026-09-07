"""Play two agent versions against each other and say whether the difference is real.

    uv run python tools/gauntlet.py --a . --b snapshots/stage5 --games 400

`harness.arena` starts a fresh process per game, which is honest but costs a full numba
compile (about 20 s) per side per game, so a run long enough to measure anything takes hours.
This loads both agents as modules once per worker and plays many games in that worker,
driving the real `harness.referee` through an in-process adapter, so the clock, the draw and
adjudication rules and the legality checks are exactly the ones `harness.arena` uses. Use
`make arena` to check that an agent survives the process protocol; use this to find out
whether a change is worth uploading.

Games are played in pairs from each opening, once with each colour, and the report carries an
Elo estimate with a 95% interval and the likelihood that A is genuinely ahead.
"""

import argparse
import contextlib
import importlib.util
import io
import math
import os
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from types import ModuleType

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.referee import FAILED_TERMINATIONS, play_match  # noqa: E402
from harness.rules import PLY_CAP  # noqa: E402

_LOADED: dict[str, ModuleType] = {}


def _load(directory: Path, alias: str) -> ModuleType:
    """Import `directory/agent.py` under its own module name so two versions can coexist."""
    path = directory.resolve() / "agent.py"
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)  # the warm-up search at import prints its iterations
    return module


class InProcess:
    """The Agent interface harness.referee expects, backed by an imported module.

    The referee only ever calls start, move and stop, and it times `move` on the wall clock,
    so a game played through this is scored by exactly the code that scores a real one.
    """

    def __init__(self, module: ModuleType) -> None:
        self.module = module
        self.stderr_tail = ""

    def start(self, init_budget_s: float) -> None:
        self.module.new_game()

    def move(self, fen: str, time_left_ms: int) -> str:
        with contextlib.redirect_stdout(io.StringIO()):
            return str(self.module.get_move(fen, time_left_ms))

    def stop(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.module.new_game()


def _initialise(a: str, b: str, ponder: bool, increment_ms: int) -> None:
    _LOADED["a"] = _load(Path(a), "agent_a")
    _LOADED["b"] = _load(Path(b), "agent_b")
    for module in _LOADED.values():
        # Both engines share this core, so pondering here measures the scheduler as much as
        # the engine. `make play` is where pondering gets exercised honestly.
        module.PONDER = ponder
        # get_move is handed a clock but never an increment, so the shipped constant is the
        # rated 500 ms. Testing at a faster control without telling the engine leaves it
        # budgeting for time it will not be given, which is precisely the thing a change to
        # time management would be measured against.
        if hasattr(module, "INCREMENT_MS"):
            module.INCREMENT_MS = float(increment_ms)


def _play(task: tuple[str, bool, int, int, int]) -> tuple[str, str, bool]:
    fen, a_is_white, base_ms, increment_ms, ply_cap = task
    a, b = InProcess(_LOADED["a"]), InProcess(_LOADED["b"])
    white, black = (a, b) if a_is_white else (b, a)
    outcome = play_match(white, black, base_ms, increment_ms, ply_cap=ply_cap, start_fen=fen)
    return outcome.result, outcome.termination, a_is_white


def _elo(score: float) -> float:
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def report(wins: int, draws: int, losses: int, terminations: Counter[str]) -> None:
    n = wins + draws + losses
    if n == 0:
        raise SystemExit("no games completed")
    score = (wins + draws / 2.0) / n
    variance = (
        wins * (1.0 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2
    ) / n
    stderr = math.sqrt(variance / n) if variance > 0 else 0.0

    print(f"\n+{wins} ={draws} -{losses}  over {n} games")
    print(f"score {score:.4f} +/- {1.96 * stderr:.4f}")
    if 0.0 < score < 1.0 and stderr > 0.0:
        # Delta method: d(Elo)/d(score) at the measured score.
        slope = 400.0 / (math.log(10.0) * score * (1.0 - score))
        margin = 1.96 * stderr * slope
        print(f"Elo  {_elo(score):+.1f} +/- {margin:.1f}  (95%)")
        los = _phi((score - 0.5) / stderr)
        print(f"likelihood A is ahead of B: {los:.1%}")
        if score - 1.96 * stderr > 0.5:
            print("verdict: A is better; the interval clears even.")
        elif score + 1.96 * stderr < 0.5:
            print("verdict: A is WORSE; do not upload this.")
        else:
            gap = abs(score - 0.5)
            needed = int((1.96 * math.sqrt(variance) / gap) ** 2) if gap > 0 else 0
            print(f"verdict: not separated yet (about {needed} games would settle it).")
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in terminations.most_common()))
    broken = {k: v for k, v in terminations.items() if k in FAILED_TERMINATIONS}
    if broken:
        raise SystemExit("an agent failed to finish a game: " + str(broken))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, default=Path("."), help="the version under test")
    parser.add_argument("--b", type=Path, default=Path("snapshots/stage5"), help="the reference")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--base-ms", type=int, default=8000)
    parser.add_argument("--increment-ms", type=int, default=80)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--openings", type=Path, default=Path("tools/openings.epd"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 3))
    parser.add_argument("--ponder", action="store_true", help="leave pondering on (noisy)")
    arguments = parser.parse_args()

    if arguments.openings.exists():
        lines = arguments.openings.read_text().splitlines()
        fens = [line.strip() for line in lines if line.strip()]
    else:
        print(f"{arguments.openings} not found; every game starts from the standard position")
        fens = [chess.STARTING_FEN]

    tasks: list[tuple[str, bool, int, int, int]] = []
    for index in range(arguments.games):
        fen = fens[(index // 2) % len(fens)]
        tasks.append(
            (fen, index % 2 == 0, arguments.base_ms, arguments.increment_ms, arguments.ply_cap)
        )

    wins = draws = losses = 0
    terminations: Counter[str] = Counter()
    print(f"{arguments.a} vs {arguments.b}: {arguments.games} games, "
          f"{arguments.base_ms}ms+{arguments.increment_ms}ms, {arguments.workers} workers")

    with Pool(
        arguments.workers,
        initializer=_initialise,
        initargs=(
            str(arguments.a), str(arguments.b), arguments.ponder, arguments.increment_ms,
        ),
    ) as pool:
        for done, (result, termination, a_is_white) in enumerate(
            pool.imap_unordered(_play, tasks), start=1
        ):
            terminations[termination] += 1
            if result in ("draw", "void"):
                draws += 1
            elif (result == "white") == a_is_white:
                wins += 1
            else:
                losses += 1
            if done % 20 == 0 or done == len(tasks):
                score = (wins + draws / 2.0) / done
                print(f"  {done}/{len(tasks)}  +{wins} ={draws} -{losses}  "
                      f"score {score:.3f}  Elo {_elo(score):+.0f}", flush=True)

    report(wins, draws, losses, terminations)


if __name__ == "__main__":
    main()
