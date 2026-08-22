"""Generate reproducible 2x2 systolic-array differential vectors."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from src.test.model import matmul_int8

FORMAT_VERSION = 1
DEFAULT_SEED = 0x5A17
DEFAULT_CASES = 128
PHYSICAL_ROWS = 2
PHYSICAL_COLUMNS = 2
MAX_K = 8
SIGNED_ENDPOINTS = (-128, -1, 0, 1, 127)


@dataclass(frozen=True)
class SystolicVector:
    m: int
    n: int
    k: int
    stall_step: int
    a_padded: tuple[int, ...]
    b_padded: tuple[int, ...]
    c_padded: tuple[int, ...]


def _random_int8(rng: random.Random) -> int:
    if rng.randrange(5) == 0:
        return rng.choice(SIGNED_ENDPOINTS)
    return rng.randint(-128, 127)


def _make_case(
    rng: random.Random,
    m: int,
    n: int,
    k: int,
    forced_a: tuple[tuple[int, ...], ...] | None = None,
    forced_b: tuple[tuple[int, ...], ...] | None = None,
) -> SystolicVector:
    matrix_a = forced_a or tuple(
        tuple(_random_int8(rng) for _ in range(k)) for _ in range(m)
    )
    matrix_b = forced_b or tuple(
        tuple(_random_int8(rng) for _ in range(n)) for _ in range(k)
    )
    matrix_c = matmul_int8(matrix_a, matrix_b)
    last_step = m + n + k - 3
    stall_step = rng.randint(0, last_step)

    a_padded = tuple(
        matrix_a[row][index] if row < m and index < k else 0
        for row in range(PHYSICAL_ROWS)
        for index in range(MAX_K)
    )
    b_padded = tuple(
        matrix_b[index][column] if index < k and column < n else 0
        for index in range(MAX_K)
        for column in range(PHYSICAL_COLUMNS)
    )
    c_padded = tuple(
        matrix_c[row][column] if row < m and column < n else 0
        for row in range(PHYSICAL_ROWS)
        for column in range(PHYSICAL_COLUMNS)
    )
    return SystolicVector(m, n, k, stall_step, a_padded, b_padded, c_padded)


def build_vectors(seed: int = DEFAULT_SEED, count: int = DEFAULT_CASES) -> tuple:
    if count < 100:
        raise ValueError("differential verification requires at least 100 cases")
    rng = random.Random(seed)
    vectors = [
        _make_case(
            rng,
            2,
            2,
            2,
            forced_a=((-128, 127), (127, -128)),
            forced_b=((127, -128), (-128, 127)),
        ),
        _make_case(
            rng,
            1,
            1,
            5,
            forced_a=((-128, -1, 0, 1, 127),),
            forced_b=((-128,), (-1,), (0,), (1,), (127,)),
        ),
    ]
    while len(vectors) < count:
        m = rng.randint(1, PHYSICAL_ROWS)
        n = rng.randint(1, PHYSICAL_COLUMNS)
        k = rng.randint(1, MAX_K)
        vectors.append(_make_case(rng, m, n, k))
    return tuple(vectors)


def write_vectors(
    output_path: Path,
    seed: int = DEFAULT_SEED,
    count: int = DEFAULT_CASES,
) -> None:
    vectors = build_vectors(seed=seed, count=count)
    lines = [
        f"{FORMAT_VERSION} {seed} {len(vectors)} "
        f"{PHYSICAL_ROWS} {PHYSICAL_COLUMNS} {MAX_K}"
    ]
    for vector in vectors:
        values = (
            (vector.m, vector.n, vector.k, vector.stall_step)
            + vector.a_padded
            + vector.b_padded
            + vector.c_padded
        )
        lines.append(" ".join(str(value) for value in values))
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cases", type=int, default=DEFAULT_CASES)
    arguments = parser.parse_args()
    write_vectors(arguments.output, arguments.seed, arguments.cases)


if __name__ == "__main__":
    main()
