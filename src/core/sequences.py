"""Integer sequences used by more than one mini-project."""

from __future__ import annotations


def fibonacci_numbers(count: int) -> list[int]:
    """Return the first ``count`` Fibonacci numbers ``1, 1, 2, 3, 5, ...``.

    Raises:
        ValueError: If ``count`` is negative.
    """
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    seq: list[int] = []
    a, b = 1, 1
    for _ in range(count):
        seq.append(a)
        a, b = b, a + b
    return seq
