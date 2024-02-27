"""Helpers to define simple network layouts."""

from typing import List


def rect(depth: int, width: int) -> List[int]:
    """Rectangular layout."""
    return [width] * depth


def tri(depth: int, start: int, end: int) -> List[int]:
    """Triangular layout."""
    widths = []
    for i in range(depth):
        z = i / (depth - 1)
        w = int(round((1 - z) * start + z * end))
        widths.append(w)
    return widths
