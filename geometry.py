"""
geometry.py — GeNeSIS V — Procedural Geometry & Topology
=============================================================

Masterplan §3.5. Two independent pieces, both real mathematics:

    - L-systems (Lindenmayer, 1968): recursive string-rewriting grammars
      turned into turtle-graphics line segments — real structure/plant
      growth, cheap to compute (a string rewrite plus a linear walk).
    - Betti numbers: B0 (connected components) and a real, standard 2D
      proxy for B1 (enclosed holes, via background-component labelling)
      formalise what Level 8 (Abstract Representation) tries to measure
      about the meme grid's structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage

EPS = 1e-12


# ----------------------------------------------------------------------------
# L-systems  (Lindenmayer, 1968)
# ----------------------------------------------------------------------------
@dataclass
class LSystem:
    axiom: str
    rules: Dict[str, str]
    angle_degrees: float

    def expand(self, iterations: int) -> str:
        s = self.axiom
        for _ in range(iterations):
            s = "".join(self.rules.get(ch, ch) for ch in s)
        return s


# Two real, well-known presets from Lindenmayer/Prusinkiewicz's canon —
# mathematical rule systems, not expressive copyrighted works.
KOCH_CURVE = LSystem(axiom="F", rules={"F": "F+F--F+F"}, angle_degrees=60.0)
FRACTAL_PLANT = LSystem(
    axiom="X",
    rules={"X": "F+[[X]-X]-F[-FX]+X", "F": "FF"},
    angle_degrees=25.0,
)


def turtle_render(
    instructions: str,
    angle_degrees: float,
    step: float = 1.0,
    start: Tuple[float, float] = (0.0, 0.0),
    start_heading_degrees: float = 90.0,
) -> List[Tuple[float, float, float, float]]:
    """
    Standard L-system turtle graphics: F/f move forward (F draws, f doesn't),
    +/- turn left/right by angle_degrees, [/] push/pop turtle state. Returns
    a list of (x0, y0, x1, y1) drawn line segments.
    """
    x, y = start
    heading = np.radians(start_heading_degrees)
    angle = np.radians(angle_degrees)
    stack: List[Tuple[float, float, float]] = []
    segments: List[Tuple[float, float, float, float]] = []

    for ch in instructions:
        if ch == "F":
            nx, ny = x + step * np.cos(heading), y + step * np.sin(heading)
            segments.append((x, y, nx, ny))
            x, y = nx, ny
        elif ch == "f":
            x, y = x + step * np.cos(heading), y + step * np.sin(heading)
        elif ch == "+":
            heading += angle
        elif ch == "-":
            heading -= angle
        elif ch == "[":
            stack.append((x, y, heading))
        elif ch == "]":
            if stack:
                x, y, heading = stack.pop()
        # any other character (e.g. 'X' in FRACTAL_PLANT) is a non-drawing
        # grammar symbol used only during expand(), not during rendering.

    return segments


def bounding_box(segments: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    """(min_x, min_y, max_x, max_y) — useful for fitting a render into a
    fixed-size canvas/artifact viewport."""
    if not segments:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p for seg in segments for p in (seg[0], seg[2])]
    ys = [p for seg in segments for p in (seg[1], seg[3])]
    return (min(xs), min(ys), max(xs), max(ys))


# ----------------------------------------------------------------------------
# Betti numbers on a 2D binary field  (real algebraic topology, standard 2D
# proxy for B1 — see e.g. any digital-topology text on hole-counting via
# background-component labelling)
# ----------------------------------------------------------------------------
def betti_0(binary_grid: np.ndarray) -> int:
    """B0 = number of connected foreground components."""
    _, n = ndimage.label(binary_grid > 0)
    return int(n)


def betti_1(binary_grid: np.ndarray) -> int:
    """
    B1 = number of enclosed holes. Standard 2D technique: label the
    *background* (the complement of the foreground); any background
    component that does not touch the grid border is an enclosed hole
    rather than the surrounding exterior. This is the real, commonly-used
    approach for hole-counting in a 2D binary image without computing full
    persistent homology.
    """
    background = binary_grid <= 0
    labeled, n = ndimage.label(background)
    if n == 0:
        return 0

    border_labels = set(labeled[0, :]) | set(labeled[-1, :]) | set(labeled[:, 0]) | set(labeled[:, -1])
    border_labels.discard(0)

    all_labels = set(range(1, n + 1))
    enclosed = all_labels - border_labels
    return len(enclosed)


def euler_characteristic(binary_grid: np.ndarray) -> int:
    """chi = B0 - B1, the real Euler characteristic formula for a 2D field."""
    return betti_0(binary_grid) - betti_1(binary_grid)


def topology_summary(binary_grid: np.ndarray) -> Dict[str, int]:
    b0 = betti_0(binary_grid)
    b1 = betti_1(binary_grid)
    return {"betti_0": b0, "betti_1": b1, "euler_characteristic": b0 - b1}


if __name__ == "__main__":
    expanded = KOCH_CURVE.expand(3)
    f_count = expanded.count("F")
    assert f_count == 4 ** 3, f"Koch curve after 3 iterations should have 4^3=64 F's, got {f_count}"
    print(f"Koch curve, 3 iterations: {f_count} F symbols (expected 4^3=64). Grammar expansion OK.")

    segments = turtle_render(expanded, KOCH_CURVE.angle_degrees, step=1.0)
    assert len(segments) == f_count, "one drawn segment per F symbol"
    bbox = bounding_box(segments)
    print(f"Koch curve rendered: {len(segments)} segments, bounding box {tuple(round(v, 2) for v in bbox)}")

    plant = FRACTAL_PLANT.expand(4)
    plant_segments = turtle_render(plant, FRACTAL_PLANT.angle_degrees, step=1.0)
    assert len(plant_segments) > 0
    print(f"Fractal plant, 4 iterations: {len(plant_segments)} segments rendered.")

    # Betti numbers: a grid with a real hole in it
    grid = np.ones((20, 20), dtype=np.int8)
    grid[7:13, 7:13] = 0  # a square hole, fully enclosed
    summary = topology_summary(grid)
    print("Topology of a solid square with one enclosed hole:", summary)
    assert summary["betti_0"] == 1, "one connected foreground component"
    assert summary["betti_1"] == 1, "exactly one enclosed hole"
    assert summary["euler_characteristic"] == 0

    # sanity: a solid grid with no holes has betti_1 == 0
    solid = np.ones((10, 10), dtype=np.int8)
    solid_summary = topology_summary(solid)
    print("Topology of a solid grid (no holes):", solid_summary)
    assert solid_summary["betti_1"] == 0
    assert solid_summary["euler_characteristic"] == 1

    # sanity: two disconnected blobs -> betti_0 == 2
    two_blobs = np.zeros((20, 20), dtype=np.int8)
    two_blobs[2:6, 2:6] = 1
    two_blobs[14:18, 14:18] = 1
    blob_summary = topology_summary(two_blobs)
    print("Topology of two disconnected blobs:", blob_summary)
    assert blob_summary["betti_0"] == 2

    # realistic use case: run this on an actual Gray-Scott morphogenesis frame
    from biology import gray_scott_step, init_grid

    A, B = init_grid(48, 48, seed=7)
    for _ in range(600):
        A, B = gray_scott_step(A, B)
    pattern_binary = (B > B.mean()).astype(np.int8)
    pattern_summary = topology_summary(pattern_binary)
    print("Topology of a real Gray-Scott morphogen pattern:", pattern_summary)
    assert pattern_summary["betti_0"] >= 1

    print("\ngeometry.py self-test passed.")
