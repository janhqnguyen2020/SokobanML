"""Generates guaranteed-solvable 1-box and 2-box Sokoban maps.

Maps use only outer-border walls (no interior walls), matching all existing
custom maps in src/utils/custom_maps.py.

ex: python src/env/map_generator.py
    python src/env/map_generator.py --n1 50 --n2 50 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
from collections import deque

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right


# BFS solvability

def _interior(r: int, c: int, h: int, w: int) -> bool:
    return 1 <= r <= h - 2 and 1 <= c <= w - 2


def solvable_1box(h: int, w: int, player, box, goal) -> bool:
    """BFS over (player, box) state space. Returns True if box can reach goal."""
    playerRow, playerCol = player
    boxRow, boxCol = box
    goalRow, goalCol = goal

    start = (playerRow, playerCol, boxRow, boxCol)
    visited = {start}
    queue = deque([start])

    while queue:
        playerRow, playerCol, boxRow, boxCol = queue.popleft()
        if (boxRow, boxCol) == (goalRow, goalCol):
            return True
        for dirRow, dirCol in _DIRS:
            nextRow, nextCol = playerRow + dirRow, playerCol + dirCol
            if not _interior(nextRow, nextCol, h, w):
                continue
            if (nextRow, nextCol) == (boxRow, boxCol):  # push
                newBoxRow, newBoxCol = boxRow + dirRow, boxCol + dirCol
                if not _interior(newBoxRow, newBoxCol, h, w):
                    continue
                state = (nextRow, nextCol, newBoxRow, newBoxCol)
            else:  # move
                state = (nextRow, nextCol, boxRow, boxCol)
            if state not in visited:
                visited.add(state)
                queue.append(state)

    return False


def solvable_2box(h: int, w: int, player, boxes, goals) -> bool:
    """BFS over (player, box1, box2) states. Any box-to-goal assignment counts."""

    def canonicalState(playerRow: int, playerCol: int, box1: tuple, box2: tuple) -> tuple:
        if box1 > box2:
            box1, box2 = box2, box1
        return (playerRow, playerCol, box1[0], box1[1], box2[0], box2[1])

    box1 = tuple(boxes[0])
    box2 = tuple(boxes[1])
    goalSet = frozenset(tuple(g) for g in goals)

    start = canonicalState(*player, box1, box2)
    visited = {start}
    queue = deque([start])

    while queue:
        playerRow, playerCol, box1Row, box1Col, box2Row, box2Col = queue.popleft()
        if frozenset([(box1Row, box1Col), (box2Row, box2Col)]) == goalSet:
            return True

        box1 = (box1Row, box1Col)
        box2 = (box2Row, box2Col)

        for dirRow, dirCol in _DIRS:
            nextRow, nextCol = playerRow + dirRow, playerCol + dirCol
            if not _interior(nextRow, nextCol, h, w):
                continue

            if (nextRow, nextCol) == box1:
                newBox1 = (box1Row + dirRow, box1Col + dirCol)
                if not _interior(*newBox1, h, w) or newBox1 == box2:
                    continue
                state = canonicalState(nextRow, nextCol, newBox1, box2)
            elif (nextRow, nextCol) == box2:
                newBox2 = (box2Row + dirRow, box2Col + dirCol)
                if not _interior(*newBox2, h, w) or newBox2 == box1:
                    continue
                state = canonicalState(nextRow, nextCol, box1, newBox2)
            else:
                state = canonicalState(nextRow, nextCol, box1, box2)

            if state not in visited:
                visited.add(state)
                queue.append(state)

    return False


# Difficulty and size profiles

_1BOX_PROFILE = [
    ("Easy-1box",   [(5, 6), (5, 7), (5, 8), (6, 7)]),
    ("Medium-1box", [(6, 8), (6, 9), (7, 8), (7, 9)]),
    ("Hard-1box",   [(8, 8), (8, 9), (8, 10), (9, 9), (9, 10)]),
]

_2BOX_PROFILE = [
    ("Easy-2box",   [(6, 7), (6, 8), (7, 8)]),
    ("Medium-2box", [(7, 9), (8, 9), (8, 10)]),
    ("Hard-2box",   [(9, 9), (9, 10), (9, 11)]),
]


def _split(n: int, k: int) -> list[int]:
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


# Map generators

def generate_1box_maps(n: int = 50, seed: int = 42) -> list[dict]:
    """Return n BFS-verified solvable 1-box maps distributed across difficulties."""
    rng = random.Random(seed)
    targets = _split(n, len(_1BOX_PROFILE))
    maps: list[dict] = []
    idx = 1

    for (difficulty, sizes), target in zip(_1BOX_PROFILE, targets):
        count = 0
        attempts = 0
        maxAttempts = max(target * 300, 500)

        while count < target and attempts < maxAttempts:
            attempts += 1
            h, w = rng.choice(sizes)
            interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
            if len(interior) < 3:
                continue

            player, box, goal = rng.sample(interior, 3)
            if box == goal:
                continue
            if not solvable_1box(h, w, player, box, goal):
                continue

            maps.append({
                "map_name": f"gen1b_{idx:03d}",
                "difficulty": difficulty,
                "height": h,
                "width": w,
                "player": player,
                "boxes": [box],
                "goals": [goal],
                "max_steps": 120,
            })
            idx += 1
            count += 1

        if count < target:
            print(f"  [warn] {difficulty}: generated {count}/{target} maps ({attempts} attempts)")

    return maps


def generate_2box_maps(n: int = 50, seed: int = 42) -> list[dict]:
    """Return n BFS-verified solvable 2-box maps distributed across difficulties."""
    rng = random.Random(seed)
    targets = _split(n, len(_2BOX_PROFILE))
    maps: list[dict] = []
    idx = 1

    for (difficulty, sizes), target in zip(_2BOX_PROFILE, targets):
        count = 0
        attempts = 0
        maxAttempts = max(target * 500, 1000)

        while count < target and attempts < maxAttempts:
            attempts += 1
            h, w = rng.choice(sizes)
            interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
            if len(interior) < 5:
                continue

            positions = rng.sample(interior, 5)
            player = positions[0]
            boxes = positions[1:3]
            goals = positions[3:5]

            if boxes[0] == boxes[1] or goals[0] == goals[1]:
                continue
            if any(b == g for b in boxes for g in goals):
                continue
            if not solvable_2box(h, w, player, boxes, goals):
                continue

            maps.append({
                "map_name": f"gen2b_{idx:03d}",
                "difficulty": difficulty,
                "height": h,
                "width": w,
                "player": player,
                "boxes": boxes,
                "goals": goals,
                "max_steps": 150,
            })
            idx += 1
            count += 1

        if count < target:
            print(f"  [warn] {difficulty}: generated {count}/{target} maps ({attempts} attempts)")

    return maps


# File output

def _fmt(m: dict) -> str:
    return (
        f'    _m("{m["map_name"]}", "{m["difficulty"]}", '
        f'{m["height"]}, {m["width"]}, '
        f'{m["player"]!r}, {m["boxes"]!r}, {m["goals"]!r}, {m["max_steps"]})'
    )


def write_generated_maps(maps_1box: list[dict], maps_2box: list[dict], output_path: str) -> None:
    """Write maps to an importable Python file at output_path."""
    lines = [
        '"""Generated Sokoban maps — BFS-verified solvable.',
        '',
        'Auto-generated by src/env/map_generator.py. Do not edit by hand.',
        'To regenerate: python src/env/map_generator.py',
        '"""',
        '',
        '',
        'def _m(name, diff, h, w, player, boxes, goals, max_steps=120):',
        '    return {"map_name": name, "difficulty": diff, "height": h, "width": w,',
        '            "player": player, "boxes": boxes, "goals": goals, "max_steps": max_steps}',
        '',
        '',
        f'def build_generated_1box_maps():',
        f'    """Returns {len(maps_1box)} BFS-verified solvable 1-box maps."""',
        '    return [',
    ]
    for m in maps_1box:
        lines.append(_fmt(m) + ',')
    lines += [
        '    ]',
        '',
        '',
        f'def build_generated_2box_maps():',
        f'    """Returns {len(maps_2box)} BFS-verified solvable 2-box maps."""',
        '    return [',
    ]
    for m in maps_2box:
        lines.append(_fmt(m) + ',')
    lines += ['    ]', '']

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Wrote {len(maps_1box)} 1-box + {len(maps_2box)} 2-box maps → {output_path}")


# Entry point

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate solvable Sokoban maps for curriculum training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n1",   type=int, default=50, help="Number of 1-box maps")
    parser.add_argument("--n2",   type=int, default=50, help="Number of 2-box maps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path (default: src/utils/generated_maps.py beside this file)",
    )
    args = parser.parse_args()

    if args.out is None:
        here = os.path.dirname(os.path.abspath(__file__))
        args.out = os.path.normpath(os.path.join(here, "..", "utils", "generated_maps.py"))

    print(f"Generating {args.n1} 1-box maps (seed={args.seed})...")
    maps_1box = generate_1box_maps(args.n1, args.seed)
    print(f"  Done: {len(maps_1box)} maps")

    print(f"Generating {args.n2} 2-box maps (seed={args.seed})...")
    maps_2box = generate_2box_maps(args.n2, args.seed)
    print(f"  Done: {len(maps_2box)} maps")

    write_generated_maps(maps_1box, maps_2box, args.out)


if __name__ == "__main__":
    main()
