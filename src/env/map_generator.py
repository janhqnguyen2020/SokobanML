# src/env/map_generator.py
"""Generates guaranteed-solvable 1-box, 2-box, and 3-box Sokoban maps.

Maps use only outer-border walls (no interior walls), matching all existing
custom maps in src/utils/custom_maps.py.

Training maps (default):
    python src/env/map_generator.py
    python src/env/map_generator.py --n1 50 --n2 50 --seed 42

FinalEval benchmark (seed=999, separate output file):
    python src/env/map_generator.py --final-eval
    python src/env/map_generator.py --final-eval --n1 50 --n2 50 --n3 50 --seed 999
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


# Player reachability (used by 3-box reverse scrambler)

def _player_reachable(player, h: int, w: int, boxes) -> frozenset:
    """Return all interior cells reachable by the player without pushing any box."""
    blocked = frozenset(map(tuple, boxes))
    start = tuple(player)
    visited = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if _interior(nr, nc, h, w) and (nr, nc) not in blocked and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return frozenset(visited)


# 3-box reverse scrambler

def _reverse_scramble_3box(h: int, w: int, goals, rng, n_scramble: int):
    """Scramble from the solved state (boxes on goals) via reverse pushes.

    Forward push in direction d:
        before: player at P,   box at P+d
        after:  player at P+d, box at P+2d

    Un-push (reverse) — box at B_cur, direction d:
        requires: player can reach B_cur - d
        result:   box moves to B_cur - d, player moves to B_cur - 2d

    Because reachability is verified at each step the player can execute
    every forward push in reverse order, so the puzzle is solvable by
    construction.  We also require every box to be displaced from its goal
    so no position is pre-solved.
    """
    interior = frozenset((r, c) for r in range(1, h - 1) for c in range(1, w - 1))
    boxes = list(map(tuple, goals))
    free = list(interior - set(boxes))
    if not free:
        return None
    player = rng.choice(free)

    success = 0
    for _ in range(n_scramble * 50):
        if success >= n_scramble:
            break
        bi = rng.randrange(3)
        d = rng.choice(_DIRS)
        box = boxes[bi]

        push_from = (box[0] - d[0], box[1] - d[1])   # box moves here; player must reach this cell
        player_dest = (box[0] - 2 * d[0], box[1] - 2 * d[1])  # player ends up here

        if push_from not in interior or player_dest not in interior:
            continue
        other = [boxes[j] for j in range(3) if j != bi]
        if push_from in other or player_dest in other:
            continue
        if push_from not in _player_reachable(player, h, w, boxes):
            continue

        boxes[bi] = push_from
        player = player_dest
        success += 1

    if success < max(n_scramble // 3, 3):
        return None

    # All boxes must be displaced — no pre-solved positions
    goal_set = set(map(tuple, goals))
    if any(b in goal_set for b in boxes):
        return None

    return tuple(player), [list(b) for b in boxes]


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

_3BOX_PROFILE = [
    ("Easy-3box",   [(7, 8), (7, 9), (8, 8)],      15),
    ("Medium-3box", [(8, 9), (8, 10), (9, 9)],     22),
    ("Hard-3box",   [(9, 10), (9, 11), (10, 10)],  30),
]


def _split(n: int, k: int) -> list[int]:
    base, rem = divmod(n, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


# Map generators

def generate_1box_maps(n: int = 50, seed: int = 42, prefix: str = "gen1b") -> list[dict]:
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
                "map_name": f"{prefix}_{idx:03d}",
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


def generate_2box_maps(n: int = 50, seed: int = 42, prefix: str = "gen2b") -> list[dict]:
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
                "map_name": f"{prefix}_{idx:03d}",
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


def generate_3box_maps(n: int = 50, seed: int = 999, prefix: str = "fe3b") -> list[dict]:
    """Return n reverse-generated solvable 3-box maps distributed across difficulties.

    Solvability is guaranteed by construction: each puzzle is built by
    un-pushing boxes from a known-solved state, with player reachability
    verified at every step.
    """
    rng = random.Random(seed)
    targets = _split(n, len(_3BOX_PROFILE))
    maps: list[dict] = []
    idx = 1

    for (difficulty, sizes, n_scramble), target in zip(_3BOX_PROFILE, targets):
        count = 0
        attempts = 0
        maxAttempts = max(target * 400, 1000)

        while count < target and attempts < maxAttempts:
            attempts += 1
            h, w = rng.choice(sizes)
            interior = list({(r, c) for r in range(1, h - 1) for c in range(1, w - 1)})
            if len(interior) < 7:
                continue

            goals = [tuple(g) for g in rng.sample(interior, 3)]
            result = _reverse_scramble_3box(h, w, goals, rng, n_scramble)
            if result is None:
                continue

            player, boxes = result
            maps.append({
                "map_name": f"{prefix}_{idx:03d}",
                "difficulty": difficulty,
                "height": h,
                "width": w,
                "player": player,
                "boxes": boxes,
                "goals": goals,
                "max_steps": 180,
            })
            idx += 1
            count += 1

        if count < target:
            print(f"  [warn] {difficulty}: generated {count}/{target} maps ({attempts} attempts)")

    return maps


# File output

def _fmt(m: dict) -> str:
    boxes = [tuple(b) for b in m["boxes"]]
    goals = [tuple(g) for g in m["goals"]]
    return (
        f'    _m("{m["map_name"]}", "{m["difficulty"]}", '
        f'{m["height"]}, {m["width"]}, '
        f'{m["player"]!r}, {boxes!r}, {goals!r}, {m["max_steps"]})'
    )


def write_generated_maps(maps_1box: list[dict], maps_2box: list[dict], output_path: str) -> None:
    """Write training maps (1-box + 2-box) to an importable Python file."""
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

    print(f"Wrote {len(maps_1box)} 1-box + {len(maps_2box)} 2-box maps -> {output_path}")


def write_final_eval_maps(
    maps_1box: list[dict],
    maps_2box: list[dict],
    maps_3box: list[dict],
    output_path: str,
    seed: int,
) -> None:
    """Write FinalEval benchmark (1-box + 2-box + 3-box) to a separate Python file."""
    total = len(maps_1box) + len(maps_2box) + len(maps_3box)

    def section(fn: str, doc: str, maps: list[dict]) -> list[str]:
        lines = [f"def {fn}():", f'    """{doc}"""', "    return ["]
        for m in maps:
            lines.append(_fmt(m) + ",")
        lines += ["    ]", "", ""]
        return lines

    lines = [
        "# src/utils/final_eval_maps.py",
        '"""Frozen FinalEval benchmark -- seed={seed}, never used in training.'.format(seed=seed),
        "",
        "Auto-generated by src/env/map_generator.py. Do not edit by hand.",
        "To regenerate: python src/env/map_generator.py --final-eval --seed {seed}".format(seed=seed),
        "",
        "1-box maps : BFS-verified solvable.",
        "2-box maps : BFS-verified solvable.",
        "3-box maps : solvable by reverse-scramble construction (reachability-checked).",
        '"""',
        "",
        "",
        "def _m(name, diff, h, w, player, boxes, goals, max_steps=120):",
        '    return {"map_name": name, "difficulty": diff, "height": h, "width": w,',
        '            "player": player, "boxes": boxes, "goals": goals, "max_steps": max_steps}',
        "",
        "",
    ]
    lines += section(
        "build_final_eval_1box_maps",
        f"Returns {len(maps_1box)} BFS-verified 1-box FinalEval maps (seed={seed}).",
        maps_1box,
    )
    lines += section(
        "build_final_eval_2box_maps",
        f"Returns {len(maps_2box)} BFS-verified 2-box FinalEval maps (seed={seed}).",
        maps_2box,
    )
    lines += section(
        "build_final_eval_3box_maps",
        f"Returns {len(maps_3box)} reverse-generated 3-box FinalEval maps (seed={seed}).",
        maps_3box,
    )
    lines += [
        "def build_all_final_eval_maps():",
        f'    """All {total} FinalEval maps: {len(maps_1box)} one-box + {len(maps_2box)} two-box + {len(maps_3box)} three-box."""',
        "    return (build_final_eval_1box_maps()",
        "            + build_final_eval_2box_maps()",
        "            + build_final_eval_3box_maps())",
        "",
    ]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Wrote {len(maps_1box)} 1-box + {len(maps_2box)} 2-box + {len(maps_3box)} 3-box -> {output_path}")


# Verification (used in --final-eval mode)

def verify_final_eval(maps_1box, maps_2box, maps_3box) -> int:
    errors = 0

    for m in maps_1box:
        h, w = m["height"], m["width"]
        p, b, g = m["player"], m["boxes"][0], m["goals"][0]
        if not (_interior(*p, h, w) and _interior(*b, h, w) and _interior(*g, h, w)):
            print(f"  [ERROR] {m['map_name']}: position outside interior")
            errors += 1
        elif b == g:
            print(f"  [ERROR] {m['map_name']}: box already on goal")
            errors += 1
        elif not solvable_1box(h, w, p, b, g):
            print(f"  [ERROR] {m['map_name']}: BFS confirms unsolvable")
            errors += 1

    for m in maps_2box:
        h, w = m["height"], m["width"]
        p, bs, gs = m["player"], m["boxes"], m["goals"]
        if not (all(_interior(*b, h, w) for b in bs) and all(_interior(*g, h, w) for g in gs)):
            print(f"  [ERROR] {m['map_name']}: position outside interior")
            errors += 1
        elif not solvable_2box(h, w, p, bs, gs):
            print(f"  [ERROR] {m['map_name']}: BFS confirms unsolvable")
            errors += 1

    for m in maps_3box:
        h, w = m["height"], m["width"]
        p = m["player"]
        bs = [tuple(b) for b in m["boxes"]]
        gs = [tuple(g) for g in m["goals"]]
        goal_set = set(gs)
        ok = True
        if not _interior(*p, h, w):
            print(f"  [ERROR] {m['map_name']}: player outside interior")
            ok = False
        if len(set(bs)) != 3:
            print(f"  [ERROR] {m['map_name']}: duplicate box positions")
            ok = False
        if tuple(p) in set(bs):
            print(f"  [ERROR] {m['map_name']}: player overlaps a box")
            ok = False
        if any(b in goal_set for b in bs):
            print(f"  [ERROR] {m['map_name']}: box pre-placed on goal")
            ok = False
        if not all(_interior(*b, h, w) for b in bs):
            print(f"  [ERROR] {m['map_name']}: box outside interior")
            ok = False
        if not ok:
            errors += 1

    return errors


# Entry point

def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(
        description="Generate solvable Sokoban maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n1",   type=int, default=50, help="Number of 1-box maps")
    parser.add_argument("--n2",   type=int, default=50, help="Number of 2-box maps")
    parser.add_argument("--n3",   type=int, default=50, help="Number of 3-box maps (--final-eval only)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--final-eval", action="store_true",
        help=(
            "Generate the FinalEval held-out benchmark (50+50+50, seed=999) "
            "instead of the training map set."
        ),
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output path (auto-selected by mode if omitted)",
    )
    args = parser.parse_args()

    if args.final_eval:
        seed = args.seed if args.seed != 42 else 999
        out = args.out or os.path.normpath(
            os.path.join(here, "..", "utils", "final_eval_maps.py")
        )

        print(f"=== FinalEval mode (seed={seed}) ===")
        print(f"Generating {args.n1} 1-box maps...")
        maps_1box = generate_1box_maps(args.n1, seed, prefix="fe1b")
        print(f"  Done: {len(maps_1box)}")

        print(f"Generating {args.n2} 2-box maps...")
        maps_2box = generate_2box_maps(args.n2, seed, prefix="fe2b")
        print(f"  Done: {len(maps_2box)}")

        print(f"Generating {args.n3} 3-box maps...")
        maps_3box = generate_3box_maps(args.n3, seed, prefix="fe3b")
        print(f"  Done: {len(maps_3box)}")

        total = len(maps_1box) + len(maps_2box) + len(maps_3box)
        print(f"\nVerifying all {total} maps...")
        errors = verify_final_eval(maps_1box, maps_2box, maps_3box)
        if errors:
            print(f"\nFAILED: {errors} error(s) found. Output NOT written.")
            raise SystemExit(1)
        print(f"  All {total} maps OK.")
        write_final_eval_maps(maps_1box, maps_2box, maps_3box, out, seed)

    else:
        seed = args.seed
        out = args.out or os.path.normpath(
            os.path.join(here, "..", "utils", "generated_maps.py")
        )

        print(f"Generating {args.n1} 1-box maps (seed={seed})...")
        maps_1box = generate_1box_maps(args.n1, seed)
        print(f"  Done: {len(maps_1box)} maps")

        print(f"Generating {args.n2} 2-box maps (seed={seed})...")
        maps_2box = generate_2box_maps(args.n2, seed)
        print(f"  Done: {len(maps_2box)} maps")

        write_generated_maps(maps_1box, maps_2box, out)


if __name__ == "__main__":
    main()
