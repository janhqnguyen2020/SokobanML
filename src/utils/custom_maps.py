# src/utils/custom_maps.py
"""Custom benchmark maps for fixed Sokoban evaluation.

Active map groups
-----------------
custom_core   — 10 open-grid maps by Shizuka (map_01..map_10), widths 12-15.
                Used for classical-planner-only benchmarks.
canvas        — 10 maps all within the DQN 10x10 observation canvas (≤9x9).
                7 new maps (canvas_01..canvas_07) + 3 original dqn_custom maps.
                Used for the full 4-algorithm comparison.

Archived map groups (kept for reference, not used in active benchmarks)
-----------------------------------------------------------------------
custom_additional — 5 maps (easy_1box, medium_1box, large_1box, medium_2box,
                    hard_3box). Replaced by the canvas set for DQN-compatible
                    evaluation. Widths up to 15x15 exceed the DQN canvas.
"""


def build_map(map_name, difficulty, height, width, player, boxes, goals):
    return {
        "map_name": map_name,
        "difficulty": difficulty,
        "height": height,
        "width": width,
        "player": player,
        "boxes": boxes,
        "goals": goals,
        "max_steps": 120,
    }


# ---------------------------------------------------------------------------
# Active groups
# ---------------------------------------------------------------------------

def build_core_custom_maps():
    """10 open-grid maps by Shizuka — classical planners only (widths 12-15)."""
    return [
        build_map("map_01", "Easy-1box",   3, 12, (1, 1),  [(1, 6)],                    [(1, 10)]),
        build_map("map_02", "Medium-1box", 5, 12, (2, 10), [(2, 6)],                    [(2, 2)]),
        build_map("map_03", "Medium-2box", 5, 13, (2, 6),  [(1, 5), (3, 7)],            [(1, 2), (3, 10)]),
        build_map("map_04", "Medium-2box", 5, 13, (2, 6),  [(1, 7), (3, 5)],            [(1, 10), (3, 2)]),
        build_map("map_05", "Hard-3box",   6, 14, (2, 7),  [(1, 5), (2, 8), (4, 6)],   [(1, 2), (2, 11), (4, 10)]),
        build_map("map_06", "Hard-3box",   6, 14, (3, 7),  [(1, 8), (3, 6), (4, 8)],   [(1, 11), (3, 2), (4, 3)]),
        build_map("map_07", "Hard-3box",   6, 14, (2, 7),  [(1, 5), (2, 8), (4, 5)],   [(1, 11), (2, 2), (4, 10)]),
        build_map("map_08", "Hard-3box",   6, 14, (3, 7),  [(1, 8), (3, 6), (4, 5)],   [(1, 3), (3, 11), (4, 2)]),
        build_map("map_09", "Hard-3box",   7, 15, (3, 8),  [(1, 6), (3, 5), (5, 9)],   [(1, 12), (3, 2), (5, 12)]),
        build_map("map_10", "Hard-3box",   7, 15, (3, 7),  [(1, 9), (3, 6), (5, 5)],   [(1, 3), (3, 12), (5, 2)]),
    ]


def build_canvas_maps():
    """7 DQN-compatible 3-box maps — all ≤9x9, obs size 412, action space 12.

    Covers a range of push complexity and navigation difficulty. All maps use
    exactly 3 boxes to match the trained model's fixed input size.
    """
    return [
        # Easy: boxes aligned, all push straight up 2 steps
        build_map("canvas_01", "Easy-3box",   6, 9, (4, 4), [(3, 2), (3, 4), (3, 6)], [(1, 2), (1, 4), (1, 6)]),
        # Easy: boxes aligned, all push straight down 1 step, player starts above
        build_map("canvas_05", "Easy-3box",   5, 9, (1, 4), [(2, 2), (2, 4), (2, 6)], [(3, 2), (3, 4), (3, 6)]),
        # Medium: boxes push in mixed directions, moderate distances
        build_map("canvas_02", "Medium-3box", 7, 9, (5, 1), [(4, 2), (2, 5), (4, 7)], [(1, 2), (5, 5), (2, 7)]),
        # Medium: all boxes push right, but player starts on the wrong (right) side
        build_map("canvas_06", "Medium-3box", 7, 8, (3, 6), [(1, 2), (3, 2), (5, 2)], [(1, 5), (3, 5), (5, 5)]),
        # Hard: long push distances, player must navigate around boxes
        build_map("canvas_03", "Hard-3box",   8, 9, (6, 1), [(5, 2), (3, 5), (1, 7)], [(1, 2), (6, 5), (6, 7)]),
        # Hard: scattered boxes, mixed push directions (right/left/right)
        build_map("canvas_07", "Hard-3box",   8, 9, (5, 5), [(2, 3), (4, 6), (6, 3)], [(2, 7), (4, 3), (6, 7)]),
        # Very Hard: full 9x9 grid, maximum push distances, most planning required
        build_map("canvas_04", "VHard-3box",  9, 9, (7, 4), [(5, 2), (4, 5), (2, 7)], [(1, 2), (7, 5), (6, 7)]),
    ]


def build_dqn_test_maps():
    """3 symmetric 3-box maps within 9x9 — original DQN training-compatible maps."""
    return [
        build_map("dqn_map_1", "Hard-3box", 7, 9, (5, 4), [(2, 4), (3, 4), (4, 4)], [(1, 4), (3, 2), (3, 6)]),
        build_map("dqn_map_2", "Hard-3box", 6, 9, (3, 4), [(2, 4), (3, 2), (3, 6)], [(1, 2), (1, 6), (4, 4)]),
        build_map("dqn_map_3", "Hard-3box", 6, 9, (4, 4), [(2, 4), (3, 2), (3, 6)], [(1, 2), (1, 4), (1, 6)]),
    ]


def build_all_canvas_maps():
    """Full 10-map canvas set: canvas_01-07 + dqn_map_1-3. All ≤9x9."""
    return build_canvas_maps() + build_dqn_test_maps()


# ---------------------------------------------------------------------------
# Archived groups (not used in active benchmarks)
# ---------------------------------------------------------------------------

def build_archived_maps():
    """Archived: original custom_additional maps (widths up to 15x15, exceed DQN canvas).

    Replaced by build_canvas_maps() + build_dqn_test_maps() for DQN-compatible
    evaluation. Kept here for reference and reproducibility of earlier results.
    """
    return [
        build_map("easy_1box",   "Easy",        3,  12, (1, 1), [(1, 4)],            [(1, 10)]),
        build_map("medium_1box", "Medium",       9,   9, (1, 1), [(2, 4)],            [(7, 7)]),
        build_map("large_1box",  "Large-1box",  15,  15, (1, 1), [(5, 7)],            [(12, 9)]),
        build_map("medium_2box", "Medium-2box", 11,  11, (1, 1), [(2, 3), (3, 6)],   [(8, 3), (8, 6)]),
        build_map("hard_3box",   "Hard-3box",   15,  15, (1, 1), [(2, 2), (3, 3), (5, 7)], [(4, 9), (10, 5), (12, 7)]),
    ]


# ---------------------------------------------------------------------------
# Curriculum group (variable box count, variable size, DQN-compatible ≤15x15)
# ---------------------------------------------------------------------------

def build_curriculum_maps():
    """10 hand-crafted maps for curriculum DQN training and evaluation.

    Difficulty progression: 2 easy (1-box) → 3 medium (2-box) →
    1 medium-hard + 4 hard (3-box). All verified solvable by a simple
    push sequence with no blocking conflicts.

    Map       Size    Boxes  Difficulty    Solution sketch
    curr_01   5x7     1      Easy          right x2, up x1
    curr_02   6x8     1      Easy          right x2, up x1
    curr_03   6x8     2      Easy-Med      up x1 each
    curr_04   7x9     2      Medium        right x4 / left x4
    curr_05   7x9     2      Medium        right x4 / left x4 (edge rows)
    curr_06   8x10    3      Medium-Hard   right x5, down x1, up x4
    curr_07   9x10    3      Hard          right x6, up x2, left x2
    curr_08   9x11    3      Hard          right x4, down x2, right x3
    curr_09   10x12   3      Hard          right x6, down x2, up x5
    curr_10   11x13   3      Hard          right x7, down x3, right x4
    """
    return [
        # --- Easy: 1 box ---
        build_map("curr_01", "Easy-1box",   5,  7, (3, 1), [(2, 2)],                    [(1, 4)]),
        build_map("curr_02", "Easy-1box",   6,  8, (4, 1), [(2, 3)],                    [(1, 5)]),
        # --- Easy-Med: 2 boxes ---
        build_map("curr_03", "EasyMed-2box", 6,  8, (4, 3), [(2, 2), (2, 5)],           [(1, 2), (1, 5)]),
        # --- Medium: 2 boxes ---
        build_map("curr_04", "Medium-2box", 7,  9, (5, 4), [(2, 2), (4, 6)],            [(2, 6), (4, 2)]),
        build_map("curr_05", "Medium-2box", 7,  9, (3, 4), [(1, 2), (5, 6)],            [(1, 6), (5, 2)]),
        # --- Medium-Hard: 3 boxes ---
        build_map("curr_06", "MedHard-3box", 8, 10, (3, 5), [(2, 2), (4, 7), (5, 3)],  [(2, 7), (5, 7), (1, 3)]),
        # --- Hard: 3 boxes ---
        build_map("curr_07", "Hard-3box",   9, 10, (5, 5), [(2, 2), (4, 5), (6, 7)],   [(2, 8), (2, 5), (6, 5)]),
        build_map("curr_08", "Hard-3box",   9, 11, (4, 5), [(2, 3), (4, 8), (6, 2)],   [(2, 7), (6, 8), (6, 5)]),
        build_map("curr_09", "Hard-3box",  10, 12, (5, 6), [(2, 3), (5, 9), (7, 2)],   [(2, 9), (7, 9), (2, 2)]),
        build_map("curr_10", "Hard-3box",  11, 13, (6, 6), [(2, 3), (6, 10), (9, 4)],  [(2, 10), (9, 10), (9, 8)]),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_all_custom_maps():
    """All active maps: core (classical) + canvas (DQN-compatible)."""
    return build_core_custom_maps() + build_all_canvas_maps()


def select_custom_maps(source_names, selected_names):
    maps = []
    if "custom_core" in source_names:
        maps.extend(build_core_custom_maps())
    if "canvas" in source_names:
        maps.extend(build_all_canvas_maps())
    if "dqn_custom" in source_names:
        maps.extend(build_dqn_test_maps())
    if "curriculum" in source_names:
        maps.extend(build_curriculum_maps())
    if "archived" in source_names:
        maps.extend(build_archived_maps())
    return filter_maps_by_name(maps, selected_names)


def filter_maps_by_name(maps, selected_names):
    if not selected_names:
        return maps
    wanted = set(selected_names)
    return [m for m in maps if m["map_name"] in wanted]