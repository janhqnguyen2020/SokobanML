# src/utils/custom_maps.py
"""Custom benchmark maps for fixed Sokoban evaluation.

Active map groups
-----------------
custom_core   — 10 open-grid maps by Shizuka (map_01..map_10), widths 12-15.
                Used for classical-planner-only benchmarks.
curriculum    — 10 hand-crafted curriculum maps (curr_01..curr_10).
                Used to design and validate the curriculum progression.
                Considered seen/training-adjacent — not a held-out test set.

Archived map groups (kept for reference, not used in active benchmarks)
-----------------------------------------------------------------------
custom_additional — 5 maps (easy_1box, medium_1box, large_1box, medium_2box,
                    hard_3box). Widths up to 15x15 exceed the DQN canvas.
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


# ---------------------------------------------------------------------------
# Archived groups (not used in active benchmarks)
# ---------------------------------------------------------------------------

def build_archived_maps():
    """Archived: original custom_additional maps (widths up to 15x15, exceed DQN canvas)."""
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
    """All active maps: core (classical) + curriculum."""
    return build_core_custom_maps() + build_curriculum_maps()


def select_custom_maps(source_names, selected_names):
    maps = []
    if "custom_core" in source_names:
        maps.extend(build_core_custom_maps())
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