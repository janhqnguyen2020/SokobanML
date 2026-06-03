# src/utils/custom_maps.py
"""Custom benchmark maps for fixed Sokoban evaluation.

Active map groups
-----------------
custom_core   — 10 open-grid maps by Shizuka (map_01..map_10), widths 12-15.
                Used for classical-planner-only benchmarks.
curriculum    — 60 progressive curriculum maps (curr_01-10 baseline +
                curr_prog_01-50 staged 1/2/3-box progression).
                Considered seen/training-adjacent — not a held-out test set.

Archived map groups (kept for reference, not used in active benchmarks)
-----------------------------------------------------------------------
custom_additional — 5 maps (easy_1box, medium_1box, large_1box, medium_2box,
                    hard_3box). Widths up to 15x15 exceed the DQN canvas.
"""


def build_map(map_name,difficulty,height,width,player,boxes,goals,walls=None,group_name=""):
    """Return one custom-map config for the benchmark runner."""
    return {
        "map_name": map_name,
        "difficulty": difficulty,
        "height": height,
        "width": width,
        "player": player,
        "boxes": boxes,
        "goals": goals,
        "walls": walls or [],
        "group_name": group_name,
        "max_steps": 200,
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
    """60 progressive curriculum maps spanning 1-box through 3-box difficulty.

    Original baseline (curr_01-10): 10 maps used in early training runs.
    Progressive stages (curr_prog_01-50): 8 staged difficulty groups.

    Stage 0 — baseline (curr_01-10):         original 1/2/3-box set
    Stage 1 — 1-box easy (curr_prog_01-05):  basic push mechanics
    Stage 2 — 1-box planning (06-10):        navigation, longer distances
    Stage 3 — 1-box CSP-lite (11-15):        single-box routing pressure
    Stage 4 — 2-box easy (16-20):            coordination basics
    Stage 5 — 2-box planning (21-30):        bottlenecks, ordering
    Stage 6 — 2-box CSP (31-40):             assignment reasoning
    Stage 7 — 3-box planning (41-45):        multi-box navigation
    Stage 8 — 3-box CSP (46-50):             full assignment + commitment
    """
    return [
        # ================================================================
        # STAGE 0 — ORIGINAL BASELINE (curr_01-10)
        # ================================================================
        build_map("curr_01", "Easy-1box",    5,  7, (3, 1), [(2, 2)],                   [(1, 4)]),
        build_map("curr_02", "Easy-1box",    6,  8, (4, 1), [(2, 3)],                   [(1, 5)]),
        build_map("curr_03", "EasyMed-2box", 6,  8, (4, 3), [(2, 2), (2, 5)],           [(1, 2), (1, 5)]),
        build_map("curr_04", "Medium-2box",  7,  9, (5, 4), [(2, 2), (4, 6)],           [(2, 6), (4, 2)]),
        build_map("curr_05", "Medium-2box",  7,  9, (3, 4), [(1, 2), (5, 6)],           [(1, 6), (5, 2)]),
        build_map("curr_06", "MedHard-3box", 8, 10, (3, 5), [(2, 2), (4, 7), (5, 3)],  [(2, 7), (5, 7), (1, 3)]),
        build_map("curr_07", "Hard-3box",    9, 10, (5, 5), [(2, 2), (4, 5), (6, 7)],  [(2, 8), (2, 5), (6, 5)]),
        build_map("curr_08", "Hard-3box",    9, 11, (4, 5), [(2, 3), (4, 8), (6, 2)],  [(2, 7), (6, 8), (6, 5)]),
        build_map("curr_09", "Hard-3box",   10, 12, (5, 6), [(2, 3), (5, 9), (7, 2)],  [(2, 9), (7, 9), (2, 2)]),
        build_map("curr_10", "Hard-3box",   11, 13, (6, 6), [(2, 3), (6, 10), (9, 4)], [(2, 10), (9, 10), (9, 8)]),

        # ================================================================
        # STAGE 1 — 1-BOX EASY (curr_prog_01-05)
        # ================================================================
        build_map("curr_prog_01", "Easy-1box",   5,  7, (3, 1), [(2, 2)], [(1, 4)]),
        build_map("curr_prog_02", "Easy-1box",   6,  8, (4, 1), [(2, 3)], [(1, 5)]),
        build_map("curr_prog_03", "Easy-1box",   6,  8, (4, 4), [(3, 2)], [(1, 6)]),
        build_map("curr_prog_04", "Easy-1box",   7,  9, (5, 3), [(4, 4)], [(1, 7)]),
        build_map("curr_prog_05", "Easy-1box",   7,  9, (5, 5), [(3, 3)], [(1, 5)]),

        # ================================================================
        # STAGE 2 — 1-BOX PLANNING (curr_prog_06-10)
        # ================================================================
        build_map("curr_prog_06", "Medium-1box",  8,  9, (6, 1), [(4, 3)], [(1, 7)]),
        build_map("curr_prog_07", "Medium-1box",  8, 10, (6, 5), [(3, 2)], [(1, 8)]),
        build_map("curr_prog_08", "Medium-1box",  9, 10, (7, 3), [(5, 5)], [(1, 7)]),
        build_map("curr_prog_09", "Medium-1box",  9, 11, (7, 8), [(4, 3)], [(1, 9)]),
        build_map("curr_prog_10", "Medium-1box", 10, 11, (8, 5), [(5, 5)], [(1, 8)]),

        # ================================================================
        # STAGE 3 — 1-BOX CSP-LITE (curr_prog_11-15)
        # ================================================================
        build_map("curr_prog_11", "CSPLite-1box",  8,  9, (6, 2), [(4, 4)], [(1, 7)]),
        build_map("curr_prog_12", "CSPLite-1box",  9, 10, (7, 1), [(4, 5)], [(1, 8)]),
        build_map("curr_prog_13", "CSPLite-1box",  9, 11, (7, 4), [(5, 3)], [(1, 9)]),
        build_map("curr_prog_14", "CSPLite-1box", 10, 11, (8, 7), [(4, 4)], [(1, 8)]),
        build_map("curr_prog_15", "CSPLite-1box", 10, 12, (8, 2), [(5, 5)], [(1, 10)]),

        # ================================================================
        # STAGE 4 — 2-BOX EASY (curr_prog_16-20)
        # ================================================================
        build_map("curr_prog_16", "Easy-2box", 6,  8, (4, 3), [(2, 2), (2, 5)], [(1, 2), (1, 5)]),
        build_map("curr_prog_17", "Easy-2box", 7,  9, (5, 4), [(2, 2), (4, 6)], [(2, 6), (4, 2)]),
        build_map("curr_prog_18", "Easy-2box", 7,  9, (3, 4), [(1, 2), (5, 6)], [(1, 6), (5, 2)]),
        build_map("curr_prog_19", "Easy-2box", 8, 10, (6, 5), [(3, 3), (5, 7)], [(1, 3), (1, 7)]),
        build_map("curr_prog_20", "Easy-2box", 8, 10, (6, 2), [(3, 4), (5, 6)], [(1, 4), (1, 8)]),

        # ================================================================
        # STAGE 5 — 2-BOX PLANNING (curr_prog_21-30)
        # ================================================================
        build_map("curr_prog_21", "Medium-2box",  8, 10, (5, 5), [(2, 2), (4, 7)],  [(2, 7), (5, 7)]),
        build_map("curr_prog_22", "Medium-2box",  9, 10, (5, 5), [(2, 2), (6, 7)],  [(2, 8), (6, 5)]),
        build_map("curr_prog_23", "Medium-2box",  9, 11, (4, 5), [(2, 3), (4, 8)],  [(2, 7), (6, 8)]),
        build_map("curr_prog_24", "Medium-2box", 10, 12, (5, 6), [(2, 3), (5, 9)],  [(2, 9), (7, 9)]),
        build_map("curr_prog_25", "Medium-2box", 11, 13, (6, 6), [(2, 3), (6, 10)], [(2, 10), (9, 10)]),
        build_map("curr_prog_26", "Medium-2box",  9, 10, (7, 3), [(2, 2), (5, 6)],  [(2, 7), (6, 7)]),
        build_map("curr_prog_27", "Medium-2box", 10, 11, (8, 2), [(2, 3), (6, 7)],  [(2, 8), (7, 5)]),
        build_map("curr_prog_28", "Medium-2box",  9, 10, (7, 6), [(2, 4), (5, 2)],  [(2, 7), (5, 8)]),
        build_map("curr_prog_29", "Medium-2box", 10, 11, (8, 8), [(2, 2), (5, 5)],  [(2, 8), (6, 2)]),
        build_map("curr_prog_30", "Medium-2box", 11, 12, (9, 6), [(2, 4), (6, 9)],  [(2, 9), (7, 4)]),

        # ================================================================
        # STAGE 6 — 2-BOX CSP (curr_prog_31-40)
        # ================================================================
        build_map("curr_prog_31", "CSP-2box",  9, 10, (7, 2), [(3, 3), (6, 7)], [(1, 2), (1, 8)]),
        build_map("curr_prog_32", "CSP-2box", 10, 11, (8, 1), [(3, 5), (6, 4)], [(1, 7), (7, 7)]),
        build_map("curr_prog_33", "CSP-2box",  9, 11, (7, 8), [(3, 3), (5, 7)], [(1, 2), (1, 9)]),
        build_map("curr_prog_34", "CSP-2box", 10, 12, (8, 3), [(3, 6), (6, 4)], [(1, 9), (1, 2)]),
        build_map("curr_prog_35", "CSP-2box",  9, 10, (7, 2), [(3, 5), (6, 3)], [(1, 7), (1, 1)]),
        build_map("curr_prog_36", "CSP-2box", 10, 11, (8, 9), [(3, 2), (6, 8)], [(1, 1), (1, 9)]),
        build_map("curr_prog_37", "CSP-2box",  9, 10, (7, 5), [(3, 2), (5, 7)], [(1, 1), (1, 8)]),
        build_map("curr_prog_38", "CSP-2box", 10, 12, (8, 6), [(3, 3), (6, 9)], [(1, 2), (1, 10)]),
        build_map("curr_prog_39", "CSP-2box", 11, 12, (9, 4), [(3, 2), (7, 9)], [(1, 1), (1, 10)]),
        build_map("curr_prog_40", "CSP-2box", 10, 11, (8, 3), [(3, 7), (6, 4)], [(1, 8), (1, 1)]),

        # ================================================================
        # STAGE 7 — 3-BOX PLANNING (curr_prog_41-45)
        # ================================================================
        build_map("curr_prog_41", "Hard-3box",  9, 10, (5, 5), [(2, 2), (4, 5), (6, 7)],   [(2, 8), (2, 5), (6, 5)]),
        build_map("curr_prog_42", "Hard-3box",  9, 11, (4, 5), [(2, 3), (4, 8), (6, 2)],   [(2, 7), (6, 8), (6, 5)]),
        build_map("curr_prog_43", "Hard-3box", 10, 12, (5, 6), [(2, 3), (5, 9), (7, 2)],   [(2, 9), (7, 9), (2, 2)]),
        build_map("curr_prog_44", "Hard-3box", 11, 13, (6, 6), [(2, 3), (6, 10), (9, 4)],  [(2, 10), (9, 10), (9, 8)]),
        build_map("curr_prog_45", "Hard-3box", 11, 12, (8, 5), [(3, 2), (5, 7), (8, 9)],   [(1, 2), (1, 8), (1, 10)]),

        # ================================================================
        # STAGE 8 — 3-BOX CSP (curr_prog_46-50)
        # ================================================================
        build_map("curr_prog_46", "CSP-3box", 10, 11, (8, 2),  [(3, 2), (5, 5), (7, 8)],  [(1, 2), (1, 6), (1, 9)]),
        build_map("curr_prog_47", "CSP-3box", 11, 12, (9, 1),  [(3, 3), (5, 7), (8, 9)],  [(1, 2), (1, 8), (1, 10)]),
        build_map("curr_prog_48", "CSP-3box", 10, 12, (8, 10), [(3, 2), (5, 5), (7, 7)],  [(1, 1), (1, 6), (1, 10)]),
        build_map("curr_prog_49", "CSP-3box", 11, 13, (9, 6),  [(3, 3), (5, 6), (8, 10)], [(1, 2), (1, 7), (1, 11)]),
        build_map("curr_prog_50", "CSP-3box", 12, 12, (10, 2), [(3, 2), (6, 5), (8, 8)],  [(1, 2), (1, 6), (1, 9)]),
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
    if "additional" in source_names:
        maps.extend(build_additional_maps())
    if "archived" in source_names:
        maps.extend(build_archived_maps())
    return filter_maps_by_name(maps, selected_names)


def filter_maps_by_name(maps, selected_names):
    if not selected_names:
        return maps
    wanted = set(selected_names)
    return [m for m in maps if m["map_name"] in wanted]


def build_additional_maps():
    """Return extra custom maps for one-off planner checks."""
    return [
        build_map(
            map_name="tue_map_01",
            difficulty="Hard-4box",
            height=7,
            width=10,
            player=(3, 2),
            boxes=[(2, 2), (3, 4), (3, 7), (4, 6)],
            goals=[(4, 2), (4, 3), (5, 2), (5, 3)],
            walls=[(1,1), (2,1), (5,1), (2, 3), (2, 4), (2, 5), (4, 4), (5, 4), (1, 7), (1,8), (4,8), (5,8)],
            group_name="additional",
        ),
    ]