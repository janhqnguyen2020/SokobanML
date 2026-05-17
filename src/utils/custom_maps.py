"""Custom benchmark maps for fixed Sokoban evaluation."""


def build_map(map_name, difficulty, height, width, player, boxes, goals):
    """Store one custom map in a simple dictionary."""
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


def build_core_custom_maps():
    """Return the main custom benchmark set."""
    return [
        build_map("map_01", "Easy-1box", 3, 12, (1, 1), [(1, 6)], [(1, 10)]),
        build_map("map_02", "Medium-1box", 5, 12, (2, 10), [(2, 6)], [(2, 2)]),
        build_map("map_03", "Medium-2box", 5, 13, (2, 6), [(1, 5), (3, 7)], [(1, 2), (3, 10)]),
        build_map("map_04", "Medium-2box", 5, 13, (2, 6), [(1, 7), (3, 5)], [(1, 10), (3, 2)]),
        build_map("map_05", "Hard-3box", 6, 14, (2, 7), [(1, 5), (2, 8), (4, 6)], [(1, 2), (2, 11), (4, 10)]),
        build_map("map_06", "Hard-3box", 6, 14, (3, 7), [(1, 8), (3, 6), (4, 8)], [(1, 11), (3, 2), (4, 3)]),
        build_map("map_07", "Hard-3box", 6, 14, (2, 7), [(1, 5), (2, 8), (4, 5)], [(1, 11), (2, 2), (4, 10)]),
        build_map("map_08", "Hard-3box", 6, 14, (3, 7), [(1, 8), (3, 6), (4, 5)], [(1, 3), (3, 11), (4, 2)]),
        build_map("map_09", "Hard-3box", 7, 15, (3, 8), [(1, 6), (3, 5), (5, 9)], [(1, 12), (3, 2), (5, 12)]),
        build_map("map_10", "Hard-3box", 7, 15, (3, 7), [(1, 9), (3, 6), (5, 5)], [(1, 3), (3, 12), (5, 2)]),
    ]


def build_additional_custom_maps():
    return [
        build_map("easy_1box", "Easy", 3, 12, (1, 1), [(1, 4)], [(1, 10)]),
        build_map("medium_1box", "Medium", 9, 9, (1, 1), [(2, 4)], [(7, 7)]),
        build_map("large_1box", "Large-1box", 15, 15, (1, 1), [(5, 7)], [(12, 9)]),
        build_map("medium_2box", "Medium-2box", 11, 11, (1, 1), [(2, 3), (3, 6)], [(8, 3), (8, 6)]),
        build_map("hard_3box", "Hard-3box", 15, 15, (1, 1), [(2, 2), (3, 3), (5, 7)], [(4, 9), (10, 5), (12, 7)]),
    ]

def build_all_custom_maps():
    """Return every custom benchmark map."""
    return build_core_custom_maps() + build_additional_custom_maps()


def select_custom_maps(source_names, selected_names):
    """Return only the requested custom maps."""
    maps = []
    if "custom_core" in source_names:
        maps.extend(build_core_custom_maps())
    if "custom_additional" in source_names:
        maps.extend(build_additional_custom_maps())
    return filter_maps_by_name(maps, selected_names)


def filter_maps_by_name(maps, selected_names):
    """Keep all maps when no names are given, otherwise filter by id."""
    if not selected_names:
        return maps
    wanted = set(selected_names)
    return [config for config in maps if config["map_name"] in wanted]