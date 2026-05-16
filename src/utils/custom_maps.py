def build_map(map_name, height, width, player, boxes, goals):
    """Store one custom map configuration in a simple dictionary."""
    return {
        "map_name": map_name,
        "height": height,
        "width": width,
        "player": player,
        "boxes": boxes,
        "goals": goals,
        "max_steps": 120,
    }


def build_custom_maps():
    """Return fixed custom maps with mixed left and right goals."""
    return [
        build_map("map_01", 3, 12, (1, 1), [(1, 6)], [(1, 10)]),
        build_map("map_02", 5, 12, (2, 10), [(2, 6)], [(2, 2)]),

        build_map("map_03", 5, 13, (2, 6), [(1, 5), (3, 7)], [(1, 2), (3, 10)]),
        build_map("map_04", 5, 13, (2, 6), [(1, 7), (3, 5)], [(1, 10), (3, 2)]),

        build_map("map_05", 6, 14, (2, 7), [(1, 5), (2, 8), (4, 6)], [(1, 2), (2, 11), (4, 10)]),
        build_map("map_06", 6, 14, (3, 7), [(1, 8), (3, 6), (4, 8)], [(1, 11), (3, 2), (4, 3)]),

        build_map("map_07", 6, 14, (2, 7), [(1, 5), (2, 8), (4, 5)], [(1, 11), (2, 2), (4, 10)]),
        build_map("map_08", 6, 14, (3, 7), [(1, 8), (3, 6), (4, 5)], [(1, 3), (3, 11), (4, 2)]),

        build_map("map_09", 7, 15, (3, 8), [(1, 6), (3, 5), (5, 9)], [(1, 12), (3, 2), (5, 12)]),
        build_map("map_10", 7, 15, (3, 7), [(1, 9), (3, 6), (5, 5)], [(1, 3), (3, 12), (5, 2)]),
    ]