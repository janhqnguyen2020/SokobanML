"""Helpers for the fixed-map subsets used during validation and periodic eval."""

from src.utils.config import (
    CURRICULUM_FIXED_PERIODIC_MAPS_PER_GROUP,
    CURRICULUM_FIXED_VAL_MAPS_PER_GROUP,
)


def build_quick_fixed_eval_maps(map_configs):
    """Pick the larger fixed subset used for stage-end validation and videos."""
    return build_balanced_fixed_eval_maps(map_configs, CURRICULUM_FIXED_VAL_MAPS_PER_GROUP)


def build_periodic_fixed_eval_maps(map_configs):
    """Pick the smaller fixed subset used by periodic checkpoint selection."""
    return build_balanced_fixed_eval_maps(map_configs, CURRICULUM_FIXED_PERIODIC_MAPS_PER_GROUP)


def build_balanced_fixed_eval_maps(map_configs, maps_per_group):
    """Pick the same number of maps from each fixed family for fair comparison."""
    return (
        select_generated_box_maps(map_configs, 1, maps_per_group)
        + select_generated_box_maps(map_configs, 2, maps_per_group)
        + select_generated_box_maps(map_configs, 3, maps_per_group)
        + select_prefix_maps(map_configs, ["custom", "wall"], maps_per_group)
        + select_prefix_maps(map_configs, ["csp"], maps_per_group)
    )


def select_generated_box_maps(map_configs, num_boxes, limit):
    """Pick the first few generated maps for one box-count family."""
    return select_matching_maps(
        map_configs,
        lambda map_config: map_config.get("map_source") == "generated" and len(map_config["boxes"]) == int(num_boxes),
        limit,
    )


def select_prefix_maps(map_configs, prefixes, limit):
    """Pick the first few fixed maps whose names match the requested family."""
    return select_matching_maps(
        map_configs,
        lambda map_config: prefix_match(map_config["map_name"], prefixes),
        limit,
    )


def select_matching_maps(map_configs, predicate, limit):
    """Return the first sorted maps that match one family rule."""
    selected_maps = []
    for map_config in sorted(map_configs, key=lambda item: item["map_name"]):
        if predicate(map_config):
            selected_maps.append(map_config)
        if len(selected_maps) == int(limit):
            return selected_maps
    return selected_maps


def prefix_match(map_name, prefixes):
    """Return True when one map name starts with any requested prefix."""
    return any(str(map_name).startswith(str(prefix)) for prefix in prefixes)
