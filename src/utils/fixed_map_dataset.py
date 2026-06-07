"""Build the merged fixed-map dataset used by imitation and curriculum runs."""

import json
import os
import random

from src.utils.generated_maps import (
    build_walled_1box_maps,
    build_walled_2box_maps,
    build_walled_3box_maps,
)

SPLIT_COUNTS_BY_BOX = {
    1: {"train": 114, "val": 28, "test": 28},
    2: {"train": 113, "val": 29, "test": 28},
    3: {"train": 113, "val": 28, "test": 29},
}


def build_fixed_dataset_splits(generated_json_path, split_seed):
    """Return the merged 510 maps plus deterministic train, val, and test splits."""
    all_maps = build_fixed_dataset_maps(generated_json_path)
    split_maps = split_fixed_maps_by_box_count(all_maps, split_seed)
    split_maps["all_maps"] = all_maps
    return split_maps


def build_fixed_dataset_maps(generated_json_path):
    """Merge repo-stored generated maps with the current solvable walled maps."""
    generated_maps = build_generated_maps_from_json(generated_json_path)
    walled_maps = build_current_walled_maps()
    merged_maps = generated_maps + walled_maps
    validate_fixed_dataset(merged_maps)
    return merged_maps


def build_generated_maps_from_json(generated_json_path):
    """Load the original generated 1/2/3-box maps from the repo JSON file."""
    generated_data = load_generated_map_json(generated_json_path)
    return flatten_generated_map_data(generated_data)


def load_generated_map_json(generated_json_path):
    """Read the repo-stored generated 1/2/3-box map JSON file."""
    with open(generated_json_path) as generated_file:
        return json.load(generated_file)


def flatten_generated_map_data(generated_data):
    """Flatten the generated 1/2/3-box lists into one annotated dataset."""
    merged_maps = []
    for box_key in ("1box", "2box", "3box"):
        merged_maps.extend(add_map_source(generated_data.get(box_key, []), "generated"))
    return merged_maps


def build_current_walled_maps():
    """Load the current branch walled/custom/CSP maps that include your solvability fixes."""
    walled_maps = []
    walled_maps.extend(add_map_source(build_walled_1box_maps(), "walled"))
    walled_maps.extend(add_map_source(build_walled_2box_maps(), "walled"))
    walled_maps.extend(add_map_source(build_walled_3box_maps(), "walled"))
    return walled_maps


def add_map_source(map_configs, map_source):
    """Attach one source label to each map without mutating the original dictionaries."""
    tagged_maps = []
    for map_config in map_configs:
        tagged_map = dict(map_config)
        tagged_map["map_source"] = str(map_source)
        tagged_maps.append(tagged_map)
    return tagged_maps


def validate_fixed_dataset(map_configs):
    """Fail fast when the merged fixed dataset is not the expected 510 unique maps."""
    map_names = [map_config["map_name"] for map_config in map_configs]
    if len(map_configs) != 510:
        raise ValueError(f"Expected 510 fixed maps but found {len(map_configs)}.")
    if len(set(map_names)) != len(map_names):
        raise ValueError("The merged fixed dataset contains duplicate map names.")


def split_fixed_maps_by_box_count(map_configs, split_seed):
    """Split the fixed maps by box count so train, val, and test stay balanced."""
    maps_by_boxes = group_maps_by_box_count(map_configs)
    split_maps = {"train_maps": [], "val_maps": [], "test_maps": []}
    for num_boxes in (1, 2, 3):
        assign_box_split(maps_by_boxes[num_boxes], num_boxes, split_seed, split_maps)
    return split_maps


def group_maps_by_box_count(map_configs):
    """Group the merged fixed maps into 1-box, 2-box, and 3-box buckets."""
    grouped_maps = {1: [], 2: [], 3: []}
    for map_config in map_configs:
        grouped_maps[len(map_config["boxes"])].append(map_config)
    return grouped_maps


def assign_box_split(map_configs, num_boxes, split_seed, split_maps):
    """Shuffle one box-count bucket and slice it into train, val, and test maps."""
    shuffled_maps = sorted(map_configs, key=lambda item: item["map_name"])
    random.Random(split_seed + num_boxes).shuffle(shuffled_maps)
    counts = SPLIT_COUNTS_BY_BOX[num_boxes]
    split_maps["train_maps"].extend(shuffled_maps[: counts["train"]])
    start_val = counts["train"]
    end_val = start_val + counts["val"]
    split_maps["val_maps"].extend(shuffled_maps[start_val:end_val])
    split_maps["test_maps"].extend(shuffled_maps[end_val : end_val + counts["test"]])


def build_split_manifest(split_maps, generated_json_path, split_seed):
    """Build one JSON-friendly manifest that records the dataset sources and split names."""
    return {
        "generated_json_path": os.path.abspath(generated_json_path),
        "split_seed": int(split_seed),
        "counts": count_split_members(split_maps),
        "train_map_names": sorted(map_config["map_name"] for map_config in split_maps["train_maps"]),
        "val_map_names": sorted(map_config["map_name"] for map_config in split_maps["val_maps"]),
        "test_map_names": sorted(map_config["map_name"] for map_config in split_maps["test_maps"]),
    }


def count_split_members(split_maps):
    """Return the map counts for the full dataset and each split."""
    return {
        "all_maps": len(split_maps["all_maps"]),
        "train_maps": len(split_maps["train_maps"]),
        "val_maps": len(split_maps["val_maps"]),
        "test_maps": len(split_maps["test_maps"]),
    }
