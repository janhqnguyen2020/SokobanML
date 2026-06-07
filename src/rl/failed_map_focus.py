"""Helpers for reweighting curriculum training toward previously failed fixed maps."""

import csv
import glob
import os


def resolve_failed_map_focus(split_maps, explicit_csv_path=None, project_root=None):
    """Load failed fixed maps from one CSV and match them back to full map configs."""
    csv_path = explicit_csv_path or latest_validation_fixed_csv(project_root)
    if csv_path is None:
        return empty_focus_payload()
    failed_names = failed_map_names(csv_path)
    return focus_payload(split_maps, csv_path, failed_names)


def latest_validation_fixed_csv(project_root):
    """Return the newest validation_fixed.csv saved by training or validation-only runs."""
    patterns = validation_csv_patterns(project_root)
    csv_paths = matching_validation_csv_paths(patterns)
    if not csv_paths:
        return None
    return max(csv_paths, key=os.path.getmtime)


def validation_csv_patterns(project_root):
    """Build the search patterns used to find saved fixed-map validation CSV files."""
    base_dir = os.path.join(project_root, "results", "rl_tests", "curriculum_dqn_imitation")
    return [
        os.path.join(base_dir, "*", "validation", "post_imitation", "validation_fixed.csv"),
        os.path.join(base_dir, "*", "validation", "phase*", "validation_fixed.csv"),
        os.path.join(base_dir, "evals", "validation_only", "*", "validation_fixed.csv"),
    ]


def matching_validation_csv_paths(patterns):
    """Collect every validation CSV path that matches the requested glob patterns."""
    csv_paths = []
    for pattern in patterns:
        csv_paths.extend(glob.glob(pattern))
    return csv_paths


def failed_map_names(csv_path):
    """Read one validation CSV and keep only the map names that were not solved."""
    with open(csv_path, newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    return [row["map_name"] for row in rows if not csv_row_solved(row)]


def csv_row_solved(row):
    """Return True when the saved validation row reports a solved fixed map."""
    return str(row.get("solved", "")).strip().lower() == "true"


def focus_payload(split_maps, csv_path, failed_names):
    """Match failed names to map configs and summarize how many came from each split."""
    matched_maps = matched_failed_maps(split_maps["all_maps"], failed_names)
    matched_names = {map_config["map_name"] for map_config in matched_maps}
    return {
        "csv_path": str(csv_path),
        "failed_names": list(failed_names),
        "focus_maps": matched_maps,
        "summary": focus_summary(split_maps, failed_names, matched_names),
    }


def matched_failed_maps(all_maps, failed_names):
    """Return the full fixed-map configs whose names appear in the failed CSV rows."""
    map_lookup = {map_config["map_name"]: map_config for map_config in all_maps}
    return [map_lookup[map_name] for map_name in failed_names if map_name in map_lookup]


def focus_summary(split_maps, failed_names, matched_names):
    """Describe how many failed maps belong to train, validation, and test splits."""
    split_sets = split_name_sets(split_maps)
    return {
        "failed_rows": int(len(failed_names)),
        "matched_fixed_maps": int(len(matched_names)),
        "train_split_matches": overlap_count(matched_names, split_sets["train"]),
        "val_split_matches": overlap_count(matched_names, split_sets["val"]),
        "test_split_matches": overlap_count(matched_names, split_sets["test"]),
    }


def split_name_sets(split_maps):
    """Build one set of map names for each fixed split."""
    return {
        "train": {map_config["map_name"] for map_config in split_maps["train_maps"]},
        "val": {map_config["map_name"] for map_config in split_maps["val_maps"]},
        "test": {map_config["map_name"] for map_config in split_maps["test_maps"]},
    }


def overlap_count(left_names, right_names):
    """Count how many map names appear in both requested collections."""
    return int(len(set(left_names) & set(right_names)))


def empty_focus_payload():
    """Return the empty focus-map payload used when no failed CSV is available."""
    return {
        "csv_path": None,
        "failed_names": [],
        "focus_maps": [],
        "summary": {
            "failed_rows": 0,
            "matched_fixed_maps": 0,
            "train_split_matches": 0,
            "val_split_matches": 0,
            "test_split_matches": 0,
        },
    }
