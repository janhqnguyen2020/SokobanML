# src/utils/result_paths.py

"""Path helpers for benchmark result files."""

import os


def build_board_csv_path(group_name, board_name):
    """Return the detailed file for one map or one seeded episode."""
    folder = os.path.join("results", group_name, board_name)
    return os.path.join(folder, "planner_runs.csv")


def build_algorithm_runs_path(algorithm_name):
    """Return the detailed file for one algorithm across all runs."""
    folder = os.path.join("results", "summaries")
    return os.path.join(folder, f"{algorithm_name}_runs.csv")


def build_algorithm_summary_path(algorithm_name):
    """Return the aggregated summary file for one algorithm."""
    folder = os.path.join("results", "summaries")
    return os.path.join(folder, f"{algorithm_name}_summary.csv")


def ensure_parent_folder(file_path):
    """Create the parent folder so CSV files can be written safely."""
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)