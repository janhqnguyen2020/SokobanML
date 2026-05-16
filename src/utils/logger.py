import csv
import os
from datetime import datetime

_FIELDS = [
    "timestamp", "planner", "mode",
    "map_height", "map_width", "num_boxes",
    "solved", "total_reward",
    "primitive_steps", "box_pushes",
    "nodes_expanded", "deadlocks_pruned", "dead_squares",
    "solve_time_ms",
]

def log_run(record, filepath="results/planner_log.csv"):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    new_file = not os.path.exists(filepath)
    record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({k: record.get(k, "") for k in _FIELDS})
