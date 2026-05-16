"""
Batch experiment runner.
Runs BFS, Greedy, and A* across all benchmark maps, prints a summary,
and saves everything to results/benchmark_results.csv
"""

import csv
import os
import re
import subprocess
from pathlib import Path

# Benchmark maps and algorithms

MAPS = [
    {
        "name":       "easy_1box",
        "difficulty": "Easy",
        "height": 3,  "width": 12,
        "player": "1,1",
        "boxes":  "1,4",
        "goals":  "1,10",
    },
    {
        "name":       "medium_1box",
        "difficulty": "Medium",
        "height": 9,  "width": 9,
        "player": "1,1",
        "boxes":  "2,4",
        "goals":  "7,7",
    },
    {
        "name":       "large_1box",
        "difficulty": "Large-1box",
        "height": 15, "width": 15,
        "player": "1,1",
        "boxes":  "5,7",
        "goals":  "12,9",
    },
    {
        "name":       "medium_2box",
        "difficulty": "Medium-2box",
        "height": 11, "width": 11,
        "player": "1,1",
        "boxes":  "2,3;3,6",
        "goals":  "8,3;8,6",
    },
    {
        "name":       "hard_3box",
        "difficulty": "Hard-3box",
        "height": 15, "width": 15,
        "player": "1,1",
        "boxes":  "2,2;3,3;5,7",
        "goals":  "4,9;10,5;12,7",
    },
]

ALGORITHMS = [
    ("BFS",    "test_custom_bfs.py"),
    ("Greedy", "test_custom_greedy.py"),
    ("A*",     "test_custom_astar.py"),
]

# Output parser  (matches the print_metrics block)
def extract(pattern, text, cast=str, default=None):
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return cast(m.group(1))
    except (ValueError, TypeError):
        return default

def parse_output(text):
    return {
        "solved":           extract(r"Solved\s+(True|False)", text),
        "total_reward":     extract(r"Total reward\s+([\d.\-]+)", text, float),
        "box_pushes":       extract(r"Box pushes\s+(\d+)", text, int),
        "primitive_steps":  extract(r"Primitive steps\s+(\d+)", text, int),
        "nodes_expanded":   extract(r"Nodes expanded\s+(\d+)", text, int),
        "deadlocks_pruned": extract(r"Deadlocks pruned\s+(\d+)", text, int),
        "dead_squares":     extract(r"Dead squares\s+(\d+)", text, int),
        "solve_time_ms":    extract(r"Solve time\s+([\d.]+)\s*ms", text, float),
    }

# --------------------------------------------------
# Run experiments
# --------------------------------------------------

# Force headless matplotlib so no windows open during batch run
batch_env = {**os.environ, "MPLBACKEND": "Agg"}

results = []
total = len(MAPS) * len(ALGORITHMS)
idx = 0

for m in MAPS:
    for algo_name, script in ALGORITHMS:
        idx += 1
        print(f"[{idx:2d}/{total}] {m['difficulty']:12s} | {algo_name:6s} ...", end=" ", flush=True)

        cmd = [
            "python", script,
            "--height", str(m["height"]),
            "--width",  str(m["width"]),
            "--player", m["player"],
            "--boxes",  m["boxes"],
            "--goals",  m["goals"],
            "--no-vis",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, env=batch_env)
        metrics = parse_output(proc.stdout + proc.stderr)

        status = "SOLVED" if metrics["solved"] == "True" else "FAILED"
        nodes  = metrics["nodes_expanded"]
        t      = metrics["solve_time_ms"]
        print(f"{status:6s}  nodes={str(nodes):>7s}  time={str(t):>8s} ms")

        results.append({
            "map":            m["name"],
            "difficulty":     m["difficulty"],
            "algorithm":      algo_name,
            "height":         m["height"],
            "width":          m["width"],
            "num_boxes":      len(m["boxes"].split(";")),
            **metrics,
        })

# Summary table
print("\n" + "=" * 72)
print(f"  {'Map':<14} {'Algo':<8} {'Solved':<8} {'Nodes':>8} {'Pushes':>7} {'Time(ms)':>10}")
print("=" * 72)
for r in results:
    print(
        f"  {r['map']:<14} {r['algorithm']:<8} {str(r['solved']):<8} "
        f"{str(r['nodes_expanded']):>8} {str(r['box_pushes']):>7} "
        f"{str(r['solve_time_ms']):>10}"
    )
print("=" * 72)

# Export CSV

FIELDNAMES = [
    "map", "difficulty", "algorithm", "height", "width", "num_boxes",
    "solved", "total_reward", "box_pushes", "primitive_steps",
    "nodes_expanded", "deadlocks_pruned", "dead_squares", "solve_time_ms",
]

out = Path("results/benchmark_results.csv")
out.parent.mkdir(exist_ok=True)

with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved to: {out.resolve()}")
print(f"Rows: {len(results)}  ({len(MAPS)} maps x {len(ALGORITHMS)} algorithms)")
