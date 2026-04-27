# src/utils/metrics.py
# Member C — Joseph Nguyen
#
# Standardized metric definitions and aggregation functions.
#
# All planners and RL evaluators output data in the same schema so
# that cross-method comparison is clean and consistent.
#
# Standard result schema (one row per episode/attempt):
#   - method: str        "bfs" | "greedy" | "ppo" | "dqn"
#   - level_id: int      which puzzle was attempted
#   - solved: bool       True if all boxes reached goals
#   - steps: int         number of actions taken (or until timeout)
#   - runtime_ms: float  wall clock time in milliseconds
#   - nodes_expanded: int  (planners only) states explored
#   - total_reward: float  (RL only) cumulative episode reward
#
# Functions to implement:
#   - make_result_row(method, level_id, solved, steps, runtime_ms, **kwargs)
#       → returns dict matching schema above
#       → kwargs for method-specific fields (nodes_expanded, total_reward)
#
#   - aggregate_results(df)
#       → takes Pandas DataFrame of result rows
#       → returns summary dict:
#           success_rate, mean_steps, median_steps, mean_runtime_ms,
#           std_steps, pct_timeout
#
#   - compare_methods(results_df)
#       → groups by "method", calls aggregate_results per group
#       → returns summary DataFrame with one row per method
#       → used directly for report tables
#
#   - print_summary(summary_df)
#       → pretty-print comparison table to stdout
#
# Notes:
#   - Keep schema in sync with planner_runner.py and evaluate.py
#   - Timeout episodes: steps = max_steps, solved = False

import numpy as np

def compute_metrics(results):
    rewards = [r for r, s in results]
    steps = [s for r, s in results]

    return {
        "success_rate": np.mean([r > 0 for r in rewards]),
        "avg_reward": np.mean(rewards),
        "avg_steps": np.mean(steps)
    }