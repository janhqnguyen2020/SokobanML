# src/utils/metrics.py
import numpy as np

def compute_metrics(results):
    """Compute simple averages from result tuples used by the evaluation scripts."""
    rewards = [r[0] for r in results]
    num_steps = [r[1] for r in results]
    solved = [r > 0 for r in rewards]
    metrics = _base_metrics(rewards, num_steps, solved, results)
    if len(results[0]) >= 3:
        metrics.update(_time_metrics(results))
    if len(results[0]) >= 6:
        metrics.update(_planner_metrics(results))
    return metrics

def _base_metrics(rewards, num_steps, solved, results):
    """Return the shared reward and step averages for any evaluation run."""
    return {
        "success_rate":   float(np.mean(solved)),
        "solved_count":   int(sum(solved)),
        "total_episodes": len(results),
        "avg_reward":     float(np.mean(rewards)),
        "best_reward":    float(np.max(rewards)),
        "avg_num_steps":  float(np.mean(num_steps)),
        "min_num_steps":  int(np.min(num_steps)),
        "max_num_steps":  int(np.max(num_steps)),
    }

def _time_metrics(results):
    """Return runtime averages when result tuples include timing data."""
    times = [r[2] for r in results]
    return {
        "avg_time_ms": float(np.mean(times)),
        "min_time_ms": float(np.min(times)),
        "max_time_ms": float(np.max(times)),
    }

def _planner_metrics(results):
    """Return planner-only averages when search counters are present."""
    return {
        "avg_nodes_expanded": float(np.mean([r[3] for r in results])),
        "avg_deadlocks_pruned": float(np.mean([r[4] for r in results])),
        "avg_dead_squares": float(np.mean([r[5] for r in results])),
    }
