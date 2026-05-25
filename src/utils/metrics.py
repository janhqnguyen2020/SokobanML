# src/utils/metrics.py
import numpy as np

def compute_metrics(results):
    # results: list of (reward, steps, time_ms, nodes_expanded, deadlocks_pruned, dead_squares_count)
    # fields 3+ are optional (planners only)
    rewards = [r[0] for r in results]
    steps   = [r[1] for r in results]
    solved  = [r > 0 for r in rewards]

    metrics = {
        "success_rate":   float(np.mean(solved)),
        "solved_count":   int(sum(solved)),
        "total_episodes": len(results),
        "avg_reward":     float(np.mean(rewards)),
        "best_reward":    float(np.max(rewards)),
        "avg_steps":      float(np.mean(steps)),
        "min_steps":      int(np.min(steps)),
        "max_steps":      int(np.max(steps)),
    }

    if len(results[0]) >= 3:
        times = [r[2] for r in results]
        metrics["avg_time_ms"] = float(np.mean(times))
        metrics["min_time_ms"] = float(np.min(times))
        metrics["max_time_ms"] = float(np.max(times))

    if len(results[0]) >= 6:
        metrics["avg_nodes_expanded"]   = float(np.mean([r[3] for r in results]))
        metrics["avg_deadlocks_pruned"] = float(np.mean([r[4] for r in results]))
        metrics["avg_dead_squares"]     = float(np.mean([r[5] for r in results]))

    return metrics
