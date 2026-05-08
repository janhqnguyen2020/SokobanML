# src/rl/evaluate.py
# Member B — Shizuka Takao / Member C — Joseph Nguyen
#
# Evaluation script for trained RL agents.
#
# Loads a saved model checkpoint and runs it on a set of test levels
# to measure performance metrics comparable to the classical planners.
# Results are saved to results/ so Member C can run cross-method analysis.
#
# Metrics collected (must match planner_runner.py schema):
#   - solved: bool — did agent reach all boxes on goals?
#   - steps: int — number of actions taken
#   - runtime_ms: float — wall time for the episode
#   - total_reward: float — cumulative reward received
#
# Functions to implement:
#   - load_model(model_path, algo)
#       → load PPO.load() or DQN.load() from Stable-Baselines3
#       → algo: "ppo" or "dqn"
#
#   - evaluate_episode(model, env)
#       → run one episode deterministically (deterministic=True)
#       → returns metrics dict for that episode
#
#   - evaluate_model(model_path, algo, level_ids, n_episodes, output_path)
#       → outer loop: for each level, run n_episodes
#       → aggregate: success rate, mean steps, mean reward
#       → save to results/rl_{algo}_results.csv
#
#   - main()
#       → CLI: --model, --algo, --levels, --episodes, --output
#
# Usage (intended):
#   python evaluate.py --model models/ppo_final.zip --algo ppo --output results/ppo_results.csv
#
# Notes:
#   - Use evaluate_policy() from SB3 for quick sanity checks
#   - For detailed per-level metrics, use the custom evaluate_model() above
#   - Deterministic=True means argmax action, no exploration

import os
import csv
import json
import time
import argparse
from stable_baselines3 import DQN, PPO
from src.env.sokoban_env import initialize_env
from src.utils.metrics import compute_metrics


def load_model(model_path, algo):
    """
    Load a trained RL model from disk.
    Parameters:
        model_path: path to a saved .zip model file
        algo: either "dqn" or "ppo"
    Returns:
        Loaded SB3 model object
    """
    algo = algo.lower()
    if algo == "dqn":
        return DQN.load(model_path)
    if algo == "ppo":
        return PPO.load(model_path)
    raise ValueError(f"Unsupported algo: {algo}")


def evaluate_episode(model, env):
    """
    Run exactly one episode using a trained model.
    The model acts deterministically, meaning it chooses the action it
    currently believes is best instead of exploring randomly.
    Returns:
        A dictionary containing solved / steps / runtime_ms / total_reward
    """
    observation = env.reset()
    done = False
    total_reward = 0
    steps = 0
    start_time = time.time()

    # Save the final info dict so we can check whether the episode ended
    # because the puzzle was solved.
    final_info = {}

    while not done:
        action, _ = model.predict(observation, deterministic=True) # deterministic=True means "pick the best-known action"
        observation, reward, done, info = env.step(int(action)) # Old Gym API returns 4 values 
        total_reward += reward
        steps += 1
        final_info = info
    runtime_ms = (time.time() - start_time) * 1000.0

    # By default, assume not solved. If the env tells us all boxes are on
    # targets, then mark the episode as solved.
    solved = False
    if isinstance(final_info, dict):
        solved = bool(final_info.get("all_boxes_on_target", False))

    return {
        "solved": solved,
        "steps": steps,
        "runtime_ms": runtime_ms,
        "total_reward": total_reward,
    }


def _build_results_rows(results, level_label):
    """
    Convert per-episode result dicts into flat rows for CSV / JSON output.
    level_label is stored so later analysis can group results by level set.
    """
    rows = []
    for episode_idx, result in enumerate(results):
        rows.append({
            "level": level_label,
            "episode": episode_idx,
            "solved": result["solved"],
            "steps": result["steps"],
            "runtime_ms": result["runtime_ms"],
            "total_reward": result["total_reward"],
        })
    return rows


def _write_results_csv(rows, output_path):
    """
    Save detailed per-episode results to a CSV file.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        # CSV header
        writer.writerow(["level", "episode", "solved", "steps", "runtime_ms", "total_reward"])
        # One row per episode
        for row in rows:
            writer.writerow([
                row["level"],
                row["episode"],
                row["solved"],
                row["steps"],
                row["runtime_ms"],
                row["total_reward"],
            ])


def _write_results_json(rows, output_path):
    """
    Save detailed per-episode results to a JSON file.
    The JSON path is derived from the CSV path.
    Example:
        results.csv -> results.json
    """
    json_path = output_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)


def _build_metric_input(results):
    """
    Convert episode result dicts into the tuple format expected by
    compute_metrics().
    compute_metrics() expects tuples like:
        (reward, steps, runtime_ms)
    """
    return [
        (
            result["total_reward"],
            result["steps"],
            result["runtime_ms"],
        )
        for result in results
    ]


def _write_summary_json(summary, output_path):
    """
    Save aggregate summary metrics to a JSON file
    Example:
        results.csv -> results_summary.json
    """
    summary_path = output_path.replace(".csv", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def evaluate_model(model_path, algo, level_ids, n_episodes, output_path, env_factory=None):
    """
    Full evaluation pipeline.

    Parameters:
        model_path: path to the trained model
        algo: "dqn" or "ppo"
        level_ids: list of labels for level groups
        n_episodes: how many episodes to run per level label
        output_path: where to save the CSV results
        env_factory: optional function/class that creates the correct env

    If env_factory is None, fall back to the normal Sokoban env.
    This keeps primitive DQN / PPO working.

    For higher-level DQN, pass:
        env_factory=HighLevelSokobanEnv
    """
    model = load_model(model_path, algo)
    all_episode_rows = []
    all_results = []

    if not level_ids:
        level_ids = ["default"]

    if env_factory is None:
        env_factory = initialize_env

    for level_id in level_ids:
        env = env_factory()
        level_results = []

        for episode_idx in range(n_episodes):
            print(f"Level {level_id} - Starting episode: {episode_idx}")
            episode_result = evaluate_episode(model, env)
            level_results.append(episode_result)
            all_results.append(episode_result)
            print("Episode result: ", episode_result)

        all_episode_rows.extend(_build_results_rows(level_results, level_id))
        env.close()

    _write_results_csv(all_episode_rows, output_path)
    _write_results_json(all_episode_rows, output_path)
    metric_input = _build_metric_input(all_results)
    summary = compute_metrics(metric_input)
    _write_summary_json(summary, output_path)
    return summary



def _parse_levels(levels_arg):
    """
    Convert a comma-separated string into a list of level labels.
    Example:
        "v0,v1,v2" -> ["v0", "v1", "v2"]
    """
    if levels_arg is None:
        return []
    levels = [level.strip() for level in levels_arg.split(",") if level.strip()]
    return levels


def main():
    """
    Command-line entry point.
    Example usage:
        python evaluate.py --model models/ppo_final.zip --algo ppo --output results/ppo_results.csv
    """
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents on Sokoban")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--algo", type=str, required=True, choices=["dqn", "ppo"])
    parser.add_argument("--levels", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    level_ids = _parse_levels(args.levels)
    summary = evaluate_model(
        model_path=args.model,
        algo=args.algo,
        level_ids=level_ids,
        n_episodes=args.episodes,
        output_path=args.output,
    )

    print("Evaluation summary:")
    print(summary)


if __name__ == "__main__":
    main()