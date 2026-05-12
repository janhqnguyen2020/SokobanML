import csv
import json
import logging
import os
import time
from stable_baselines3 import DQN, PPO
from src.env.sokoban_env import initialize_env
from src.rl.masked_dqn import MaskedDQN
from src.utils.config import MAX_STEPS
from src.utils.metrics import compute_metrics

LOGGER = logging.getLogger(__name__)
EPISODE_LOG_INTERVAL = 25
RESULT_FIELDS = [
    "level",
    "episode",
    "solved",
    "steps",
    "runtime_ms",
    "total_reward",
    "invalid_macro_actions",
    "truncated",
    "boxes_on_target",
    "best_boxes_on_target",
    "termination_reason",
]

def load_model(model_path, algo):
    algo = algo.lower()
    if algo == "dqn":
        try:
            return MaskedDQN.load(model_path)
        except Exception as exc:
            LOGGER.warning("Falling back to plain DQN loader for %s: %s", model_path, exc)
            return DQN.load(model_path)
    if algo == "ppo":
        return PPO.load(model_path)
    raise ValueError(f"Unsupported algo: {algo}")

def _episode_result(final_info, steps, total_reward, invalid_macro_actions, best_boxes_on_target, runtime_ms):
    return {
        "solved": bool(final_info.get("all_boxes_on_target", False)),
        "steps": steps,
        "runtime_ms": runtime_ms,
        "total_reward": total_reward,
        "invalid_macro_actions": invalid_macro_actions,
        "truncated": bool(final_info.get("truncated", False)),
        "boxes_on_target": int(final_info.get("boxes_on_target", 0)),
        "best_boxes_on_target": best_boxes_on_target,
        "termination_reason": str(final_info.get("termination_reason", "unknown")),
    }

def evaluate_episode(model, env, max_steps=MAX_STEPS, log_progress=True, reset_seed=None):
    observation = env.reset() if reset_seed is None else env.reset(seed=int(reset_seed))
    done = False
    total_reward = 0.0
    steps = 0
    invalid_macro_actions = 0
    best_boxes_on_target = 0
    final_info = {}
    start_time = time.time()
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(int(action))
        info = info if isinstance(info, dict) else {}
        total_reward += float(reward)
        steps += 1
        final_info = info
        if info.get("invalid_macro_action"):
            invalid_macro_actions += 1
        best_boxes_on_target = max(best_boxes_on_target, int(info.get("best_boxes_on_target", info.get("boxes_on_target", 0))))
        if log_progress and (steps == 1 or steps % EPISODE_LOG_INTERVAL == 0):
            LOGGER.info(
                "Episode progress: steps=%s reward=%.2f last_action=%s done=%s invalid_macro_actions=%s",
                steps,
                total_reward,
                int(action),
                done,
                invalid_macro_actions,
            )
        if steps >= max_steps and not done:
            done = True
            final_info["truncated"] = True
    runtime_ms = (time.time() - start_time) * 1000.0
    return _episode_result(final_info, steps, total_reward, invalid_macro_actions, best_boxes_on_target, runtime_ms)

def _result_row(level_label, episode_idx, result):
    return {
        "level": level_label,
        "episode": episode_idx,
        "solved": result["solved"],
        "steps": result["steps"],
        "runtime_ms": result["runtime_ms"],
        "total_reward": result["total_reward"],
        "invalid_macro_actions": result["invalid_macro_actions"],
        "truncated": result["truncated"],
        "boxes_on_target": result["boxes_on_target"],
        "best_boxes_on_target": result["best_boxes_on_target"],
        "termination_reason": result["termination_reason"],
    }

def _write_results_csv(rows, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def summarize_results(results):
    summary = compute_metrics([
        (result["total_reward"], result["steps"], result["runtime_ms"], result["solved"])
        for result in results
    ])
    total = max(len(results), 1)
    summary["success_rate"] = float(sum(1 for result in results if result["solved"]) / total)
    summary["solved_count"] = int(sum(1 for result in results if result["solved"]))
    summary["total_episodes"] = len(results)
    summary["avg_boxes_on_target"] = float(sum(result["boxes_on_target"] for result in results) / total)
    summary["avg_best_boxes_on_target"] = float(sum(result["best_boxes_on_target"] for result in results) / total)
    summary["one_step_dead_end_count"] = int(
        sum(
            1
            for result in results
            if result["termination_reason"] == "dead_end" and int(result["steps"]) <= 1
        )
    )
    termination_counts = {}
    for result in results:
        reason = str(result.get("termination_reason", "unknown"))
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    summary["termination_counts"] = termination_counts
    return summary

def _evaluate_level(model, env_factory, level_id, n_episodes, episode_seeds):
    results = []
    for episode_idx in range(n_episodes):
        reset_seed = None if episode_seeds is None else episode_seeds[episode_idx]
        LOGGER.info("Creating evaluation environment for level %s episode %s", level_id, episode_idx)
        env = env_factory()
        try:
            LOGGER.info("Level %s - starting episode %s", level_id, episode_idx)
            result = evaluate_episode(model, env, log_progress=True, reset_seed=reset_seed)
        finally:
            env.close()
        results.append(result)
        LOGGER.info("Level %s - episode %s result: %s", level_id, episode_idx, result)
    return results

def evaluate_model(model_path, algo, level_ids, n_episodes, output_path, env_factory=None, episode_seeds=None):
    LOGGER.info("Loading model from %s using algo=%s", model_path, algo)
    model = load_model(model_path, algo)
    env_factory = env_factory or initialize_env
    level_ids = level_ids or ["default"]
    all_results = []
    all_rows = []
    for level_id in level_ids:
        level_results = _evaluate_level(model, env_factory, level_id, n_episodes, episode_seeds)
        all_results.extend(level_results)
        all_rows.extend(_result_row(level_id, episode_idx, result) for episode_idx, result in enumerate(level_results))
    _write_results_csv(all_rows, output_path)
    _write_json(output_path.replace(".csv", ".json"), all_rows)
    summary = summarize_results(all_results)
    _write_json(output_path.replace(".csv", "_summary.json"), summary)
    LOGGER.info("Evaluation summary: %s", summary)
    return summary
