# src/rl/evaluate.py
import csv
import json
import logging
import os
import time
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import _convert_space
from stable_baselines3.common.save_util import load_from_zip_file
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
    "num_steps",
    "num_pushes",
    "runtime_ms",
    "total_reward",
    "invalid_macro_actions",
    "truncated",
    "boxes_on_target",
    "best_boxes_on_target",
    "termination_reason",
]

def load_model(model_path, algo):
    """Load one saved model and fall back to weight-only restore for old DQN checkpoints."""
    algo = algo.lower()
    if algo == "dqn":
        try:
            return MaskedDQN.load(model_path)
        except Exception as exc:
            LOGGER.warning("Falling back to compatibility DQN loader for %s: %s", model_path, exc)
            return load_dqn_without_optimizer(model_path)
    if algo == "ppo":
        return PPO.load(model_path)
    raise ValueError(f"Unsupported algo: {algo}")


def saved_model_max_boxes(model_path):
    """Read the saved action-space size and convert it into one max-box count."""
    action_size = saved_action_space_size(model_path)
    if action_size is None:
        return None
    return int(action_size) // 4


def saved_action_space_size(model_path):
    """Return the saved discrete action-space size for one checkpoint when available."""
    data, _, _ = load_from_zip_file(model_path, device="cpu")
    if data is None or "action_space" not in data:
        return None
    action_space = _convert_space(data["action_space"])
    return int(action_space.n)


def load_dqn_without_optimizer(model_path):
    """Load one DQN checkpoint for evaluation while skipping mismatched optimizer state."""
    data, params, _ = load_from_zip_file(model_path, device="auto")
    normalized_data = normalized_dqn_data(data)
    model = build_dqn_from_saved_data(normalized_data)
    model.set_parameters(params_without_optimizer(params), exact_match=False, device="auto")
    return model


def normalized_dqn_data(data):
    """Clean saved DQN metadata so older checkpoints rebuild cleanly on this codebase."""
    normalized = dict(data)
    normalized["observation_space"] = _convert_space(normalized["observation_space"])
    normalized["action_space"] = _convert_space(normalized["action_space"])
    normalized["policy_kwargs"] = normalized_policy_kwargs(normalized.get("policy_kwargs", {}))
    return normalized


def normalized_policy_kwargs(policy_kwargs):
    """Remove stale device fields and unwrap older net-arch save formats."""
    normalized = dict(policy_kwargs)
    normalized.pop("device", None)
    if policy_uses_legacy_net_arch(normalized):
        normalized["net_arch"] = normalized["net_arch"][0]
    return normalized


def policy_uses_legacy_net_arch(policy_kwargs):
    """Return True when one saved policy uses the older wrapped net-arch format."""
    net_arch = policy_kwargs.get("net_arch", [])
    return bool(net_arch) and isinstance(net_arch[0], dict)


def build_dqn_from_saved_data(data):
    """Rebuild one DQN model shell from saved metadata before loading weights."""
    model = MaskedDQN(
        policy=data["policy_class"],
        env=data.get("env"),
        device="auto",
        _init_setup_model=False,
    )
    model.__dict__.update(data)
    model._setup_model()
    return model


def params_without_optimizer(params):
    """Drop optimizer state so older checkpoints can still be used for prediction."""
    filtered = {}
    for name, state_dict in params.items():
        if str(name).endswith("optimizer"):
            continue
        filtered[name] = state_dict
    return filtered

def _episode_result(final_info, num_steps, num_pushes, total_reward, invalid_macro_actions, best_boxes_on_target, runtime_ms):
    """Build one evaluation row using explicit step and push counters."""
    return {
        "solved": bool(final_info.get("all_boxes_on_target", False)),
        "num_steps": num_steps,
        "num_pushes": num_pushes,
        "runtime_ms": runtime_ms,
        "total_reward": total_reward,
        "invalid_macro_actions": invalid_macro_actions,
        "truncated": bool(final_info.get("truncated", False)),
        "boxes_on_target": int(final_info.get("boxes_on_target", 0)),
        "best_boxes_on_target": best_boxes_on_target,
        "termination_reason": str(final_info.get("termination_reason", "unknown")),
    }

def evaluate_episode(model, env, max_steps=MAX_STEPS, log_progress=True, reset_seed=None):
    """Run one DQN episode and count both macro steps and executed pushes."""
    observation = env.reset() if reset_seed is None else env.reset(seed=int(reset_seed))
    done = False
    total_reward = 0.0
    num_steps = 0
    num_pushes = 0
    invalid_macro_actions = 0
    best_boxes_on_target = 0
    final_info = {}
    start_time = time.time()
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(int(action))
        info = info if isinstance(info, dict) else {}
        total_reward += float(reward)
        num_steps += 1
        final_info = info
        if info.get("invalid_macro_action"):
            invalid_macro_actions += 1
        if info.get("executed_push"):
            num_pushes += 1
        best_boxes_on_target = max(best_boxes_on_target, int(info.get("best_boxes_on_target", info.get("boxes_on_target", 0))))
        if log_progress and (num_steps == 1 or num_steps % EPISODE_LOG_INTERVAL == 0):
            LOGGER.info(
                "Episode progress: num_steps=%s num_pushes=%s reward=%.2f last_action=%s done=%s invalid_macro_actions=%s",
                num_steps,
                num_pushes,
                total_reward,
                int(action),
                done,
                invalid_macro_actions,
            )
        if num_steps >= max_steps and not done:
            done = True
            final_info["truncated"] = True
    runtime_ms = (time.time() - start_time) * 1000.0
    return _episode_result(final_info, num_steps, num_pushes, total_reward, invalid_macro_actions, best_boxes_on_target, runtime_ms)

def _result_row(level_label, episode_idx, result):
    """Attach level metadata to one DQN evaluation result row."""
    return {
        "level": level_label,
        "episode": episode_idx,
        "solved": result["solved"],
        "num_steps": result["num_steps"],
        "num_pushes": result["num_pushes"],
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
    """Compute summary metrics for a list of DQN evaluation episodes."""
    if not results:
        return empty_result_summary()
    summary = compute_metrics([
        (result["total_reward"], result["num_steps"], result["runtime_ms"], result["solved"])
        for result in results
    ])
    total = max(len(results), 1)
    summary["success_rate"] = float(sum(1 for result in results if result["solved"]) / total)
    summary["solved_count"] = int(sum(1 for result in results if result["solved"]))
    summary["total_episodes"] = len(results)
    summary["avg_num_pushes"] = float(sum(result["num_pushes"] for result in results) / total)
    summary["avg_boxes_on_target"] = float(sum(result["boxes_on_target"] for result in results) / total)
    summary["avg_best_boxes_on_target"] = float(sum(result["best_boxes_on_target"] for result in results) / total)
    summary["one_step_dead_end_count"] = int(
        sum(
            1
            for result in results
            if result["termination_reason"] == "dead_end" and int(result["num_steps"]) <= 1
        )
    )
    termination_counts = {}
    for result in results:
        reason = str(result.get("termination_reason", "unknown"))
        termination_counts[reason] = termination_counts.get(reason, 0) + 1
    summary["termination_counts"] = termination_counts
    return summary


def empty_result_summary():
    """Return one zero-filled summary when an evaluation group has no episodes."""
    summary = empty_metric_summary()
    summary["avg_num_pushes"] = 0.0
    summary["avg_boxes_on_target"] = 0.0
    summary["avg_best_boxes_on_target"] = 0.0
    summary["one_step_dead_end_count"] = 0
    summary["termination_counts"] = {}
    return summary


def empty_metric_summary():
    """Return the zero-filled metric fields shared by every evaluation summary."""
    return {
        "success_rate": 0.0,
        "solved_count": 0,
        "total_episodes": 0,
        "avg_reward": 0.0,
        "best_reward": 0.0,
        "avg_num_steps": 0.0,
        "min_num_steps": 0,
        "max_num_steps": 0,
        "avg_time_ms": 0.0,
        "min_time_ms": 0.0,
        "max_time_ms": 0.0,
    }

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
