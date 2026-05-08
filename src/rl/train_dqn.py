# src/rl/train_dqn.py
# Member B — Shizuka Takao
#
# Training script for DQN (Deep Q-Network) agent on Sokoban.
#
# DQN learns a Q-function Q(s, a) that estimates expected cumulative reward
# for taking action a in state s. Uses a replay buffer and target network
# to stabilize training.
#
# This script uses Stable-Baselines3's DQN, or optionally a custom
# implementation via network.py if SB3's DQN is too limiting.
#
# DQN vs PPO tradeoffs:
#   DQN  → sample efficient (replay buffer), slower to converge on sparse rewards
#   PPO  → less sample efficient, more stable on long-horizon tasks
#   → run both, compare in results
#
# Responsibilities:
#   - Set up single Sokoban env (DQN doesn't use vectorized envs well)
#   - Configure DQN hyperparameters (lr, buffer_size, exploration schedule)
#   - Attach callbacks for checkpointing
#   - Run model.learn()
#   - Save model to models/dqn_final.zip
#
# Key hyperparameters:
#   - learning_rate: 1e-4
#   - buffer_size: 50_000 (replay buffer capacity)
#   - learning_starts: 10_000 (steps before first gradient update)
#   - exploration_fraction: 0.2 (fraction of training spent exploring)
#   - exploration_final_eps: 0.05 (final epsilon for epsilon-greedy)
#   - target_update_interval: 1000 steps
#
# CLI args to support:
#   --timesteps, --levels, --save_path, --log_dir
#
# Notes:
#   - DQN expects flat or CNN observations — match network.py architecture
#   - Logs go to results/ for Member C's evaluation scripts

"""
Higher-level DQN training.
The DQN agent acts over abstract push decisions instead of raw primitive actions.
Choose which box to move and which direction to push
"""

import os
import json
from datetime import datetime
from stable_baselines3 import DQN
from src.rl.high_level_env import HighLevelSokobanEnv
from src.utils.config import DQN_TOTAL_STEPS, DQN_BUFFER_SIZE


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _build_run_paths(run_id):
    results_dir = os.path.join(PROJECT_ROOT, "results")
    run_dir = os.path.join(results_dir, "rl_tests", "high_level_dqn", run_id)
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    model_path = os.path.join(run_dir, "high_level_dqn_final")
    config_path = os.path.join(run_dir, "config.json")
    status_path = os.path.join(run_dir, "train_status.json")
    return {
        "run_dir": run_dir,
        "tensorboard_dir": tensorboard_dir,
        "model_path": model_path,
        "config_path": config_path,
        "status_path": status_path,
    }


def train():
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    paths = _build_run_paths(run_id)
    os.makedirs(paths["run_dir"], exist_ok=True)
    os.makedirs(paths["tensorboard_dir"], exist_ok=True)
    env = HighLevelSokobanEnv()
    config = {
        "algo": "high_level_dqn",
        "run_id": run_id,
        "env_id": getattr(getattr(env.env.unwrapped, "spec", None), "id", "unknown"),
        "policy": "CnnPolicy",
        "macro_action_space": "box_index * 4 + direction_index",
        "num_boxes": env.num_boxes,
        "buffer_size": DQN_BUFFER_SIZE,
        "learning_rate": 1e-4,
        "learning_starts": 1000,
        "batch_size": 32,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.25,
        "exploration_final_eps": 0.05,
        "total_timesteps": DQN_TOTAL_STEPS,
        "tensorboard_log": paths["tensorboard_dir"],
        "model_output": paths["model_path"],
    }

    with open(paths["config_path"], "w") as f:
        json.dump(config, f, indent=2)

    model = DQN(
        "CnnPolicy",
        env,
        buffer_size=DQN_BUFFER_SIZE,
        learning_rate=1e-4,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.25,
        exploration_final_eps=0.05,
        tensorboard_log=paths["tensorboard_dir"],
        verbose=1,
        device="auto",
    )

    model.learn(total_timesteps=DQN_TOTAL_STEPS)
    model.save(paths["model_path"])
    with open(paths["status_path"], "w") as f:
        json.dump(
            {
                "status": "completed",
                "model_saved_to": paths["model_path"] + ".zip",
            },
            f,
            indent=2,
        )
    return model, paths["run_dir"]
