"""
DQN training using MlpPolicy + HintWrapper.
The HintWrapper converts raw Sokoban pixels into a flat hint vector
(box->goal assignments + board info), allowing MlpPolicy to learn
from structured reasoning signals instead of raw pixels.
"""

import os
import json
from datetime import datetime
from stable_baselines3 import DQN
from src.env.sokoban_env import initialize_env
from src.rl.hint_wrapper import HintWrapper
from src.utils.config import DQN_TOTAL_STEPS, DQN_BUFFER_SIZE, SEED


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _build_run_paths(run_id):
    run_dir = os.path.join(PROJECT_ROOT, "results", "rl_tests", "dqn", run_id)
    return {
        "run_dir":     run_dir,
        "tensorboard": os.path.join(run_dir, "tensorboard"),
        "model_path":  os.path.join(run_dir, "dqn_final"),
        "config_path": os.path.join(run_dir, "config.json"),
        "status_path": os.path.join(run_dir, "train_status.json"),
    }


def train():
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    paths = _build_run_paths(run_id)
    os.makedirs(paths["run_dir"], exist_ok=True)
    os.makedirs(paths["tensorboard"], exist_ok=True)

    env = initialize_env()
    env = HintWrapper(env)

    config = {
        "algo": "dqn",
        "policy": "MlpPolicy",
        "observation": "HintWrapper flat vector (pixels + box-goal assignments)",
        "buffer_size": DQN_BUFFER_SIZE,
        "learning_rate": 1e-4,
        "learning_starts": 5000,
        "batch_size": 64,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 1000,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "total_timesteps": DQN_TOTAL_STEPS,
        "seed": SEED,
    }
    with open(paths["config_path"], "w") as f:
        json.dump(config, f, indent=2)

    model = DQN(
        "MlpPolicy",
        env,
        buffer_size=DQN_BUFFER_SIZE,
        learning_rate=1e-4,
        learning_starts=5000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        tensorboard_log=paths["tensorboard"],
        verbose=1,
        seed=SEED,
        device="auto",
    )

    model.learn(total_timesteps=DQN_TOTAL_STEPS)
    model.save(paths["model_path"])

    with open(paths["status_path"], "w") as f:
        json.dump({"status": "completed", "model": paths["model_path"] + ".zip"}, f, indent=2)

    env.close()
    return model, paths["run_dir"]
