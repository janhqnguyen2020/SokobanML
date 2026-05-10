"""
DQN training using MlpPolicy + HintWrapper.
The HintWrapper converts raw Sokoban pixels into a flat hint vector
(box->goal assignments + board info), allowing MlpPolicy to learn
from structured reasoning signals instead of raw pixels.
"""

import os
import json
from datetime import datetime

from stable_baselines3 import DQN#actual DQN 
from src.env.sokoban_env import initialize_env
from src.rl.hint_wrapper import HintWrapper# high abstract state representation that gives the agent hints about box-goal assignments
from src.utils.config import DQN_TOTAL_STEPS, DQN_BUFFER_SIZE, SEED


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FIXED_LEVEL = False   # True = train on one fixed puzzle; False = random levels every episode


def _build_run_paths(run_id, fixed_level):
    label = "dqn_fixed" if fixed_level else "dqn"
    run_dir = os.path.join(PROJECT_ROOT, "results", "rl_tests", label, run_id)
    return {
        "run_dir":     run_dir,
        "tensorboard": os.path.join(run_dir, "tensorboard"),
        "model_path":  os.path.join(run_dir, "dqn_final"),
        "config_path": os.path.join(run_dir, "config.json"),
        "status_path": os.path.join(run_dir, "train_status.json"),
    }


def train():
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    paths = _build_run_paths(run_id, FIXED_LEVEL)
    os.makedirs(paths["run_dir"], exist_ok=True)
    os.makedirs(paths["tensorboard"], exist_ok=True)

    env = initialize_env()
    env = HintWrapper(env, fixed_level=FIXED_LEVEL, level_seed=SEED)

    total_steps = 100_000 if FIXED_LEVEL else DQN_TOTAL_STEPS

    config = {
        "algo": "dqn",
        "policy": "MlpPolicy",
        "fixed_level": FIXED_LEVEL,
        "observation": "HintWrapper flat vector (room_state + room_fixed + hints + semantics)",
        "buffer_size": DQN_BUFFER_SIZE,
        "learning_rate": 1e-4,
        "learning_starts": 500,
        "batch_size": 64,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 500,
        "exploration_fraction": 0.5,
        "exploration_final_eps": 0.1,
        "total_timesteps": total_steps,
        "seed": SEED,
    }
    with open(paths["config_path"], "w") as f:
        json.dump(config, f, indent=2)

    model = DQN(
        "MlpPolicy",
        env,
        buffer_size=DQN_BUFFER_SIZE,
        learning_rate=1e-4,
        learning_starts=500,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.5,   # explore longer on a fixed level to find the solution
        exploration_final_eps=0.1,  # keep 10% random late so it doesn't get stuck
        tensorboard_log=paths["tensorboard"],
        verbose=1,
        seed=SEED,
        device="auto",
    )

    model.learn(total_timesteps=total_steps)
    model.save(paths["model_path"])

    with open(paths["status_path"], "w") as f:
        json.dump({"status": "completed", "model": paths["model_path"] + ".zip"}, f, indent=2)

    env.close()
    return model, paths["run_dir"]
