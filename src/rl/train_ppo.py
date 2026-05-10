"""
PPO training using MlpPolicy + HintWrapper.
The HintWrapper converts raw Sokoban pixels into a flat hint vector
(box->goal assignments + board info), allowing MlpPolicy to learn
from structured reasoning signals instead of raw pixels.
PPO is more stable than DQN on sparse rewards due to its clipped objective
and longer rollout collection before each update.
"""

import os
import json
from datetime import datetime
from stable_baselines3 import PPO
from src.env.sokoban_env import initialize_env
from src.rl.hint_wrapper import HintWrapper
from src.utils.config import PPO_TOTAL_STEPS, SEED


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _build_run_paths(run_id):
    run_dir = os.path.join(PROJECT_ROOT, "results", "rl_tests", "ppo", run_id)
    return {
        "run_dir":     run_dir,
        "tensorboard": os.path.join(run_dir, "tensorboard"),
        "model_path":  os.path.join(run_dir, "ppo_final"),
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
        "algo": "ppo",
        "policy": "MlpPolicy",
        "observation": "HintWrapper flat vector (pixels + box-goal assignments)",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "total_timesteps": PPO_TOTAL_STEPS,
        "seed": SEED,
    }
    with open(paths["config_path"], "w") as f:
        json.dump(config, f, indent=2)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,      # entropy bonus keeps exploration alive on sparse rewards
        tensorboard_log=paths["tensorboard"],
        verbose=1,
        seed=SEED,
        device="auto",
    )

    model.learn(total_timesteps=PPO_TOTAL_STEPS)
    model.save(paths["model_path"])

    with open(paths["status_path"], "w") as f:
        json.dump({"status": "completed", "model": paths["model_path"] + ".zip"}, f, indent=2)

    env.close()
    return model, paths["run_dir"]
