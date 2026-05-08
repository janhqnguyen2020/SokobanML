"""
Primitive DQN Agent for solving Sokoban
This file trains a Deep Q-Network using the original Sokoban actions.
The agent directly learns from primitive actions such as up, down, left, and right.
This is the first RL baseline before adding box-push reasoning.
"""

import os
import json
from datetime import datetime
from stable_baselines3 import DQN
from src.utils.config import DQN_TOTAL_STEPS, DQN_BUFFER_SIZE


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _build_run_paths(run_id):
    """
    Build all important paths for one training run.
    """
    results_dir = os.path.join(PROJECT_ROOT, "results")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    run_dir = os.path.join(results_dir, "rl_tests", "primitive_dqn", run_id)

    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    model_path = os.path.join(run_dir, "dqn_primitive_final")
    config_path = os.path.join(run_dir, "config.json")
    status_path = os.path.join(run_dir, "train_status.json")

    return {
        "results_dir": results_dir,
        "models_dir": models_dir,
        "run_dir": run_dir,
        "tensorboard_dir": tensorboard_dir,
        "model_path": model_path,
        "config_path": config_path,
        "status_path": status_path,
    }


def _ensure_directories(paths):
    """
    Create all directories needed for the run.
    """
    os.makedirs(paths["results_dir"], exist_ok=True)
    os.makedirs(paths["models_dir"], exist_ok=True)
    os.makedirs(paths["run_dir"], exist_ok=True)
    os.makedirs(paths["tensorboard_dir"], exist_ok=True)


def _build_config(env, run_id, paths):
    """
    Build the config dictionary saved for this experiment.
    """
    return {
        "algo": "primitive_dqn",
        "run_id": run_id,
        "env_id": getattr(getattr(env.unwrapped, "spec", None), "id", "unknown"),
        "policy": "CnnPolicy",
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


def _save_config(config, config_path):
    """
    Save the run config as JSON.
    """
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def _save_train_status(status_path, model_path):
    """
    Save a tiny JSON marker showing that training finished.
    """
    with open(status_path, "w") as f:
        json.dump(
            {
                "status": "completed",
                "model_saved_to": model_path + ".zip",
            },
            f,
            indent=2,
        )


def train_primitive_dqn(env):
    """
    Train a primitive-action DQN on Sokoban.
    Returns:
        model: trained SB3 DQN model
        run_dir: folder where this run's artifacts were saved
    """
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    paths = _build_run_paths(run_id)
    _ensure_directories(paths)
    config = _build_config(env, run_id, paths)
    _save_config(config, paths["config_path"])

    model = DQN(
        "CnnPolicy",
        env,

        # replay buffer stores old experience for training
        buffer_size=DQN_BUFFER_SIZE,

        # learning settings
        learning_rate=1e-4,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,

        # how often model updates
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,

        # exploration settings
        exploration_fraction=0.25,
        exploration_final_eps=0.05,

        # logging and device
        tensorboard_log=paths["tensorboard_dir"],
        verbose=1,
        device="auto",
    )

    model.learn(total_timesteps=DQN_TOTAL_STEPS)
    model.save(paths["model_path"])
    _save_train_status(paths["status_path"], paths["model_path"])

    return model, paths["run_dir"]

