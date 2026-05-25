# src/rl/train_dqn.py
import logging
import os
from datetime import datetime
import torch as th
from stable_baselines3.common.utils import set_random_seed
from src.rl.callbacks import PeriodicEvalCallback
from src.rl.high_level_env import HighLevelSokobanEnv
from src.rl.masked_dqn import MaskedDQN
from src.rl.network import HighLevelBoardExtractor
from src.utils.config import (
    HIGH_LEVEL_DQN_BACKBONE,
    HIGH_LEVEL_DQN_BATCH_SIZE,
    HIGH_LEVEL_DQN_BUFFER_SIZE,
    HIGH_LEVEL_DQN_CNN_FEATURES_DIM,
    HIGH_LEVEL_DQN_EARLY_STOP_MIN_TIMESTEPS,
    HIGH_LEVEL_DQN_EARLY_STOP_PATIENCE_EVALS,
    HIGH_LEVEL_DQN_EVAL_EPISODES,
    HIGH_LEVEL_DQN_EVAL_FREQ,
    HIGH_LEVEL_DQN_EXPLORATION_FINAL_EPS,
    HIGH_LEVEL_DQN_EXPLORATION_FRACTION,
    HIGH_LEVEL_DQN_EXPLORATION_INITIAL_EPS,
    HIGH_LEVEL_DQN_GAMMA,
    HIGH_LEVEL_DQN_GRADIENT_STEPS,
    HIGH_LEVEL_DQN_INIT_MODEL_PATH,
    HIGH_LEVEL_DQN_LEARNING_RATE,
    HIGH_LEVEL_DQN_LEARNING_STARTS,
    HIGH_LEVEL_DQN_POLICY_HIDDEN_SIZES,
    HIGH_LEVEL_DQN_TARGET_UPDATE_INTERVAL,
    HIGH_LEVEL_DQN_TOTAL_STEPS,
    HIGH_LEVEL_DQN_TRAIN_FREQ,
    HIGH_LEVEL_OBS_CANVAS_SHAPE,
    HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES,
    SEED,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOGGER = logging.getLogger(__name__)

def _build_run_paths(run_id):
    run_dir = os.path.join(PROJECT_ROOT, "results", "rl_tests", "high_level_dqn", run_id)
    return {
        "run_dir": run_dir,
        "tensorboard_dir": os.path.join(run_dir, "tensorboard"),
        "model_path": os.path.join(run_dir, "high_level_dqn_final"),
        "best_model_path": os.path.join(run_dir, "high_level_dqn_best"),
    }

def _seed_env(env, seed):
    if seed is None:
        return env
    env.seed(seed)
    env.action_space.seed(seed)
    return env

def _create_high_level_env(use_shaped_reward, seed=None):
    env = HighLevelSokobanEnv(
        observation_board_shape=HIGH_LEVEL_OBS_CANVAS_SHAPE,
        use_extra_scalar_features=HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES,
        use_shaped_reward=use_shaped_reward,
    )
    return _seed_env(env, seed)

def make_train_env():
    return _create_high_level_env(use_shaped_reward=True, seed=SEED)

def make_eval_env():
    return _create_high_level_env(use_shaped_reward=False, seed=SEED + 1)

def _build_policy_kwargs(env):
    policy_kwargs = {"net_arch": HIGH_LEVEL_DQN_POLICY_HIDDEN_SIZES}
    if HIGH_LEVEL_DQN_BACKBONE == "cnn":
        policy_kwargs["features_extractor_class"] = HighLevelBoardExtractor
        policy_kwargs["features_extractor_kwargs"] = {
            "board_shape": env.observation_board_shape,
            "action_mask_size": env.action_space.n,
            "scalar_feature_size": env.scalar_feature_size,
            "features_dim": HIGH_LEVEL_DQN_CNN_FEATURES_DIM,
        }
    elif HIGH_LEVEL_DQN_BACKBONE != "mlp":
        raise ValueError(f"Unsupported HIGH_LEVEL_DQN_BACKBONE: {HIGH_LEVEL_DQN_BACKBONE}")
    return policy_kwargs

def _build_model_kwargs(env):
    return {
        "buffer_size": HIGH_LEVEL_DQN_BUFFER_SIZE,
        "learning_rate": HIGH_LEVEL_DQN_LEARNING_RATE,
        "learning_starts": HIGH_LEVEL_DQN_LEARNING_STARTS,
        "batch_size": HIGH_LEVEL_DQN_BATCH_SIZE,
        "gamma": HIGH_LEVEL_DQN_GAMMA,
        "train_freq": HIGH_LEVEL_DQN_TRAIN_FREQ,
        "gradient_steps": HIGH_LEVEL_DQN_GRADIENT_STEPS,
        "target_update_interval": HIGH_LEVEL_DQN_TARGET_UPDATE_INTERVAL,
        "exploration_initial_eps": HIGH_LEVEL_DQN_EXPLORATION_INITIAL_EPS,
        "exploration_fraction": HIGH_LEVEL_DQN_EXPLORATION_FRACTION,
        "exploration_final_eps": HIGH_LEVEL_DQN_EXPLORATION_FINAL_EPS,
        "policy_kwargs": _build_policy_kwargs(env),
        "verbose": 1,
        "device": "auto",
        "seed": SEED,
    }

def _remap_first_layer_weights(source_weight, target_weight, source_mask_size, target_mask_size):
    if source_weight.shape[0] != target_weight.shape[0]:
        raise ValueError(
            f"Incompatible hidden width in warm start: {tuple(source_weight.shape)} vs {tuple(target_weight.shape)}"
        )
    if source_mask_size != target_mask_size:
        raise ValueError(
            f"Incompatible action mask sizes in warm start: {source_mask_size} vs {target_mask_size}"
        )
    remapped = th.zeros_like(target_weight)
    shared_prefix = min(source_weight.shape[1] - source_mask_size, target_weight.shape[1] - target_mask_size)
    remapped[:, :shared_prefix] = source_weight[:, :shared_prefix]
    remapped[:, -target_mask_size:] = source_weight[:, -source_mask_size:]
    return remapped

def _warm_start_policy_weights(target_model, checkpoint_path):
    source_model = MaskedDQN.load(checkpoint_path, device="cpu")
    source_state = source_model.policy.state_dict()
    target_state = target_model.policy.state_dict()
    source_mask_size = int(source_model.action_space.n)
    target_mask_size = int(target_model.action_space.n)
    remapped_state = {}
    for key, target_tensor in target_state.items():
        if key not in source_state:
            raise KeyError(f"Checkpoint is missing policy parameter '{key}'")
        source_tensor = source_state[key]
        if tuple(source_tensor.shape) == tuple(target_tensor.shape):
            remapped_state[key] = source_tensor
            continue
        if key in {"q_net.q_net.0.weight", "q_net_target.q_net.0.weight"}:
            remapped_state[key] = _remap_first_layer_weights(
                source_tensor,
                target_tensor,
                source_mask_size=source_mask_size,
                target_mask_size=target_mask_size,
            )
            continue
        raise ValueError(
            f"Checkpoint parameter '{key}' has incompatible shape {tuple(source_tensor.shape)} for target {tuple(target_tensor.shape)}"
        )
    target_model.policy.load_state_dict(remapped_state, strict=True)
    LOGGER.info(
        "Warm-started policy weights from %s (source_obs=%s target_obs=%s)",
        checkpoint_path,
        tuple(source_model.observation_space.shape),
        tuple(target_model.observation_space.shape),
    )

def _prepare_run():
    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}_seed{SEED}"
    paths = _build_run_paths(run_id)
    os.makedirs(paths["run_dir"], exist_ok=True)
    os.makedirs(paths["tensorboard_dir"], exist_ok=True)
    env = make_train_env()
    return paths, env

def _build_model(env, paths):
    model = MaskedDQN(
        "MlpPolicy",
        env,
        action_mask_size=env.action_space.n,
        tensorboard_log=paths["tensorboard_dir"],
        **_build_model_kwargs(env),
    )
    if HIGH_LEVEL_DQN_INIT_MODEL_PATH:
        LOGGER.info("Initializing model weights from %s", HIGH_LEVEL_DQN_INIT_MODEL_PATH)
        _warm_start_policy_weights(model, HIGH_LEVEL_DQN_INIT_MODEL_PATH)
    return model

def _build_callback(paths):
    return PeriodicEvalCallback(
        best_model_path=paths["best_model_path"],
        eval_env_factory=make_eval_env,
        eval_seed_base=SEED + 10_000,
        eval_freq=HIGH_LEVEL_DQN_EVAL_FREQ,
        n_eval_episodes=HIGH_LEVEL_DQN_EVAL_EPISODES,
        early_stop_patience_evals=HIGH_LEVEL_DQN_EARLY_STOP_PATIENCE_EVALS,
        early_stop_min_timesteps=HIGH_LEVEL_DQN_EARLY_STOP_MIN_TIMESTEPS,
    )

def train():
    set_random_seed(SEED)
    paths, env = _prepare_run()
    LOGGER.info("Starting high-level DQN training")
    LOGGER.info("Run directory: %s", paths["run_dir"])
    model = _build_model(env, paths)
    callback = _build_callback(paths)
    LOGGER.info("Stable-Baselines3 selected device: %s", model.device)
    LOGGER.info("Beginning training for %s timesteps", HIGH_LEVEL_DQN_TOTAL_STEPS)
    model.learn(
        total_timesteps=HIGH_LEVEL_DQN_TOTAL_STEPS,
        callback=callback,
        reset_num_timesteps=True,
    )
    model.save(paths["model_path"])
    LOGGER.info("Training finished")
    LOGGER.info("Saved final model to %s.zip", paths["model_path"])
    env.close()
    return model, paths["run_dir"]
