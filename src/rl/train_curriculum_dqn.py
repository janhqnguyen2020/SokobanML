"""Training pipeline for the curriculum / generalized DQN.

Unlike train_dqn.py (which trains only on Sokoban-small-v1), this pipeline
mixes procedural small-v1 episodes with fixed curriculum maps so the agent
learns across 1-, 2-, and 3-box layouts of varying sizes.

Key differences from train_dqn.py:
  - Canvas: 15x15  (vs 10x10)
  - Action space: always 12 = MAX_BOXES*4  (padded with zeros for <3-box maps)
  - Observation: 912 dims = 4*225 + 12  (vs 412)
  - Training env samples from procedural + curriculum maps each episode
  - Results saved under results/rl_tests/curriculum_dqn/
"""

import logging
import os
import random
from datetime import datetime

from stable_baselines3.common.utils import set_random_seed

from src.rl.callbacks import PeriodicEvalCallback
from src.rl.high_level_env import HighLevelSokobanEnv
from src.rl.masked_dqn import MaskedDQN
from src.rl.network import HighLevelBoardExtractor
from src.utils.config import (
    CURRICULUM_DQN_BATCH_SIZE,
    CURRICULUM_DQN_BUFFER_SIZE,
    CURRICULUM_DQN_CANVAS_SHAPE,
    CURRICULUM_DQN_EARLY_STOP_MIN_TIMESTEPS,
    CURRICULUM_DQN_EARLY_STOP_PATIENCE_EVALS,
    CURRICULUM_DQN_EVAL_EPISODES,
    CURRICULUM_DQN_EVAL_FREQ,
    CURRICULUM_DQN_LEARNING_RATE,
    CURRICULUM_DQN_LEARNING_STARTS,
    CURRICULUM_DQN_MAX_BOXES,
    CURRICULUM_DQN_PROCEDURAL_FRACTION,
    CURRICULUM_DQN_SELECTION_EPISODES,
    CURRICULUM_DQN_TOTAL_STEPS,
    HIGH_LEVEL_DQN_BACKBONE,
    HIGH_LEVEL_DQN_CNN_FEATURES_DIM,
    HIGH_LEVEL_DQN_EXPLORATION_FINAL_EPS,
    HIGH_LEVEL_DQN_EXPLORATION_FRACTION,
    HIGH_LEVEL_DQN_EXPLORATION_INITIAL_EPS,
    HIGH_LEVEL_DQN_GAMMA,
    HIGH_LEVEL_DQN_GRADIENT_STEPS,
    HIGH_LEVEL_DQN_POLICY_HIDDEN_SIZES,
    HIGH_LEVEL_DQN_TARGET_UPDATE_INTERVAL,
    HIGH_LEVEL_DQN_TRAIN_FREQ,
    HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES,
    SEED,
)
from src.utils.custom_maps import build_curriculum_maps

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curriculum map sampler
# ---------------------------------------------------------------------------

class CurriculumSampler:
    """Callable that returns None (procedural) or a map_config dict each call.

    Curriculum schedule — fraction of episodes from each difficulty tier
    changes as training progresses:

      Phase          | Procedural | Easy (1-box) | Medium (2-box) | Hard (3-box)
      ---------------+-----------+--------------+----------------+-------------
      early  (0-30%) |    40%    |     35%      |      20%       |      5%
      mid   (30-70%) |    40%    |     15%      |      25%       |     20%
      late  (70-100%)|    40%    |      5%      |      15%       |     40%
    """

    def __init__(self, maps, procedural_fraction=CURRICULUM_DQN_PROCEDURAL_FRACTION):
        self._maps = maps
        self._procedural_fraction = procedural_fraction
        self._step = 0
        self._total_steps = max(CURRICULUM_DQN_TOTAL_STEPS, 1)

        # pre-split by box count for curriculum weighting
        self._easy   = [m for m in maps if len(m["boxes"]) == 1]
        self._medium = [m for m in maps if len(m["boxes"]) == 2]
        self._hard   = [m for m in maps if len(m["boxes"]) == 3]

    def update_step(self, step):
        self._step = step

    def __call__(self):
        if random.random() < self._procedural_fraction:
            return None  # use procedural small-v1

        progress = self._step / self._total_steps
        if progress < 0.30:
            weights = (0.35, 0.20, 0.05)
        elif progress < 0.70:
            weights = (0.15, 0.25, 0.20)
        else:
            weights = (0.05, 0.15, 0.40)

        pool = random.choices(
            [self._easy, self._medium, self._hard],
            weights=weights,
            k=1,
        )[0]
        if not pool:
            pool = self._maps
        return random.choice(pool)


# ---------------------------------------------------------------------------
# Env factories
# ---------------------------------------------------------------------------

def _build_env(use_shaped_reward, map_sampler=None, seed=None):
    env = HighLevelSokobanEnv(
        observation_board_shape=CURRICULUM_DQN_CANVAS_SHAPE,
        use_extra_scalar_features=HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES,
        use_shaped_reward=use_shaped_reward,
        max_boxes=CURRICULUM_DQN_MAX_BOXES,
        map_sampler=map_sampler,
    )
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    return env


def make_train_env(sampler):
    return _build_env(use_shaped_reward=True, map_sampler=sampler, seed=SEED)


def make_eval_env():
    """Eval env uses procedural small-v1 only (no curriculum) for fair comparison."""
    return _build_env(use_shaped_reward=False, map_sampler=None, seed=SEED + 1)


def make_curriculum_eval_env(maps):
    """Eval env that cycles through all curriculum maps deterministically."""
    idx = {"i": 0}

    def round_robin():
        config = maps[idx["i"] % len(maps)]
        idx["i"] += 1
        return config

    return _build_env(use_shaped_reward=False, map_sampler=round_robin, seed=SEED + 2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _build_run_paths(run_id):
    run_dir = os.path.join(PROJECT_ROOT, "results", "rl_tests", "curriculum_dqn", run_id)
    return {
        "run_dir": run_dir,
        "tensorboard_dir": os.path.join(run_dir, "tensorboard"),
        "model_path": os.path.join(run_dir, "curriculum_dqn_final"),
        "best_model_path": os.path.join(run_dir, "curriculum_dqn_best"),
    }


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
    return policy_kwargs


def _build_model(env, paths, sampler):
    model = MaskedDQN(
        "MlpPolicy",
        env,
        action_mask_size=env.action_space.n,
        tensorboard_log=paths["tensorboard_dir"],
        buffer_size=CURRICULUM_DQN_BUFFER_SIZE,
        learning_rate=CURRICULUM_DQN_LEARNING_RATE,
        learning_starts=CURRICULUM_DQN_LEARNING_STARTS,
        batch_size=CURRICULUM_DQN_BATCH_SIZE,
        gamma=HIGH_LEVEL_DQN_GAMMA,
        train_freq=HIGH_LEVEL_DQN_TRAIN_FREQ,
        gradient_steps=HIGH_LEVEL_DQN_GRADIENT_STEPS,
        target_update_interval=HIGH_LEVEL_DQN_TARGET_UPDATE_INTERVAL,
        exploration_initial_eps=HIGH_LEVEL_DQN_EXPLORATION_INITIAL_EPS,
        exploration_fraction=HIGH_LEVEL_DQN_EXPLORATION_FRACTION,
        exploration_final_eps=HIGH_LEVEL_DQN_EXPLORATION_FINAL_EPS,
        policy_kwargs=_build_policy_kwargs(env),
        verbose=1,
        device="auto",
        seed=SEED,
    )
    return model


class _StepSyncCallback:
    """Keeps the CurriculumSampler's step counter in sync with training."""

    def __init__(self, sampler):
        self._sampler = sampler

    def on_step(self, num_timesteps):
        self._sampler.update_step(num_timesteps)


def train():
    set_random_seed(SEED)

    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}_seed{SEED}"
    paths = _build_run_paths(run_id)
    os.makedirs(paths["run_dir"], exist_ok=True)
    os.makedirs(paths["tensorboard_dir"], exist_ok=True)

    curriculum_maps = build_curriculum_maps()
    sampler = CurriculumSampler(curriculum_maps)

    env = make_train_env(sampler)
    model = _build_model(env, paths, sampler)

    callback = PeriodicEvalCallback(
        best_model_path=paths["best_model_path"],
        eval_env_factory=make_eval_env,
        eval_seed_base=SEED + 10_000,
        eval_freq=CURRICULUM_DQN_EVAL_FREQ,
        n_eval_episodes=CURRICULUM_DQN_EVAL_EPISODES,
        early_stop_patience_evals=CURRICULUM_DQN_EARLY_STOP_PATIENCE_EVALS,
        early_stop_min_timesteps=CURRICULUM_DQN_EARLY_STOP_MIN_TIMESTEPS,
    )

    LOGGER.info("Starting curriculum DQN training  run_dir=%s", paths["run_dir"])
    LOGGER.info(
        "Canvas=%s  max_boxes=%d  obs_dims=%d  action_dims=%d",
        CURRICULUM_DQN_CANVAS_SHAPE,
        CURRICULUM_DQN_MAX_BOXES,
        env.observation_space.shape[0],
        env.action_space.n,
    )
    LOGGER.info(
        "Training on %d curriculum maps + %.0f%% procedural small-v1",
        len(curriculum_maps),
        CURRICULUM_DQN_PROCEDURAL_FRACTION * 100,
    )

    model.learn(
        total_timesteps=CURRICULUM_DQN_TOTAL_STEPS,
        callback=callback,
        reset_num_timesteps=True,
    )

    model.save(paths["model_path"])
    env.close()

    LOGGER.info("Curriculum DQN training finished — saved to %s", paths["run_dir"])
    return model, paths["run_dir"]
