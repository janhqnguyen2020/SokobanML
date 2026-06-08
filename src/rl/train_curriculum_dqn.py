# src/rl/train_curriculum_dqn.py
"""
Curriculum DQN training pipeline.

Key differences from train_dqn.py:
  - Canvas: 15x15 (vs 10x10)
  - Action space: MAX_BOXES*4, padded for smaller maps
  - Difficulty phases from easy (1-box) to hard (3-box) as training progresses
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
    CURRICULUM_DQN_TOTAL_STEPS,
    CURRICULUM_EPS_STAGE_SCHEDULE,
    CURRICULUM_EPS_USE_STAGE_RESET,
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

from src.utils.generated_maps import (
    build_generated_1box_maps, build_generated_2box_maps, build_generated_3box_maps,
    build_walled_1box_maps, build_walled_2box_maps, build_walled_3box_maps,
)
from src.rl.video_wrapper import VideoWrapper
from src.rl.video_recorder import EpisodeVideoRecorder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOGGER = logging.getLogger(__name__)


# Curriculum teacher — decides which map difficulty to show each episode

class CurriculumTeacher:
    """Decides what to train on each episode based on how far training has progressed.

    Difficulty schedule:
      Phase          | Procedural | 1-box | 2-box | 3-box
      early  (0-30%) |    40%    |  35%  |  20%  |   5%
      mid   (30-70%) |    40%    |  15%  |  25%  |  20%
      late  (70-100%)|    40%    |   5%  |  15%  |  40%
    """

    def __init__(self, maps, proceduralFraction=CURRICULUM_DQN_PROCEDURAL_FRACTION):
        self.allMaps = maps
        self.proceduralFraction = proceduralFraction
        self.currentTrainingStep = 0
        self.totalTrainingSteps = max(CURRICULUM_DQN_TOTAL_STEPS, 1)

        self.oneBoxMaps   = [m for m in maps if len(m["boxes"]) == 1]
        self.twoBoxMaps   = [m for m in maps if len(m["boxes"]) == 2]
        self.threeBoxMaps = [m for m in maps if len(m["boxes"]) == 3]

    def updateTrainingProgress(self, step):
        self.currentTrainingStep = step

    def sampleNextMap(self):
        if random.random() < self.proceduralFraction:
            return None  # fall back to procedural small-v1

        trainingProgressPercent = self.currentTrainingStep / self.totalTrainingSteps
        if trainingProgressPercent < 0.30:
            difficultyWeights = (0.35, 0.20, 0.05)
        elif trainingProgressPercent < 0.70:
            difficultyWeights = (0.15, 0.25, 0.20)
        else:
            difficultyWeights = (0.05, 0.15, 0.40)

        selectedMapPool = random.choices(
            [self.oneBoxMaps, self.twoBoxMaps, self.threeBoxMaps],
            weights=difficultyWeights,
            k=1,
        )[0]
        if not selectedMapPool:
            selectedMapPool = self.allMaps
        return random.choice(selectedMapPool)

    def __call__(self):
        return self.sampleNextMap()


# Callback that keeps CurriculumTeacher's step counter in sync with training

class CurriculumProgressCallback:
    """Notifies the CurriculumTeacher of training progress each step."""

    def __init__(self, curriculumTeacher):
        self.curriculumTeacher = curriculumTeacher

    def on_step(self, num_timesteps):
        self.curriculumTeacher.updateTrainingProgress(num_timesteps)


# Environment factories

def createSokobanEnvironment(use_shaped_reward, curriculumTeacher=None, seed=None):
    env = HighLevelSokobanEnv(
        observation_board_shape=CURRICULUM_DQN_CANVAS_SHAPE,
        use_extra_scalar_features=HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES,
        use_shaped_reward=use_shaped_reward,
        max_boxes=CURRICULUM_DQN_MAX_BOXES,
        map_sampler=curriculumTeacher,
    )
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    return env


def createTrainingEnvironment(curriculumTeacher, run_id=None):
    if run_id is None:
        run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}_seed{SEED}"

    save_dir = os.path.join(
        "results",
        "rl_tests",
        "curriculum_dqn",
        run_id,
        "videos"
    )

    recorder = EpisodeVideoRecorder(
        save_dir=save_dir,
        fps=5,
    )

    env = createSokobanEnvironment(
        use_shaped_reward=True,
        curriculumTeacher=curriculumTeacher,
        seed=SEED,
    )

    return VideoWrapper(env, recorder)


def createEvaluationEnvironment():
    """Procedural small-v1 only — no curriculum, for fair baseline comparison."""
    return createSokobanEnvironment(use_shaped_reward=False, curriculumTeacher=None, seed=SEED + 1)


def createCurriculumEvalEnvironment(trainingCurriculumMaps):
    """Cycles through all curriculum maps deterministically for consistent eval."""
    idx = {"i": 0}

    def cycleThroughMaps():
        config = trainingCurriculumMaps[idx["i"] % len(trainingCurriculumMaps)]
        idx["i"] += 1
        return config

    return createSokobanEnvironment(use_shaped_reward=False, curriculumTeacher=cycleThroughMaps, seed=SEED + 2)


# Model and path builders

def buildRunPaths(run_id):
    runDirectory = os.path.join(PROJECT_ROOT, "results", "rl_tests", "curriculum_dqn", run_id)
    return {
        "run_dir":         runDirectory,
        "tensorboard_dir": os.path.join(runDirectory, "tensorboard"),
        "model_path":      os.path.join(runDirectory, "curriculum_dqn_final"),
        "best_model_path": os.path.join(runDirectory, "curriculum_dqn_best"),
    }


def buildPolicyConfig(env):
    policyConfig = {"net_arch": HIGH_LEVEL_DQN_POLICY_HIDDEN_SIZES}
    if HIGH_LEVEL_DQN_BACKBONE == "cnn":
        policyConfig["features_extractor_class"] = HighLevelBoardExtractor
        policyConfig["features_extractor_kwargs"] = {
            "board_shape":         env.observation_board_shape,
            "action_mask_size":    env.action_space.n,
            "scalar_feature_size": env.scalar_feature_size,
            "features_dim":        HIGH_LEVEL_DQN_CNN_FEATURES_DIM,
        }
    return policyConfig


def createMaskedDQNModel(env, trainingOutputPaths, init_model_path=None):
    curriculum_eps = CURRICULUM_EPS_STAGE_SCHEDULE if CURRICULUM_EPS_USE_STAGE_RESET else None

    if init_model_path:
        LOGGER.info("Warm-starting weights from %s", init_model_path)
        model = MaskedDQN.load(
            init_model_path,
            env=env,
            tensorboard_log=trainingOutputPaths["tensorboard_dir"],
            device="auto",
        )
        model._curriculum_eps_schedule = None
        model.exploration_initial_eps = HIGH_LEVEL_DQN_EXPLORATION_FINAL_EPS
        model.exploration_rate = HIGH_LEVEL_DQN_EXPLORATION_FINAL_EPS
        model.exploration_final_eps = HIGH_LEVEL_DQN_EXPLORATION_FINAL_EPS
        return model

    return MaskedDQN(
        "MlpPolicy",
        env,
        action_mask_size=env.action_space.n,
        curriculum_eps_schedule=curriculum_eps,
        tensorboard_log=trainingOutputPaths["tensorboard_dir"],
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
        policy_kwargs=buildPolicyConfig(env),
        verbose=1,
        device="auto",
        seed=SEED,
    )


# Main training entry point

def train(init_model_path=None):
    set_random_seed(SEED)

    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}_seed{SEED}"
    trainingOutputPaths = buildRunPaths(run_id)
    os.makedirs(trainingOutputPaths["run_dir"], exist_ok=True)
    os.makedirs(trainingOutputPaths["tensorboard_dir"], exist_ok=True)

    trainingCurriculumMaps = (
        build_generated_1box_maps() + build_walled_1box_maps() +
        build_generated_2box_maps() + build_walled_2box_maps() +
        build_generated_3box_maps() + build_walled_3box_maps()
    )
    curriculumTeacher = CurriculumTeacher(trainingCurriculumMaps, proceduralFraction=CURRICULUM_DQN_PROCEDURAL_FRACTION)

    env = createTrainingEnvironment(curriculumTeacher, run_id=run_id)
    model = createMaskedDQNModel(env, trainingOutputPaths, init_model_path=init_model_path)

    # Eval on the same pool used for training so success_rate is meaningful
    evalCallback = PeriodicEvalCallback(
        best_model_path=trainingOutputPaths["best_model_path"],
        eval_env_factory=lambda: createCurriculumEvalEnvironment(trainingCurriculumMaps),
        eval_seed_base=SEED + 10_000,
        eval_freq=CURRICULUM_DQN_EVAL_FREQ,
        n_eval_episodes=CURRICULUM_DQN_EVAL_EPISODES,
        early_stop_patience_evals=CURRICULUM_DQN_EARLY_STOP_PATIENCE_EVALS,
        early_stop_min_timesteps=CURRICULUM_DQN_EARLY_STOP_MIN_TIMESTEPS,
    )

    LOGGER.info("Starting curriculum DQN  run_dir=%s", trainingOutputPaths["run_dir"])
    LOGGER.info(
        "Canvas=%s  max_boxes=%d  obs_dims=%d  action_dims=%d",
        CURRICULUM_DQN_CANVAS_SHAPE,
        CURRICULUM_DQN_MAX_BOXES,
        env.observation_space.shape[0],
        env.action_space.n,
    )
    LOGGER.info(
        "Training on %d curriculum maps + %.0f%% procedural small-v1",
        len(trainingCurriculumMaps),
        CURRICULUM_DQN_PROCEDURAL_FRACTION * 100,
    )

    model.learn(
        total_timesteps=CURRICULUM_DQN_TOTAL_STEPS,
        callback=evalCallback,
        reset_num_timesteps=True,
    )

    model.save(trainingOutputPaths["model_path"])
    env.close()

    LOGGER.info("Curriculum DQN finished — saved to %s", trainingOutputPaths["run_dir"])
    return model, trainingOutputPaths["run_dir"]
