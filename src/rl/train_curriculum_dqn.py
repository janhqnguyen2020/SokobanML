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
FIXED_FAMILY_NAMES = [
    "generated_1box",
    "generated_2box",
    "generated_3box",
    "walled",
    "csp",
]


# Curriculum teacher — decides which map difficulty to show each episode

class CurriculumTeacher:
    """Decides what to train on each episode based on how far training has progressed.

    Difficulty schedule:
      Phase          | Procedural | 1-box | 2-box | 3-box
      early  (0-30%) |    40%    |  35%  |  20%  |   5%
      mid   (30-70%) |    40%    |  15%  |  25%  |  20%
      late  (70-100%)|    40%    |   5%  |  15%  |  40%
    """

    def __init__(self, maps, proceduralFraction=CURRICULUM_DQN_PROCEDURAL_FRACTION, phase_config=None, focusMaps=None, focusMapBoost=1.0):
        self.allMaps = maps
        self.proceduralFraction = proceduralFraction
        self.phaseConfig = phase_config
        self.focusMaps = list(focusMaps or [])
        self.focusMapBoost = float(max(focusMapBoost, 1.0))
        self.currentTrainingStep = 0
        self.totalTrainingSteps = max(CURRICULUM_DQN_TOTAL_STEPS, 1)
        self.proceduralEpisodeCount = 0
        self.oneBoxMaps = [m for m in maps if len(m["boxes"]) == 1]
        self.twoBoxMaps = [m for m in maps if len(m["boxes"]) == 2]
        self.threeBoxMaps = [m for m in maps if len(m["boxes"]) == 3]
        self.focusOneBoxMaps = [m for m in self.focusMaps if len(m["boxes"]) == 1]
        self.focusTwoBoxMaps = [m for m in self.focusMaps if len(m["boxes"]) == 2]
        self.focusThreeBoxMaps = [m for m in self.focusMaps if len(m["boxes"]) == 3]
        self.focusMapNames = {map_config["map_name"] for map_config in self.focusMaps}
        self.totalSampledEpisodes = 0
        self.phaseSampledEpisodes = 0
        self.phaseFixedCounts = build_box_count_totals()
        self.phaseProceduralCounts = {}
        self.phaseFocusedFixedSamples = 0
        self.phaseAdaptiveBoxWeights = build_box_weight_totals()
        self.phaseAdaptiveFamilyWeights = build_family_weight_totals()
        self.phaseAdaptiveProceduralWeights = build_procedural_weight_totals(self.phaseConfig)
        self.phaseAdaptiveFailedMapNames = set()
        self.phaseAdaptiveFailedMapBoost = 1.0

    def updateTrainingProgress(self, step):
        """Track training progress for the original percentage-based schedule."""
        self.currentTrainingStep = step

    def setPhaseConfig(self, phase_config):
        """Switch the teacher to one explicit curriculum phase configuration."""
        self.phaseConfig = phase_config
        self._reset_phase_sampling_stats()

    def sampleNextMap(self):
        """Return either one fixed map or one procedural env spec for this episode."""
        sampled_item = self._sample_phase_item() if self.phaseConfig is not None else self._sample_progress_item()
        self._record_sampled_item(sampled_item)
        return sampled_item


    def applyPeriodicValidationSummary(self, summary):
        """Use periodic validation to boost weak fixed maps and weak procedural envs."""
        if "type_summary" not in summary:
            return
        fixed_summary = summary["type_summary"].get("fixed", {})
        procedural_summary = summary["type_summary"].get("procedural", {})
        self._set_adaptive_box_weights(fixed_summary)
        self._set_adaptive_family_weights(fixed_summary)
        self._set_adaptive_procedural_weights(procedural_summary)
        self._set_adaptive_failed_map_names(fixed_summary)

    def _sample_phase_item(self):
        """Sample from the explicit phase weights and procedural env list."""
        if random.random() > self.phaseConfig["fixed_fraction"]:
            return self._sample_procedural_spec(self.phaseConfig["procedural_env_ids"])
        return self._sample_weighted_fixed_map(self.phaseConfig["box_weights"])

    def _sample_progress_item(self):
        """Keep the original percentage-based schedule for baseline curriculum runs."""
        if random.random() < self.proceduralFraction:
            return None
        return self._sample_weighted_fixed_map(self._progress_box_weights())

    def _progress_box_weights(self):
        """Return the original box-count weights based on global training progress."""
        trainingProgressPercent = self.currentTrainingStep / self.totalTrainingSteps
        if trainingProgressPercent < 0.30:
            return {1: 0.35, 2: 0.20, 3: 0.05}
        if trainingProgressPercent < 0.70:
            return {1: 0.15, 2: 0.25, 3: 0.20}
        return {1: 0.05, 2: 0.15, 3: 0.40}

    def _sample_procedural_spec(self, procedural_env_ids):
        """Build one procedural env request for the current training episode."""
        if not procedural_env_ids:
            return None
        env_id = self._sample_weighted_procedural_env_id(procedural_env_ids)
        self.proceduralEpisodeCount += 1
        return {"env_id": env_id, "seed": SEED + 50_000 + self.proceduralEpisodeCount}

    def _sample_weighted_procedural_env_id(self, procedural_env_ids):
        """Pick one enabled procedural env using recent procedural validation weights."""
        weights = procedural_sampling_weights(procedural_env_ids, self.phaseAdaptiveProceduralWeights)
        return random.choices(procedural_env_ids, weights=weights, k=1)[0]

    def _sample_weighted_fixed_map(self, box_weights):
        """Pick one fixed map using box weights plus adaptive family and failure boosts."""
        weighted_maps = fixed_map_sampling_weights(
            self.allMaps,
            box_weights,
            self.phaseAdaptiveBoxWeights,
            self.phaseAdaptiveFamilyWeights,
            self.focusMapNames,
            self.focusMapBoost,
            self.phaseAdaptiveFailedMapNames,
            self.phaseAdaptiveFailedMapBoost,
        )
        return random.choices(self.allMaps, weights=weighted_maps, k=1)[0]


    def _set_adaptive_box_weights(self, fixed_summary):
        """Boost weaker 1-box, 2-box, or 3-box groups across all fixed families."""
        self.phaseAdaptiveBoxWeights = {
            1: adaptive_success_weight(fixed_family_success(fixed_summary, "generated_1box")),
            2: adaptive_success_weight(fixed_family_success(fixed_summary, "generated_2box")),
            3: adaptive_success_weight(fixed_family_success(fixed_summary, "generated_3box")),
        }


    def _set_adaptive_family_weights(self, fixed_summary):
        """Boost fixed families whose recent periodic validation is currently weak."""
        self.phaseAdaptiveFamilyWeights = {
            family_name: adaptive_success_weight(fixed_family_success(fixed_summary, family_name))
            for family_name in FIXED_FAMILY_NAMES
        }

    def _set_adaptive_procedural_weights(self, procedural_summary):
        """Boost weaker enabled procedural envs until they catch back up."""
        env_ids = phase_procedural_env_ids(self.phaseConfig)
        boost_scale = procedural_boost_scale(self.phaseConfig)
        self.phaseAdaptiveProceduralWeights = adaptive_procedural_weights(procedural_summary, env_ids, boost_scale)


    def _set_adaptive_failed_map_names(self, fixed_summary):
        """Focus on currently failing fixed maps so the boost disappears after recovery."""
        self.phaseAdaptiveFailedMapNames = failed_fixed_case_names(fixed_summary)
        self.phaseAdaptiveFailedMapBoost = 3.0 if self.phaseAdaptiveFailedMapNames else 1.0

    def _reset_phase_sampling_stats(self):
        """Clear the episode counters used to describe the current phase."""
        self.phaseSampledEpisodes = 0
        self.phaseFixedCounts = build_box_count_totals()
        self.phaseProceduralCounts = {}
        self.phaseFocusedFixedSamples = 0
        self.phaseAdaptiveBoxWeights = build_box_weight_totals()
        self.phaseAdaptiveFamilyWeights = build_family_weight_totals()
        self.phaseAdaptiveProceduralWeights = build_procedural_weight_totals(self.phaseConfig)
        self.phaseAdaptiveFailedMapNames = set()
        self.phaseAdaptiveFailedMapBoost = 1.0

    def _record_sampled_item(self, sampled_item):
        """Track what kind of episode the curriculum just sampled."""
        self.totalSampledEpisodes += 1
        self.phaseSampledEpisodes += 1
        if is_procedural_sample(sampled_item):
            self._record_procedural_sample(sampled_item)
            return
        self._record_fixed_sample(sampled_item)

    def _record_procedural_sample(self, procedural_spec):
        """Increment the counter for one sampled procedural environment id."""
        env_id = procedural_spec["env_id"] if procedural_spec is not None else "default_procedural"
        current_count = self.phaseProceduralCounts.get(env_id, 0)
        self.phaseProceduralCounts[env_id] = current_count + 1

    def _record_fixed_sample(self, map_config):
        """Increment the counter for one sampled fixed-map box count."""
        num_boxes = len(map_config["boxes"])
        self.phaseFixedCounts[num_boxes] = self.phaseFixedCounts[num_boxes] + 1
        if map_config["map_name"] in self.focusMapNames:
            self.phaseFocusedFixedSamples += 1

    def phaseProgressSummary(self):
        """Return the current phase counts in a simple dictionary."""
        return {
            "phase_sampled_episodes": int(self.phaseSampledEpisodes),
            "total_sampled_episodes": int(self.totalSampledEpisodes),
            "fixed_counts": fixed_count_summary(self.phaseFixedCounts),
            "procedural_counts": sorted_procedural_counts(self.phaseProceduralCounts),
            "focused_fixed_samples": int(self.phaseFocusedFixedSamples),
            "adaptive_box_weights": box_weight_summary(self.phaseAdaptiveBoxWeights),
            "adaptive_family_weights": family_weight_summary(self.phaseAdaptiveFamilyWeights),
            "adaptive_procedural_weights": procedural_weight_summary(self.phaseAdaptiveProceduralWeights),
            "adaptive_failed_maps": sorted(self.phaseAdaptiveFailedMapNames),
        }

    def __call__(self):
        return self.sampleNextMap()


def build_box_count_totals():
    """Start one empty fixed-map counter for 1-box, 2-box, and 3-box episodes."""
    return {1: 0, 2: 0, 3: 0}


def build_box_weight_totals():
    """Start one neutral adaptive weight for each fixed-map box-count family."""
    return {1: 1.0, 2: 1.0, 3: 1.0}


def build_family_weight_totals():
    """Start one neutral adaptive weight for each tracked fixed-map family."""
    return {family_name: 1.0 for family_name in FIXED_FAMILY_NAMES}


def build_procedural_weight_totals(phase_config):
    """Start one neutral adaptive weight for each enabled procedural env."""
    return {env_id: 1.0 for env_id in phase_procedural_env_ids(phase_config)}


def phase_procedural_env_ids(phase_config):
    """Return the procedural env ids enabled for the current curriculum phase."""
    if phase_config is None:
        return []
    return list(phase_config["procedural_env_ids"])


def is_procedural_sample(sampled_item):
    """Return True when the sampled curriculum item is one procedural env spec."""
    return sampled_item is None or "env_id" in sampled_item


def fixed_count_summary(box_counts):
    """Convert internal fixed-map counters into readable 1/2/3-box labels."""
    return {
        "1box": int(box_counts[1]),
        "2box": int(box_counts[2]),
        "3box": int(box_counts[3]),
    }


def sorted_procedural_counts(procedural_counts):
    """Return procedural env counters in a stable key order for printing."""
    return {env_id: int(procedural_counts[env_id]) for env_id in sorted(procedural_counts.keys())}


def procedural_weight_summary(procedural_weights):
    """Round procedural weights so phase logs stay compact and easy to read."""
    return {env_id: round(float(procedural_weights[env_id]), 3) for env_id in sorted(procedural_weights.keys())}


def procedural_sampling_weights(procedural_env_ids, adaptive_procedural_weights):
    """Return the sampling weight for each enabled procedural env id."""
    return [procedural_sampling_weight(env_id, adaptive_procedural_weights) for env_id in procedural_env_ids]


def procedural_sampling_weight(env_id, adaptive_procedural_weights):
    """Read one procedural env weight and fall back to a neutral weight."""
    return float(adaptive_procedural_weights.get(str(env_id), 1.0))


def box_weight_summary(box_weights):
    """Round adaptive box weights so progress logs stay compact and readable."""
    return {f"{box_count}box": round(float(box_weights[box_count]), 3) for box_count in sorted(box_weights.keys())}


def family_weight_summary(family_weights):
    """Round adaptive family weights so progress logs stay compact and readable."""
    return {family_name: round(float(family_weights[family_name]), 3) for family_name in sorted(family_weights.keys())}


def fixed_map_sampling_weights(all_maps, box_weights, adaptive_box_weights, adaptive_family_weights, focus_names, focus_boost, failed_map_names, failed_map_boost):
    """Build one sampling weight per fixed map from box, family, and failure signals."""
    return [
        fixed_map_sampling_weight(
            map_config,
            box_weights,
            adaptive_box_weights,
            adaptive_family_weights,
            focus_names,
            focus_boost,
            failed_map_names,
            failed_map_boost,
        )
        for map_config in all_maps
    ]


def fixed_map_sampling_weight(map_config, box_weights, adaptive_box_weights, adaptive_family_weights, focus_names, focus_boost, failed_map_names, failed_map_boost):
    """Combine the base box weight with family and failing-map boosts for one map."""
    num_boxes = len(map_config["boxes"])
    family_name = fixed_family_name(map_config)
    weight = float(box_weights[num_boxes]) * float(adaptive_box_weights[num_boxes])
    weight *= float(adaptive_family_weights[family_name])
    weight *= focused_weight(map_config, focus_names, focus_boost)
    weight *= failed_map_weight(map_config, failed_map_names, failed_map_boost)
    return weight


def fixed_family_name(map_config):
    """Map one fixed training map to the family names used in validation summaries."""
    if map_config.get("map_source") == "generated":
        return f"generated_{len(map_config['boxes'])}box"
    if prefix_match(map_config["map_name"], ["csp"]):
        return "csp"
    return "walled"


def focused_weight(map_config, focus_names, focus_boost):
    """Return the extra sampling weight for one map in the focus set."""
    if map_config["map_name"] in focus_names:
        return float(focus_boost)
    return 1.0


def failed_map_weight(map_config, failed_map_names, failed_map_boost):
    """Return the temporary boost for one currently failing fixed map."""
    if map_config["map_name"] in failed_map_names:
        return float(failed_map_boost)
    return 1.0


def adaptive_success_weight(success_rate):
    """Convert one recent success rate into a replay boost for weaker map groups."""
    return scaled_adaptive_success_weight(success_rate, 2.0)


def scaled_adaptive_success_weight(success_rate, boost_scale):
    """Convert one success rate into a replay boost using the requested strength."""
    return 1.0 + max(0.0, 1.0 - float(success_rate)) * float(boost_scale)


def fixed_family_success(fixed_summary, family_name):
    """Read one fixed-family success rate from the periodic type summary."""
    family_summary = fixed_summary.get(str(family_name), {})
    return float(family_summary.get("success_rate", 1.0))


def procedural_family_success(procedural_summary, env_id):
    """Read one procedural-env success rate from the periodic type summary."""
    env_summary = procedural_summary.get(str(env_id), {})
    return float(env_summary.get("success_rate", 1.0))


def adaptive_procedural_weights(procedural_summary, env_ids, boost_scale):
    """Build one adaptive weight per enabled procedural env id."""
    return {
        env_id: scaled_adaptive_success_weight(procedural_family_success(procedural_summary, env_id), boost_scale)
        for env_id in env_ids
    }


def procedural_boost_scale(phase_config):
    """Return the procedural replay boost strength for one curriculum phase."""
    if phase_config is None or "procedural_weight_scale" not in phase_config:
        return 2.0
    return float(phase_config["procedural_weight_scale"])


def failed_fixed_case_names(fixed_summary):
    """Collect the currently failing fixed maps from all tracked fixed families."""
    failed_names = set()
    for family_name in FIXED_FAMILY_NAMES:
        failed_names.update(fixed_summary.get(family_name, {}).get("failed_case_names", []))
    return failed_names


def prefix_match(map_name, prefixes):
    """Return True when one fixed-map name starts with any requested prefix."""
    return any(str(map_name).startswith(str(prefix)) for prefix in prefixes)


# Callback that keeps CurriculumTeacher's step counter in sync with training

class CurriculumProgressCallback:
    """Notifies the CurriculumTeacher of training progress each step."""

    def __init__(self, curriculumTeacher):
        self.curriculumTeacher = curriculumTeacher

    def on_step(self, num_timesteps):
        self.curriculumTeacher.updateTrainingProgress(num_timesteps)


# Environment factories

def createSokobanEnvironment(use_shaped_reward, curriculumTeacher=None, seed=None, procedural_env_id=None, max_boxes=None):
    """Build one high-level Sokoban env around either a default or chosen procedural env."""
    env = HighLevelSokobanEnv(
        env=create_procedural_env(procedural_env_id, seed) if procedural_env_id is not None else None,
        observation_board_shape=CURRICULUM_DQN_CANVAS_SHAPE,
        use_extra_scalar_features=HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES,
        use_shaped_reward=use_shaped_reward,
        max_boxes=resolved_curriculum_max_boxes(max_boxes),
        map_sampler=curriculumTeacher,
    )
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    return env


def resolved_curriculum_max_boxes(max_boxes):
    """Use the requested max-box count or fall back to the current curriculum default."""
    if max_boxes is None:
        return int(CURRICULUM_DQN_MAX_BOXES)
    return int(max_boxes)


def create_procedural_env(procedural_env_id, seed):
    """Create one procedural gym_sokoban env for the requested env id."""
    from src.env.sokoban_env import initialize_env

    return initialize_env(env_id=procedural_env_id, seed=seed)


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


def createEvaluationEnvironment(env_id=None, seed=None, max_boxes=None):
    """Build one procedural evaluation env for the requested gym_sokoban id."""
    return createSokobanEnvironment(
        use_shaped_reward=False,
        curriculumTeacher=None,
        seed=SEED + 1 if seed is None else seed,
        procedural_env_id=env_id,
        max_boxes=max_boxes,
    )


def createCurriculumEvalEnvironment(trainingCurriculumMaps, max_boxes=None):
    """Cycles through all curriculum maps deterministically for consistent eval."""
    idx = {"i": 0}

    def cycleThroughMaps():
        config = trainingCurriculumMaps[idx["i"] % len(trainingCurriculumMaps)]
        idx["i"] += 1
        return config

    return createSokobanEnvironment(
        use_shaped_reward=False,
        curriculumTeacher=cycleThroughMaps,
        seed=SEED + 2,
        max_boxes=max_boxes,
    )


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


def createMaskedDQNModel(env, trainingOutputPaths):
    return MaskedDQN(
        "MlpPolicy",
        env,
        action_mask_size=env.action_space.n,
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

def train():
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
    model = createMaskedDQNModel(env, trainingOutputPaths)

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
