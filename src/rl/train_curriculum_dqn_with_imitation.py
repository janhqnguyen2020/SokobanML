"""Train curriculum DQN with fixed-map splits, phase checkpoints, and separate evaluation."""

import argparse
import csv
import glob
import json
import os
from datetime import datetime

from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import set_random_seed

from src.rl.callbacks import PeriodicEvalCallback, PhaseProgressCallback
from src.rl.curriculum_schedule import build_curriculum_phase_configs, find_phase_config
from src.rl.evaluate import evaluate_episode, load_model, summarize_results
from src.rl.astar_reference_videos import save_astar_reference_videos
from src.rl.failed_map_focus import resolve_failed_map_focus
from src.rl.fixed_eval_subset import build_full_fixed_eval_maps, build_quick_fixed_eval_maps
from src.rl.imitation import (
    DEFAULT_DEMO_CACHE_PATH,
    DEFAULT_PROCEDURAL_DEMO_CACHE_PATH,
    load_or_generate_demonstrations,
    load_or_generate_procedural_demonstrations,
    limit_demonstration_samples,
    merge_demonstration_payloads,
    print_demo_summary,
    run_macro_replay_sanity_check,
    save_demonstration_payload,
    validate_demonstration_actions,
)
from src.rl.imitation_audit import save_astar_audit_videos
from src.rl.masked_dqn import MaskedDQN
from src.rl.type_summary import build_phase_history_row, build_type_summary, phase_history_columns, write_type_summary
from src.rl.train_curriculum_dqn import (
    CurriculumTeacher,
    createCurriculumEvalEnvironment,
    createEvaluationEnvironment,
    createMaskedDQNModel,
    createSokobanEnvironment,
)
from src.rl.validation_videos import save_validation_videos
from src.rl.video_recorder import EpisodeVideoRecorder
from src.rl.video_wrapper import VideoWrapper
from src.utils.config import (
    CURRICULUM_DQN_CANVAS_SHAPE,
    CURRICULUM_DQN_EARLY_STOP_MIN_TIMESTEPS,
    CURRICULUM_DQN_EARLY_STOP_PATIENCE_EVALS,
    CURRICULUM_DQN_EVAL_FREQ,
    CURRICULUM_FIXED_GENERATED_JSON_PATH,
    CURRICULUM_DQN_MAX_BOXES,
    CURRICULUM_DQN_PHASE_TIMESTEPS,
    CURRICULUM_DQN_PROCEDURAL_FRACTION,
    CURRICULUM_PROCEDURAL_DEMO_TARGET_PER_ENV,
    CURRICULUM_PROCEDURAL_ENV_IDS,
    CURRICULUM_PROCEDURAL_TEST_EPISODES_PER_ENV,
    CURRICULUM_PROCEDURAL_VAL_EPISODES_PER_ENV,
    HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES,
    SEED,
)
from src.utils.fixed_map_dataset import build_fixed_dataset_splits, build_split_manifest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "rl_tests", "curriculum_dqn_imitation")
PHASE_HISTORY_CSV_PATH = os.path.join(RESULTS_ROOT, "phase_type_history.csv")


def parse_args():
    """Parse the training, resume, audit, validation, and final-test options."""
    parser = argparse.ArgumentParser(description="Train curriculum DQN with A* imitation")
    parser.add_argument("--generated-map-json", default=CURRICULUM_FIXED_GENERATED_JSON_PATH)
    parser.add_argument("--demo-cache", default=DEFAULT_DEMO_CACHE_PATH)
    parser.add_argument("--procedural-demo-cache", default=DEFAULT_PROCEDURAL_DEMO_CACHE_PATH)
    parser.add_argument("--imitation-epochs", type=int, default=8)
    parser.add_argument("--imitation-batch-size", type=int, default=64)
    parser.add_argument("--imitation-learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-demonstrations", type=int, default=None)
    parser.add_argument("--regenerate-demos", action="store_true")
    parser.add_argument("--save-astar-audit-videos", action="store_true")
    parser.add_argument("--show-astar-audit", action="store_true")
    parser.add_argument("--astar-audit-only", action="store_true")
    parser.add_argument("--astar-audit-successes", type=int, default=5)
    parser.add_argument("--astar-audit-fps", type=int, default=4)
    parser.add_argument("--astar-audit-search-stride", type=int, default=50)
    parser.add_argument("--skip-imitation", action="store_true")
    parser.add_argument("--resume-model", default=None)
    parser.add_argument("--resume-replay-buffer", default=None)
    parser.add_argument("--fresh-run-id-on-resume", action="store_true")
    parser.add_argument("--focus-failed-maps", action="store_true")
    parser.add_argument("--failed-map-csv", default=None)
    parser.add_argument("--failed-map-boost", type=float, default=3.0)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--final-test-only", action="store_true")
    parser.add_argument("--eval-model", default=None)
    parser.add_argument("--full-validation-fixed", action="store_true")
    parser.add_argument("--validation-procedural-env-ids", nargs="*", default=None)
    parser.add_argument("--additional-timesteps", type=int, default=None)
    parser.add_argument("--start-phase", type=int, default=1)
    parser.add_argument("--stop-after-phase", type=int, default=len(CURRICULUM_DQN_PHASE_TIMESTEPS))
    parser.add_argument("--procedural-demo-target-per-env", type=int, default=CURRICULUM_PROCEDURAL_DEMO_TARGET_PER_ENV)
    parser.add_argument("--procedural-val-episodes-per-env", type=int, default=CURRICULUM_PROCEDURAL_VAL_EPISODES_PER_ENV)
    parser.add_argument("--procedural-test-episodes-per-env", type=int, default=CURRICULUM_PROCEDURAL_TEST_EPISODES_PER_ENV)
    return parser.parse_args()


def build_run_paths(run_id):
    """Return the folders and core artifact paths for one imitation-training run."""
    run_dir = os.path.join(RESULTS_ROOT, run_id)
    return {
        "run_dir": run_dir,
        "tensorboard_dir": os.path.join(run_dir, "tensorboard"),
        "video_dir": os.path.join(run_dir, "videos"),
        "astar_audit_dir": os.path.join(run_dir, "astar_audit"),
        "checkpoints_dir": os.path.join(run_dir, "checkpoints"),
        "validation_dir": os.path.join(run_dir, "validation"),
        "test_dir": os.path.join(run_dir, "final_test"),
        "astar_reference_dir": os.path.join(run_dir, "validation", "astar_reference"),
        "split_manifest_path": os.path.join(run_dir, "fixed_split_manifest.json"),
        "post_imitation_model_path": os.path.join(run_dir, "checkpoints", "post_imitation"),
        "run_summary_json": os.path.join(run_dir, "run_summary.json"),
    }


def ensure_run_directories(run_paths):
    """Create the directories needed for models, validation, videos, and summaries."""
    for key in ("run_dir", "tensorboard_dir", "video_dir", "astar_audit_dir", "checkpoints_dir", "validation_dir", "test_dir", "astar_reference_dir"):
        os.makedirs(run_paths[key], exist_ok=True)


def timestamp_run_id():
    """Build one readable timestamped run id for a fresh training launch."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f") + f"_seed{SEED}"


def normalized_path_parts(path):
    """Split one path into normalized parts so run ids can be recovered cleanly."""
    if path is None:
        return []
    return os.path.normpath(os.path.abspath(path)).split(os.sep)


def curriculum_run_id_from_path(path):
    """Return the original run id when a checkpoint path lives inside one run folder."""
    path_parts = normalized_path_parts(path)
    if "curriculum_dqn_imitation" not in path_parts:
        return None
    run_index = path_parts.index("curriculum_dqn_imitation") + 1
    if run_index >= len(path_parts) or path_parts[run_index] == "evals":
        return None
    return path_parts[run_index]


def checkpoint_stage_name(model_path):
    """Map one checkpoint filename back to the stage folder that should hold its outputs."""
    checkpoint_name = checkpoint_file_stem(model_path)
    if checkpoint_name == "post_imitation":
        return checkpoint_name
    if checkpoint_name.endswith("_best") or checkpoint_name.endswith("_end"):
        return checkpoint_name.rsplit("_", 1)[0]
    step_stage_name, _ = step_checkpoint_parts(checkpoint_name)
    if step_stage_name is not None:
        return step_stage_name
    return None


def checkpoint_file_stem(model_path):
    """Return one checkpoint filename without its file extension."""
    return os.path.splitext(os.path.basename(str(model_path)))[0]


def step_checkpoint_parts(checkpoint_name):
    """Split one step snapshot name into its stage name and saved phase timesteps."""
    if "_step" not in str(checkpoint_name):
        return None, 0
    stage_name, step_text = str(checkpoint_name).rsplit("_step", 1)
    if not step_text.isdigit():
        return None, 0
    return stage_name, int(step_text)


def checkpoint_step_timesteps(model_path):
    """Return the completed phase timesteps stored in one step snapshot name."""
    _, saved_timesteps = step_checkpoint_parts(checkpoint_file_stem(model_path))
    return int(saved_timesteps)


def resumed_phase_elapsed_timesteps(args, phase_config):
    """Return the saved in-phase progress when resuming from one step snapshot."""
    if args.resume_model is None:
        return 0
    if checkpoint_stage_name(args.resume_model) != phase_stage_name(phase_config):
        return 0
    return checkpoint_step_timesteps(args.resume_model)


def phase_exploration_timesteps(phase_config, phase_timesteps, phase_elapsed_timesteps):
    """Use the full original phase schedule when resuming from inside that phase."""
    if int(phase_elapsed_timesteps) > 0:
        return int(phase_config["timesteps"])
    return int(phase_timesteps)


def resolve_training_run_id(args):
    """Reuse the original run folder for resumed training or create a fresh one."""
    if should_use_fresh_run_id_on_resume(args):
        return timestamp_run_id()
    existing_run_id = resumed_run_id_to_reuse(args)
    if existing_run_id is not None:
        return existing_run_id
    return timestamp_run_id()


def should_use_fresh_run_id_on_resume(args):
    """Return True when resumed training should avoid overwriting older outputs."""
    if args.resume_model is None:
        return False
    return bool(args.fresh_run_id_on_resume)


def resumed_run_id_to_reuse(args):
    """Return one original run id only when the caller is truly continuing later phases."""
    if args.resume_model is None:
        return None
    if int(args.start_phase) <= 1:
        return None
    if checkpoint_stage_name(args.resume_model) == "post_imitation":
        return None
    return curriculum_run_id_from_path(args.resume_model)


def fallback_eval_output_dir(eval_name):
    """Return one timestamped eval folder when no original run folder can be recovered."""
    return os.path.join(RESULTS_ROOT, "evals", eval_name, timestamp_run_id())


def eval_root_dir(run_paths, split_name):
    """Choose the validation or final-test root folder inside one training run."""
    if split_name == "final_test":
        return run_paths["test_dir"]
    return run_paths["validation_dir"]


def stage_output_dir(root_dir, stage_name, eval_name):
    """Place one eval inside its stage folder, or fall back to a readable manual folder."""
    if stage_name is None:
        return os.path.join(root_dir, eval_name)
    return os.path.join(root_dir, stage_name)


def eval_astar_reference_dir(model_path, split_name, eval_dir):
    """Reuse the run-level A* reference folder when the checkpoint belongs to one run."""
    run_id = curriculum_run_id_from_path(model_path)
    if run_id is None:
        return os.path.join(eval_dir, "astar_reference")
    run_paths = build_run_paths(run_id)
    ensure_run_directories(run_paths)
    if split_name == "final_test":
        return os.path.join(run_paths["test_dir"], "astar_reference")
    return run_paths["astar_reference_dir"]


def astar_reference_summary_path(output_dir):
    """Return the summary file that marks one saved A* reference showcase."""
    return os.path.join(output_dir, "astar_reference_videos.json")


def procedural_eval_env_ids_for_model(model_path):
    """Return the procedural env ids that match one saved checkpoint's curriculum phase."""
    phase_config = phase_config_for_model_path(model_path)
    if phase_config is None:
        return list(CURRICULUM_PROCEDURAL_ENV_IDS)
    return list(phase_config["procedural_env_ids"])


def validation_procedural_env_ids(args, model_path):
    """Use the caller's validation env override when provided, else follow the checkpoint phase."""
    if args.validation_procedural_env_ids is not None:
        return list(args.validation_procedural_env_ids)
    return procedural_eval_env_ids_for_model(model_path)


def phase_config_for_model_path(model_path):
    """Return the matching phase config when one checkpoint belongs to one curriculum phase."""
    stage_name = checkpoint_stage_name(model_path)
    if stage_name is None or stage_name == "post_imitation":
        return None
    for phase_config in build_curriculum_phase_configs():
        if phase_stage_name(phase_config) == stage_name:
            return phase_config
    return None


def phase_stage_name(phase_config):
    """Build the folder and checkpoint label used by one curriculum phase."""
    return f"phase{phase_config['phase_id']}_{phase_config['phase_name']}"


def ensure_directory(path):
    """Create one directory path when later saves depend on it."""
    os.makedirs(path, exist_ok=True)


def resolve_eval_output_dir(model_path, split_name, eval_name):
    """Reuse the checkpoint's original run folder when possible for manual validation/test runs."""
    run_id = curriculum_run_id_from_path(model_path)
    if run_id is None:
        return fallback_eval_output_dir(eval_name)
    run_paths = build_run_paths(run_id)
    ensure_run_directories(run_paths)
    return stage_output_dir(eval_root_dir(run_paths, split_name), checkpoint_stage_name(model_path), eval_name)


def validation_fixed_maps(args, split_maps):
    """Choose either the quick validation subset or the full held-out validation split."""
    if args.full_validation_fixed:
        return build_full_fixed_eval_maps(split_maps["val_maps"])
    return build_quick_fixed_eval_maps(split_maps["val_maps"])


def create_training_environment(curriculum_teacher, video_dir):
    """Wrap the curriculum env with the existing episode video recorder."""
    recorder = EpisodeVideoRecorder(save_dir=video_dir, fps=5)
    env = createSokobanEnvironment(
        use_shaped_reward=True,
        curriculumTeacher=curriculum_teacher,
        seed=SEED,
    )
    return VideoWrapper(env, recorder)


def build_training_components(run_paths, train_maps, phase_config, focus_maps, focus_map_boost):
    """Create the phase-aware teacher, env, and masked DQN model."""
    teacher = CurriculumTeacher(
        train_maps,
        proceduralFraction=CURRICULUM_DQN_PROCEDURAL_FRACTION,
        phase_config=phase_config,
        focusMaps=focus_maps,
        focusMapBoost=focus_map_boost,
    )
    env = create_training_environment(teacher, run_paths["video_dir"])
    model = createMaskedDQNModel(env, run_paths)
    return teacher, env, model


def build_or_load_model(args, run_paths, train_maps, phase_config, focus_maps):
    """Create a fresh model or load one saved checkpoint for continued training."""
    teacher, env, model = build_training_components(
        run_paths,
        train_maps,
        phase_config,
        focus_maps,
        args.failed_map_boost,
    )
    if args.resume_model is None:
        return teacher, env, model
    loaded_model = MaskedDQN.load(
        args.resume_model,
        env=env,
        tensorboard_log=run_paths["tensorboard_dir"],
    )
    maybe_load_replay_buffer(loaded_model, args.resume_replay_buffer or derived_replay_buffer_path(args.resume_model))
    return teacher, env, loaded_model


def maybe_load_replay_buffer(model, replay_buffer_path):
    """Load one replay buffer when the resume artifacts include it."""
    if replay_buffer_path is None or not os.path.exists(replay_buffer_path):
        return
    model.load_replay_buffer(replay_buffer_path)


def derived_replay_buffer_path(model_path):
    """Return the replay buffer path that matches one saved checkpoint path."""
    if model_path is None:
        return None
    return os.path.splitext(model_path)[0] + "_replay_buffer.pkl"


def build_dataset_splits(args):
    """Load the merged 510 fixed maps and split them into train, val, and test."""
    split_maps = build_fixed_dataset_splits(args.generated_map_json, SEED)
    split_maps["manifest"] = build_split_manifest(split_maps, args.generated_map_json, SEED)
    return split_maps


def write_split_manifest(run_paths, split_maps):
    """Save the deterministic fixed-map split that backs one training run."""
    write_json(run_paths["split_manifest_path"], split_maps["manifest"])


def run_sanity_check(train_maps):
    """Verify macro replay on the simplest fixed training map before training begins."""
    sanity_map = sorted(train_maps, key=lambda item: (len(item["boxes"]), item["map_name"]))[0]
    sanity_result = run_macro_replay_sanity_check(sanity_map)
    if not sanity_result["success"]:
        raise ValueError(f"Sanity check failed on {sanity_result['map_name']}: {sanity_result['failure_reason']}")
    return sanity_result


def load_fixed_demo_payload(args, train_maps):
    """Load or regenerate the fixed-map A* demos for the 340 training maps."""
    payload = load_or_generate_demonstrations(
        train_maps,
        cache_path=args.demo_cache,
        regenerate=args.regenerate_demos,
    )
    if not fixed_demo_cache_matches_train_maps(payload, train_maps):
        payload = load_or_generate_demonstrations(
            train_maps,
            cache_path=args.demo_cache,
            regenerate=True,
        )
    print_demo_summary(payload["summary"])
    return payload


def fixed_demo_cache_matches_train_maps(payload, train_maps):
    """Return True when the cached fixed demos were built from the current train split."""
    if not demo_payload_matches_action_space(payload):
        return False
    cached_names = sorted(result["map_name"] for result in payload["map_results"])
    requested_names = sorted(map_config["map_name"] for map_config in train_maps)
    return cached_names == requested_names


def demo_payload_matches_action_space(payload):
    """Return True when cached demos use the current padded action-mask size."""
    if not payload["samples"]:
        return True
    return len(payload["samples"][0]["action_mask"]) == CURRICULUM_DQN_MAX_BOXES * 4


def should_run_sanity_check(args):
    """Run the A* sanity check only for fresh training launches."""
    return args.resume_model is None


def should_load_fixed_demo_payload(args):
    """Load fixed A* demos for fresh training or explicit A* audit requests."""
    if args.resume_model is None:
        return True
    return bool(args.save_astar_audit_videos or args.show_astar_audit or args.astar_audit_only)


def should_save_run_validation_astar_references(args):
    """Save run-level A* validation references only for fresh training launches."""
    return args.resume_model is None


def load_training_demo_payload(args, train_maps):
    """Load the merged fixed and procedural imitation payload used before RL."""
    fixed_payload = load_fixed_demo_payload(args, train_maps)
    procedural_payload = load_or_generate_procedural_demonstrations(
        CURRICULUM_PROCEDURAL_ENV_IDS,
        cache_path=args.procedural_demo_cache,
        regenerate=args.regenerate_demos,
        target_per_env=args.procedural_demo_target_per_env,
    )
    if not demo_payload_matches_action_space(procedural_payload):
        procedural_payload = load_or_generate_procedural_demonstrations(
            CURRICULUM_PROCEDURAL_ENV_IDS,
            cache_path=args.procedural_demo_cache,
            regenerate=True,
            target_per_env=args.procedural_demo_target_per_env,
        )
    merged_payload = merge_demonstration_payloads(fixed_payload, procedural_payload)
    merged_payload = limit_demonstration_samples(merged_payload, args.max_demonstrations)
    print_demo_summary(merged_payload["summary"])
    validate_demo_payload(merged_payload)
    return merged_payload


def validate_demo_payload(payload):
    """Fail fast when the merged imitation payload is empty or mask-invalid."""
    if not payload["samples"]:
        raise ValueError("No expert demonstrations were collected.")
    if not validate_demonstration_actions(payload["samples"]):
        raise ValueError("At least one expert action is invalid under its saved action mask.")


def resolve_focus_maps(args, split_maps):
    """Load failed fixed maps for resumed curriculum focus when the flag is enabled."""
    if not args.focus_failed_maps:
        return []
    focus_payload = resolve_failed_map_focus(
        split_maps,
        explicit_csv_path=args.failed_map_csv,
        project_root=PROJECT_ROOT,
    )
    print_focus_map_summary(focus_payload, args.failed_map_boost)
    return focus_payload["focus_maps"]


def print_focus_map_summary(focus_payload, focus_map_boost):
    """Print the focus-map source and explain how many maps were matched."""
    if not focus_payload["focus_maps"]:
        print("[Curriculum] Failed-map focus requested, but no failed fixed maps were loaded.")
        return
    summary = focus_payload["summary"]
    print(
        f"[Curriculum] Failed-map focus enabled | csv={focus_payload['csv_path']} | "
        f"matched={summary['matched_fixed_maps']} | boost={focus_map_boost:.2f}"
    )
    print(
        f"[Curriculum] Focus split matches | train={summary['train_split_matches']} | "
        f"val={summary['val_split_matches']} | test={summary['test_split_matches']}"
    )
    if summary["val_split_matches"] or summary["test_split_matches"]:
        print("[Curriculum] Focus includes held-out maps, so later validation/test on those maps is no longer clean.")


def maybe_save_astar_audit_cases(args, train_maps, fixed_demo_payload, run_paths):
    """Save audit videos for selected fixed training maps and return the saved rows."""
    should_save = args.save_astar_audit_videos or args.show_astar_audit or args.astar_audit_only
    if not should_save:
        return []
    saved_rows = save_astar_audit_videos(
        train_maps,
        fixed_demo_payload,
        run_paths["astar_audit_dir"],
        max_success_cases=args.astar_audit_successes,
        show_ui=args.show_astar_audit,
        fps=args.astar_audit_fps,
        search_stride=args.astar_audit_search_stride,
    )
    print(f"Saved A* audit videos to: {run_paths['astar_audit_dir']}")
    return saved_rows


def print_audit_only_summary(run_paths, audit_rows):
    """Print the small audit summary before exiting the audit-only mode."""
    print("A* audit complete")
    print(f"  audit folder: {run_paths['astar_audit_dir']}")
    print(f"  reference folder: {run_paths['astar_reference_dir']}")
    print(f"  saved videos: {len(audit_rows)}")
    print("  next step: run the training command without --astar-audit-only")


def save_validation_astar_reference_videos(run_paths, fixed_maps, split_label):
    """Save one expert-reference video set for the deterministic validation showcase."""
    save_astar_reference_videos(fixed_maps, run_paths["astar_reference_dir"], split_label)


def save_eval_astar_reference_videos(model_path, fixed_maps, split_label, split_name, eval_dir):
    """Save one separate A* reference folder for the checkpoint being inspected."""
    output_dir = eval_astar_reference_dir(model_path, split_name, eval_dir)
    if os.path.exists(astar_reference_summary_path(output_dir)):
        return
    save_astar_reference_videos(fixed_maps, output_dir, split_label)


def pretrain_model_with_demos(model, demo_payload, args):
    """Run behavior cloning on cached expert state-action pairs."""
    return model.behavior_clone_pretrain(
        demo_payload["samples"],
        epochs=args.imitation_epochs,
        batch_size=args.imitation_batch_size,
        learning_rate=args.imitation_learning_rate,
    )


def load_phase_training_demo_payload(args, train_maps, pretrain_payload):
    """Return the demo payload that should keep guiding RL after pretraining ends."""
    if args.skip_imitation:
        return None
    if pretrain_payload is not None:
        return pretrain_payload
    return load_training_demo_payload(args, train_maps)


def phase_demo_guidance_weight(phase_config):
    """Keep the imitation anchor fully on so later phases do not forget earlier skill."""
    return 1.0


def phase_exploration_start_eps(phase_config):
    """Use a gentler exploration bump once the curriculum starts stressing retention."""
    phase_id = int(phase_config["phase_id"])
    if phase_id <= 2:
        return 0.20
    return 0.15


def configure_phase_demo_guidance(model, demo_payload, args, phase_config):
    """Set the expert mini-batch weight used during one curriculum phase."""
    if demo_payload is None:
        model.clear_demo_guidance()
        return 0.0
    loss_weight = phase_demo_guidance_weight(phase_config)
    model.set_demo_guidance(demo_payload["samples"], args.imitation_batch_size, loss_weight)
    return loss_weight


def save_model_artifacts(model, model_path):
    """Save one checkpoint plus its replay buffer sidecar file."""
    model.save(model_path)
    model.save_replay_buffer(model_path + "_replay_buffer.pkl")


def build_phase_paths(run_paths, phase_config):
    """Return the checkpoint and validation paths for one curriculum phase."""
    phase_name = f"phase{phase_config['phase_id']}_{phase_config['phase_name']}"
    return {
        "phase_model_path": os.path.join(run_paths["checkpoints_dir"], f"{phase_name}_end"),
        "best_model_path": os.path.join(run_paths["checkpoints_dir"], f"{phase_name}_best"),
        "snapshot_base_path": os.path.join(run_paths["checkpoints_dir"], phase_name),
        "validation_dir": os.path.join(run_paths["validation_dir"], phase_name),
        "summary_path": os.path.join(run_paths["validation_dir"], f"{phase_name}_summary.json"),
        "training_summary_path": os.path.join(run_paths["validation_dir"], f"{phase_name}_training_summary.json"),
        "periodic_history_path": os.path.join(run_paths["validation_dir"], phase_name, "periodic_eval_history.json"),
        "periodic_type_summary_dir": os.path.join(run_paths["validation_dir"], phase_name, "periodic_type_summaries"),
        "best_summary_path": os.path.join(run_paths["validation_dir"], phase_name, "best_summary.json"),
    }


def build_phase_eval_callback(phase_paths, periodic_fixed_maps, periodic_procedural_specs, phase_config, phase_elapsed_timesteps, teacher):
    """Create the deterministic periodic checkpoint check used during one RL phase."""
    return PeriodicEvalCallback(
        best_model_path=phase_paths["best_model_path"],
        eval_env_factory=lambda: createCurriculumEvalEnvironment(periodic_fixed_maps),
        eval_seed_base=SEED + 20_000 + int(phase_config["phase_id"]) * 1_000,
        eval_freq=CURRICULUM_DQN_EVAL_FREQ,
        n_eval_episodes=len(periodic_fixed_maps),
        early_stop_patience_evals=CURRICULUM_DQN_EARLY_STOP_PATIENCE_EVALS,
        early_stop_min_timesteps=CURRICULUM_DQN_EARLY_STOP_MIN_TIMESTEPS,
        periodic_history_path=phase_paths["periodic_history_path"],
        periodic_type_summary_dir=phase_paths["periodic_type_summary_dir"],
        best_summary_path=phase_paths["best_summary_path"],
        snapshot_base_path=phase_paths["snapshot_base_path"],
        phase_elapsed_timesteps=phase_elapsed_timesteps,
        summary_factory=lambda model: run_periodic_eval(model, periodic_fixed_maps, periodic_procedural_specs),
        summary_consumer=teacher.applyPeriodicValidationSummary,
    )


def build_phase_progress_callback(teacher, phase_config, phase_timesteps, completed_run_timesteps, total_run_timesteps, phase_elapsed_timesteps):
    """Create the progress printer that reports sampled episodes during one phase."""
    return PhaseProgressCallback(
        curriculum_teacher=teacher,
        phase_config=phase_config,
        phase_timesteps=phase_timesteps,
        completed_run_timesteps=completed_run_timesteps,
        total_run_timesteps=total_run_timesteps,
        log_every_timesteps=max(phase_timesteps // 10, 5_000),
        phase_elapsed_timesteps=phase_elapsed_timesteps,
    )


def run_training_phases(args, run_paths, teacher, model, val_maps, demo_payload):
    """Train the selected curriculum phases and save validation at each phase end."""
    phase_configs = select_phase_configs(args)
    quick_val_maps = build_quick_fixed_eval_maps(val_maps)
    total_run_timesteps = total_selected_phase_timesteps(args, phase_configs)
    completed_run_timesteps = 0
    for phase_config in phase_configs:
        phase_timesteps = resolve_phase_timesteps(args, phase_config)
        phase_elapsed_timesteps = resumed_phase_elapsed_timesteps(args, phase_config)
        phase_schedule_timesteps = phase_exploration_timesteps(phase_config, phase_timesteps, phase_elapsed_timesteps)
        teacher.setPhaseConfig(phase_config)
        demo_guidance_weight = configure_phase_demo_guidance(model, demo_payload, args, phase_config)
        start_eps = phase_exploration_start_eps(phase_config)
        model.start_phase_exploration(
            start_eps,
            phase_schedule_timesteps,
            elapsed_phase_timesteps=phase_elapsed_timesteps,
        )
        phase_paths = build_phase_paths(run_paths, phase_config)
        phase_procedural_specs = build_phase_procedural_eval_specs(phase_config, args)
        phase_callback = build_phase_eval_callback(
            phase_paths,
            quick_val_maps,
            phase_procedural_specs,
            phase_config,
            phase_elapsed_timesteps,
            teacher,
        )
        progress_callback = build_phase_progress_callback(
            teacher,
            phase_config,
            phase_timesteps,
            completed_run_timesteps,
            total_run_timesteps,
            phase_elapsed_timesteps,
        )
        model.learn(
            total_timesteps=phase_timesteps,
            callback=CallbackList([progress_callback, phase_callback]),
            reset_num_timesteps=False,
        )
        save_model_artifacts(model, phase_paths["phase_model_path"])
        phase_validation = run_validation_suite(
            phase_paths["phase_model_path"] + ".zip",
            quick_val_maps,
            phase_config["procedural_env_ids"],
            args.procedural_val_episodes_per_env,
            "validation",
            phase_paths["validation_dir"],
        )
        append_phase_history_row(run_paths["run_dir"], phase_config, phase_validation["type_summary"])
        write_json(
            phase_paths["training_summary_path"],
            build_phase_training_summary(
                teacher,
                phase_config,
                phase_timesteps,
                completed_run_timesteps,
                total_run_timesteps,
                demo_guidance_weight,
            ),
        )
        write_json(phase_paths["summary_path"], phase_validation)
        print_phase_completion(phase_config, phase_paths, phase_validation)
        completed_run_timesteps += phase_timesteps


def select_phase_configs(args):
    """Return the requested inclusive range of curriculum phases."""
    phase_configs = build_curriculum_phase_configs()
    selected_configs = []
    for phase_id in range(int(args.start_phase), int(args.stop_after_phase) + 1):
        selected_configs.append(find_phase_config(phase_configs, phase_id))
    return selected_configs


def total_selected_phase_timesteps(args, phase_configs):
    """Return the total timesteps planned for the current curriculum run."""
    return sum(resolve_phase_timesteps(args, phase_config) for phase_config in phase_configs)


def resolve_phase_timesteps(args, phase_config):
    """Use the phase default timesteps unless the caller asked for an override."""
    if args.additional_timesteps is not None:
        return int(args.additional_timesteps)
    return int(phase_config["timesteps"])


def build_phase_training_summary(teacher, phase_config, phase_timesteps, completed_run_timesteps, total_run_timesteps, demo_guidance_weight):
    """Save the main counters that describe how one curriculum phase was sampled."""
    return {
        "phase_id": int(phase_config["phase_id"]),
        "phase_name": str(phase_config["phase_name"]),
        "phase_timesteps": int(phase_timesteps),
        "run_timesteps_before_phase": int(completed_run_timesteps),
        "run_timesteps_after_phase": int(completed_run_timesteps + phase_timesteps),
        "total_run_timesteps": int(total_run_timesteps),
        "demo_guidance_weight": float(demo_guidance_weight),
        "sampling_summary": teacher.phaseProgressSummary(),
    }


def print_phase_completion(phase_config, phase_paths, phase_validation):
    """Print the main outputs once one curriculum phase finishes and validates."""
    print(
        f"[Curriculum] Phase {phase_config['phase_id']} complete | "
        f"checkpoint={phase_paths['phase_model_path']}.zip | "
        f"validation_fixed={phase_validation['fixed']['success_rate']:.3f} | "
        f"validation_procedural={phase_validation['procedural']['success_rate']:.3f}"
    )
    print(f"[Curriculum] Phase {phase_config['phase_id']} validation dir: {phase_paths['validation_dir']}")


def build_procedural_eval_specs(env_ids, episodes_per_env, seed_base):
    """Return one deterministic evaluation spec for every env-id and episode slot."""
    eval_specs = []
    seed_offset = 0
    for env_id in env_ids:
        for episode_index in range(int(episodes_per_env)):
            eval_specs.append({"env_id": env_id, "seed": int(seed_base + seed_offset), "episode_index": episode_index})
            seed_offset += 1
    return eval_specs


def build_phase_procedural_eval_specs(phase_config, args):
    """Return the phase-aware procedural validation set shared by periodic and end checks."""
    return build_procedural_eval_specs(
        phase_config["procedural_env_ids"],
        args.procedural_val_episodes_per_env,
        procedural_seed_base("validation"),
    )


def run_periodic_eval(model, fixed_maps, procedural_specs):
    """Evaluate one in-memory model on the shared phase validation subset."""
    fixed_result = evaluate_fixed_results_with_model(model, fixed_maps)
    procedural_result = evaluate_procedural_results_with_model(model, procedural_specs)
    return periodic_eval_summary(fixed_result, procedural_result, fixed_maps)


def evaluate_fixed_results_with_model(model, map_configs):
    """Return fixed periodic rows, raw results, and summary stats for one checkpoint."""
    rows = []
    results = []
    for map_config in map_configs:
        result = evaluate_fixed_map_with_model(model, map_config)
        rows.append(fixed_result_row(map_config, "periodic", result))
        results.append(result)
    return evaluation_group(rows, results)


def evaluate_fixed_map_with_model(model, map_config):
    """Evaluate one model on one fixed map and return the raw episode result."""
    env = createCurriculumEvalEnvironment([map_config])
    try:
        return evaluate_episode(model, env, log_progress=False)
    finally:
        env.close()


def evaluate_procedural_results_with_model(model, procedural_specs):
    """Return procedural periodic rows, raw results, and summary stats for one checkpoint."""
    rows = []
    results = []
    for procedural_spec in procedural_specs:
        result = evaluate_procedural_spec_with_model(model, procedural_spec)
        rows.append(procedural_result_row(procedural_spec, "periodic", result))
        results.append(result)
    return evaluation_group(rows, results)


def evaluate_procedural_spec_with_model(model, procedural_spec):
    """Evaluate one model on one deterministic procedural environment case."""
    env = createEvaluationEnvironment(env_id=procedural_spec["env_id"], seed=procedural_spec["seed"])
    try:
        return evaluate_episode(model, env, log_progress=False, reset_seed=procedural_spec["seed"])
    finally:
        env.close()


def evaluation_group(rows, results):
    """Bundle rows, raw episode results, and one aggregate summary together."""
    return {"rows": rows, "results": results, "summary": summarize_results(results)}


def periodic_eval_summary(fixed_result, procedural_result, fixed_maps):
    """Combine periodic aggregate metrics with the same type summary used at phase end."""
    merged_results = list(fixed_result["results"]) + list(procedural_result["results"])
    summary = summarize_results(merged_results)
    summary["fixed"] = fixed_result["summary"]
    summary["procedural"] = procedural_result["summary"]
    summary["type_summary"] = build_type_summary(fixed_result["rows"], procedural_result["rows"], fixed_maps)
    return summary


def run_validation_suite(model_path, fixed_maps, env_ids, episodes_per_env, split_label, output_dir):
    """Evaluate one checkpoint on fixed maps and procedural envs, then save both."""
    fixed_result = evaluate_fixed_maps(model_path, fixed_maps, split_label)
    procedural_specs = build_procedural_eval_specs(env_ids, episodes_per_env, procedural_seed_base(split_label))
    procedural_result = evaluate_procedural_specs(model_path, procedural_specs, split_label)
    save_split_evaluation(output_dir, split_label, fixed_result, procedural_result)
    type_summary = write_type_summary(output_dir, fixed_result["rows"], procedural_result["rows"], fixed_maps)
    save_validation_videos(model_path, fixed_maps, procedural_specs, os.path.join(output_dir, "videos"), split_label)
    return {
        "fixed": fixed_result["summary"],
        "procedural": procedural_result["summary"],
        "type_summary": type_summary,
    }


def append_phase_history_row(run_dir, phase_config, type_summary):
    """Append one phase-end type-summary row to the shared curriculum history CSV."""
    run_id = os.path.basename(str(run_dir))
    row = build_phase_history_row(run_id, phase_config, type_summary)
    ensure_directory(os.path.dirname(PHASE_HISTORY_CSV_PATH))
    write_phase_history_row(PHASE_HISTORY_CSV_PATH, row)


def write_phase_history_row(csv_path, row):
    """Write one row to the phase history CSV and add the header when needed."""
    has_header = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=phase_history_columns())
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def procedural_seed_base(split_label):
    """Use different deterministic seed ranges for validation and final test."""
    if str(split_label) == "final_test":
        return SEED + 40_000
    return SEED + 30_000


def evaluate_fixed_maps(model_path, map_configs, split_label):
    """Evaluate one checkpoint on a fixed list of held-out custom map dictionaries."""
    model = load_model(model_path, "dqn")
    rows = []
    results = []
    for map_config in map_configs:
        env = createCurriculumEvalEnvironment([map_config])
        try:
            result = evaluate_episode(model, env, log_progress=False)
        finally:
            env.close()
        rows.append(fixed_result_row(map_config, split_label, result))
        results.append(result)
    return {"rows": rows, "summary": summarize_results(results)}


def fixed_result_row(map_config, split_label, result):
    """Attach fixed-map metadata to one held-out evaluation result."""
    row = dict(result)
    row["model_split"] = str(split_label)
    row["map_name"] = map_config["map_name"]
    row["difficulty"] = map_config.get("difficulty", "unknown")
    row["num_boxes"] = len(map_config["boxes"])
    row["map_source"] = map_config.get("map_source", "unknown")
    return row


def evaluate_procedural_specs(model_path, procedural_specs, split_label):
    """Evaluate one checkpoint on a deterministic list of procedural env episodes."""
    model = load_model(model_path, "dqn")
    rows = []
    results = []
    for procedural_spec in procedural_specs:
        env = createEvaluationEnvironment(env_id=procedural_spec["env_id"], seed=procedural_spec["seed"])
        try:
            result = evaluate_episode(model, env, log_progress=False, reset_seed=procedural_spec["seed"])
        finally:
            env.close()
        rows.append(procedural_result_row(procedural_spec, split_label, result))
        results.append(result)
    return {"rows": rows, "summary": summarize_results(results)}


def procedural_result_row(procedural_spec, split_label, result):
    """Attach env-id metadata to one procedural held-out evaluation result."""
    row = dict(result)
    row["model_split"] = str(split_label)
    row["env_id"] = procedural_spec["env_id"]
    row["map_name"] = f"{procedural_spec['env_id']}_ep_{int(procedural_spec['episode_index']) + 1:03d}"
    row["episode_index"] = int(procedural_spec["episode_index"])
    return row


def save_split_evaluation(output_dir, split_label, fixed_result, procedural_result):
    """Write fixed rows, procedural rows, and one combined summary for a split."""
    os.makedirs(output_dir, exist_ok=True)
    write_csv(os.path.join(output_dir, f"{split_label}_fixed.csv"), fixed_result["rows"])
    write_csv(os.path.join(output_dir, f"{split_label}_procedural.csv"), procedural_result["rows"])
    write_json(
        os.path.join(output_dir, f"{split_label}_summary.json"),
        {"fixed": fixed_result["summary"], "procedural": procedural_result["summary"]},
    )


def write_csv(path, rows):
    """Write a list of dictionaries to CSV using the union of their keys."""
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    """Write one indented JSON file for easy inspection."""
    with open(path, "w") as output_file:
        json.dump(payload, output_file, indent=2)


def build_run_summary(run_paths, split_maps, sanity_result, demo_payload):
    """Build the top-level run summary saved for one imitation-training run."""
    return {
        "run_dir": run_paths["run_dir"],
        "split_manifest_path": run_paths["split_manifest_path"],
        "counts": split_maps["manifest"]["counts"],
        "sanity_check": sanity_result,
        "demo_summary": demo_payload["summary"] if demo_payload is not None else None,
    }


def resolve_eval_model_path(explicit_model_path):
    """Use the caller-provided eval checkpoint or fall back to the latest saved one."""
    if explicit_model_path is not None:
        return explicit_model_path
    patterns = [
        os.path.join(PROJECT_ROOT, "results", "rl_tests", "curriculum_dqn_imitation", "*", "checkpoints", "phase*_end.zip"),
        os.path.join(PROJECT_ROOT, "results", "rl_tests", "curriculum_dqn_imitation", "*", "checkpoints", "post_imitation.zip"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    raise FileNotFoundError("No imitation-learning checkpoint was found. Pass --eval-model <path>.")


def run_validation_only(args, split_maps):
    """Evaluate one checkpoint on the held-out validation splits and save the results."""
    model_path = resolve_eval_model_path(args.eval_model)
    procedural_env_ids = validation_procedural_env_ids(args, model_path)
    eval_dir = resolve_eval_output_dir(model_path, "validation", "validation_only")
    ensure_directory(eval_dir)
    quick_val_maps = validation_fixed_maps(args, split_maps)
    save_eval_astar_reference_videos(model_path, split_maps["val_maps"], "validation", "validation", eval_dir)
    result = run_validation_suite(
        model_path,
        quick_val_maps,
        procedural_env_ids,
        args.procedural_val_episodes_per_env,
        "validation",
        eval_dir,
    )
    print_split_summary("Validation", model_path, result, eval_dir)


def run_final_test_only(args, split_maps):
    """Evaluate one checkpoint on the held-out final-test splits and save the results."""
    model_path = resolve_eval_model_path(args.eval_model)
    procedural_env_ids = procedural_eval_env_ids_for_model(model_path)
    eval_dir = resolve_eval_output_dir(model_path, "final_test", "final_test_only")
    ensure_directory(eval_dir)
    save_eval_astar_reference_videos(model_path, split_maps["test_maps"], "final_test", "final_test", eval_dir)
    result = run_validation_suite(
        model_path,
        split_maps["test_maps"],
        procedural_env_ids,
        args.procedural_test_episodes_per_env,
        "final_test",
        eval_dir,
    )
    print_split_summary("Final test", model_path, result, eval_dir)


def build_eval_output_dir(eval_name):
    """Return one timestamped folder for validation-only or final-test-only runs."""
    return fallback_eval_output_dir(eval_name)


def print_split_summary(label, model_path, result, output_dir):
    """Print the main saved paths plus a small by-type success table for the user."""
    print(f"{label} complete")
    print(f"  model: {model_path}")
    print(f"  fixed success: {result['fixed']['success_rate']:.3f}")
    print(f"  procedural success: {result['procedural']['success_rate']:.3f}")
    print(f"  output dir: {output_dir}")
    print(f"  A* reference dir: {eval_astar_reference_dir(model_path, label.lower().replace(' ', '_'), output_dir)}")
    print_type_success_rows(result["type_summary"])


def print_type_success_rows(type_summary):
    """Print the fixed and procedural success rows in a beginner-friendly format."""
    print("  type summary:")
    for label_name, success_rate in type_success_rows(type_summary):
        print(f"    {label_name}: {success_rate:.3f}")


def type_success_rows(type_summary):
    """Return the fixed and procedural success rows shown in the console summary."""
    return total_fixed_box_rows(type_summary) + fixed_structure_rows(type_summary) + procedural_family_rows(type_summary)


def type_success_row(label_name, success_rate):
    """Bundle one printed type label with its success rate."""
    return str(label_name), float(success_rate)


def total_fixed_box_rows(type_summary):
    """Show total fixed success for 1-box, 2-box, and 3-box maps across all fixed families."""
    return [
        type_success_row("1 box total", total_fixed_box_success(type_summary, 1)),
        type_success_row("2 box total", total_fixed_box_success(type_summary, 2)),
        type_success_row("3 box total", total_fixed_box_success(type_summary, 3)),
    ]


def fixed_structure_rows(type_summary):
    """Show fixed-map success grouped by box count and inner-wall presence."""
    return [
        type_success_row("1 box no inner wall", fixed_structure_success(type_summary, "1box_no_inner_wall")),
        type_success_row("1 box inner wall", fixed_structure_success(type_summary, "1box_inner_wall")),
        type_success_row("2 box no inner wall", fixed_structure_success(type_summary, "2box_no_inner_wall")),
        type_success_row("2 box inner wall", fixed_structure_success(type_summary, "2box_inner_wall")),
        type_success_row("3 box no inner wall", fixed_structure_success(type_summary, "3box_no_inner_wall")),
        type_success_row("3 box inner wall", fixed_structure_success(type_summary, "3box_inner_wall")),
    ]


def procedural_family_rows(type_summary):
    """Show the key procedural validation env success rates in a stable console order."""
    return [
        type_success_row("small-v0", procedural_type_success(type_summary, "Sokoban-small-v0")),
        type_success_row("small-v1", procedural_type_success(type_summary, "Sokoban-small-v1")),
        type_success_row("v0", procedural_type_success(type_summary, "Sokoban-v0")),
    ]


def fixed_type_success(type_summary, family_name):
    """Read one fixed-family success rate and use zero when that family is absent."""
    fixed_summary = type_summary["fixed"]
    family_summary = fixed_summary.get(str(family_name), {})
    return float(family_summary.get("success_rate", 0.0))


def fixed_structure_success(type_summary, group_name):
    """Read one structure-group success rate and use zero when it is absent."""
    structure_summary = type_summary.get("fixed_structure", {})
    group_summary = structure_summary.get(str(group_name), {})
    return float(group_summary.get("success_rate", 0.0))


def procedural_type_success(type_summary, env_id):
    """Read one procedural-env success rate and use zero when that env is absent."""
    procedural_summary = type_summary["procedural"]
    env_summary = procedural_summary.get(str(env_id), {})
    return float(env_summary.get("success_rate", 0.0))


def total_fixed_box_success(type_summary, box_count):
    """Compute full fixed success for one box count using all fixed families in the summary."""
    case_names = fixed_case_names(type_summary, box_count, "case_names")
    failed_case_names = fixed_case_names(type_summary, box_count, "failed_case_names")
    return success_rate_from_case_names(case_names, failed_case_names)


def fixed_case_names(type_summary, box_count, field_name):
    """Collect the fixed case names whose map names match the requested box count."""
    selected_names = []
    for family_summary in type_summary["fixed"].values():
        selected_names.extend(case_names_with_box_count(family_summary, box_count, field_name))
    return selected_names


def case_names_with_box_count(family_summary, box_count, field_name):
    """Keep only the case names whose labels show the requested 1b, 2b, or 3b pattern."""
    selected_names = []
    for case_name in family_summary.get(str(field_name), []):
        if box_count_from_case_name(case_name) == int(box_count):
            selected_names.append(str(case_name))
    return selected_names


def box_count_from_case_name(case_name):
    """Read the box count from the saved case name label and return zero when it is unknown."""
    case_name = str(case_name)
    if "1b_" in case_name:
        return 1
    if "2b_" in case_name:
        return 2
    if "3b_" in case_name:
        return 3
    return 0


def success_rate_from_case_names(case_names, failed_case_names):
    """Convert saved attempted and failed case-name lists into one simple success rate."""
    attempted_cases = len(case_names)
    if attempted_cases == 0:
        return 0.0
    solved_cases = attempted_cases - len(failed_case_names)
    return float(solved_cases) / float(attempted_cases)


def run_training(args, split_maps):
    """Run sanity check, imitation pretraining, phase training, and validation writes."""
    run_id = resolve_training_run_id(args)
    run_paths = build_run_paths(run_id)
    ensure_run_directories(run_paths)
    write_split_manifest(run_paths, split_maps)
    sanity_result = maybe_run_sanity_check(args, split_maps["train_maps"])
    fixed_demo_payload = maybe_load_resume_aware_fixed_demos(args, split_maps["train_maps"])
    audit_rows = maybe_save_resume_aware_astar_audit(args, split_maps["train_maps"], fixed_demo_payload, run_paths)
    maybe_save_resume_aware_astar_references(args, run_paths, split_maps["val_maps"])
    if args.astar_audit_only:
        print_audit_only_summary(run_paths, audit_rows)
        return
    demo_payload = None
    focus_maps = resolve_focus_maps(args, split_maps)
    teacher, env, model = build_or_load_model(
        args,
        run_paths,
        split_maps["train_maps"],
        build_curriculum_phase_configs()[0],
        focus_maps,
    )
    try:
        if args.resume_model is None and not args.skip_imitation:
            demo_payload = load_training_demo_payload(args, split_maps["train_maps"])
            pretrain_model_with_demos(model, demo_payload, args)
            save_model_artifacts(model, run_paths["post_imitation_model_path"])
            run_post_imitation_validation(args, run_paths, split_maps)
        training_demo_payload = load_phase_training_demo_payload(args, split_maps["train_maps"], demo_payload)
        run_training_phases(args, run_paths, teacher, model, split_maps["val_maps"], training_demo_payload)
    finally:
        env.close()
    run_summary = build_run_summary(run_paths, split_maps, sanity_result, demo_payload)
    write_json(run_paths["run_summary_json"], run_summary)
    print_training_summary(run_paths)


def run_post_imitation_validation(args, run_paths, split_maps):
    """Evaluate the post-imitation checkpoint before any RL phase begins."""
    output_dir = os.path.join(run_paths["validation_dir"], "post_imitation")
    quick_val_maps = build_quick_fixed_eval_maps(split_maps["val_maps"])
    run_validation_suite(
        run_paths["post_imitation_model_path"] + ".zip",
        quick_val_maps,
        CURRICULUM_PROCEDURAL_ENV_IDS,
        args.procedural_val_episodes_per_env,
        "validation",
        output_dir,
    )


def print_training_summary(run_paths):
    """Print the most important training outputs for the user."""
    print("Imitation curriculum training complete")
    print(f"  run dir: {run_paths['run_dir']}")
    print(f"  checkpoints: {run_paths['checkpoints_dir']}")
    print(f"  validation summaries: {run_paths['validation_dir']}")
    print(f"  A* reference videos: {run_paths['astar_reference_dir']}")
    print("  final test stays separate and is not run automatically")


def maybe_run_sanity_check(args, train_maps):
    """Skip the A* sanity check when resuming from an existing checkpoint."""
    if not should_run_sanity_check(args):
        return None
    return run_sanity_check(train_maps)


def maybe_load_resume_aware_fixed_demos(args, train_maps):
    """Skip fixed A* demo loading unless this launch still needs it."""
    if not should_load_fixed_demo_payload(args):
        return None
    return load_fixed_demo_payload(args, train_maps)


def maybe_save_resume_aware_astar_audit(args, train_maps, fixed_demo_payload, run_paths):
    """Skip A* audit generation unless fixed demos were loaded for this launch."""
    if fixed_demo_payload is None:
        return []
    return maybe_save_astar_audit_cases(args, train_maps, fixed_demo_payload, run_paths)


def maybe_save_resume_aware_astar_references(args, run_paths, fixed_maps):
    """Skip run-level A* reference videos when resuming an existing checkpoint."""
    if not should_save_run_validation_astar_references(args):
        return
    save_validation_astar_reference_videos(run_paths, fixed_maps, "validation")


def main():
    """Run the requested training, validation-only, final-test-only, or audit flow."""
    args = parse_args()
    set_random_seed(SEED)
    split_maps = build_dataset_splits(args)
    if args.validation_only:
        run_validation_only(args, split_maps)
        return
    if args.final_test_only:
        run_final_test_only(args, split_maps)
        return
    run_training(args, split_maps)


if __name__ == "__main__":
    main()
