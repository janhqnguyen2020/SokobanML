"""Train curriculum DQN with A* imitation pretraining and comparison support."""

import argparse
import csv
import glob
import json
import os
from datetime import datetime

from stable_baselines3.common.utils import set_random_seed

from src.rl.callbacks import PeriodicEvalCallback, summary_rank
from src.rl.evaluate import evaluate_episode, load_model, summarize_results
from src.rl.imitation import (
    DEFAULT_DEMO_CACHE_PATH,
    DEFAULT_HARD_CASE_CACHE_PATH,
    build_imitation_curriculum_maps,
    failed_map_configs,
    limit_demonstration_samples,
    load_or_generate_demonstrations,
    merge_demonstration_payloads,
    print_demo_summary,
    run_macro_replay_sanity_check,
    save_demonstration_payload,
    validate_demonstration_actions,
    generate_demonstration_payload,
)
from src.rl.imitation_audit import save_astar_audit_videos
from src.rl.train_curriculum_dqn import (
    CurriculumTeacher,
    createCurriculumEvalEnvironment,
    createEvaluationEnvironment,
    createMaskedDQNModel,
    createSokobanEnvironment,
)
from src.rl.video_recorder import EpisodeVideoRecorder
from src.rl.video_wrapper import VideoWrapper
from src.utils.config import (
    CURRICULUM_DQN_EARLY_STOP_MIN_TIMESTEPS,
    CURRICULUM_DQN_EARLY_STOP_PATIENCE_EVALS,
    CURRICULUM_DQN_EVAL_EPISODES,
    CURRICULUM_DQN_EVAL_FREQ,
    CURRICULUM_DQN_PROCEDURAL_FRACTION,
    CURRICULUM_DQN_SELECTION_EPISODES,
    CURRICULUM_DQN_TOTAL_STEPS,
    SEED,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def parse_args():
    """Parse command-line options for imitation pretraining and evaluation."""
    parser = argparse.ArgumentParser(description="Train curriculum DQN with A* imitation")
    parser.add_argument("--demo-cache", default=DEFAULT_DEMO_CACHE_PATH)
    parser.add_argument("--hard-case-cache", default=DEFAULT_HARD_CASE_CACHE_PATH)
    parser.add_argument("--baseline-model", default=None)
    parser.add_argument("--imitation-epochs", type=int, default=8)
    parser.add_argument("--imitation-batch-size", type=int, default=64)
    parser.add_argument("--imitation-learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-demonstrations", type=int, default=None)
    parser.add_argument("--procedural-eval-episodes", type=int, default=CURRICULUM_DQN_SELECTION_EPISODES)
    parser.add_argument("--curriculum-eval-repeats", type=int, default=1)
    parser.add_argument("--regenerate-demos", action="store_true")
    parser.add_argument("--use-hard-case-cache", action="store_true")
    parser.add_argument("--refresh-hard-case-cache", action="store_true")
    parser.add_argument("--include-handmade-curriculum", action="store_true")
    parser.add_argument("--sanity-check-only", action="store_true")
    parser.add_argument("--epsilon-reset-on-stage-change", action="store_true")
    parser.add_argument("--save-astar-audit-videos", action="store_true")
    parser.add_argument("--show-astar-audit", action="store_true")
    parser.add_argument("--astar-audit-only", action="store_true")
    parser.add_argument("--astar-audit-successes", type=int, default=5)
    parser.add_argument("--astar-audit-fps", type=int, default=4)
    parser.add_argument("--astar-audit-search-stride", type=int, default=50)
    return parser.parse_args()


def build_run_paths(run_id):
    """Build the artifact paths for one imitation-training run."""
    run_dir = os.path.join(PROJECT_ROOT, "results", "rl_tests", "curriculum_dqn_imitation", run_id)
    return {
        "run_dir": run_dir,
        "tensorboard_dir": os.path.join(run_dir, "tensorboard"),
        "video_dir": os.path.join(run_dir, "videos"),
        "astar_audit_dir": os.path.join(run_dir, "astar_audit"),
        "model_path": os.path.join(run_dir, "curriculum_dqn_imitation_final"),
        "best_model_path": os.path.join(run_dir, "curriculum_dqn_imitation_best"),
        "procedural_eval_csv": os.path.join(run_dir, "eval_results_procedural.csv"),
        "curriculum_eval_csv": os.path.join(run_dir, "eval_results_curriculum.csv"),
        "comparison_csv": os.path.join(run_dir, "comparison_summary.csv"),
        "run_summary_json": os.path.join(run_dir, "run_summary.json"),
    }


def ensure_run_directories(run_paths):
    """Create the folders needed for models, videos, and evaluation files."""
    os.makedirs(run_paths["run_dir"], exist_ok=True)
    os.makedirs(run_paths["tensorboard_dir"], exist_ok=True)
    os.makedirs(run_paths["video_dir"], exist_ok=True)
    os.makedirs(run_paths["astar_audit_dir"], exist_ok=True)


def create_training_environment(curriculum_teacher, video_dir):
    """Build the video-wrapped curriculum training environment for imitation runs."""
    recorder = EpisodeVideoRecorder(save_dir=video_dir, fps=5)
    env = createSokobanEnvironment(
        use_shaped_reward=True,
        curriculumTeacher=curriculum_teacher,
        seed=SEED,
    )
    return VideoWrapper(env, recorder)


def procedural_episode_seeds(num_episodes):
    """Build reproducible seeds for fair procedural-model comparison."""
    return [30_000 + episode_index for episode_index in range(int(num_episodes))]


def run_sanity_check(curriculum_maps):
    """Verify that one simple map can be solved and replayed as macro actions."""
    sanity_map = sorted(curriculum_maps, key=lambda item: (len(item["boxes"]), item["map_name"]))[0]
    sanity_result = run_macro_replay_sanity_check(sanity_map)
    if not sanity_result["success"]:
        raise ValueError(f"Sanity check failed on {sanity_result['map_name']}: {sanity_result['failure_reason']}")
    return sanity_result


def maybe_warn_about_epsilon_reset(args):
    """Explain that stage-reset epsilon still needs a dedicated DQN schedule hook."""
    if args.epsilon_reset_on_stage_change:
        print("TODO: epsilon stage resets are not implemented yet.")
        print("TODO: the clean hook is MaskedDQN._on_step() so schedule resets stay separate from imitation.")


def load_training_demonstrations(args, curriculum_maps):
    """Load, regenerate, merge, and validate the imitation dataset."""
    payload = load_or_generate_demonstrations(
        curriculum_maps,
        cache_path=args.demo_cache,
        regenerate=args.regenerate_demos,
    )
    if args.use_hard_case_cache and os.path.exists(args.hard_case_cache):
        hard_case_payload = load_or_generate_demonstrations([], cache_path=args.hard_case_cache, regenerate=False)
        payload = merge_demonstration_payloads(payload, hard_case_payload)
    payload = limit_demonstration_samples(payload, args.max_demonstrations)
    print_demo_summary(payload["summary"])
    if not validate_demonstration_actions(payload["samples"]):
        raise ValueError("At least one expert action is invalid under its saved action mask.")
    if not payload["samples"]:
        raise ValueError("No expert demonstrations were collected.")
    return payload


def maybe_save_astar_audit_cases(args, curriculum_maps, demo_payload, run_paths):
    """Save a small visual audit set for selected A* success and failure cases."""
    should_save = (
        args.save_astar_audit_videos
        or args.regenerate_demos
        or args.show_astar_audit
        or args.astar_audit_only
    )
    if not should_save:
        return []
    saved_rows = save_astar_audit_videos(
        curriculum_maps,
        demo_payload,
        run_paths["astar_audit_dir"],
        max_success_cases=args.astar_audit_successes,
        show_ui=args.show_astar_audit,
        fps=args.astar_audit_fps,
        search_stride=args.astar_audit_search_stride,
    )
    print(f"Saved A* audit videos to: {run_paths['astar_audit_dir']}")
    return saved_rows


def print_audit_only_summary(run_paths, audit_rows):
    """Print the key audit-only outputs before returning early."""
    print("A* audit complete")
    print(f"  audit folder: {run_paths['astar_audit_dir']}")
    print(f"  saved videos: {len(audit_rows)}")
    print("  next step: run the training command without --astar-audit-only")


def create_training_components(run_paths, curriculum_maps):
    """Build the curriculum teacher, environment, and masked DQN model."""
    curriculum_teacher = CurriculumTeacher(
        curriculum_maps,
        proceduralFraction=CURRICULUM_DQN_PROCEDURAL_FRACTION,
    )
    env = create_training_environment(curriculum_teacher, run_paths["video_dir"])
    model = createMaskedDQNModel(env, run_paths)
    return curriculum_teacher, env, model


def pretrain_model_with_demos(model, demo_payload, args):
    """Run behavior cloning on cached A* demonstrations before RL begins."""
    return model.behavior_clone_pretrain(
        demo_payload["samples"],
        epochs=args.imitation_epochs,
        batch_size=args.imitation_batch_size,
        learning_rate=args.imitation_learning_rate,
    )


def train_model(model, curriculum_maps, run_paths):
    """Continue from behavior cloning into the normal curriculum DQN phase."""
    eval_callback = PeriodicEvalCallback(
        best_model_path=run_paths["best_model_path"],
        eval_env_factory=lambda: createCurriculumEvalEnvironment(curriculum_maps),
        eval_seed_base=SEED + 10_000,
        eval_freq=CURRICULUM_DQN_EVAL_FREQ,
        n_eval_episodes=CURRICULUM_DQN_EVAL_EPISODES,
        early_stop_patience_evals=CURRICULUM_DQN_EARLY_STOP_PATIENCE_EVALS,
        early_stop_min_timesteps=CURRICULUM_DQN_EARLY_STOP_MIN_TIMESTEPS,
    )
    model.learn(
        total_timesteps=CURRICULUM_DQN_TOTAL_STEPS,
        callback=eval_callback,
        reset_num_timesteps=True,
    )
    model.save(run_paths["model_path"])


def select_checkpoint(run_paths, procedural_eval_episodes):
    """Choose between the final and best imitation checkpoints using procedural eval."""
    seeds = procedural_episode_seeds(procedural_eval_episodes)
    final_path = run_paths["model_path"] + ".zip"
    best_path = run_paths["best_model_path"] + ".zip"
    final_eval = evaluate_procedural_model(final_path, seeds)
    selected = {"label": "final", "model_path": final_path, "summary": final_eval["summary"]}
    if os.path.exists(best_path):
        best_eval = evaluate_procedural_model(best_path, seeds)
        if summary_rank(best_eval["summary"]) > summary_rank(final_eval["summary"]):
            selected = {"label": "best", "model_path": best_path, "summary": best_eval["summary"]}
    return selected


def evaluate_procedural_model(model_path, seeds):
    """Evaluate one checkpoint on procedural Sokoban-small-v1 episodes."""
    model = load_model(model_path, "dqn")
    results = []
    for reset_seed in seeds:
        env = createEvaluationEnvironment()
        try:
            results.append(evaluate_episode(model, env, log_progress=False, reset_seed=reset_seed))
        finally:
            env.close()
    return {"rows": procedural_rows(results), "summary": summarize_results(results)}


def procedural_rows(results):
    """Convert procedural evaluation episodes into CSV-friendly rows."""
    rows = []
    for episode_index, result in enumerate(results):
        row = dict(result)
        row["model_split"] = "procedural"
        row["map_name"] = f"procedural_ep_{episode_index:03d}"
        rows.append(row)
    return rows


def evaluate_curriculum_model(model_path, curriculum_maps, repeats):
    """Evaluate one checkpoint on the fixed curriculum maps used during training."""
    model = load_model(model_path, "dqn")
    rows = []
    results = []
    for map_config in curriculum_maps:
        for repeat_index in range(int(repeats)):
            env = createCurriculumEvalEnvironment([map_config])
            try:
                result = evaluate_episode(model, env, log_progress=False)
            finally:
                env.close()
            rows.append(curriculum_row(map_config, repeat_index, result))
            results.append(result)
    return {"rows": rows, "summary": summarize_results(results)}


def curriculum_row(map_config, repeat_index, result):
    """Attach fixed-map metadata to one curriculum evaluation result."""
    row = dict(result)
    row["model_split"] = "curriculum"
    row["map_name"] = map_config["map_name"]
    row["difficulty"] = map_config.get("difficulty", "unknown")
    row["num_boxes"] = len(map_config["boxes"])
    row["repeat_index"] = int(repeat_index)
    return row


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
    """Write one JSON artifact with indentation for readability."""
    with open(path, "w") as output_file:
        json.dump(payload, output_file, indent=2)


def compare_model_summaries(run_paths, baseline_result, imitation_result):
    """Save a flat CSV that makes baseline versus imitation easy to compare."""
    rows = [comparison_row("baseline_curriculum_dqn", baseline_result), comparison_row("curriculum_dqn_with_imitation", imitation_result)]
    rows = [row for row in rows if row is not None]
    write_csv(run_paths["comparison_csv"], rows)
    return rows


def comparison_row(label, evaluation_result):
    """Flatten one model summary into a single comparison row."""
    if evaluation_result is None:
        return None
    row = {"model_label": label}
    row.update(flat_summary("procedural", evaluation_result["procedural"]["summary"]))
    row.update(flat_summary("curriculum", evaluation_result["curriculum"]["summary"]))
    return row


def flat_summary(prefix, summary):
    """Prefix summary keys so multiple splits fit in one CSV row."""
    flat = {}
    for key, value in summary.items():
        if isinstance(value, dict):
            flat[f"{prefix}_{key}"] = json.dumps(value, sort_keys=True)
        else:
            flat[f"{prefix}_{key}"] = value
    return flat


def evaluate_selected_model(model_path, curriculum_maps, run_paths, procedural_eval_episodes, curriculum_eval_repeats, file_prefix):
    """Run both procedural and curriculum evaluation for one chosen checkpoint."""
    seeds = procedural_episode_seeds(procedural_eval_episodes)
    procedural_result = evaluate_procedural_model(model_path, seeds)
    curriculum_result = evaluate_curriculum_model(model_path, curriculum_maps, curriculum_eval_repeats)
    procedural_csv = run_paths["procedural_eval_csv"].replace(".csv", f"_{file_prefix}.csv")
    curriculum_csv = run_paths["curriculum_eval_csv"].replace(".csv", f"_{file_prefix}.csv")
    write_csv(procedural_csv, procedural_result["rows"])
    write_csv(curriculum_csv, curriculum_result["rows"])
    return {"procedural": procedural_result, "curriculum": curriculum_result}


def find_latest_baseline_model(explicit_path):
    """Resolve the baseline checkpoint path from either CLI input or result folders."""
    if explicit_path:
        return explicit_path
    patterns = [
        os.path.join(PROJECT_ROOT, "results", "rl_tests", "curriculum_dqn", "*", "curriculum_dqn_best.zip"),
        os.path.join(PROJECT_ROOT, "results", "rl_tests", "curriculum_dqn", "*", "curriculum_dqn_final.zip"),
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


def maybe_refresh_hard_case_cache(args, curriculum_maps, imitation_curriculum_result):
    """Rebuild the optional hard-case cache from maps the imitation model still fails."""
    if not args.refresh_hard_case_cache:
        return None
    failed_maps = failed_map_configs(curriculum_maps, imitation_curriculum_result["curriculum"]["rows"])
    hard_case_payload = generate_demonstration_payload(failed_maps)
    save_demonstration_payload(args.hard_case_cache, hard_case_payload)
    return hard_case_payload["summary"]


def write_run_summary(run_paths, selected_checkpoint, sanity_result, bc_result, demo_payload, baseline_model_path, baseline_result, imitation_result, hard_case_summary, astar_audit_rows):
    """Save one JSON artifact that records the full imitation-training outcome."""
    payload = {
        "selected_checkpoint": selected_checkpoint,
        "sanity_check": sanity_result,
        "behavior_cloning": bc_result,
        "demo_summary": demo_payload["summary"],
        "astar_audit_dir": run_paths["astar_audit_dir"],
        "astar_audit_cases": astar_audit_rows,
        "baseline_model_path": baseline_model_path,
        "baseline_result": baseline_result,
        "imitation_result": imitation_result,
        "hard_case_summary": hard_case_summary,
    }
    write_json(run_paths["run_summary_json"], payload)


def print_training_summary(run_paths, selected_checkpoint, baseline_model_path, baseline_result, imitation_result):
    """Print the most important outputs and where they were saved."""
    print("Imitation training complete")
    print(f"  run dir: {run_paths['run_dir']}")
    print(f"  A* audit videos: {run_paths['astar_audit_dir']}")
    print(f"  selected checkpoint: {selected_checkpoint['model_path']}")
    print(f"  procedural success: {imitation_result['procedural']['summary']['success_rate']:.3f}")
    print(f"  curriculum success: {imitation_result['curriculum']['summary']['success_rate']:.3f}")
    if baseline_model_path and baseline_result is not None:
        print(f"  baseline compared: {baseline_model_path}")
        print(f"  comparison csv: {run_paths['comparison_csv']}")


def run_training_stage(args, run_paths, curriculum_maps, demo_payload):
    """Create the model, pretrain it on demos, and continue with RL."""
    _, env, model = create_training_components(run_paths, curriculum_maps)
    try:
        bc_result = pretrain_model_with_demos(model, demo_payload, args)
        train_model(model, curriculum_maps, run_paths)
    finally:
        env.close()
    return bc_result


def evaluate_models_for_run(args, curriculum_maps, run_paths, selected_checkpoint):
    """Evaluate the imitation run and the optional baseline checkpoint."""
    imitation_result = evaluate_selected_model(
        selected_checkpoint["model_path"],
        curriculum_maps,
        run_paths,
        args.procedural_eval_episodes,
        args.curriculum_eval_repeats,
        "imitation",
    )
    baseline_model_path = find_latest_baseline_model(args.baseline_model)
    baseline_result = evaluate_baseline_if_available(
        baseline_model_path,
        curriculum_maps,
        run_paths,
        args.procedural_eval_episodes,
        args.curriculum_eval_repeats,
    )
    return baseline_model_path, baseline_result, imitation_result


def train_with_imitation(args):
    """Run the full imitation-pretrain, RL-train, evaluate, and compare pipeline."""
    set_random_seed(SEED)
    maybe_warn_about_epsilon_reset(args)
    run_id = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')}_seed{SEED}"
    run_paths = build_run_paths(run_id)
    ensure_run_directories(run_paths)
    curriculum_maps = build_imitation_curriculum_maps(args.include_handmade_curriculum)
    sanity_result = run_sanity_check(curriculum_maps)
    if args.sanity_check_only:
        print(json.dumps(sanity_result, indent=2))
        return {"run_paths": run_paths, "sanity_result": sanity_result}
    demo_payload = load_training_demonstrations(args, curriculum_maps)
    audit_rows = maybe_save_astar_audit_cases(args, curriculum_maps, demo_payload, run_paths)
    if args.astar_audit_only:
        print_audit_only_summary(run_paths, audit_rows)
        return {"run_paths": run_paths, "astar_audit_rows": audit_rows}
    bc_result = run_training_stage(args, run_paths, curriculum_maps, demo_payload)
    selected_checkpoint = select_checkpoint(run_paths, args.procedural_eval_episodes)
    baseline_model_path, baseline_result, imitation_result = evaluate_models_for_run(
        args,
        curriculum_maps,
        run_paths,
        selected_checkpoint,
    )
    compare_model_summaries(run_paths, baseline_result, imitation_result)
    hard_case_summary = maybe_refresh_hard_case_cache(args, curriculum_maps, imitation_result)
    write_run_summary(
        run_paths,
        selected_checkpoint,
        sanity_result,
        bc_result,
        demo_payload,
        baseline_model_path,
        baseline_result,
        imitation_result,
        hard_case_summary,
        audit_rows,
    )
    print_training_summary(run_paths, selected_checkpoint, baseline_model_path, baseline_result, imitation_result)
    return {"run_paths": run_paths, "selected_checkpoint": selected_checkpoint, "astar_audit_rows": audit_rows}


def evaluate_baseline_if_available(baseline_model_path, curriculum_maps, run_paths, procedural_eval_episodes, curriculum_eval_repeats):
    """Evaluate the baseline curriculum DQN when a checkpoint is available."""
    if baseline_model_path is None:
        return None
    return evaluate_selected_model(
        baseline_model_path,
        curriculum_maps,
        run_paths,
        procedural_eval_episodes,
        curriculum_eval_repeats,
        "baseline",
    )


def main():
    """Run the imitation-training pipeline from the command line."""
    train_with_imitation(parse_args())


if __name__ == "__main__":
    main()
