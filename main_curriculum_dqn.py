# main_curriculum_dqn.py
"""Entry point for the curriculum / generalized DQN experiment.

Trains a new DQN model that generalises across:
  - 1-box maps  (Easy)
  - 2-box maps  (Medium)
  - 3-box maps  (Hard)
  - procedural Sokoban-small-v1 maps (40% of episodes)

Results are saved to results/rl_tests/curriculum_dqn/<run>/  and do NOT
affect Shizuka's existing high_level_dqn results.

Usage:
    python main_curriculum_dqn.py
"""

import argparse
import json
import logging
import os
import shutil

from src.rl.evaluate import evaluate_episode, evaluate_model, load_model, summarize_results
from src.rl.callbacks import summary_rank
from src.rl.train_curriculum_dqn import (
    createCurriculumEvalEnvironment as make_curriculum_eval_env,
    createEvaluationEnvironment as make_eval_env,
    train,
)
from src.utils.config import CURRICULUM_DQN_SELECTION_EPISODES
from src.utils.generated_maps import (
    build_generated_1box_maps, build_generated_2box_maps, build_generated_3box_maps,
    build_walled_1box_maps, build_walled_2box_maps, build_walled_3box_maps,
)

def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _make_episode_seeds(base_seed, n):
    return [base_seed + i for i in range(n)]


def _copy_eval_artifacts(src, dst):
    for suffix in ("", ".replace('.csv', '.json')", "_summary.json"):
        s = src if suffix == "" else src.replace(".csv", suffix.strip("\"'"))
        d = dst if suffix == "" else dst.replace(".csv", suffix.strip("\"'"))
        if os.path.exists(s):
            shutil.copyfile(s, d)


def _evaluate_checkpoint(model_path, output_path, episode_seeds, env_factory):
    from src.utils.config import CURRICULUM_DQN_SELECTION_EPISODES as N
    return evaluate_model(
        model_path=model_path,
        algo="dqn",
        level_ids=["curriculum_eval"],
        n_episodes=N,
        output_path=output_path,
        env_factory=env_factory,
        episode_seeds=episode_seeds,
    )


def _select_checkpoint(run_dir, final_path, best_path, seeds, env_factory):
    final_out = os.path.join(run_dir, "eval_results_final.csv")
    final_summary = _evaluate_checkpoint(final_path, final_out, seeds, env_factory)

    selected = {
        "label": "final",
        "model_path": final_path,
        "output_path": final_out,
        "summary": final_summary,
    }
    best_summary = None

    if os.path.exists(best_path):
        best_out = os.path.join(run_dir, "eval_results_best.csv")
        best_summary = _evaluate_checkpoint(best_path, best_out, seeds, env_factory)
        if summary_rank(best_summary) > summary_rank(final_summary):
            selected = {
                "label": "best",
                "model_path": best_path,
                "output_path": best_out,
                "summary": best_summary,
            }

    return selected, final_summary, best_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--init-model",
        default=None,
        metavar="PATH",
        help="Warm-start from an existing .zip checkpoint (weights only, replay buffer resets).",
    )
    args = parser.parse_args()

    _configure_logging()
    print("=== CURRICULUM DQN ===")
    if args.init_model:
        print(f"Warm-starting from: {args.init_model}")

    _, run_dir = train(init_model_path=args.init_model)

    final_path = os.path.join(run_dir, "curriculum_dqn_final.zip")
    best_path  = os.path.join(run_dir, "curriculum_dqn_best.zip")
    seeds = _make_episode_seeds(30_000, CURRICULUM_DQN_SELECTION_EPISODES)

    # Full training pool — 1-box through 3-box, with and without interior walls
    curriculum_maps = (
        build_generated_1box_maps() + build_walled_1box_maps() +
        build_generated_2box_maps() + build_walled_2box_maps() +
        build_generated_3box_maps() + build_walled_3box_maps()
    )

    # evaluate on procedural small-v1 (same distribution as Shizuka's baseline)
    selected, final_summary, best_summary = _select_checkpoint(
        run_dir, final_path, best_path, seeds, make_eval_env
    )

    canonical = os.path.join(run_dir, "eval_results.csv")
    shutil.copyfile(selected["output_path"], canonical)
    for ext in (".json", "_summary.json"):
        src = selected["output_path"].replace(".csv", ext)
        if os.path.exists(src):
            shutil.copyfile(src, canonical.replace(".csv", ext))

    # also evaluate on all curriculum maps — one env reused across all episodes
    # so the map cycler advances (map 0, 1, 2 ...) instead of resetting each time
    curriculum_eval_out = os.path.join(run_dir, "eval_results_curriculum.csv")
    curriculum_model = load_model(selected["model_path"], "dqn")
    curr_env = make_curriculum_eval_env(curriculum_maps)
    try:
        curr_results = [evaluate_episode(curriculum_model, curr_env, log_progress=False)
                        for _ in range(len(curriculum_maps))]
    finally:
        curr_env.close()
    curriculum_summary = summarize_results(curr_results)

    with open(os.path.join(run_dir, "selected_model.json"), "w") as f:
        json.dump(
            {
                "selected_label": selected["label"],
                "selected_model_path": selected["model_path"],
                "procedural_eval_summary": selected["summary"],
                "curriculum_eval_summary": curriculum_summary,
                "final_model_path": final_path,
                "final_summary": final_summary,
                "best_model_path": best_path if os.path.exists(best_path) else None,
                "best_summary": best_summary,
            },
            f,
            indent=2,
        )

    print("\nProcedural eval summary:", selected["summary"])
    print("Curriculum eval summary:", curriculum_summary)
    print("Selected checkpoint:", selected["model_path"])
    print("Run artifacts saved to:", run_dir)


if __name__ == "__main__":
    main()
