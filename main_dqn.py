# main_dqn.py
import json
import logging
import os
import shutil
from src.rl.callbacks import summary_rank
from src.rl.evaluate import evaluate_model
from src.rl.train_dqn import make_eval_env, train
from src.utils.config import ENV_ID, HIGH_LEVEL_DQN_SELECTION_EPISODES

def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

def _make_episode_seeds(base_seed, n_episodes):
    return [base_seed + episode_idx for episode_idx in range(n_episodes)]

def _copy_eval_artifacts(source_csv_path, target_csv_path):
    shutil.copyfile(source_csv_path, target_csv_path)
    shutil.copyfile(source_csv_path.replace(".csv", ".json"), target_csv_path.replace(".csv", ".json"))
    shutil.copyfile(
        source_csv_path.replace(".csv", "_summary.json"),
        target_csv_path.replace(".csv", "_summary.json"),
    )

def _evaluate_checkpoint(model_path, output_path, episode_seeds):
    return evaluate_model(
        model_path=model_path,
        algo="dqn",
        level_ids=[ENV_ID],
        n_episodes=HIGH_LEVEL_DQN_SELECTION_EPISODES,
        output_path=output_path,
        env_factory=make_eval_env,
        episode_seeds=episode_seeds,
    )

def _select_checkpoint(run_dir, final_model_path, best_model_path, episode_seeds):
    final_output_path = os.path.join(run_dir, "eval_results_final.csv")
    final_summary = _evaluate_checkpoint(final_model_path, final_output_path, episode_seeds)
    selected_label = "final"
    selected_model_path = final_model_path
    selected_output_path = final_output_path
    selected_summary = final_summary
    best_summary = None
    if os.path.exists(best_model_path):
        best_output_path = os.path.join(run_dir, "eval_results_best.csv")
        best_summary = _evaluate_checkpoint(best_model_path, best_output_path, episode_seeds)
        if summary_rank(best_summary) > summary_rank(final_summary):
            selected_label = "best"
            selected_model_path = best_model_path
            selected_output_path = best_output_path
            selected_summary = best_summary
    return {
        "selected_label": selected_label,
        "selected_model_path": selected_model_path,
        "selected_output_path": selected_output_path,
        "selected_summary": selected_summary,
        "final_model_path": final_model_path,
        "final_summary": final_summary,
        "best_model_path": best_model_path if os.path.exists(best_model_path) else None,
        "best_summary": best_summary,
    }

def main():
    _configure_logging()
    print("=== HIGH-LEVEL DQN ===")
    _, run_dir = train()
    final_model_path = os.path.join(run_dir, "high_level_dqn_final.zip")
    best_model_path = os.path.join(run_dir, "high_level_dqn_best.zip")
    episode_seeds = _make_episode_seeds(20_000, HIGH_LEVEL_DQN_SELECTION_EPISODES)
    selection = _select_checkpoint(run_dir, final_model_path, best_model_path, episode_seeds)
    canonical_output_path = os.path.join(run_dir, "eval_results.csv")
    _copy_eval_artifacts(selection["selected_output_path"], canonical_output_path)
    with open(os.path.join(run_dir, "selected_model.json"), "w") as f:
        json.dump(
            {
                "selected_label": selection["selected_label"],
                "selected_model_path": selection["selected_model_path"],
                "selected_summary": selection["selected_summary"],
                "final_model_path": selection["final_model_path"],
                "final_summary": selection["final_summary"],
                "best_model_path": selection["best_model_path"],
                "best_summary": selection["best_summary"],
            },
            f,
            indent=2,
        )
    print("Evaluation summary:")
    print(selection["selected_summary"])
    print("Selected checkpoint:", selection["selected_model_path"])
    print("Saved run artifacts to:", run_dir)

if __name__ == "__main__":
    main()
