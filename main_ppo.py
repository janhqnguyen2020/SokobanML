"""
PPO experiment entrypoint (MlpPolicy + HintWrapper)
"""

import os
from src.rl.train_ppo import train
from src.rl.evaluate import evaluate_model
from src.utils.config import NUM_EPISODES, ENV_ID


def main():
    print("=== PPO (MlpPolicy + HintWrapper) ===")
    _, run_dir = train()

    model_path  = os.path.join(run_dir, "ppo_final.zip")
    output_path = os.path.join(run_dir, "eval_results.csv")

    summary = evaluate_model(
        model_path=model_path,
        algo="ppo",
        level_ids=[ENV_ID],
        n_episodes=NUM_EPISODES,
        output_path=output_path,
    )

    print("Evaluation summary:", summary)
    print("Artifacts saved to:", run_dir)


if __name__ == "__main__":
    main()
