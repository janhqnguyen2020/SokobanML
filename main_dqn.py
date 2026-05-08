"""
Higher-level DQN experiment entrypoint
"""

import os
from src.rl.train_dqn import train
from src.rl.evaluate import evaluate_model
from src.utils.config import NUM_EPISODES, ENV_ID
from src.rl.high_level_env import HighLevelSokobanEnv


def main():
    print("=== HIGH-LEVEL DQN ===")
    _, run_dir = train()
    model_path = os.path.join(run_dir, "high_level_dqn_final.zip")
    output_path = os.path.join(run_dir, "eval_results.csv")
    summary = evaluate_model(
        model_path=model_path,
        algo="dqn",
        level_ids=[ENV_ID],
        n_episodes=NUM_EPISODES,
        output_path=output_path,
        env_factory=HighLevelSokobanEnv,
    )

    print("Evaluation summary:")
    print(summary)
    print("Saved run artifacts to:", run_dir)


if __name__ == "__main__":
    main()
