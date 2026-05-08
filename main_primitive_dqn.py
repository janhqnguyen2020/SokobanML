"""
Primitive DQN test: for training and evaluating primitive-action DQN
This file trains DQN on the original Sokoban environment actions.
It does not use box-push abstraction yet.
"""

import os
from src.env.sokoban_env import initialize_env
from src.rl.train_primitive_dqn import train_primitive_dqn
from src.rl.evaluate import evaluate_model
from src.utils.config import NUM_EPISODES, ENV_ID


def main():
    env = initialize_env()
    print("=== PRIMITIVE DQN ===")
    model, run_dir = train_primitive_dqn(env)

    # build saved model path
    model_path = os.path.join(run_dir, "dqn_primitive_final.zip")
    output_path = os.path.join(run_dir, "eval_results.csv")

    # evaluate saved model using shared evaluation pipeline
    summary = evaluate_model(
        model_path=model_path,
        algo="dqn",
        level_ids=[ENV_ID],
        n_episodes=NUM_EPISODES,
        output_path=output_path,
)

    print("Evaluation summary:")
    print(summary)
    print("Saved run artifacts to:", run_dir)
    env.close()


if __name__ == "__main__":
    main()
