"""Entry point for curriculum DQN training with A* imitation pretraining."""

from src.rl.train_curriculum_dqn_with_imitation import main


def run():
    """Launch the imitation-training command-line entry point."""
    main()


if __name__ == "__main__":
    run()
