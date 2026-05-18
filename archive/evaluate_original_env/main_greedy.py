from src.env.sokoban_env import initialize_env
from src.planners.greedy import GreedyAgent
from src.planners.planner_runner import run_experiments


def main():
    """Run Greedy search on the original Sokoban environment."""
    env = initialize_env()
    metrics = run_experiments(
        env=env,
        policy_function=GreedyAgent(env),
        method_name="GreedyAgent",
        show_ui=True,
        delay=0.5,
    )
    print(metrics)
    env.close()


if __name__ == "__main__":
    main()