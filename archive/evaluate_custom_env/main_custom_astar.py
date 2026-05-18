from src.planners.astar import AStarAgent
from src.utils.custom_runner import run_custom_evaluation


def main():
    """Run A* across the fixed custom benchmark maps."""
    run_custom_evaluation(
        planner_class=AStarAgent,
        method_name="AStarAgent",
    )


if __name__ == "__main__":
    main()