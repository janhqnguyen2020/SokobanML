from src.planners.greedy import GreedyAgent
from src.utils.custom_runner import run_custom_evaluation


run_custom_evaluation(
    planner_class=GreedyAgent,
    method_name="GreedyAgent",
)