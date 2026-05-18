from src.planners.bfs import BFSAgent
from src.utils.custom_runner import run_custom_evaluation


run_custom_evaluation(
    planner_class=BFSAgent,
    method_name="BFSAgent",
)