# Planning (BFS/Greedy test): for testing planning algorithms and belongs in experiment script only
from src.env.sokoban_env import initialize_env
from src.planners.planner_runner import run_experiments
from src.planners.bfs import bfs_policy
from src.planners.greedy import greedy_policy


env = initialize_env()
print("=== BFS ===")
bfs_metrics = run_experiments(env, bfs_policy)
print(bfs_metrics)
env.close()

env = initialize_env()
print("\n=== GREEDY BFS ===")
greedy_metrics = run_experiments(env, greedy_policy)
print(greedy_metrics)
env.close()