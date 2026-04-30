# src/planners/planner_runner.py
# Member C — Joseph Nguyen
#
# Entry point for running classical planners across a set of Sokoban levels.
#
# This script ties together bfs.py, greedy.py, and deadlock.py to run
# batch experiments and collect metrics for the final comparison with RL agents.
#
# Responsibilities:
#   - Load a list of levels (from Gymnasium or custom level files)
#   - Run each planner (BFS, Greedy BFS) on each level with a timeout
#   - Record per-level results: solved?, steps taken, runtime (ms), nodes expanded
#   - Save results to results/planner_results.csv via logger.py
#
# Functions to implement:
#   - load_levels(level_dir)
#       → returns list of level configs (board layout, wall/goal/box positions)
#
#   - run_planner(planner_fn, env, timeout_sec)
#       → calls planner_fn(env), enforces time limit
#       → returns dict: {solved, steps, runtime_ms, nodes_expanded}
#
#   - run_all(planners, levels, output_path)
#       → outer loop over planners × levels
#       → calls run_planner() for each combination
#       → appends rows to results CSV
#
#   - main()
#       → CLI entry: parse args (--levels, --output, --timeout)
#       → call run_all() with BFS and Greedy BFS
#       → print summary table to stdout
#
# Usage (intended):
#   python planner_runner.py --levels data/levels/ --output results/planner_results.csv
#
# Notes:
#   - Use signal or threading for timeouts (Windows-safe: use threading)
#   - Metrics schema matches src/utils/metrics.py definitions

import time
from src.planners.bfs import BFSAgent
from src.planners.greedy import GreedyAgent
from src.utils.metrics import compute_metrics
from src.utils.config import NUM_EPISODES

def run_episode(env, policy_function):
    observation = env.reset()

    #tell the agent a new epsiode started so it clears its cached plan
    if hasattr(policy_function, 'reset'):
        policy_function.reset()

    done = False
    
    total_reward = 0
    steps = 0
    start_time = time.time()

    while not done:
        action = policy_function(observation)
        observation, reward, done, info = env.step(action)

        total_reward += reward
        steps += 1
    
    elapsed_ms = (time.time() - start_time) * 1000

    return total_reward, steps, elapsed_ms

def run_experiments(env, policy_function, number_episodes=NUM_EPISODES):
    results = []

    for i in range(number_episodes):
        print("Starting episode: ", i)
        results.append(run_episode(env, policy_function))

    return compute_metrics(results)