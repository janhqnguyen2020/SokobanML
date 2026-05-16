# src/planners/planner_runner.py
"""
runs agents, collects results, and computes metrics for evaluation
"""

import time#measure how long each episode took
import os

from src.planners.bfs import BFSAgent
from src.planners.greedy import GreedyAgent

from src.utils.metrics import compute_metrics
from src.utils.config import NUM_EPISODES, ENV_ID

from src.utils.logger import init_csv, append_row

PLANNER_FIELDS = [ 'method', 'env_version', 'episode', 'solved', 'total_reward',
                  'steps', 'runtime_ms', 'avg_runtime_ms',
                  'nodes_expanded', 'deadlocks_pruned', 'dead_squares',
                  'pruning_rate']

#runs one full game of Sokoban with the given agent, returns results for that episode
def run_episode(env, policy_function):
    observation = env.reset()#resets environment to initial state, gets first observation

    #checks if agent has reset() method and calls it to clear any internal state before starting episode
    if hasattr(policy_function, 'reset'):
        policy_function.reset()

    #init tracking variables for episode
    done = False
    total_reward = 0
    steps = 0
    start_time = time.time()

    #keep playing until game finishes
    while not done:
        action = policy_function(observation)#get action from agent based on current observation
        observation, reward, done, info = env.step(action)#applies action to environment, gets new observation and reward
        
        total_reward += reward
        steps += 1

    elapsed_ms = (time.time() - start_time) * 1000

    nodes_expanded   = getattr(policy_function, 'nodes_expanded',   0)#how many states explored
    deadlocks_pruned = getattr(policy_function, 'deadlocks_pruned', 0)#how many states skipped
    dead_squares     = getattr(policy_function, 'dead_squares_count', 0)#number of bad positions

    return total_reward, steps, elapsed_ms, nodes_expanded, deadlocks_pruned, dead_squares

#runs multiple episodes and computes aggregate metrics for the agent's performance
def run_experiments(env, policy_function, number_episodes=NUM_EPISODES, output_path = 'results/planner_results.csv', method_name = None, env_version = ENV_ID):
    if method_name is None:
        method_name = type(policy_function).__name__

    if not os.path.exists(output_path):
        init_csv(output_path, fieldnames = PLANNER_FIELDS)

    results = []
    cumulative_time = 0.0

    #repeat episode multiple times to get average performance metrics, since some levels may have random elements or variability in solution paths
    for i in range(number_episodes):
        print("Starting episode: ", i)
        result = run_episode(env, policy_function)
        results.append(result)

        total_reward, steps, elapsed_ms, nodes_expanded, deadlocks_pruned, dead_squares = result

        cumulative_time += elapsed_ms
        avg_runtime_ms = cumulative_time / (i + 1)
        pruning_rate = round(deadlocks_pruned / nodes_expanded, 4) if nodes_expanded > 0 else 0.0


        append_row(output_path, {
            'method': method_name,
            'env_version': env_version,
            'episode': i,
            'solved': total_reward > 0, #assuming positive reward means level was solved
            'total_reward': round(total_reward, 4),
            'steps': steps,
            'runtime_ms': round(elapsed_ms, 4),
            'avg_runtime_ms': round(avg_runtime_ms, 4),
            'nodes_expanded': nodes_expanded,
            'deadlocks_pruned': deadlocks_pruned,
            'dead_squares': dead_squares,
            'pruning_rate': pruning_rate
        })

    return compute_metrics(results)