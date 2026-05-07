# src/planners/planner_runner.py
"""
runs agents, collects results, and computes metrics for evaluation
"""

import time#measure how long each episode took

from src.planners.bfs import BFSAgent
from src.planners.greedy import GreedyAgent

from src.utils.metrics import compute_metrics
from src.utils.config import NUM_EPISODES

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
def run_experiments(env, policy_function, number_episodes=NUM_EPISODES):
    results = []

    #repeat episode multiple times to get average performance metrics, since some levels may have random elements or variability in solution paths
    for i in range(number_episodes):
        print("Starting episode: ", i)
        results.append(run_episode(env, policy_function))
    return compute_metrics(results)
