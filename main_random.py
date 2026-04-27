# Random agent test: baseline sanity check
from src.env.sokoban_env import initialize_env
from src.rl.evaluate import run_episode
from src.utils.metrics import compute_metrics
from src.utils.config import NUM_EPISODES

import numpy as np

def random_agent(obs, env):
    return env.action_space.sample()

env = initialize_env()

results = []

for i in range(NUM_EPISODES):
    print("Starting episode: ", i)
    episode_result = run_episode(env, lambda obs: random_agent(obs, env))
    results.append(episode_result)

print("Results: ", results)