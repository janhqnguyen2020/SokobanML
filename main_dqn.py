# src/main_dqn.py
# DQN test: for training and evaluating pipeline
from src.env.sokoban_env import initialize_env
from src.rl.train_dqn import train
from src.rl.evaluate import run_episode
from src.utils.config import NUM_EPISODES
from src.rl.hint_wrapper import HintWrapper

env = initialize_env()
env = HintWrapper(env)

model = train(env)

def agent(obs):
    return model.predict(obs)[0]

results = []
for i in range(NUM_EPISODES):
    print("Starting episode: ", i)
    results.append(run_episode(env, agent))

print(results)