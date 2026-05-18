#For debugging environment and running sokoban simulator
from src.env.sokoban_env import initialize_env
from src.utils.config import MAX_STEPS

env = initialize_env()

observation = env.reset()
print("Reset succeed.")
print("Observation shape:", observation)
frame = env.render(mode="rgb_array")

action = env.action_space.sample()
observation, reward, done, info = env.step(action)
print("Step succeed.")

#### For manual implementing both step limit and time runs out instead of using TimeLimit in wrappers ###
max_steps = MAX_STEPS
step_count = 0
done = False
while not done and step_count < max_steps:
    action = env.action_space.sample()

    observation, reward, done, info = env.step(action)

    step_count += 1

    if step_count >= max_steps:
        print("Reached time limit")
        break
########

env.close()