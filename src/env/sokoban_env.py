import gym
import gym_sokoban
from src.utils.config import ENV_ID, MAX_STEPS

def initialize_env():
    env = gym.make(ENV_ID, disable_env_checker=True)
    env.unwrapped.max_steps = MAX_STEPS
    env.reset(render_mode="tiny_rgb_array")
    return env
