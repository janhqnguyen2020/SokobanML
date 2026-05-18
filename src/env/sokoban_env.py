import gym
import gym_sokoban
from src.utils.config import ENV_ID

def initialize_env(env_id=None, seed=None):
    env = gym.make(env_id or ENV_ID, disable_env_checker=True)
    if seed is not None:
        env.seed(seed)
    env.reset(render_mode="tiny_rgb_array")
    return env
