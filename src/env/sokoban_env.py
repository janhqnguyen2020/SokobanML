import gym
import gym_sokoban
from src.utils.config import ENV_ID

def initialize_env():
    env = gym.make(ENV_ID, disable_env_checker=True) 
    #env = gym.wrappers.TimeLimit(env, max_episode_steps=300) #TimeLimit use the new Gymnasium which requires 5 parameters, while we are using the old gym which only have 4 parameters. Therefore, it is an incompatible issue.
    
    # Reset the environment with the "tiny_rgb_array" render mode to have a resized 84x84 observation from 160x160
    env.reset(render_mode="tiny_rgb_array")
    return env
