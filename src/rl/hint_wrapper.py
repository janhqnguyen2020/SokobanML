#src/rl/hint_wrapper.py
"""
This wrapper augments RL observations with structured reasoning signals.
DQN/PPO will see: Original board + box-goal reasoning + structural hints instead of raw pixels only. (box -> goal pairing)
This will help DQN/PPO learn faster by reducing relational reasoning burden.
"""
import gym
import numpy as np

from src.planners.reasoning import ReasoningPlanner
from gym import spaces

class HintWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reasoning = ReasoningPlanner(env)
        original_shape = env.observation_space.shape
        flat_size = int(np.prod(original_shape))

        #estimate max hint size
        #4 numbers per box-goal pair
        max_boxes = 10
        hint_size = max_boxes * 4

        total_size = flat_size + hint_size

        self.observation_space = spaces.Box(low=0, high=255, shape=(total_size,), dtype=np.float32)

    def reset(self):
        observation = self.env.reset()
        return self._augment(observation)
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        return (self._augment(observation), reward, done, info)
    
    def _augment(self, obs):
        room_state = self.env.unwrapped.room_state
        room_fixed = self.env.unwrapped.room_fixed

        boxes = [tuple(p) for p in np.argwhere((room_state == 3) | (room_state == 4))]
        goals = [tuple(p) for p in np.argwhere(room_fixed == 2)]

        assignment = self.reasoning.plan(boxes, goals)
        hint_vector = []

        for box, goal in assignment.items():
            hint_vector.extend([box[0], box[1], goal[0], goal[1]])

        #fixed-size padding
        max_hint_size = 40

        while len(hint_vector) < max_hint_size:
            hint_vector.append(0)

        hint_vector = np.array(hint_vector, dtype=np.float32)

        obs_flat = np.array(obs, dtype=np.float32).flatten()

        return np.concatenate([obs_flat, hint_vector]) 