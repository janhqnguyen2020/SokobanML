#src/rl/hint_wrapper.py
"""
Augments RL observations with structured reasoning signals.
Uses room_state + room_fixed (compact integer grids) instead of raw RGB pixels,
so MlpPolicy gets a small, information-dense input instead of a 76k-element
pixel vector that would require a CNN and a massive replay buffer.

Observation layout (flat float32 vector):
  [room_state flat | room_fixed flat | hint_vector]
  e.g. 10x10 board: 100 + 100 + 40 = 240 values
"""
import gym
import numpy as np
from gym import spaces
from src.planners.reasoning import ReasoningPlanner

MAX_HINT_SIZE = 40  # 10 boxes * 4 values (box_row, box_col, goal_row, goal_col)


class HintWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reasoning = ReasoningPlanner(env)

        room_state = env.unwrapped.room_state
        board_flat  = int(np.prod(room_state.shape))  # e.g. 100 for a 10x10 board

        # obs = room_state flat + room_fixed flat + hint vector
        total_size = board_flat * 2 + MAX_HINT_SIZE

        self.observation_space = spaces.Box(
            low=0, high=10, shape=(total_size,), dtype=np.float32
        )

    def reset(self):
        self.env.reset()
        return self._augment()

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        return self._augment(), reward, done, info

    def _augment(self):
        room_state = self.env.unwrapped.room_state
        room_fixed = self.env.unwrapped.room_fixed

        boxes = [tuple(p) for p in np.argwhere((room_state == 3) | (room_state == 4))]
        goals = [tuple(p) for p in np.argwhere(room_fixed == 2)]

        hint_vector = []
        if boxes and goals:
            assignment = self.reasoning.plan(boxes, goals)
            for box, goal in assignment.items():
                hint_vector.extend([box[0], box[1], goal[0], goal[1]])

        while len(hint_vector) < MAX_HINT_SIZE:
            hint_vector.append(0)

        return np.concatenate([
            room_state.flatten().astype(np.float32),
            room_fixed.flatten().astype(np.float32),
            np.array(hint_vector[:MAX_HINT_SIZE], dtype=np.float32),
        ])
