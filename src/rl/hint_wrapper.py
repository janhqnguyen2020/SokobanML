#src/rl/hint_wrapper.py
"""
Augments RL observations with structured reasoning signals.
Uses room_state + room_fixed (compact integer grids) instead of raw RGB pixels,
so MlpPolicy gets a small, information-dense input instead of a 76k-element
pixel vector that would require a CNN and a massive replay buffer.

Observation layout (flat float32 vector):
  [room_state flat | room_fixed flat | hint_vector (40) | semantic_features (13)]
  e.g. 10x10 board: 100 + 100 + 40 + 13 = 253 values

  hint_vector:       10 boxes x (box_row, box_col, goal_row, goal_col)
  semantic_features: [num_on_goal, num_nearly_placed, num_deadlocked,
                      box_0_flag, ..., box_9_flag]

Reward shaping (applied every step):
  +SHAPE_SCALE * (prev_dist - new_dist)    reward progress toward goals
  -DEADLOCK_PENALTY per new deadlocked box  punish irreversible mistakes
"""
import gym
import numpy as np
from gym import spaces
from src.planners.reasoning import ReasoningPlanner
from src.planners.heuristics import manhattan
from src.planners.backward_hints import BackwardHintGenerator
from src.planners.deadlock import (
    precompute_dead_squares,
    is_corner_deadlock,
    is_freeze_deadlock,
)

MAX_HINT_SIZE    = 40   # 10 boxes x 4 values (box_row, box_col, goal_row, goal_col)
SEMANTIC_SIZE    = 13   # num_on_goal + num_nearly_placed + num_deadlocked + 10 per-box flags
SHAPE_SCALE      = 1.0  # reward per unit of manhattan improvement
DEADLOCK_PENALTY = 0.05 # penalty each time a new box becomes deadlocked
BOX_ON_GOAL_BONUS = 10.0 # bonus each time a new box lands on a goal


class HintWrapper(gym.Wrapper):
    def __init__(self, env, fixed_level=False, level_seed=42):
        super().__init__(env)
        self.reasoning = ReasoningPlanner(env)
        self.backward_hints = BackwardHintGenerator()
        self._fixed_level = fixed_level
        self._level_seed = level_seed

        room_state = env.unwrapped.room_state
        board_flat = int(np.prod(room_state.shape))

        total_size = board_flat * 2 + MAX_HINT_SIZE + SEMANTIC_SIZE
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(total_size,), dtype=np.float32
        )

        self._dead_squares = None
        self._prev_total_dist = 0.0
        self._prev_deadlocked = 0
        self._prev_on_goal = 0

    def reset(self):
        if self._fixed_level:
            np.random.seed(self._level_seed)
        self.env.reset()
        self._dead_squares = self._compute_dead_squares()
        self._prev_total_dist = self._total_box_goal_dist()
        self._prev_deadlocked = self._count_deadlocked()
        self._prev_on_goal = self._count_on_goal()
        return self._augment()

    def step(self, action):
        _, reward, done, info = self.env.step(action)

        new_dist = self._total_box_goal_dist()
        new_deadlocked = self._count_deadlocked()
        new_on_goal = self._count_on_goal()

        shaped = reward
        shaped += SHAPE_SCALE * (self._prev_total_dist - new_dist)
        if new_deadlocked > self._prev_deadlocked:
            shaped -= DEADLOCK_PENALTY * (new_deadlocked - self._prev_deadlocked)
        if new_on_goal > self._prev_on_goal:
            shaped += BOX_ON_GOAL_BONUS * (new_on_goal - self._prev_on_goal)

        self._prev_total_dist = new_dist
        self._prev_deadlocked = new_deadlocked
        self._prev_on_goal = new_on_goal

        return self._augment(), shaped, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_dead_squares(self):
        """Precompute all squares where a box can never reach any goal."""
        room_fixed = self.env.unwrapped.room_fixed
        walls = frozenset(
            (int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 0)
        )
        goals = [(int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 2)]
        return precompute_dead_squares(walls, goals, room_fixed.shape)

    def _get_boxes_goals_walls(self):
        room_state = self.env.unwrapped.room_state
        room_fixed = self.env.unwrapped.room_fixed
        boxes = frozenset(
            (int(p[0]), int(p[1])) for p in np.argwhere((room_state == 3) | (room_state == 4))
        )
        goals = frozenset(
            (int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 2)
        )
        walls = frozenset(
            (int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 0)
        )
        return boxes, goals, walls

    def _count_deadlocked(self):
        """Count boxes in an irreversible position that are not on a goal."""
        if self._dead_squares is None:
            return 0
        boxes, goals, walls = self._get_boxes_goals_walls()
        count = 0
        for box in boxes - goals:
            if (box in self._dead_squares
                    or is_corner_deadlock(box, walls)
                    or is_freeze_deadlock(box, boxes, walls)):
                count += 1
        return count

    def _count_on_goal(self):
        """Count how many boxes are currently sitting on a goal."""
        boxes, goals, _ = self._get_boxes_goals_walls()
        return len(boxes & goals)

    def _total_box_goal_dist(self):
        """Sum of manhattan distances from each box to its assigned goal."""
        room_state = self.env.unwrapped.room_state
        room_fixed = self.env.unwrapped.room_fixed
        boxes = [(int(p[0]), int(p[1])) for p in np.argwhere((room_state == 3) | (room_state == 4))]
        goals = [(int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 2)]
        if not boxes or not goals:
            return 0.0
        assignment = self.reasoning.plan(boxes, goals)
        return float(sum(manhattan(b, g) for b, g in assignment.items()))

    def _augment(self):
        room_state = self.env.unwrapped.room_state
        room_fixed = self.env.unwrapped.room_fixed
        boxes, goals, walls = self._get_boxes_goals_walls()
        boxes_list = sorted(boxes)

        # --- hint vector: box-goal assignments ---
        hint_vector = []
        if boxes_list and goals:
            assignment = self.reasoning.plan(boxes_list, list(goals))
            for box, goal in assignment.items():
                hint_vector.extend([box[0], box[1], goal[0], goal[1]])
        while len(hint_vector) < MAX_HINT_SIZE:
            hint_vector.append(0)

        # --- semantic features ---
        num_on_goal = len(boxes & goals)
        num_nearly_placed = self.backward_hints.compute_overlap(boxes_list, list(goals))

        per_box_flags = []
        if self._dead_squares is not None:
            for box in boxes_list[:10]:
                is_dl = (
                    box in self._dead_squares
                    or is_corner_deadlock(box, walls)
                    or is_freeze_deadlock(box, boxes, walls)
                )
                per_box_flags.append(float(is_dl))
        while len(per_box_flags) < 10:
            per_box_flags.append(0.0)

        num_deadlocked = sum(per_box_flags)
        semantic = np.array(
            [float(num_on_goal), float(num_nearly_placed), float(num_deadlocked)] + per_box_flags,
            dtype=np.float32,
        )

        return np.concatenate([
            room_state.flatten().astype(np.float32),
            room_fixed.flatten().astype(np.float32),
            np.array(hint_vector[:MAX_HINT_SIZE], dtype=np.float32),
            semantic,
        ])
