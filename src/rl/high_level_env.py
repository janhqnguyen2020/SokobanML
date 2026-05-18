"""
Higher-level Sokoban wrapper for DQN.

One macro action means:
1. choose a box
2. choose a push direction for that box
3. walk behind the box
4. execute one real push
"""

import random
import gym
import numpy as np
from gym.spaces import Box, Discrete

from src.env.sokoban_env import initialize_env
from src.planners.deadlock import precompute_dead_squares
from src.rl.high_level_env_parts.action_profile import board_profile
from src.rl.high_level_env_parts.observation import encode_observation
from src.rl.high_level_env_parts.constants import (
    DEAD_END_PENALTY,
    DIRECTION_DELTAS,
    INVALID_ACTION_STREAK_LIMIT,
    NO_PROGRESS_STEP_LIMIT,
    REPEATED_STATE_LIMIT,
    RESET_RETRY_LIMIT,
)
from src.rl.high_level_env_parts.state import (
    count_boxes_on_target,
    progress_snapshot,
    read_board,
    record_state_visit,
    state_key,
)
from src.rl.high_level_env_parts.transition import (
    dead_end_info,
    done_flags,
    invalid_action_info,
    select_reward,
    step_info,
    update_no_progress,
)


class HighLevelSokobanEnv(gym.Env):
    """
    High-level Sokoban environment where each action is one push
    """

    def __init__(
        self,
        env=None,
        observation_board_shape=None,
        use_extra_scalar_features=False,
        use_shaped_reward=True,
    ):
        super().__init__()
        self.env = env if env is not None else initialize_env()
        self.num_boxes = self.env.unwrapped.num_boxes
        self.action_space = Discrete(self.num_boxes * 4)
        self.board_shape = tuple(self.env.unwrapped.room_state.shape)
        self.observation_board_shape = tuple(observation_board_shape or self.board_shape)
        self.use_extra_scalar_features = bool(use_extra_scalar_features)
        self.use_shaped_reward = bool(use_shaped_reward)

        if any(canvas < actual for canvas, actual in zip(self.observation_board_shape, self.board_shape)):
            raise ValueError(f"Observation canvas {self.observation_board_shape} cannot be smaller than board {self.board_shape}")

        self.scalar_feature_size = 10 if self.use_extra_scalar_features else 0
        board_feature_size = int(np.prod(self.observation_board_shape) * 4)
        observation_size = board_feature_size + self.scalar_feature_size + self.action_space.n
        self.observation_space = Box(low=0.0, high=1.0, shape=(observation_size,), dtype=np.float32)
        self.dead_squares = set()
        self.invalid_action_streak = 0
        self.no_progress_steps = 0
        self.best_boxes_on_target = 0
        self.state_visit_counts = {}


    def seed(self, seed=None):
        """
        Seed wrapper, ensure compatibility with original Sokoban env
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return self.env.seed(seed)


    def reset(self, **kwargs):
        """
        Reset until the generator gives us a board with a valid macro move (eg. not dead-square)
        """
        seed = kwargs.pop("seed", None)
        kwargs.pop("options", None)
        if seed is not None:
            self.seed(seed)

        # reset all counters used for reward/penality
        self.invalid_action_streak = 0
        self.no_progress_steps = 0
        self.best_boxes_on_target = 0
        self.state_visit_counts = {}
        
        # reset the board until there is at least one valid macro-move
        player_pos, box_positions, goals, _, action_profile = self._reset_to_valid_state()
        self.state_visit_counts = {state_key(player_pos, box_positions): 1} # mark the starting state as visited
        self.best_boxes_on_target = count_boxes_on_target(box_positions, goals) # count how many boxes are on goals initially for later comparison used to determine penalities/rewards
        return self._observation(action_profile)


    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


    def close(self):
        self.env.close()


    def step(self, action):
        """
        Apply one macro push action and return the next state
        """
        action = int(action)
        _, box_positions, goals, _, action_profile = board_profile(self.env, self.dead_squares, self.num_boxes, DIRECTION_DELTAS)
        if not action_profile["selected"]:
            return self._dead_end_result(action_profile) # if there are no valid macro-action, the state is a dead-end
        if action not in action_profile["selected"]:
            return self._invalid_action_result(action_profile) # if agent chose an action that is not valid (eg. lead to dead-square)
        self.invalid_action_streak = 0

        # log current board, execute next move, and log new board
        before_progress = progress_snapshot(box_positions, goals) 
        raw_reward, env_done, env_info = self._execute_macro_action(action_profile["selected"][action]) # raw reward is the reward in original Sokoban env (eg. -0.1 for each step)
        next_player_pos, next_box_positions, goals, _, next_action_profile = board_profile(self.env, self.dead_squares, self.num_boxes, DIRECTION_DELTAS)
        after_progress = progress_snapshot(next_box_positions, goals)

        # compute additional rewards
        self.best_boxes_on_target = max(self.best_boxes_on_target, after_progress["boxes_on_target"])
        self.no_progress_steps, box_progress_delta, distance_progress = update_no_progress(self.no_progress_steps, before_progress, after_progress)
        state_visit_count = record_state_visit(self.state_visit_counts, next_player_pos, next_box_positions)
        solved = after_progress["boxes_on_target"] == self.num_boxes
        done, repeated_state, no_progress, dead_end = done_flags(env_done, solved, state_visit_count, next_action_profile, self.no_progress_steps)
        returned_reward = select_reward(
            self.use_shaped_reward,
            raw_reward,
            box_progress_delta,
            distance_progress,
            solved,
            repeated_state,
            no_progress,
            dead_end,
            state_visit_count,
        )

        # log for observational purposes
        info = step_info(
            action,
            action_profile,
            next_action_profile,
            after_progress,
            raw_reward,
            returned_reward,
            box_progress_delta,
            distance_progress,
            solved,
            repeated_state,
            no_progress,
            dead_end,
            state_visit_count,
            self.no_progress_steps,
            self.best_boxes_on_target,
            env_info,
        )

        return self._observation(next_action_profile), returned_reward, done, info


    def _reset_to_valid_state(self):
        for _ in range(RESET_RETRY_LIMIT):
            self.env.reset()
            player_pos, box_positions, goals, walls, action_profile = board_profile(self.env, self.dead_squares, self.num_boxes, DIRECTION_DELTAS)
            self.dead_squares = precompute_dead_squares(walls, goals, self.board_shape)
            player_pos, box_positions, goals, walls, action_profile = board_profile(self.env, self.dead_squares, self.num_boxes, DIRECTION_DELTAS)
            if action_profile["selected"]:
                return player_pos, box_positions, goals, walls, action_profile  # return promising states if available
        return player_pos, box_positions, goals, walls, action_profile  # if all retry fails, return the state even if it leads to dead-end or no valid move


    def _execute_macro_action(self, action_data):
        """
        Run the walk sequence first, then the final push action
        """
        total_reward = 0.0
        done = False
        info = {}
        for primitive_action in action_data["walk_actions"]:
            _, reward, done, info = self.env.step(primitive_action)
            total_reward += float(reward)
            if done:
                return total_reward, done, info
        _, reward, done, info = self.env.step(action_data["direction"])
        total_reward += float(reward)
        return total_reward, done, info


    def _invalid_action_result(self, action_profile):
        _, box_positions, goals, _ = read_board(self.env)
        self.invalid_action_streak += 1
        boxes_on_target = count_boxes_on_target(box_positions, goals)
        info, done, reward = invalid_action_info(action_profile, boxes_on_target, self.best_boxes_on_target, self.invalid_action_streak)
        return self._observation(action_profile), reward, done, info


    def _dead_end_result(self, action_profile):
        _, box_positions, goals, _ = read_board(self.env)
        boxes_on_target = count_boxes_on_target(box_positions, goals)
        info = dead_end_info(action_profile, boxes_on_target, self.best_boxes_on_target)
        return self._observation(action_profile), DEAD_END_PENALTY, True, info


    def _observation(self, action_profile):
        """
        Used for logging, observational purposes
        """
        return encode_observation(
            self.env,
            action_profile,
            self.observation_board_shape,
            self.use_extra_scalar_features,
            self.num_boxes,
            self.action_space.n,
            self.best_boxes_on_target,
            self.no_progress_steps,
            self.invalid_action_streak,
            self.state_visit_counts,
            NO_PROGRESS_STEP_LIMIT,
            INVALID_ACTION_STREAK_LIMIT,
            REPEATED_STATE_LIMIT,
        )
