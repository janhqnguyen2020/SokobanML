# src/rl/high_level_env.py
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

from src.env.custom_env import SimpleCustomSokobanEnv
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
    forced_box_failure,
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
        max_boxes=None,
        map_sampler=None,
    ):
        """
        max_boxes:   Fix the action-space size to max_boxes*4 regardless of how
                     many boxes the current episode has.  Required when training
                     across maps with different box counts so the network input
                     stays constant.  Defaults to the box count of the first env.
        map_sampler: Callable() → None | map_config_dict.  Called at the start
                     of every reset().  None means use the procedural env;
                     a dict is forwarded to SimpleCustomSokobanEnv.
        """
        super().__init__()
        self._map_sampler = map_sampler
        self._procedural_env = env if env is not None else initialize_env()
        self._owns_active_env = False
        self.env = self._procedural_env

        self.num_boxes = self.env.unwrapped.num_boxes
        _max_boxes = max_boxes if max_boxes is not None else self.num_boxes
        self.action_space = Discrete(_max_boxes * 4)

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
        self.capturePrimitiveVideoFrames = False
        self.lastMacroStepFrames = []


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

        # swap underlying env when a map_sampler is configured
        if self._map_sampler is not None:
            self._swap_sampled_env(self._map_sampler())

        # reset all counters used for reward/penalty
        self.invalid_action_streak = 0
        self.no_progress_steps = 0
        self.best_boxes_on_target = 0
        self.state_visit_counts = {}
        self.last_macro_action = None # for adding previous move tracking to avoid RL oscillations 
        self.lastMacroStepFrames = []

        # reset the board until there is at least one valid macro-move
        player_pos, box_positions, goals, _, action_profile = self._reset_to_valid_state()
        self.state_visit_counts = {state_key(player_pos, box_positions): 1}
        self.best_boxes_on_target = count_boxes_on_target(box_positions, goals)
        return self._observation(action_profile)


    def _swap_sampled_env(self, sampled_config):
        """Switch to the sampled fixed map or procedural env for the next episode."""
        self._close_active_env()
        if sampled_config is None:
            self._use_default_procedural_env()
        elif "env_id" in sampled_config:
            self._use_sampled_procedural_env(sampled_config)
        else:
            self._use_sampled_fixed_env(sampled_config)
        self._refresh_env_shape()


    def _close_active_env(self):
        """Close a temporary env created for the previous sampled episode."""
        if self._owns_active_env and self.env is not None:
            self.env.close()
        self._owns_active_env = False


    def _use_default_procedural_env(self):
        """Reuse the default procedural env when no special sample was requested."""
        self.env = self._procedural_env


    def _use_sampled_procedural_env(self, procedural_spec):
        """Create one procedural env for the sampled env id and seed."""
        self.env = initialize_env(
            env_id=procedural_spec["env_id"],
            seed=procedural_spec.get("seed"),
        )
        self._owns_active_env = True


    def _use_sampled_fixed_env(self, map_config):
        """Create one fixed custom env from a sampled map dictionary."""
        self.env = SimpleCustomSokobanEnv(
            height=map_config["height"],
            width=map_config["width"],
            player_position=map_config["player"],
            box_positions=map_config["boxes"],
            goal_positions=map_config["goals"],
            wall_positions=map_config.get("walls", []),
            max_steps=map_config.get("max_steps", 200),
        )
        self._owns_active_env = True


    def _refresh_env_shape(self):
        """Refresh cached box counts and board size after the active env changes."""
        self.num_boxes = self.env.unwrapped.num_boxes
        self.board_shape = tuple(self.env.unwrapped.room_state.shape)


    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)


    def enable_primitive_video_frames(self):
        """Capture one render after each primitive walk or push during macro replay."""
        self.capturePrimitiveVideoFrames = True


    def disable_primitive_video_frames(self):
        """Turn off primitive-step video capture when only fast stepping is needed."""
        self.capturePrimitiveVideoFrames = False
        self.lastMacroStepFrames = []


    def consume_primitive_video_frames(self):
        """Return and clear the primitive-step frames from the last macro action."""
        frames = list(self.lastMacroStepFrames)
        self.lastMacroStepFrames = []
        return frames


    def close(self):
        self._close_active_env()
        self._procedural_env.close()


    def _current_board_profile(self):
        """Read the board and compute the current macro-action choices."""
        return board_profile(
            self.env,
            self.dead_squares,
            self.num_boxes,
            DIRECTION_DELTAS,
        )


    def step(self, action):
        """
        Apply one macro push action and return the next state
        """
        action = int(action)
        _, box_positions, goals, walls, action_profile = self._current_board_profile()
        if not action_profile["selected"]:
            return self._dead_end_result(action_profile) # if there are no valid macro-action, the state is a dead-end
        if action not in action_profile["selected"]:
            return self._invalid_action_result(action_profile) # if agent chose an action that is not valid (eg. lead to dead-square)
        action_data = action_profile["selected"][action]
        return self._step_from_action_data(
            action,
            action_data,
            box_positions,
            goals,
            walls,
            action_profile,
        )


    def step_with_macro_action_data(self, action, action_data):
        """Replay one chosen macro action without trimming it through top-k again."""
        action = int(action)
        _, box_positions, goals, walls, action_profile = self._current_board_profile()
        return self._step_from_action_data(
            action,
            action_data,
            box_positions,
            goals,
            walls,
            action_profile,
        )


    def _step_from_action_data(self, action, action_data, box_positions, goals, walls, action_profile):
        """Apply one caller-supplied macro action and update the env state."""
        self.invalid_action_streak = 0
        self.lastMacroStepFrames = []
        before_progress = progress_snapshot(box_positions, goals) 
        raw_reward, env_done, env_info = self._execute_macro_action(action_data)
        reverse_move = self._is_reverse_move(action_data)
        self.last_macro_action = action_data
        next_player_pos, next_box_positions, goals, _, next_action_profile = self._current_board_profile()
        after_progress = progress_snapshot(next_box_positions, goals)
        solvability_damage = forced_box_failure(
            box_positions,
            next_box_positions,
            goals,
            walls,
            DIRECTION_DELTAS
        )

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
            reverse_move,
            solvability_damage
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


    def _is_reverse_move(self, current_action_data):
        """Return True when the new push undoes the last push on the same box."""
        if self.last_macro_action is None:
            return False
        opposite = {1: 2, 2: 1, 3: 4, 4: 3}
        previous_box = self.last_macro_action["box_index"]
        previous_dir = self.last_macro_action["direction"]
        current_box = current_action_data["box_index"]
        current_dir = current_action_data["direction"]
        return previous_box == current_box and opposite.get(previous_dir) == current_dir


    def _reset_to_valid_state(self):
        for _ in range(RESET_RETRY_LIMIT):
            self.env.reset()
            player_pos, box_positions, goals, walls, action_profile = self._current_board_profile()
            self.dead_squares = precompute_dead_squares(walls, goals, self.board_shape)
            player_pos, box_positions, goals, walls, action_profile = self._current_board_profile()
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
            self._capture_primitive_video_frame()
            if done:
                return total_reward, done, info
        _, reward, done, info = self.env.step(action_data["direction"])
        total_reward += float(reward)
        self._capture_primitive_video_frame()
        return total_reward, done, info


    def _capture_primitive_video_frame(self):
        """Save one rgb frame when primitive-step video capture is active."""
        if not self.capturePrimitiveVideoFrames:
            return
        self.lastMacroStepFrames.append(self.render(mode="rgb_array"))


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
