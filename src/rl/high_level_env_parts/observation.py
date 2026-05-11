"""Observation-building helpers for the high-level Sokoban environment."""

import numpy as np

from src.rl.high_level_env_parts.state import count_boxes_on_target, read_board, state_key, sum_min_goal_distance


def pad_layer(layer, observation_board_shape):
    """Center one board layer on the configured observation canvas."""
    if tuple(layer.shape) == observation_board_shape:
        return layer

    padded = np.zeros(observation_board_shape, dtype=np.float32)
    row_offset = (observation_board_shape[0] - layer.shape[0]) // 2
    col_offset = (observation_board_shape[1] - layer.shape[1]) // 2
    padded[row_offset : row_offset + layer.shape[0], col_offset : col_offset + layer.shape[1]] = layer
    return padded


def observation_layers(room_state, room_fixed, observation_board_shape):
    """Build the four padded board layers used by the high-level agent."""
    return [
        pad_layer((room_fixed == 0).astype(np.float32), observation_board_shape),
        pad_layer((room_fixed == 2).astype(np.float32), observation_board_shape),
        pad_layer(np.isin(room_state, [3, 4]).astype(np.float32), observation_board_shape),
        pad_layer(np.isin(room_state, [5, 6]).astype(np.float32), observation_board_shape),
    ]


def build_action_mask(action_space_n, selected_actions):
    """Build the binary valid-action mask stored at the observation tail."""
    action_mask = np.zeros(action_space_n, dtype=np.float32)
    for action in selected_actions:
        action_mask[int(action)] = 1.0
    return action_mask


def scalar_features(use_extra_scalar_features, player_pos, box_positions, goal_positions, action_profile, num_boxes, observation_board_shape, action_space_n, best_boxes_on_target, no_progress_steps, invalid_action_streak, state_visit_counts, no_progress_step_limit, invalid_action_streak_limit, repeated_state_limit):
    """Build the optional scalar observation tail."""
    if not use_extra_scalar_features:
        return np.asarray([], dtype=np.float32)
    current_boxes_on_target = count_boxes_on_target(box_positions, goal_positions)
    current_goal_distance = sum_min_goal_distance(box_positions, goal_positions)
    visit_count = state_visit_counts.get(state_key(player_pos, box_positions), 1)
    goal_distance_scale = max(1.0, float(num_boxes * sum(observation_board_shape)))
    action_scale = max(1.0, float(action_space_n))
    return np.asarray(
        [
            current_boxes_on_target / max(num_boxes, 1),
            best_boxes_on_target / max(num_boxes, 1),
            current_goal_distance / goal_distance_scale,
            min(no_progress_steps / max(no_progress_step_limit, 1), 1.0),
            min(invalid_action_streak / max(invalid_action_streak_limit, 1), 1.0),
            min(visit_count / max(repeated_state_limit, 1), 1.0),
            len(action_profile["selected"]) / action_scale,
            action_profile["physical_count"] / action_scale,
            action_profile["safe_count"] / action_scale,
            action_profile["viable_count"] / action_scale,
        ],
        dtype=np.float32,
    )


def encode_observation(env, action_profile, observation_board_shape, use_extra_scalar_features, num_boxes, action_space_n, best_boxes_on_target, no_progress_steps, invalid_action_streak, state_visit_counts, no_progress_step_limit, invalid_action_streak_limit, repeated_state_limit):
    """Build the flat observation used by the high-level agent."""
    room_state = env.unwrapped.room_state
    room_fixed = env.unwrapped.room_fixed
    player_pos, box_positions, goal_positions, _ = read_board(env)
    board_layers = observation_layers(room_state, room_fixed, observation_board_shape)
    scalar_tail = scalar_features(
        use_extra_scalar_features,
        player_pos,
        box_positions,
        goal_positions,
        action_profile,
        num_boxes,
        observation_board_shape,
        action_space_n,
        best_boxes_on_target,
        no_progress_steps,
        invalid_action_streak,
        state_visit_counts,
        no_progress_step_limit,
        invalid_action_streak_limit,
        repeated_state_limit,
    )
    action_mask = build_action_mask(action_space_n, action_profile["selected"])
    return np.concatenate([layer.flatten() for layer in board_layers] + [scalar_tail, action_mask]).astype(np.float32)
