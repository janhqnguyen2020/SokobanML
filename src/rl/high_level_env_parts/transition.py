# src/rl/high_level_env_parts/transition.py
"""
Helps decide what happens after one macro-action
1. whether the episode should end
2. what reward/penalities to give
"""

from src.rl.high_level_env_parts.state import termination_reason
from src.rl.high_level_env_parts.constants import (
    BOX_LEFT_GOAL_PENALTY,
    BOX_PROGRESS_REWARD,
    DEAD_END_PENALTY,
    DISTANCE_PROGRESS_REWARD,
    INVALID_ACTION_PENALTY,
    INVALID_STREAK_TERMINATION_PENALTY,
    INVALID_ACTION_STREAK_LIMIT,
    NO_PROGRESS_STEP_LIMIT,
    REPEATED_STATE_LIMIT,
    REPEATED_STATE_PENALTY,
    STATE_REVISIT_PENALTY,
    SUCCESS_REWARD,
    REVERSE_MOVE_PENALTY,
)

def update_no_progress(no_progress_steps, before_progress, after_progress):
    """
    Check whether the move improved the state 
    """
    box_progress_delta = after_progress["boxes_on_target"] - before_progress["boxes_on_target"] # how many more boxes are on target after the move
    distance_progress = before_progress["goal_distance"] - after_progress["goal_distance"] # are the boxes closer to the goals overall
    no_progress_steps = 0 if box_progress_delta > 0 or distance_progress > 0 else no_progress_steps + 1 # tracks how many consecutive steps had no progress
    return no_progress_steps, box_progress_delta, distance_progress


def done_flags(env_done, solved, state_visit_count, next_action_profile, no_progress_steps):
    """
    Terminated? : loop, no progress, dead-end, solved
    """
    repeated_state = state_visit_count >= REPEATED_STATE_LIMIT
    no_progress = no_progress_steps >= NO_PROGRESS_STEP_LIMIT
    dead_end = not solved and not env_done and not next_action_profile["selected"]
    done = env_done or solved or repeated_state or no_progress or dead_end # env_done is defined in Sokoban environment -- either soled or timed out by max steps
    return done, repeated_state, no_progress, dead_end


def select_reward(use_shaped_reward, raw_reward, box_progress_delta, distance_progress, solved, repeated_state, no_progress, dead_end, state_visit_count, reverse_move, solvability_damage):
    """
    Return the training reward or the raw env reward
    """
    if not use_shaped_reward: 
        return raw_reward   # use original reward from original Sokoban env
    reward = raw_reward + BOX_PROGRESS_REWARD * box_progress_delta + DISTANCE_PROGRESS_REWARD * distance_progress
    if box_progress_delta < 0:
        reward += BOX_LEFT_GOAL_PENALTY
    if state_visit_count > 1:
        reward += STATE_REVISIT_PENALTY * (state_visit_count - 1)
    if reverse_move:
        reward += REVERSE_MOVE_PENALTY
    if solvability_damage > 0:
        reward -= 18 * solvability_damage
    if solved:
        return reward + SUCCESS_REWARD
    if repeated_state:
        return reward + REPEATED_STATE_PENALTY
    return reward + DEAD_END_PENALTY if no_progress or dead_end else reward


def action_pool_info(action_profile, prefix=""):
    """Return small action-pool counts that are useful for logs and debugging."""
    return {
        f"{prefix}physical_macro_actions_count": action_profile["physical_count"],
        f"{prefix}safe_macro_actions_count": action_profile["safe_count"],
        f"{prefix}viable_safe_macro_actions_count": action_profile["viable_count"],
        f"{prefix}valid_macro_action_pool": action_profile["pool_name"],
    }


def step_info(action, action_profile, next_action_profile, progress, raw_reward, returned_reward, box_progress_delta, distance_progress, solved, repeated_state, no_progress, dead_end, state_visit_count, no_progress_steps, best_boxes_on_target, env_info):
    """
    Used for observational purposes during training & eval
    """
    info = dict(env_info or {})
    info.update(action_pool_info(next_action_profile, prefix="next_"))
    info.update(action_pool_info(action_profile, prefix="selected_"))
    info["invalid_macro_action"] = False
    info["executed_push"] = True
    info["selected_macro_action"] = int(action)
    info["valid_macro_actions_count"] = len(next_action_profile["selected"])
    info["boxes_on_target"] = progress["boxes_on_target"]
    info["best_boxes_on_target"] = best_boxes_on_target
    info["goal_distance"] = progress["goal_distance"]
    info["box_progress_delta"] = box_progress_delta
    info["distance_progress"] = distance_progress
    info["no_progress_steps"] = no_progress_steps
    info["state_visit_count"] = state_visit_count
    info["all_boxes_on_target"] = solved
    info["dead_end_termination"] = dead_end
    info["repeated_state_termination"] = repeated_state
    info["no_progress_termination"] = no_progress
    info["termination_reason"] = termination_reason(solved, dead_end, repeated_state, no_progress)
    info["raw_reward"] = raw_reward
    info["returned_reward"] = returned_reward
    return info


def invalid_action_info(action_profile, boxes_on_target, best_boxes_on_target, invalid_action_streak):
    """
    Used for observational purposes during training & eval
    """
    done = invalid_action_streak >= INVALID_ACTION_STREAK_LIMIT
    info = {
        "invalid_macro_action": True,
        "executed_push": False,
        "invalid_action_streak": invalid_action_streak,
        "valid_macro_actions_count": len(action_profile["selected"]),
        "boxes_on_target": boxes_on_target,
        "best_boxes_on_target": best_boxes_on_target,
        "termination_reason": "invalid_action_streak_limit" if done else "invalid_action",
    }
    info.update(action_pool_info(action_profile))
    return info, done, INVALID_STREAK_TERMINATION_PENALTY if done else INVALID_ACTION_PENALTY


def dead_end_info(action_profile, boxes_on_target, best_boxes_on_target):
    """
    Used for observational purposes during training & eval
    """
    info = {
        "invalid_macro_action": False,
        "executed_push": False,
        "valid_macro_actions_count": len(action_profile["selected"]),
        "boxes_on_target": boxes_on_target,
        "best_boxes_on_target": best_boxes_on_target,
        "dead_end_termination": True,
        "termination_reason": "dead_end",
    }
    info.update(action_pool_info(action_profile))
    return info
