"""
Action helpers for the high-level Sokoban environment
"""

from src.planners.deadlock import has_deadlock
from src.rl.high_level_env_parts.state import count_boxes_on_target, read_board
from src.rl.high_level_env_parts.movement import (
    blocked_push,
    get_reachable_parents,
    new_box_positions,
    push_positions,
    reconstruct_path,
)

def build_macro_action(reachable_parents, selected_box_position, box_index, direction_index, box_positions, wall_positions, direction_deltas):
    """
    Build one macro action when the push is physically legal
    """
    push_direction = direction_index + 1
    player_push_position, box_destination_position = push_positions(direction_deltas, selected_box_position, push_direction)
    if blocked_push(player_push_position, box_destination_position, box_positions, wall_positions, reachable_parents):
        return None
    return {
        "macro_action": box_index * 4 + direction_index,
        "direction": push_direction,
        "walk_actions": reconstruct_path(reachable_parents, player_push_position),
        "new_player_pos": selected_box_position,
        "new_boxes": new_box_positions(box_positions, selected_box_position, box_destination_position),
    }


def add_box_pushes(valid_action_data, reachable_parents, selected_box_position, box_index, box_positions, wall_positions, direction_deltas):
    """
    Add every legal push for one chosen box
    """
    for direction_index in range(4):
        action_data = build_macro_action(
            reachable_parents,
            selected_box_position,
            box_index,
            direction_index,
            box_positions,
            wall_positions,
            direction_deltas,
        )
        if action_data is not None:
            valid_action_data[action_data["macro_action"]] = action_data


def build_physical_action_data(player_pos, box_positions, wall_positions, direction_deltas):
    """
    Return every one-push action the player can physically execute
    """
    reachable_parents = get_reachable_parents(player_pos, box_positions, wall_positions, direction_deltas)
    valid_action_data = {}
    for box_index, selected_box_position in enumerate(sorted(list(box_positions))):
        add_box_pushes(valid_action_data, reachable_parents, selected_box_position, box_index, box_positions, wall_positions, direction_deltas)
    return valid_action_data


def future_push_count(action_data, wall_positions, direction_deltas):
    """
    Count physical pushes available after one candidate push
    """
    next_actions = build_physical_action_data(action_data["new_player_pos"], action_data["new_boxes"], wall_positions, direction_deltas)
    return len(next_actions)


def safe_action_data_entry(action_data, goal_positions, wall_positions, dead_squares, num_boxes, direction_deltas):
    """
    Return enriched safe-action data or None when the push deadlocks
    """
    if dead_squares and has_deadlock(action_data["new_boxes"], dead_squares, wall_positions, goal_positions): # action_data["new_boxes"] stores the box positions after the current candidate push
        return None 
    enriched_action_data = dict(action_data)
    solved = count_boxes_on_target(action_data["new_boxes"], goal_positions) == num_boxes
    enriched_action_data["future_physical_pushes"] = 0 if solved else future_push_count(action_data, wall_positions, direction_deltas) # after this push, how many push is possible?
    return enriched_action_data


def is_viable_action(action_data, goal_positions, num_boxes):
    """
    Return True when the push solves the puzzle or keeps future pushes open
    """
    solved = count_boxes_on_target(action_data["new_boxes"], goal_positions) == num_boxes
    return solved or action_data["future_physical_pushes"] > 0


def split_safe_actions(physical_action_data, goal_positions, wall_positions, dead_squares, num_boxes, direction_deltas):
    """
    Groups valid pushes into better quality groups
    """
    safe_action_data = {}
    viable_safe_action_data = {}
    for action, action_data in physical_action_data.items():
        enriched_action_data = safe_action_data_entry(action_data, goal_positions, wall_positions, dead_squares, num_boxes, direction_deltas)
        if enriched_action_data is None: # if deadlock, then None
            continue
        safe_action_data[action] = enriched_action_data
        if is_viable_action(enriched_action_data, goal_positions, num_boxes):
            viable_safe_action_data[action] = enriched_action_data
    return safe_action_data, viable_safe_action_data # safe_action_data stores the pushes that does not lead to a deadlock, viable_safe_action data stores the promising pushes based on viability rule


def action_profile_summary(selected, pool_name, physical_action_data, safe_action_data, viable_safe_action_data):
    """
    Return one consistent action-profile dictionary
    """
    return {
        "selected": selected,
        "pool_name": pool_name,
        "physical_count": len(physical_action_data),
        "safe_count": len(safe_action_data),
        "viable_count": len(viable_safe_action_data),
    }


def build_action_profile(player_pos, box_positions, goal_positions, wall_positions, dead_squares, num_boxes, direction):
    """
    Build the selected macro-action pool for the current board
    """
    physical_action_data = build_physical_action_data(player_pos, box_positions, wall_positions, direction)
    safe_action_data, viable_safe_action_data = split_safe_actions(
        physical_action_data,
        goal_positions,
        wall_positions,
        dead_squares,
        num_boxes,
        direction,
    )
    if viable_safe_action_data:
        return action_profile_summary(viable_safe_action_data, "viable_safe", physical_action_data, safe_action_data, viable_safe_action_data)
    if safe_action_data:
        return action_profile_summary(safe_action_data, "safe", physical_action_data, safe_action_data, viable_safe_action_data)
    pool_name = "physical_fallback" if physical_action_data else "none"
    return action_profile_summary(physical_action_data, pool_name, physical_action_data, safe_action_data, viable_safe_action_data)


def board_profile(env, dead_squares, num_boxes, direction):
    """
    Read the board and build the current action profile
    """
    player_pos, box_positions, goal_positions, wall_positions = read_board(env)
    action_profile = build_action_profile(player_pos, box_positions, goal_positions, wall_positions, dead_squares, num_boxes, direction) # valid macro-actions
    return player_pos, box_positions, goal_positions, wall_positions, action_profile











