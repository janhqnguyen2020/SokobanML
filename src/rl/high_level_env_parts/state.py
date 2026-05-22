"""
Board-state helpers for the high-level Sokoban environment
"""

import numpy as np
from src.rl.high_level_env_parts.constants import FILLED_GOAL_DISTANCE_PENALTY


def read_board(env):
    """
    Read player, box, goal, and wall positions from the wrapped Sokoban env
    """
    room_state = env.unwrapped.room_state
    room_fixed = env.unwrapped.room_fixed
    player_arr = np.argwhere(np.isin(room_state, [5, 6]))
    player_pos = (int(player_arr[0][0]), int(player_arr[0][1]))
    box_positions = frozenset((int(cell[0]), int(cell[1])) for cell in np.argwhere((room_state == 3) | (room_state == 4)))
    goal_positions = frozenset((int(cell[0]), int(cell[1])) for cell in np.argwhere(room_fixed == 2))
    wall_positions = frozenset((int(cell[0]), int(cell[1])) for cell in np.argwhere(room_fixed == 0))
    return player_pos, box_positions, goal_positions, wall_positions


def count_boxes_on_target(box_positions, goal_positions):
    """
    Count how many boxes currently sit on goal squares
    """
    return sum(1 for box_position in box_positions if box_position in goal_positions)


def unsolved_boxes(box_positions, goal_positions):
    """
    Return boxes that are not already on goals.
    """
    return [box_position for box_position in box_positions if box_position not in goal_positions]


def sum_min_goal_distance(box_positions, goal_positions):
    """
    Prefer open goals, but still allow filled goals with an extra cost.
    """
    remaining_boxes = sorted(unsolved_boxes(box_positions, goal_positions))
    remaining_goals = sorted(list(goal_positions))
    total_distance = 0

    for box_position in remaining_boxes:
        if not remaining_goals:
            break
        distances = []
        for goal_position in remaining_goals:
            distance = abs(box_position[0] - goal_position[0]) + abs(box_position[1] - goal_position[1])
            if goal_position in box_positions:
                distance += FILLED_GOAL_DISTANCE_PENALTY
            distances.append(distance)
        best_goal_index = int(np.argmin(distances))
        total_distance += int(distances[best_goal_index])
        remaining_goals.pop(best_goal_index)
    return total_distance


def progress_snapshot(box_positions, goal_positions):
    """
    Bundle the progress signals used by reward shaping and logging
    """
    return {
        "boxes_on_target": count_boxes_on_target(box_positions, goal_positions),
        "goal_distance": sum_min_goal_distance(box_positions, goal_positions),
    }


def state_key(player_pos, box_positions):
    """
    Build the abstract state key used for repeated-state tracking
    Used for detecting agents getting back to the same position,
    which will be penalized after a certain threshold. This is to 
    ensure that the agent will not be stuck in a loop.
    """
    return player_pos, tuple(sorted(box_positions))


def record_state_visit(state_visit_counts, player_pos, box_positions):
    """
    Keep track of how many times the agent has visited the same state.
    Used for detecting agents getting back to the same position.
    """
    current_state_key = state_key(player_pos, box_positions)
    state_visit_count = state_visit_counts.get(current_state_key, 0) + 1
    state_visit_counts[current_state_key] = state_visit_count
    return state_visit_count


def termination_reason(solved, dead_end, repeated_state, no_progress):
    """
    Convert boolean termination flags into one readable label
    """
    if solved:
        return "solved"
    if dead_end:
        return "dead_end"
    if repeated_state:
        return "repeated_state"
    if no_progress:
        return "no_progress"
    return "env_done"
