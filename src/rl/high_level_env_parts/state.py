# src/rl/high_level_env_parts/state.py
"""
Board-state helpers for the high-level Sokoban environment
"""

import numpy as np
from src.rl.high_level_env_parts.constants import FILLED_GOAL_DISTANCE_PENALTY
from src.planners.heuristics import hungarian_matching
from collections import deque
from src.rl.high_level_env_parts.movement import blocked_push, get_reachable_parents, new_box_positions, push_positions

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

def feasible_goals_per_box(box, goals, walls, box_positions, direction):
    """
    Approximate which goals this box can still reach after future legal pushes.
    """
    reachable_goals = []
    for goal in goals:
        if is_simple_path_possible(box, goal, walls, box_positions, direction):
            reachable_goals.append(goal)
    return reachable_goals

def is_simple_path_possible(box, goal, walls, box_positions, direction):
    """
    Check whether repeated legal pushes can move one box to one goal.
    """
    start_state = initial_box_search_state(box, box_positions)
    visited = set()
    queue = deque([start_state])
    while queue:
        box_pos, player_pos, current_boxes = queue.popleft()
        if box_pos == goal:
            return True
        state_key = (box_pos, player_pos, current_boxes)
        if state_key in visited:
            continue
        visited.add(state_key)
        queue.extend(next_box_search_states(box_pos, player_pos, current_boxes, walls, direction))
    return False


def initial_box_search_state(box_position, box_positions):
    """
    Build the first simplified future-push search state for one box.
    """
    return box_position, box_position, frozenset(box_positions)


def next_box_search_states(box_position, player_position, current_boxes, walls, direction):
    """
    Return every one-push future state that stays legal on the updated board.
    """
    reachable_parents = get_reachable_parents(player_position, current_boxes, walls, direction)
    next_states = []
    for push_direction in direction:
        next_state = next_box_search_state(
            box_position,
            current_boxes,
            walls,
            direction,
            reachable_parents,
            push_direction,
        )
        if next_state is not None:
            next_states.append(next_state)
    return next_states


def next_box_search_state(box_position, current_boxes, walls, direction, reachable_parents, push_direction):
    """
    Return the next simplified future-push state for one legal push.
    """
    player_push_position, box_destination_position = push_positions(direction, box_position, push_direction)
    if blocked_push(player_push_position, box_destination_position, current_boxes, walls, reachable_parents):
        return None
    updated_boxes = new_box_positions(current_boxes, box_position, box_destination_position)
    return box_destination_position, box_position, updated_boxes

def forced_box_failure(before_boxes,
                       after_boxes,
                       goals,
                       walls,
                       direction):
    """
    Detect if a move destroyed the only reachable goal
    for some box.
    """

    before_domains = {}
    after_domains = {}

    # domains BEFORE move
    for box in before_boxes:
        before_domains[box] = feasible_goals_per_box(
            box,
            goals,
            walls,
            before_boxes,
            direction
        )

    # domains AFTER move
    for box in after_boxes:
        after_domains[box] = feasible_goals_per_box(
            box,
            goals,
            walls,
            after_boxes,
            direction
        )

    penalty_count = 0

    for before_box in before_boxes:

        before_domain = before_domains[before_box]

        # only care about forced boxes
        if len(before_domain) != 1:
            continue

        forced_goal = before_domain[0]

        # try matching same box after move
        matching_after_box = None

        for after_box in after_boxes:
            if after_box == before_box:
                matching_after_box = after_box
                break

        # box moved
        if matching_after_box is None:
            continue

        after_domain = after_domains[matching_after_box]

        # forced goal disappeared
        if forced_goal not in after_domain:
            penalty_count += 1

    return penalty_count
