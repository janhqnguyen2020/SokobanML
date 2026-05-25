# src/rl/high_level_env_parts/movement.py
"""
Movement helpers for the high-level Sokoban environment
"""

from collections import deque


def push_positions(direction, selected_box_position, push_direction):
    """
    Return the player stand cell and box destination for one push
    """
    row, col = direction[push_direction] # returns tuple of coordinates of push direction
    player_push_position = (selected_box_position[0] - row, selected_box_position[1] - col)
    box_destination_position = (selected_box_position[0] + row, selected_box_position[1] + col)
    return player_push_position, box_destination_position


def blocked_push(player_push_position, box_destination_position, box_positions, wall_positions, reachable_parents):
    """
    Return True when the player cannot stand behind the box or the box cannot move
    """
    if box_destination_position in wall_positions or box_destination_position in box_positions:
        return True
    if player_push_position in wall_positions or player_push_position in box_positions:
        return True
    return player_push_position not in reachable_parents


def new_box_positions(box_positions, selected_box_position, box_destination_position):
    """
    Build the next box set after pushing one chosen box
    """
    return frozenset(box_destination_position if box_position == selected_box_position else box_position for box_position in box_positions)


def get_reachable_parents(start_position, box_positions, wall_positions, direction):
    """
    Run BFS and return parent pointers for every reachable player square
    """
    queue = deque([start_position])
    parents = {start_position: None}
    while queue:
        current_position = queue.popleft()
        for primitive_action, (row, col) in direction.items():
            next_position = (current_position[0] + row, current_position[1] + col)
            if next_position in parents or next_position in wall_positions or next_position in box_positions:
                continue
            parents[next_position] = (current_position, primitive_action)
            queue.append(next_position)
    return parents


def reconstruct_path(parents, target_position):
    """
    Uses can_walk() to reconstruct the path that leads from start to target positions
    """
    if target_position not in parents or parents[target_position] is None:
        return []

    walk_actions = []
    current_position = target_position
    while parents[current_position] is not None:
        previous_position, primitive_action = parents[current_position]
        walk_actions.append(primitive_action)
        current_position = previous_position
    return list(reversed(walk_actions))
