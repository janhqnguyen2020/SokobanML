# src/planners/deadlock.py
"""
Detect states where puzzle is in AUTOMATIC DEADLOCK, meaning no sequence of actions can lead to a solution
- examples include: being stuck in a corner, trapped between obstacles, etc

Prune (Skip)
"""

from collections import deque
from src.planners.heuristics import hungarian_matching

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --- Main wrapper ---
def has_deadlock(box_positions, dead_squares, walls, goals):
    """Return True if any box (not on a goal) is in an irreversible deadlock."""
    for box in box_positions - goals:
        if box in dead_squares:
            return True
        if is_corner_deadlock(box, walls):
            return True
        # if is_freeze_deadlock(box, box_positions, walls):
        #     return True
        
    if assignment_deadlock(box_positions, goals):
        return True
    
    return False

# --- Helpers for has_deadlock() ---
def precompute_dead_squares(walls, goals, board_shape):
    """Find ALL squares where a box can never be pushed to any goal."""
    rows, cols = board_shape
    reachable = set(goals)  # all squares where a box could possibly be pushed from and eventually reach a goal
    queue = deque(goals)

    while queue:
        row, col = queue.popleft()
        for dirRow, dirCol in _DIRS:
            box_from  = (row - dirRow, col - dirCol)  # where box came from
            player_at = (box_from[0] - dirRow, box_from[1] - dirCol)

            if box_from in walls or player_at in walls:
                continue
            if not (0 <= box_from[0] < rows and 0 <= box_from[1] < cols):
                continue
            if box_from not in reachable:
                reachable.add(box_from)
                queue.append(box_from)

    return {
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in walls and (r, c) not in reachable
    }   # return squares that cannot reach a goal


def is_corner_deadlock(box_pos, walls):
    """Is the box stuck in a corner formed by two perpendicular walls?"""
    row, col = box_pos
    up    = (row - 1, col) in walls
    down  = (row + 1, col) in walls
    left  = (row, col - 1) in walls
    right = (row, col + 1) in walls
    return (up and left) or (up and right) or (down and left) or (down and right)


def is_freeze_deadlock(box_pos, box_positions, walls):
    """Return True when the current box has no push direction left."""
    return not (
        can_eventually_push_left(box_pos, box_positions, walls)
        or can_eventually_push_right(box_pos, box_positions, walls)
        or can_eventually_push_up(box_pos, box_positions, walls)
        or can_eventually_push_down(box_pos, box_positions, walls)
    )


# --- Helpers for is_freeze_deadlock() ---
def can_eventually_push_left(box_pos, box_positions, walls):
    """Return True when left push is possible now or after a movable box clears."""
    row, col = box_pos
    destination_pos = (row, col - 1)
    pusher_pos = (row, col + 1)
    return can_use_horizontal_side(destination_pos, box_positions, walls, "left") and can_use_horizontal_side(
        pusher_pos, box_positions, walls, "right"
    )


def can_eventually_push_right(box_pos, box_positions, walls):
    """Return True when right push is possible now or after a movable box clears."""
    row, col = box_pos
    destination_pos = (row, col + 1)
    pusher_pos = (row, col - 1)
    return can_use_horizontal_side(destination_pos, box_positions, walls, "right") and can_use_horizontal_side(
        pusher_pos, box_positions, walls, "left"
    )


def can_eventually_push_up(box_pos, box_positions, walls):
    """Return True when up push is possible now or after a movable box clears."""
    row, col = box_pos
    destination_pos = (row - 1, col)
    pusher_pos = (row + 1, col)
    return can_use_vertical_side(destination_pos, box_positions, walls, "up") and can_use_vertical_side(
        pusher_pos, box_positions, walls, "down"
    )


def can_eventually_push_down(box_pos, box_positions, walls):
    """Return True when down push is possible now or after a movable box clears."""
    row, col = box_pos
    destination_pos = (row + 1, col)
    pusher_pos = (row - 1, col)
    return can_use_vertical_side(destination_pos, box_positions, walls, "down") and can_use_vertical_side(
        pusher_pos, box_positions, walls, "up"
    )


def can_use_horizontal_side(adjacent_pos, box_positions, walls, side_name):
    """Return True when a left or right side is open or blocked by a movable box."""
    if adjacent_pos in walls:
        return False
    if adjacent_pos not in box_positions:
        return True
    return can_box_move_vertically(adjacent_pos, box_positions, walls)


def can_use_vertical_side(adjacent_pos, box_positions, walls, side_name):
    """Return True when an up or down side is open or blocked by a movable box."""
    if adjacent_pos in walls:
        return False
    if adjacent_pos not in box_positions:
        return True
    return can_box_move_horizontally(adjacent_pos, box_positions, walls)


def can_box_move_vertically(box_pos, box_positions, walls):
    """Return True when a neighboring box has a real up or down push."""
    return can_push_up(box_pos, box_positions, walls) or can_push_down(box_pos, box_positions, walls)


def can_box_move_horizontally(box_pos, box_positions, walls):
    """Return True when a neighboring box has a real left or right push."""
    return can_push_left(box_pos, box_positions, walls) or can_push_right(box_pos, box_positions, walls)


def can_push_left(box_pos, box_positions, walls):
    """Return True when this box can be pushed one square to the left."""
    row, col = box_pos
    destination_open = (row, col - 1) not in walls and (row, col - 1) not in box_positions
    pusher_space_open = (row, col + 1) not in walls and (row, col + 1) not in box_positions
    return destination_open and pusher_space_open


def can_push_right(box_pos, box_positions, walls):
    """Return True when this box can be pushed one square to the right."""
    row, col = box_pos
    destination_open = (row, col + 1) not in walls and (row, col + 1) not in box_positions
    pusher_space_open = (row, col - 1) not in walls and (row, col - 1) not in box_positions
    return destination_open and pusher_space_open


def can_push_up(box_pos, box_positions, walls):
    """Return True when this box can be pushed one square upward."""
    row, col = box_pos
    destination_open = (row - 1, col) not in walls and (row - 1, col) not in box_positions
    pusher_space_open = (row + 1, col) not in walls and (row + 1, col) not in box_positions
    return destination_open and pusher_space_open


def can_push_down(box_pos, box_positions, walls):
    """Return True when this box can be pushed one square downward."""
    row, col = box_pos
    destination_open = (row + 1, col) not in walls and (row + 1, col) not in box_positions
    pusher_space_open = (row - 1, col) not in walls and (row - 1, col) not in box_positions
    return destination_open and pusher_space_open


def assignment_deadlock(box_positions, goal_positions):

    #Detect impossible global assignments.

    unresolved_boxes = [
        box for box in box_positions
        if box not in goal_positions
    ]

    unresolved_goals = [
        goal for goal in goal_positions
        if goal not in box_positions
    ]

    if len(unresolved_boxes) != len(unresolved_goals):
        return False
    
    if not unresolved_boxes:
        return False
    
    #impossible if any box cannot reasonably reach any goal
    for box in unresolved_boxes:
        reachable = False

        for goal in unresolved_goals:
            dist = abs(box[0] - goal[0]) + abs(box[1] - goal[1])

            if dist < 999:
                reachable = True
                break

        if not reachable:
            return True
        
    return False