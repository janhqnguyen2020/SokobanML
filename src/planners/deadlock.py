# src/planners/deadlock.py
"""
Detect states where puzzle is in AUTOMATIC DEADLOCK, meaning no sequence of actions can lead to a solution
- examples include: being stuck in a corner, trapped between obstacles, etc

Prune (Skip)
"""

from collections import deque

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def precompute_dead_squares(walls, goals, board_shape):
    """Find ALL squares where a box can never be pushed to any goal."""
    rows, cols = board_shape

    reachable = set(goals)
    queue = deque(goals)

    while queue:
        row, col = queue.popleft()
        for dirRow, dirCol in _DIRS:
            box_from  = (row - dirRow, col - dirCol)  # where box came from
            player_at = (row + dirRow, col + dirCol)  # where player must stand

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
    }


def is_corner_deadlock(box_pos, walls):
    """Is the box stuck in a corner formed by two perpendicular walls?"""
    row, col = box_pos
    up    = (row - 1, col) in walls
    down  = (row + 1, col) in walls
    left  = (row, col - 1) in walls
    right = (row, col + 1) in walls
    return (up and left) or (up and right) or (down and left) or (down and right)


def is_freeze_deadlock(box_pos, box_positions, walls):
    """Is the box blocked on both axes by walls or other boxes?"""
    row, col = box_pos
    blocked_h = (
        ((row, col - 1) in walls or (row, col - 1) in box_positions) and
        ((row, col + 1) in walls or (row, col + 1) in box_positions)
    )
    blocked_v = (
        ((row - 1, col) in walls or (row - 1, col) in box_positions) and
        ((row + 1, col) in walls or (row + 1, col) in box_positions)
    )
    return blocked_h and blocked_v


def has_deadlock(box_positions, dead_squares, walls, goals):
    """Return True if any box (not on a goal) is in an irreversible deadlock."""
    for box in box_positions - goals:
        if box in dead_squares:
            return True
        if is_corner_deadlock(box, walls):
            return True
        if is_freeze_deadlock(box, box_positions, walls):
            return True
    return False
