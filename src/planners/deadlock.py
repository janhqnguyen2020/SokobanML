# src/planners/deadlock.py
# Member C — Joseph Nguyen
#
# Deadlock detection for Sokoban states.
#
# In Sokoban, boxes can only be pushed — never pulled. This means certain
# configurations are permanently unsolvable (deadlocks). Detecting and pruning
# these early dramatically reduces the search space.
#
# Types of deadlocks to detect:
#
#   Corner deadlock (static):
#       A box is in a corner formed by two walls and is not on a goal.
#       Example: box at (r,c) with wall above and wall to the left → stuck forever.
#
#   Freeze deadlock (dynamic):
#       A box is blocked on two axes — cannot be pushed horizontally or vertically.
#       Often caused by other boxes or walls on both sides.
#
#   Simple deadlock squares (precomputed):
#       Squares from which no box can ever reach any goal, regardless of other boxes.
#       Can be precomputed once per level by reverse-BFS from goal squares.
#
# Functions to implement:
#   - precompute_dead_squares(walls, goals, board_shape)
#       → run reverse pushes from each goal to find all reachable squares
#       → any square not reachable is a dead square
#       → returns a set of (row, col) positions
#
#   - is_corner_deadlock(box_pos, walls)
#       → returns True if box is in an unescapable corner
#
#   - is_freeze_deadlock(box_pos, box_positions, walls)
#       → returns True if box is frozen on both axes
#
#   - has_deadlock(state, dead_squares, walls)
#       → master check: returns True if any box is on a dead square
#          or in a freeze/corner deadlock
#       → called inside bfs.py and greedy.py before adding state to frontier
#
# Notes:
#   - dead_squares precomputed once at puzzle load, passed into solvers
#   - Positions as (row, col) tuples
