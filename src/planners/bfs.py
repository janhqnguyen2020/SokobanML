# src/planners/bfs.py
# Member C — Joseph Nguyen
#
# Breadth-First Search solver for Sokoban.
#
# BFS explores all states level by level, guaranteeing the shortest solution
# (fewest moves) when one exists. It is complete but memory-intensive on
# larger boards because it must store every visited state.
#
# Responsibilities:
#   - Represent the Sokoban board as a hashable state (player pos + box positions)
#   - Expand states by trying all four moves (up, down, left, right)
#   - Track visited states to avoid cycles
#   - Return the action sequence that leads to the goal, or None if unsolvable
#
# Key data structures:
#   - Queue (collections.deque) for the frontier
#   - Set of frozensets for visited states
#   - Dict mapping state -> (parent_state, action) for path reconstruction
#
# Functions to implement:
#   - get_initial_state(env)     → extract player pos and box positions from env
#   - is_goal(state, goals)      → check if all boxes are on goal squares
#   - get_neighbors(state, walls, goals) → generate valid next states + actions
#   - reconstruct_path(came_from, goal_state) → trace back action sequence
#   - bfs_solve(env)             → main entry point, returns list of actions or None
#
# Notes:
#   - State = (player_pos, frozenset(box_positions))
#   - Positions as (row, col) tuples
#   - Should integrate with deadlock.py to prune dead states early
