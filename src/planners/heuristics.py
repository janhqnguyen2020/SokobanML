# src/planners/heuristics.py
# Member C — Joseph Nguyen
#
# Heuristic functions for informed search (Greedy BFS, A*).
#
# A heuristic estimates how far a state is from the goal without solving it.
# For Sokoban, the most natural heuristic is the total cost to push every
# box to its nearest goal square (lower bound on moves remaining).
#
# Heuristics implemented here must be:
#   - Fast to compute (called at every expanded node)
#   - Admissible for A* (never overestimate true cost)
#   - Informative enough to cut down the search space
#
# Functions to implement:
#   - manhattan_distance(a, b)
#       → |row_a - row_b| + |col_a - col_b|
#
#   - min_box_to_goal(box_positions, goal_positions)
#       → for each box, find its nearest goal; sum those distances
#       → simple, fast, admissible
#
#   - hungarian_matching(box_positions, goal_positions)
#       → optimal 1-to-1 assignment of boxes to goals using the Hungarian algorithm
#       → tighter lower bound than min_box_to_goal
#       → use scipy.optimize.linear_sum_assignment
#
#   - player_to_nearest_box(player_pos, box_positions)
#       → distance from player to the closest unresolved box
#       → small tiebreaker to prefer states where player is near work
#
#   - combined_heuristic(state, goals)
#       → combines box-goal cost + player-to-box bonus
#       → primary heuristic used by greedy.py
#
# Notes:
#   - All positions as (row, col) tuples
#   - Goal positions are fixed per puzzle; pass them in or read from config

def manhattan(a,b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def min_box_to_goal(box_positions, goal_positions):
    #for each box, find distance to nearest goal, sum those distances
    total = 0
    for box in box_positions:
        nearest = min(manhattan(box, goal) for goal in goal_positions)
        total += nearest
    return total

