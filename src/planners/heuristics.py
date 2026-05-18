import numpy as np
from scipy.optimize import linear_sum_assignment

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def min_box_to_goal(box_positions, goal_positions):
    total = 0
    for box in box_positions:
        nearest = min(manhattan(box, goal) for goal in goal_positions)
        total += nearest
    return total

def hungarian_matching(box_positions, goal_positions):
    boxes = list(box_positions)
    goals = list(goal_positions)
    cost_matrix = np.array([
        [manhattan(b, g) for g in goals]
        for b in boxes
    ])
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return int(cost_matrix[row_ind, col_ind].sum())

def player_to_nearest_box(player_pos, box_positions, goal_positions):
    unresolved = box_positions - goal_positions
    if not unresolved:
        return 0
    return min(manhattan(player_pos, box) for box in unresolved)

def assignment_heuristic(box_positions, assignment):
    total = 0
    for box, goal in assignment.items():
        total += manhattan(box, goal)
    return total