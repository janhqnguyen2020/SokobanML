# src/planners/reasoning.py
"""
High-level reasoning module.

Chooses:
    box -> target goal assignments

This is not low-level planning.
It only provides guidance.
"""

from src.planners.heuristics import manhattan

class ReasoningPlanner:
    def __init__(self, env):
        self.env = env

    def plan(self, box_positions, goals):
        """
        Returns: { box_position: assigned_goal_position}
        """

        remain_goals = set(goals)
        assignment = {}

        # greedy nearest-goal matching
        for box in box_positions:
            best_goal = min(remain_goals, key=lambda g: manhattan(box, g))
            assignment[box] = best_goal
            remain_goals.remove(best_goal)

        return assignment