# src/utils/benchmark_registry.py
"""Planner registry for benchmark selection."""

from src.planners.astar import AStarAgent
from src.planners.bfs import BFSAgent
from src.planners.greedy import GreedyAgent


PLANNER_REGISTRY = {
    "bfs": ("BFSAgent", BFSAgent),
    "greedy": ("GreedyAgent", GreedyAgent),
    "astar": ("AStarAgent", AStarAgent),
}