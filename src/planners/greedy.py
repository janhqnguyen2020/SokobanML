# src/planners/greedy.py
# Member C — Joseph Nguyen
#
# Greedy Best-First Search solver for Sokoban.
#
# Greedy BFS always expands the state that looks closest to the goal
# according to the heuristic, without tracking actual cost so far.
# It is faster than BFS in practice but not guaranteed to find the
# shortest path, and can get stuck in dead ends.
#
# Comparison with BFS:
#   BFS    → optimal, slow, memory-heavy
#   Greedy → fast, not optimal, can fail where BFS succeeds
#   A*     → optimal + informed (future work / stretch goal)
#
# Responsibilities:
#   - Use a min-heap (heapq) ordered by heuristic value h(state)
#   - Call heuristics.combined_heuristic() to score each state
#   - Track visited states to avoid re-expansion
#   - Return action sequence or None
#
# Functions to implement:
#   - greedy_solve(env)
#       → main entry point
#       → uses same state representation as bfs.py
#       → returns list of actions or None
#
#   - _expand(state, walls, goals)
#       → generate (next_state, action, h_value) for each valid move
#
# Notes:
#   - Reuse get_neighbors() and reconstruct_path() from bfs.py or utils
#   - Heap entries: (h_value, state, came_from)
#   - Tie-break by insertion order to keep heap stable
