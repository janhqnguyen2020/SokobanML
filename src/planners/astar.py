"""
A* Agent for solving Sokoban

f(n) = g(n) + h(n)  where g = box pushes so far, h = hungarian_matching estimate
"""

from collections import deque
import heapq
import numpy as np

from src.planners.heuristics import hungarian_matching
from src.planners.deadlock import precompute_dead_squares, has_deadlock

_DIR = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


class AStarAgent:
    def __init__(self, env, mode="push"):
        self.env = env
        self.mode = mode
        self.action_queue = []
        self.nodes_expanded = 0
        self.deadlocks_pruned = 0
        self.dead_squares_count = 0
        self.pushes = 0
        self.solution_length = 0
        self.failure_reason = ""
        self._failed = False

    def reset(self):
        self.action_queue = []
        self.nodes_expanded = 0
        self.deadlocks_pruned = 0
        self.dead_squares_count = 0
        self.pushes = 0
        self.solution_length = 0
        self.failure_reason = ""
        self._failed = False

    def __call__(self, _obs):
        if not self.action_queue and not self._failed:
            actions = self._solve()
            if actions:
                self.action_queue = list(actions)
            else:
                self._failed = True
        if self.action_queue:
            return self.action_queue.pop(0)
        return 0

    def _get_board(self):
        room_state = self.env.unwrapped.room_state
        room_fixed = self.env.unwrapped.room_fixed

        player_arr = np.argwhere(room_state == 5)
        player_pos = (int(player_arr[0][0]), int(player_arr[0][1]))

        box_positions = frozenset(
            (int(p[0]), int(p[1])) for p in np.argwhere((room_state == 3) | (room_state == 4))
        )
        goals = frozenset((int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 2))
        walls = frozenset((int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 0))
        board_shape = room_state.shape

        return player_pos, box_positions, goals, walls, board_shape

    def _can_reach(self, start, target, boxes, walls):
        if start == target:
            return True
        queue = deque([start])
        visited = {start}
        while queue:
            pos = queue.popleft()
            for dirRow, dirCol in _DIR.values():
                nxt = (pos[0] + dirRow, pos[1] + dirCol)
                if nxt in visited or nxt in walls or nxt in boxes:
                    continue
                if nxt == target:
                    return True
                visited.add(nxt)
                queue.append(nxt)
        return False

    def _walk_to(self, start, target, boxes, walls):
        if start == target:
            return []
        queue = deque([start])
        visited = {start: None}
        while queue:
            pos = queue.popleft()
            for action, (dirRow, dirCol) in _DIR.items():
                nxt = (pos[0] + dirRow, pos[1] + dirCol)
                if nxt in visited or nxt in walls or nxt in boxes:
                    continue
                visited[nxt] = (pos, action)
                if nxt == target:
                    path = []
                    cur = nxt
                    while visited[cur] is not None:
                        parent, act = visited[cur]
                        path.append(act)
                        cur = parent
                    return list(reversed(path))
                queue.append(nxt)
        return []

    def _solve(self, max_nodes=200_000):
        player_pos, box_positions, goals, walls, board_shape = self._get_board()
        dead_squares = precompute_dead_squares(walls, goals, board_shape)

        self.dead_squares_count = len(dead_squares)
        self.nodes_expanded = 0
        self.deadlocks_pruned = 0

        if box_positions == goals:
            self.failure_reason = "solved"
            return []

        init_state = (player_pos, box_positions)
        h0 = hungarian_matching(box_positions, goals)

        counter = 0
        heap = [(h0, 0, counter, init_state)]
        came_from = {init_state: (None, None)}
        g_cost = {init_state: 0}

        while heap:
            if self.nodes_expanded >= max_nodes:
                self.failure_reason = "node_limit"
                return None

            f, g, _, state = heapq.heappop(heap)

            if g > g_cost.get(state, float('inf')):
                continue

            self.nodes_expanded += 1
            player, boxes = state

            for box in boxes:
                for action, (dirRow, dirCol) in _DIR.items():
                    push_from = (box[0] - dirRow, box[1] - dirCol)
                    box_target = (box[0] + dirRow, box[1] + dirCol)

                    if box_target in walls or box_target in boxes:
                        continue
                    if push_from in walls or push_from in boxes:
                        continue
                    if not self._can_reach(player, push_from, boxes, walls):
                        continue

                    new_boxes = frozenset((box_target if b == box else b) for b in boxes)

                    if has_deadlock(new_boxes, dead_squares, walls, goals):
                        self.deadlocks_pruned += 1
                        continue

                    next_state = (box, new_boxes)
                    new_g = g + 1

                    if new_g >= g_cost.get(next_state, float('inf')):
                        continue

                    g_cost[next_state] = new_g
                    came_from[next_state] = (state, action)

                    if new_boxes == goals:
                        result = self._reconstruct(came_from, next_state, walls)
                        self.solution_length = len(result)
                        return result

                    h = hungarian_matching(new_boxes, goals)
                    counter += 1
                    heapq.heappush(heap, (new_g + h, new_g, counter, next_state))

        self.failure_reason = "no_solution_found"
        return None

    def _reconstruct(self, came_from, goal_state, walls):
        sequence = []
        state = goal_state
        while came_from[state][0] is not None:
            parent, action = came_from[state]
            sequence.append((parent, state, action))
            state = parent
        sequence.reverse()
        self.pushes = len(sequence)

        all_actions = []
        for from_state, to_state, action in sequence:
            player, boxes = from_state
            dr, dc = _DIR[action]
            push_from = (to_state[0][0] - dr, to_state[0][1] - dc)
            all_actions.extend(self._walk_to(player, push_from, boxes, walls))
            all_actions.append(action)

        return all_actions
