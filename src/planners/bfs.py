# src/planners/bfs.py
# Member C — Joseph Nguyen
#
# Breadth-First Search solver for Sokoban.
#
# Uses push-based BFS: each transition is a box push, not a player step.
# Before allowing a push, _can_reach confirms the player can walk to the
# required position without moving any box. This shrinks the search space
# dramatically versus move-based BFS while remaining complete.
#
# After finding the push sequence, _reconstruct expands each push into
# the player walk actions + the push action so the returned list can be
# fed directly to env.step().
#
# State = (player_pos, frozenset(box_positions))
# player_pos after a push = the square the box just vacated
# Positions are plain Python int tuples (row, col)
from collections import deque
import numpy as np

DIRECTIONS = {
    1: (-1, 0),  # push up
    2: (1, 0),   # push down
    3: (0, -1),  # push left
    4: (0, 1),   # push right
}

class BFSAgent:
    def __init__(self, env):
        self.env = env
        self.action_queue = []

    def reset(self):
        self.action_queue = []

    def __call__(self, obs):
        if not self.action_queue:
            actions = self._solve()
            self.action_queue = list(actions) if actions else []

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

        return player_pos, box_positions, goals, walls

    def _can_reach(self, start, target, boxes, walls):
        """BFS check: can the player walk from start to target without moving boxes?"""
        if start == target:
            return True
        queue = deque([start])
        visited = {start}
        while queue:
            pos = queue.popleft()
            for dr, dc in DIRECTIONS.values():
                nxt = (pos[0] + dr, pos[1] + dc)
                if nxt in visited or nxt in walls or nxt in boxes:
                    continue
                if nxt == target:
                    return True
                visited.add(nxt)
                queue.append(nxt)
        return False

    def _walk_to(self, start, target, boxes, walls):
        """Return the action sequence for the player to walk from start to target."""
        if start == target:
            return []
        queue = deque([start])
        visited = {start: None}  # pos -> (parent_pos, action)
        while queue:
            pos = queue.popleft()
            for action, (dr, dc) in DIRECTIONS.items():
                nxt = (pos[0] + dr, pos[1] + dc)
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

    def _solve(self, max_nodes=100_000):
        player_pos, box_positions, goals, walls = self._get_board()
        init_state = (player_pos, box_positions)

        if box_positions == goals:
            return []

        queue = deque([init_state])
        came_from = {init_state: (None, None)}
        nodes_expanded = 0

        while queue:
            if nodes_expanded >= max_nodes:
                return None
            nodes_expanded += 1

            state = queue.popleft()
            player, boxes = state

            for box in boxes:
                for action, (dr, dc) in DIRECTIONS.items():
                    # Player must stand one step behind the box to push it
                    push_from = (box[0] - dr, box[1] - dc)
                    box_target = (box[0] + dr, box[1] + dc)

                    if box_target in walls or box_target in boxes:
                        continue
                    if push_from in walls or push_from in boxes:
                        continue
                    if not self._can_reach(player, push_from, boxes, walls):
                        continue

                    new_boxes = frozenset((box_target if b == box else b) for b in boxes)
                    # After the push the player stands on the square the box just left
                    next_state = (box, new_boxes)

                    if next_state in came_from:
                        continue

                    came_from[next_state] = (state, action)

                    if new_boxes == goals:
                        return self._reconstruct(came_from, next_state, walls)

                    queue.append(next_state)

        return None

    def _reconstruct(self, came_from, goal_state, walls):
        sequence = []
        state = goal_state
        while came_from[state][0] is not None:
            parent, action = came_from[state]
            sequence.append((parent, state, action))
            state = parent
        sequence.reverse()

        all_actions = []
        for from_state, to_state, action in sequence:
            player, boxes = from_state
            dr, dc = DIRECTIONS[action]
            # to_state[0] is where the player ends up = old box position
            push_from = (to_state[0][0] - dr, to_state[0][1] - dc)
            all_actions.extend(self._walk_to(player, push_from, boxes, walls))
            all_actions.append(action)

        return all_actions


# kept for backward compatibility with old imports
def bfs_policy(obs):
    return 0
