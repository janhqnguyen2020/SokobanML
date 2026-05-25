# src/planners/greedy.py
"""
Greedy Agent for solving Sokoban
This file implements a greedy search over box configurations,
while using BFS to validate movement and deadlock detection to prune bad states.
"""


from collections import deque
import heapq #priority queue needed for greedy
import numpy as np

from src.planners.heuristics import assignment_heuristic
from src.planners.reasoning import ReasoningPlanner
from src.planners.deadlock import precompute_dead_squares, has_deadlock

_DIR = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

class GreedyAgent:
    #constructor - initializes agent with environment and sets up variables to track performance
    def __init__(self, env):
        self.env = env
        self.action_queue = []
        self.nodes_expanded = 0
        self.deadlocks_pruned = 0
        self.dead_squares_count = 0
        self.failure_reason = ""
        self._failed = False
        self.reasoning = ReasoningPlanner(env)

    def reset(self):
        #clear everything for new episode
        self.action_queue = []
        self.nodes_expanded = 0
        self.deadlocks_pruned = 0
        self.dead_squares_count = 0
        self.failure_reason = ""
        self._failed = False

    #how environment interacts with agent, returns action to take
    def __call__(self, _obs):

        #if no plan exists, compute one
        if not self.action_queue and not self._failed:
            actions = self._solve()
            if actions:
                self.action_queue = list(actions)
            else:
                self._failed = True
                self.failure_reason = "node_limit"

        if self.action_queue:
            return self.action_queue.pop(0) #return one action at a time
        return 0

    #reads sokoban grid from environment
    def _get_board(self):
        
        room_state = self.env.unwrapped.room_state
        room_fixed = self.env.unwrapped.room_fixed

        player_arr = np.argwhere(room_state == 5)# find player in grid
        player_pos = (int(player_arr[0][0]), int(player_arr[0][1]))

        #store all box locations with a hash
        box_positions = frozenset(
            (int(p[0]), int(p[1])) for p in np.argwhere((room_state == 3) | (room_state == 4))
        )
        goals = frozenset((int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 2))
        walls = frozenset((int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 0))
        board_shape = room_state.shape

        return player_pos, box_positions, goals, walls, board_shape

    #helper function (BFS), can player walk from A to B
    def _can_reach(self, start, target, boxes, walls):
        if start == target:
            return True
        queue = deque([start])
        visited = {start}
        while queue:
            pos = queue.popleft()
            for dr, dc in _DIR.values():
                nxt = (pos[0] + dr, pos[1] + dc)

                #cannot walk thru walls & boxes
                if nxt in visited or nxt in walls or nxt in boxes:
                    continue
                if nxt == target:
                    return True
                visited.add(nxt)
                queue.append(nxt)
        return False

    #finds actual movement path, returning list of actions
    def _walk_to(self, start, target, boxes, walls):
        if start == target:
            return []
        queue = deque([start])
        visited = {start: None}
        while queue:
            pos = queue.popleft()
            for action, (dr, dc) in _DIR.items():
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

    #core solver
    def _solve(self, max_nodes=200_000):
        #get board
        player_pos, box_positions, goals, walls, board_shape = self._get_board()

        #find all positions hwere box is stuck forever
        dead_squares = precompute_dead_squares(walls, goals, board_shape)
        self.dead_squares_count = len(dead_squares)
        self.nodes_expanded = 0
        self.deadlocks_pruned = 0

        init_state = (player_pos, box_positions)#starting point of search

        if box_positions == goals:#if already solved, stop
            return []

        counter = 0
        # assignment_heuristic with ReasoningPlanner: assigns each box a unique goal
        # (no two boxes compete for the same goal), giving a better distance estimate
        # than min_box_to_goal which allows conflicts.
        assignment = self.reasoning.plan(box_positions, goals)
        heuristicValue = assignment_heuristic(box_positions, assignment)

        #creates a priority queue (stores states sorted by heuristic)
        heap = [(heuristicValue, counter, init_state)]
        
        came_from = {init_state: (None, None)}

        while heap:
            if self.nodes_expanded >= max_nodes:
                return None
            self.nodes_expanded += 1

            _, _, state = heapq.heappop(heap) #pop best state (always choose lowest heuristic)
            player, boxes = state

            #try pushing box in every direction
            for box in boxes:
                for action, (dirRow, dirCol) in _DIR.items():

                    #compute push positions
                    push_from = (box[0] - dirRow, box[1] - dirCol)
                    box_target = (box[0] + dirRow, box[1] + dirCol)

                    #make sure we cant push into obstacle
                    if box_target in walls or box_target in boxes:
                        continue
                    if push_from in walls or push_from in boxes:
                        continue

                    #ensure player can actually get there
                    if not self._can_reach(player, push_from, boxes, walls):
                        continue
                    
                    #create new state to simulate push
                    new_boxes = frozenset((box_target if b == box else b) for b in boxes)

                    if has_deadlock(new_boxes, dead_squares, walls, goals):
                        self.deadlocks_pruned += 1
                        continue

                    next_state = (box, new_boxes)

                    #avoid duplicates (no loops)
                    if next_state in came_from:
                        continue

                    #parent tracking - stores where each state came from & what action led there
                    came_from[next_state] = (state, action)

                    #reconstruct solution
                    if new_boxes == goals:
                        return self._reconstruct(came_from, next_state, walls)

                    counter += 1
                    new_assignment = self.reasoning.plan(new_boxes, goals)
                    h = assignment_heuristic(new_boxes, new_assignment)
                    heapq.heappush(heap, (h, counter, next_state))

        return None

    #backtracks using bfs  and builds full action sequence
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
            dirRow, dirCol = _DIR[action]
            push_from = (to_state[0][0] - dirRow, to_state[0][1] - dirCol)
            
            #convert abstract pushes -> real movements
            all_actions.extend(self._walk_to(player, push_from, boxes, walls))
            all_actions.append(action)

        return all_actions


def greedy_policy(_obs):
    return 0
