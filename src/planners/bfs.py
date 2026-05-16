# src/planners/bfs.py
"""
BFS Agent for solving Sokoban
This file implements a breadth-first search over box configurations,
while using BFS to validate movement and deadlock detection to prune bad states. 
"""

from collections import deque # queue structure (FIFO needed for bfs)
import numpy as np
from src.planners.deadlock import precompute_dead_squares, has_deadlock
from src.planners.reasoning import ReasoningPlanner
from src.planners.heuristics import manhattan

_DIR = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

class BFSAgent:
    #constructor - initializes agent with environment and sets up variables to track performance
    def __init__(self, env, reasoning=None):
        self.env = env
        self.action_queue = []
        self.nodes_expanded = 0
        self.deadlocks_pruned = 0
        self.dead_squares_count = 0
        self.reasoning = reasoning or ReasoningPlanner(env)

    def reset(self):
        #clear everything for new episode
        self.action_queue = []
        self.nodes_expanded = 0
        self.deadlocks_pruned = 0
        self.dead_squares_count = 0

    #how environment interacts with agent, returns action to take
    def __call__(self, _obs):

        #if no plan exists, compute one
        if not self.action_queue:
            actions = self._solve()#run solver
            self.action_queue = list(actions) if actions else []
        
        if self.action_queue:
            return self.action_queue.pop(0)#return one action at a time
        return 0

    #reads sokoban grid from environment
    def _get_board(self):
        room_state = self.env.unwrapped.room_state
        room_fixed = self.env.unwrapped.room_fixed

        player_arr = np.argwhere(room_state == 5)#find player in grid
        player_pos = (int(player_arr[0][0]), int(player_arr[0][1]))

        #store all box locations with a hash
        box_positions = frozenset(
            (int(p[0]), int(p[1])) for p in np.argwhere((room_state == 3) | (room_state == 4))
        )
        goals = frozenset((int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 2))
        walls = frozenset((int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 0))
        board_shape = room_state.shape

        return player_pos, box_positions, goals, walls, board_shape

    #helper function (BFS) to check if player can reach a position without moving boxes 
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

    #finds path for player (actual movement path, returning list of actions)
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

    #core solver
    def _solve(self, max_nodes=200_000):
        #get board info from environment
        player_pos, box_positions, goals, walls, board_shape = self._get_board()
        if box_positions == goals:#if already solved, return empty plan
            return []

        #find dead squares for pruning
        dead_squares = precompute_dead_squares(walls, goals, board_shape)
        self.dead_squares_count = len(dead_squares)
        self.nodes_expanded = 0
        self.deadlocks_pruned = 0

        init_state = (player_pos, box_positions)#start state is player position + box positions
        queue = deque([init_state])#defines BFS (FIFO queue)
        came_from = {init_state: (None, None)}
        while queue:#keep exploring until solution found or queue empty
            if self.nodes_expanded >= max_nodes:
                return None
            self.nodes_expanded += 1

            state = queue.popleft()#removes oldest state from queue for exploration
            player, boxes = state# expand state 

            # Explore promising boxes first by action ordering
            assignment = self.reasoning.plan(boxes, goals) # returns dictionary with key=position of boxes, val=position of goals -- Note that box-to-goal matching can change depending on the current state
            sorted_boxes = sorted(boxes, key=lambda b: manhattan(b, assignment[b]))

            for box in sorted_boxes:
                #try pushing boxes in every direction
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

                    # ensure this push does not lead to deadlock
                    if has_deadlock(new_boxes, dead_squares, walls, goals):
                        self.deadlocks_pruned += 1
                        continue

                    # after push, player stands in where the box was, and the boxes are in new_boxes (list of coordinates)
                    next_state = (box, new_boxes)

                    #avoid duplicates (no loops)
                    if next_state in came_from:
                        continue

                    #parent tracking - stores where each state came from & what action led there
                    came_from[next_state] = (state, action)

                    #reconstruct solution if we reached goal
                    if new_boxes == goals:
                        return self._reconstruct(came_from, next_state, walls)

                    queue.append(next_state)

        return None

    def _reconstruct(self, came_from, goal_state, walls):
        sequence = []
        state = goal_state

        while came_from[state][0] is not None:#walk backwards from goal to loc
            parent, action = came_from[state]#state came from parent via action
            sequence.append((parent, state, action))#build sequence
            state = parent
        sequence.reverse()

        # after reconstructing the path, move the player following the sequence
        all_actions = []
        for from_state, to_state, action in sequence:
            player, boxes = from_state
            dirRow, dirCol = _DIR[action]
            push_from = (to_state[0][0] - dirRow, to_state[0][1] - dirCol)
            
            #convert abtract pushes -> real movements
            all_actions.extend(self._walk_to(player, push_from, boxes, walls))
            all_actions.append(action)

        return all_actions


def bfs_policy(obs):
    return 0