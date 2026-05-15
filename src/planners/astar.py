"""
A* Agent for solving Sokoban

f(n) = g(n) + h(n)  where g = box pushes so far, h = hungarian_matching estimate
"""

from collections import deque #BFS for player reachability and pathfinding
import heapq #priority queue (A* expands lowest-cost node first)
import numpy as np

from src.planners.heuristics import hungarian_matching#estimates how far boxes are from goals h(n)
from src.planners.deadlock import precompute_dead_squares, has_deadlock#avoid searching states that are impossible

_DIR = {
    1: (-1, 0), #up
    2: (1, 0), #down
    3: (0, -1), #left
    4: (0, 1), #right
}

class AStarAgent:
    def __init__(self, env):
        self.env = env
        self.action_queue = [] #planned actions
        self.nodes_expanded = 0 #how many states explored
        self.deadlocks_pruned = 0 #how many states discarded
        self.dead_squares_count = 0 #num of deadlock squares

    def reset(self):#reset stats and planned actions when episode ends
        self.action_queue = []
        self.nodes_expanded = 0
        self.deadlocks_pruned = 0
        self.dead_squares_count = 0

    def __call__(self, _obs):

        if not self.action_queue:
            actions = self._solve()#run a*
            self.action_queue = list(actions) if actions else []#store full solution

        if self.action_queue:
            return self.action_queue.pop(0) #execute one action at a time
        
        return 0 #fallback action/ nno action

    def _get_board(self):#extract info from env
        #static features (walls, goals) vs dynamic features (player, boxes)
        room_state = self.env.unwrapped.room_state
        room_fixed = self.env.unwrapped.room_fixed

        player_arr = np.argwhere(room_state == 5)#find player cell
        player_pos = (int(player_arr[0][0]), int(player_arr[0][1]))

        #goal location
        box_positions = frozenset(
            (int(p[0]), int(p[1])) for p in np.argwhere((room_state == 3) | (room_state == 4))
        )

        goals = frozenset((int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 2))
        walls = frozenset((int(p[0]), int(p[1])) for p in np.argwhere(room_fixed == 0))
        
        #board dim
        board_shape = room_state.shape

        return player_pos, box_positions, goals, walls, board_shape

    #bfs for player walking from start to target (is it possible)
    def _can_reach(self, start, target, boxes, walls):
        
        if start == target:#already there
            return True
        
        queue = deque([start])#init bfs queue
        visited = {start}#track explored cells

        while queue:
            pos = queue.popleft()#get next cell to explore
            for dirRow, dirCol in _DIR.values():#explore neighbors
                nxt = (pos[0] + dirRow, pos[1] + dirCol)#compute adjacent cells
                
                #cannot revisit, walk through walls, or walk through boxes
                if nxt in visited or nxt in walls or nxt in boxes:
                    continue

                if nxt == target:#found target!
                    return True
                
                visited.add(nxt)#add to visited to avoid reprocessing
                queue.append(nxt)#continue bfs from this cell

        return False

    #returns actual path 
    def _walk_to(self, start, target, boxes, walls):

        if start == target:#already there
            return []
        
        queue = deque([start])#bfs queue for pathfinding
        visited = {start: None}#track explored

        while queue:
            pos = queue.popleft()#get next cell to explore

            #explore neighbors
            for action, (dirRow, dirCol) in _DIR.items():
                nxt = (pos[0] + dirRow, pos[1] + dirCol)

                #cannot revisit, walk through walls, or walk through boxes
                if nxt in visited or nxt in walls or nxt in boxes:
                    continue

                visited[nxt] = (pos, action)#tracks previous node and action taken

                if nxt == target:
                    path = []
                    cur = nxt

                    #walk backward through parents
                    while visited[cur] is not None:
                        parent, act = visited[cur]
                        path.append(act)
                        cur = parent
                    return list(reversed(path))
                queue.append(nxt)#continue bfs from this cell
        return []

    #core a* algo
    def _solve(self, max_nodes=200_000):
        #load init state
        player_pos, box_positions, goals, walls, board_shape = self._get_board()
        dead_squares = precompute_dead_squares(walls, goals, board_shape)

        self.dead_squares_count = len(dead_squares)
        self.nodes_expanded = 0
        self.deadlocks_pruned = 0

        #is it already solved?
        if box_positions == goals:
            return []

        #state representation: (player_pos, frozenset(box_positions))
        init_state = (player_pos, box_positions)

        #initial heuristic
        h0 = hungarian_matching(box_positions, goals)

        counter = 0
        #heap stores (f, g, counter, state) — counter breaks ties to avoid comparing states
        heap = [(h0, 0, counter, init_state)]

        #stores search tree for reconstructing solution path once we find the goal
        came_from = {init_state: (None, None)}
        
        g_cost = {init_state: 0}#minimum pushes to reach this state found so far

        while heap:
            if self.nodes_expanded >= max_nodes:#prevent infinite search
                return None

            f, g, _, state = heapq.heappop(heap)#expand lowest f-value

            #stale entry — we already found a cheaper path to this state
            if g > g_cost.get(state, float('inf')):
                continue

            self.nodes_expanded += 1
            player, boxes = state

            #generate successor states by trying to push each box in each direction
            for box in boxes:
                for action, (dirRow, dirCol) in _DIR.items():

                    push_from = (box[0] - dirRow, box[1] - dirCol)#required player pos
                    
                    box_target = (box[0] + dirRow, box[1] + dirCol)#where box moves after
                    
                    #check if push is valid: target cell must be free, player must be able to get into position to push
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

                    #only update if this path to next_state is cheaper than any we've seen
                    if new_g >= g_cost.get(next_state, float('inf')):
                        continue

                    g_cost[next_state] = new_g
                    came_from[next_state] = (state, action)

                    if new_boxes == goals:
                        return self._reconstruct(came_from, next_state, walls)

                    h = hungarian_matching(new_boxes, goals)
                    counter += 1
                    heapq.heappush(heap, (new_g + h, new_g, counter, next_state))

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
            dr, dc = _DIR[action]
            push_from = (to_state[0][0] - dr, to_state[0][1] - dc)
            all_actions.extend(self._walk_to(player, push_from, boxes, walls))
            all_actions.append(action)

        return all_actions


def astar_policy(_obs):
    return 0
