"""
Higher-level Sokoban wrapper for DQN
Function in the following manner: 
1. Choose which box to act on
2. Choose which direction to push that box 
"""

from collections import deque
import gym
import numpy as np
from gym.spaces.discrete import Discrete
from src.env.sokoban_env import initialize_env


_DIR = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}


class HighLevelSokobanEnv(gym.Env):
    def __init__(self, env=None):
        super().__init__()
        self.env = env if env is not None else initialize_env()

        # Choose one box, choose one action
        self.num_boxes = self.env.unwrapped.num_boxes
        self.action_space = Discrete(self.num_boxes * 4) # eg. if there are 3 boxes, then there are 12 possible actions
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        self.env.close()

    def step(self, action):
        """
        Convert one macro action into:
            choose box index
            choose push direction
            walk into position
            push once
        """
        box_index = int(action) // 4
        direction_index = int(action) % 4

        # map 0..3 -> planner-style directions 1..4
        direction = direction_index + 1
        player_pos, box_positions, goals, walls, board_shape = self._get_board()

        # stable ordering so the box index is consistent
        boxes_list = sorted(list(box_positions))
        if box_index >= len(boxes_list):
            return self._invalid_action_result()

        box = boxes_list[box_index]
        dir_row, dir_col = _DIR[direction]
        push_from = (box[0] - dir_row, box[1] - dir_col)
        box_target = (box[0] + dir_row, box[1] + dir_col)

        # invalid if push target or player setup square is blocked
        if box_target in walls or box_target in box_positions:
            return self._invalid_action_result()
        if push_from in walls or push_from in box_positions:
            return self._invalid_action_result()

        # player must be able to reach the square behind the box
        if not self._can_reach(player_pos, push_from, box_positions, walls):
            return self._invalid_action_result()

        walk_actions = self._walk_to(player_pos, push_from, box_positions, walls)
        total_reward = 0.0
        done = False
        observation = None
        info = {}

        # execute primitive walking actions first
        for primitive_action in walk_actions:
            observation, reward, done, info = self.env.step(primitive_action)
            total_reward += reward
            if done:
                return observation, total_reward, done, info

        # then execute the real push action
        observation, reward, done, info = self.env.step(direction)
        total_reward += reward

        return observation, total_reward, done, info

    def _invalid_action_result(self):
        """
        Invalid macro actions do not change the board.
        They simply return the current observation with a small penalty.
        """
        observation = self.env.render(mode="rgb_array")
        reward = -1.0
        done = False
        info = {"invalid_macro_action": True}
        return observation, reward, done, info

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
            for dir_row, dir_col in _DIR.values():
                nxt = (pos[0] + dir_row, pos[1] + dir_col)
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
            for action, (dir_row, dir_col) in _DIR.items():
                nxt = (pos[0] + dir_row, pos[1] + dir_col)
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
