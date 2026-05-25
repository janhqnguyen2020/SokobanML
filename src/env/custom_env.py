# src/env/custome_env.py
import numpy as np
from gym_sokoban.envs.sokoban_env import SokobanEnv


class SimpleCustomSokobanEnv(SokobanEnv):
    """
    Simple custom Sokoban environment
    The user chooses:
        - height
        - width
        - player position
        - box positions
        - goal positions
    Walls are automatically placed around the outside border

    --- Encoding --- 
    room_fixed:
        0 = wall
        1 = floor
        2 = goal
    room_state:
        0 = wall
        1 = empty floor
        2 = empty goal
        3 = box on goal
        4 = box
        5 = player
    """

    def __init__(
        self,
        height=3,
        width=12,
        player_position=(1, 1),
        box_positions= [(1, 5)],
        goal_positions=[(1, 11)],
        max_steps=200,
    ):

        self.height = height
        self.width = width
        self.initial_player_position = tuple(player_position)
        self.initial_box_positions = [tuple(pos) for pos in box_positions]
        self.initial_goal_positions = [tuple(pos) for pos in goal_positions]

        # inherit from SokobenEnv - initialize base Sokoban environment
        # Reference: https://github.com/mpSchrader/gym-sokoban/blob/default/gym_sokoban/envs/sokoban_env.py
        super().__init__(
            dim_room=(height, width),
            max_steps=max_steps,
            num_boxes=len(self.initial_box_positions),
            reset=False,
        )
        self.reset() # build starting state of custom room


    def reset(self, second_player=False, render_mode="rgb_array"):
        """
        Build custom room
        """
        self._build_room()
        self._reset_counters()
        return self.render(render_mode)


    def _build_room(self):
        """
        room_fixed : permanent background (eg. wall, goal, floor)
        room_state : current board state
        """
        self._validate_config()

        # Add floor
        self.room_fixed = np.ones((self.height, self.width), dtype=np.uint8)
        self.room_state = np.ones((self.height, self.width), dtype=np.uint8)

        # Add outer walls
        self.room_fixed[0, :] = 0
        self.room_fixed[-1, :] = 0
        self.room_fixed[:, 0] = 0
        self.room_fixed[:, -1] = 0

        self.room_state[0, :] = 0
        self.room_state[-1, :] = 0
        self.room_state[:, 0] = 0
        self.room_state[:, -1] = 0

        # Add goals
        for r, c in self.initial_goal_positions:
            self.room_fixed[r, c] = 2
            self.room_state[r, c] = 2

        # Add boxes
        boxes_on_target = 0
        for r, c in self.initial_box_positions:
            if self.room_fixed[r, c] == 2:
                self.room_state[r, c] = 3   # box on goal
                boxes_on_target += 1
            else:
                self.room_state[r, c] = 4   # box on floor

        # Add player
        pr, pc = self.initial_player_position
        self.room_state[pr, pc] = 5
        self.player_position = np.array([pr, pc]) # store current player location

        # how many boxes are already on goals? helps in knowing whether puzzle is solved
        self.boxes_on_target = boxes_on_target


    def _reset_counters(self):
        self.num_env_steps = 0
        self.reward_last = 0
        self.box_mapping = {}


    def _validate_config(self):
        """
        Check that the chosen positions are valid
        """
        def inside_board(pos):
            r, c = pos
            return 0 <= r < self.height and 0 <= c < self.width
        def on_outer_wall(pos):
            r, c = pos
            return r == 0 or r == self.height - 1 or c == 0 or c == self.width - 1
        
        # Check player, goal  and boxes are confined within the walls of the map
        all_positions = ([self.initial_player_position] + self.initial_box_positions + self.initial_goal_positions)
        for pos in all_positions:
            if not inside_board(pos):
                raise ValueError(f"Position {pos} is outside the board")
            if on_outer_wall(pos):
                raise ValueError(
                    f"Position {pos} is on the outer wal")
        
        # Check number of boxes == number of goals
        if len(self.initial_goal_positions) != len(self.initial_box_positions):
            raise ValueError("Need at least as many goals as boxes"
                f"num_boxes={len(self.initial_box_positions)}, "
                f"num_goals={len(self.initial_goal_positions)}."
            )
        
        # Check player is not on box
        if self.initial_player_position in self.initial_box_positions:
            raise ValueError("Player cannot start on a box.")
        
        # Check duplicate boxes & goals
        if len(set(self.initial_box_positions)) != len(self.initial_box_positions):
            raise ValueError("Duplicate box positions are not allowed.")
        if len(set(self.initial_goal_positions)) != len(self.initial_goal_positions):
            raise ValueError("Duplicate goal positions are not allowed.")#