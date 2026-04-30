# src/env/observation_utils.py
# Member A — Quang Dinh Tue Tran
#
# Utility functions to convert raw Gymnasium observations into usable formats.
#
# Gymnasium Sokoban returns observations as one-hot encoded 3D arrays
# (height x width x channels). This module converts that into:
#   - Structured board dicts (for planners)
#   - Flat tensors (for neural networks)
#
# Channel layout (Gymnasium Sokoban default):
#   0 → wall
#   1 → goal (empty)
#   2 → box on goal
#   3 → box (not on goal)
#   4 → player
#   5 → player on goal
#
# Functions to implement:
#   - obs_to_board_dict(obs)
#       → parse one-hot array into:
#         { "walls": set, "goals": set, "boxes": set, "player": (r,c) }
#       → used by planners to extract state for BFS/Greedy
#
#   - obs_to_tensor(obs)
#       → convert one-hot array to float32 PyTorch tensor
#       → shape: (C, H, W) channel-first for CNN input
#       → used by RL networks
#
#   - board_dict_to_state_tuple(board_dict)
#       → convert board dict to hashable (player_pos, frozenset(boxes))
#       → used as dict keys and set members in BFS
#
#   - normalize_obs(obs)
#       → scale pixel values to [0, 1]
#       → may be needed depending on Gymnasium version
#
# Notes:
#   - All positions as (row, col) tuples
#   - This module is imported by both src/planners/ and src/rl/
