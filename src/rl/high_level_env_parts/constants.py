"""
Shared constants for the high-level Sokoban environment
"""

# direction
DIRECTION_DELTAS = {
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}

# limits
INVALID_ACTION_STREAK_LIMIT = 10
RESET_RETRY_LIMIT = 50 # reset until a map with at least one valid action is generated 
NO_PROGRESS_STEP_LIMIT = 16
REPEATED_STATE_LIMIT = 3

# penalities/rewards
INVALID_ACTION_PENALTY = -2.0
INVALID_STREAK_TERMINATION_PENALTY = -8.0 # penality if more than 10 macro-steps attempts leads to invalid actions
DEAD_END_PENALTY = -10.0
REPEATED_STATE_PENALTY = -14.0
STATE_REVISIT_PENALTY = -0.75
BOX_PROGRESS_REWARD = 12.0
DISTANCE_PROGRESS_REWARD = 1.0
SUCCESS_REWARD = 60.0
