# src/utils/config.py
# Shared — all members
#
# Central configuration for hyperparameters, paths, and experiment settings.
#
# Keeping all magic numbers and file paths here means changing one value
# updates the whole project. Every other module should import from here
# rather than hardcoding values.
#
# Sections to define:
#
#   PATHS:
#     RESULTS_DIR = "results/"
#     MODELS_DIR  = "models/"
#     VIDEOS_DIR  = "videos/"
#     LEVELS_DIR  = "data/levels/"
#
#   ENVIRONMENT:
#     ENV_ID       = "Sokoban-v2"
#     MAX_STEPS    = 200        ← max actions per episode
#     NUM_ENVS     = 4          ← parallel envs for PPO
#     LEVEL_IDS    = list(range(0, 100))   ← training levels
#     TEST_LEVELS  = list(range(900, 1000)) ← held-out eval levels
#
#   PPO HYPERPARAMS:
#     PPO_LR            = 3e-4
#     PPO_N_STEPS       = 2048
#     PPO_N_EPOCHS      = 10
#     PPO_CLIP_RANGE    = 0.2
#     PPO_ENT_COEF      = 0.01
#     PPO_TOTAL_STEPS   = 1_000_000
#
#   DQN HYPERPARAMS:
#     DQN_LR                = 1e-4
#     DQN_BUFFER_SIZE       = 50_000
#     DQN_LEARNING_STARTS   = 10_000
#     DQN_EXPLORATION_FRAC  = 0.2
#     DQN_EPS_FINAL         = 0.05
#     DQN_TARGET_UPDATE     = 1000
#     DQN_TOTAL_STEPS       = 500_000
#
#   PLANNER:
#     PLANNER_TIMEOUT_SEC = 30   ← max time per level for BFS/Greedy
#
#   REPRODUCIBILITY:
#     SEED = 42
#
# Notes:
#   - Import as: from src.utils.config import MAX_STEPS, PPO_LR, etc.
#   - Do not store secrets or paths here — those go in .env
