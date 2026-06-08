# src/utils/config.py
"""
Shared configuration for Sokoban experiments.
"""

import os

# Environment
# Keep training on the smaller 7x7 3-box distribution first. It remains the
# cleanest place to build a stronger policy before another transfer attempt.
ENV_ID = "Sokoban-small-v1"
MAX_STEPS = 200
NUM_EPISODES = 20
HIGH_LEVEL_OBS_CANVAS_SHAPE = (15, 15)
# The richer 422-dim observation has not beaten the simpler 412-dim setup yet,
# so keep the stronger 412-dim observation family.
HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES = False

# Primitive DQN baseline defaults
DQN_BUFFER_SIZE = 10_000
DQN_TOTAL_STEPS = 50_000

# High-level masked DQN defaults
# The flat MLP path appears capped in the low-teens on fair benchmarks. The
# next experiment keeps the same 412-dim observation format, but switches to a
# spatial encoder so the policy can exploit board structure directly.
HIGH_LEVEL_DQN_BACKBONE = "cnn"
HIGH_LEVEL_DQN_CNN_FEATURES_DIM = 256
HIGH_LEVEL_DQN_BUFFER_SIZE = 50_000
HIGH_LEVEL_DQN_TOTAL_STEPS = 20_000
HIGH_LEVEL_DQN_LEARNING_RATE = 1e-4
HIGH_LEVEL_DQN_LEARNING_STARTS = 500
HIGH_LEVEL_DQN_BATCH_SIZE = 64
HIGH_LEVEL_DQN_GAMMA = 0.99
HIGH_LEVEL_DQN_TRAIN_FREQ = 4
HIGH_LEVEL_DQN_GRADIENT_STEPS = 1
HIGH_LEVEL_DQN_TARGET_UPDATE_INTERVAL = 2_000
# Fresh training needs broad exploration again.
HIGH_LEVEL_DQN_EXPLORATION_INITIAL_EPS = 1.0
HIGH_LEVEL_DQN_EXPLORATION_FRACTION = 0.30
HIGH_LEVEL_DQN_EXPLORATION_FINAL_EPS = 0.02
HIGH_LEVEL_DQN_POLICY_HIDDEN_SIZES = [256, 256]
HIGH_LEVEL_DQN_EVAL_FREQ = 10_000
# Use a larger validation sample so checkpoint selection is less noisy than the
# earlier 20-episode runs that swung between 0% and 30%.
HIGH_LEVEL_DQN_EVAL_EPISODES = 40
HIGH_LEVEL_DQN_SELECTION_EPISODES = 100
HIGH_LEVEL_DQN_EARLY_STOP_PATIENCE_EVALS = 6
HIGH_LEVEL_DQN_EARLY_STOP_MIN_TIMESTEPS = 100_000
HIGH_LEVEL_DQN_FINAL_VERIFY_EPISODES = 100
# The CNN backbone is a clean architecture shift, so the next run starts fresh.
HIGH_LEVEL_DQN_INIT_MODEL_PATH = None

# Curriculum / Generalized DQN  (parallel model — does NOT affect high_level_dqn)
# Canvas is 15x15 so all 10 curriculum maps fit (largest is 11x13).
# Action mask always padded to MAX_BOXES*4=12 so 1-box and 2-box episodes are
# compatible with the same network weights as 3-box episodes.
CURRICULUM_DQN_CANVAS_SHAPE = (15, 15)
CURRICULUM_DQN_MAX_BOXES = 3               # fixed for all phases: 1-box→2-box→5-box
CURRICULUM_DQN_PROCEDURAL_FRACTION = 0.40  # fraction of episodes from small-v1
CURRICULUM_DQN_TOTAL_STEPS = 300_000
CURRICULUM_DQN_BUFFER_SIZE = 100_000
CURRICULUM_DQN_LEARNING_RATE = 1e-4
CURRICULUM_DQN_LEARNING_STARTS = 1_000
CURRICULUM_DQN_BATCH_SIZE = 64
CURRICULUM_DQN_EVAL_FREQ = 20_000
CURRICULUM_DQN_EVAL_EPISODES = 40
CURRICULUM_DQN_EARLY_STOP_PATIENCE_EVALS = 8
CURRICULUM_DQN_EARLY_STOP_MIN_TIMESTEPS = 80_000
CURRICULUM_DQN_SELECTION_EPISODES = 100

# Curriculum epsilon per-stage schedule.
# Each tuple: (stage_start_fraction, stage_end_fraction, eps_start, eps_end)
# Stage boundaries mirror CurriculumTeacher: 0-30% easy, 30-70% mid, 70-100% hard.
# When CURRICULUM_EPS_USE_STAGE_RESET=True, epsilon resets to eps_start at each
# stage boundary instead of continuing the global 1.0→0.02 decay.
CURRICULUM_EPS_STAGE_SCHEDULE = [
    (0.00, 0.30, 1.0,  0.10),   # Stage 1: 1-box heavy,  1.0 → 0.10
    (0.30, 0.70, 0.5,  0.05),   # Stage 2: 2-box heavy,  reset to 0.5 → 0.05
    (0.70, 1.00, 0.3,  0.02),   # Stage 3: 3-box heavy,  reset to 0.3 → 0.02
]
CURRICULUM_EPS_USE_STAGE_RESET = True   # False = baseline (global 1.0→0.02 decay)

# Reproducibility
# Allow fast multi-seed sweeps from PowerShell, e.g.:
#   $env:SOKO_SEED='43'; python main_dqn.py
SEED = int(os.getenv("SOKO_SEED", "42"))
