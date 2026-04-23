# src/rl/callbacks.py
# Member B — Shizuka Takao
#
# Custom Stable-Baselines3 callbacks for training monitoring and checkpointing.
#
# SB3 callbacks are hooks that fire at specific points during training
# (every N steps, every episode end, etc.). They enable saving checkpoints,
# logging custom metrics, and early stopping.
#
# Callbacks to implement:
#
#   class CheckpointCallback(BaseCallback):
#       → saves model every N timesteps to models/checkpoint_{step}.zip
#       → keeps only the last K checkpoints to save disk space
#
#   class RewardLoggerCallback(BaseCallback):
#       → logs episode reward and length to results/training_log.csv
#       → called at episode end via locals["infos"]
#       → used to generate learning curves in notebooks/
#
#   class SuccessRateCallback(BaseCallback):
#       → runs eval episodes every N steps on a held-out level set
#       → logs success rate over time
#       → saves best model seen so far to models/best_model.zip
#
#   class EarlyStoppingCallback(BaseCallback):
#       → stops training if success rate exceeds threshold
#       → optional: useful for curriculum learning experiments
#
# Notes:
#   - All callbacks extend BaseCallback from stable_baselines3.common.callbacks
#   - self.model, self.num_timesteps available inside callback methods
#   - Log to CSV in results/ for later analysis, not just tensorboard
