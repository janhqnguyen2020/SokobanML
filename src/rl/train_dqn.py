# src/rl/train_dqn.py
# Member B — Shizuka Takao
#
# Training script for DQN (Deep Q-Network) agent on Sokoban.
#
# DQN learns a Q-function Q(s, a) that estimates expected cumulative reward
# for taking action a in state s. Uses a replay buffer and target network
# to stabilize training.
#
# This script uses Stable-Baselines3's DQN, or optionally a custom
# implementation via network.py if SB3's DQN is too limiting.
#
# DQN vs PPO tradeoffs:
#   DQN  → sample efficient (replay buffer), slower to converge on sparse rewards
#   PPO  → less sample efficient, more stable on long-horizon tasks
#   → run both, compare in results
#
# Responsibilities:
#   - Set up single Sokoban env (DQN doesn't use vectorized envs well)
#   - Configure DQN hyperparameters (lr, buffer_size, exploration schedule)
#   - Attach callbacks for checkpointing
#   - Run model.learn()
#   - Save model to models/dqn_final.zip
#
# Key hyperparameters:
#   - learning_rate: 1e-4
#   - buffer_size: 50_000 (replay buffer capacity)
#   - learning_starts: 10_000 (steps before first gradient update)
#   - exploration_fraction: 0.2 (fraction of training spent exploring)
#   - exploration_final_eps: 0.05 (final epsilon for epsilon-greedy)
#   - target_update_interval: 1000 steps
#
# CLI args to support:
#   --timesteps, --levels, --save_path, --log_dir
#
# Notes:
#   - DQN expects flat or CNN observations — match network.py architecture
#   - Logs go to results/ for Member C's evaluation scripts
