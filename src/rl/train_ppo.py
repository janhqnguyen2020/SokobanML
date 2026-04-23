# src/rl/train_ppo.py
# Member B — Shizuka Takao
#
# Training script for PPO (Proximal Policy Optimization) agent on Sokoban.
#
# Uses Stable-Baselines3's PPO implementation, which handles the training loop,
# rollout collection, advantage estimation, and policy updates internally.
# This script configures the environment, hyperparameters, and callbacks,
# then launches training and saves the final model.
#
# PPO is preferred over DQN here because:
#   - Works well with discrete action spaces
#   - More stable training via clipped objective
#   - Handles sparse rewards better with longer rollouts
#
# Responsibilities:
#   - Create vectorized training environment (SokobanEnvWrapper × N)
#   - Configure PPO hyperparameters (lr, n_steps, clip_range, etc.)
#   - Attach callbacks: checkpoint saver, reward logger
#   - Run model.learn(total_timesteps)
#   - Save final model to models/ppo_final.zip
#
# Key hyperparameters to tune:
#   - learning_rate: 3e-4 default
#   - n_steps: rollout length per update (e.g. 2048)
#   - n_epochs: gradient steps per update (e.g. 10)
#   - clip_range: PPO clip epsilon (0.2 default)
#   - ent_coef: entropy bonus to encourage exploration
#
# CLI args to support:
#   --timesteps   total training steps (default 1_000_000)
#   --levels      which level set to train on
#   --save_path   where to write final model
#   --log_dir     tensorboard / CSV log directory
#
# Notes:
#   - Use SubprocVecEnv for parallel envs if CPU allows
#   - Logs go to results/ for later analysis by Member C
