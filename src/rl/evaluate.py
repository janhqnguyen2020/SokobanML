# src/rl/evaluate.py
# Member B — Shizuka Takao / Member C — Joseph Nguyen
#
# Evaluation script for trained RL agents.
#
# Loads a saved model checkpoint and runs it on a set of test levels
# to measure performance metrics comparable to the classical planners.
# Results are saved to results/ so Member C can run cross-method analysis.
#
# Metrics collected (must match planner_runner.py schema):
#   - solved: bool — did agent reach all boxes on goals?
#   - steps: int — number of actions taken
#   - runtime_ms: float — wall time for the episode
#   - total_reward: float — cumulative reward received
#
# Functions to implement:
#   - load_model(model_path, algo)
#       → load PPO.load() or DQN.load() from Stable-Baselines3
#       → algo: "ppo" or "dqn"
#
#   - evaluate_episode(model, env)
#       → run one episode deterministically (deterministic=True)
#       → returns metrics dict for that episode
#
#   - evaluate_model(model_path, algo, level_ids, n_episodes, output_path)
#       → outer loop: for each level, run n_episodes
#       → aggregate: success rate, mean steps, mean reward
#       → save to results/rl_{algo}_results.csv
#
#   - main()
#       → CLI: --model, --algo, --levels, --episodes, --output
#
# Usage (intended):
#   python evaluate.py --model models/ppo_final.zip --algo ppo --output results/ppo_results.csv
#
# Notes:
#   - Use evaluate_policy() from SB3 for quick sanity checks
#   - For detailed per-level metrics, use the custom evaluate_model() above
#   - Deterministic=True means argmax action, no exploration
