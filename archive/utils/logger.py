# src/utils/logger.py
# Member C — Joseph Nguyen
#
# CSV logging utilities for experiment results.
#
# Both planners and RL evaluation scripts call these functions to persist
# results to disk. Keeping logging centralized ensures consistent file
# formats that the notebooks can read without preprocessing.
#
# Functions to implement:
#   - init_csv(output_path, fieldnames)
#       → create a new CSV file with header row
#       → if file exists, warn but overwrite (or append with flag)
#
#   - append_row(output_path, row_dict)
#       → append one result dict to existing CSV
#       → used inside planner_runner.py and evaluate.py loops
#
#   - load_results(csv_path)
#       → read CSV into Pandas DataFrame
#       → auto-cast columns to correct types (bool, int, float)
#       → returns DataFrame ready for metrics.py
#
#   - log_training_step(log_path, timestep, reward, success)
#       → lightweight logger for RL training curves
#       → appends one row: {timestep, mean_reward, success_rate}
#       → called from callbacks.py
#
# Notes:
#   - Use Python's csv module for writing (no Pandas dep for writing)
#   - Use Pandas for reading (load_results) — richer dtype handling
#   - All output files go to results/
#   - Thread-safe writes not required (single-process logging)
