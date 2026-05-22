# src/env/render.py
# Member A — Quang Dinh Tue Tran
#
# Renders Sokoban gameplay and saves output as GIF or MP4.
#
# Used to create demo videos for the final presentation and report.
# Should be callable with a list of actions (from a planner or RL agent)
# and produce a visual replay of the solution.
#
# Responsibilities:
#   - Step through an action sequence and capture each frame
#   - Save frames as animated GIF (imageio) or MP4 (opencv)
#   - Support side-by-side rendering for BFS vs RL comparison videos
#
# Functions to implement:
#   - capture_frames(env, actions)
#       → reset env, step through each action, collect RGB frames
#       → returns list of numpy arrays (H, W, 3)
#
#   - save_gif(frames, output_path, fps)
#       → write frames to .gif using imageio
#
#   - save_mp4(frames, output_path, fps)
#       → write frames to .mp4 using cv2.VideoWriter
#
#   - render_solution(env, actions, output_path, fmt)
#       → wrapper: calls capture_frames then save_gif or save_mp4
#       → fmt: "gif" or "mp4"
#
#   - render_comparison(env, bfs_actions, rl_actions, output_path)
#       → renders both solutions side by side in a single video
#       → concatenate frames horizontally with numpy
#
# Output goes to: videos/
#
# Dependencies:
#   - imageio (for GIF)
#   - opencv-python (for MP4)
#   - numpy
