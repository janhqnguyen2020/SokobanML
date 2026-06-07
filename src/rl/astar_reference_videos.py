"""Save A* solution videos for the same deterministic validation showcase cases."""

import json
import os

from src.rl.imitation import (
    create_custom_sokoban_env,
    create_procedural_map_config,
    solve_map_with_astar,
)
from src.rl.validation_videos import (
    build_validation_video_specs,
    build_video_path,
    normalize_frame,
    write_video_file,
)

REFERENCE_HOLD_FRAMES = 3


def save_astar_reference_videos(map_configs, output_dir, split_label):
    """Save one expert A* replay video for every deterministic validation case."""
    os.makedirs(output_dir, exist_ok=True)
    saved_rows = []
    for video_spec in build_validation_video_specs(map_configs):
        saved_rows.append(save_one_astar_reference_video(video_spec, output_dir, split_label))
    write_astar_reference_summary(output_dir, saved_rows)
    return saved_rows


def save_one_astar_reference_video(video_spec, output_dir, split_label):
    """Solve one validation case with A* and save its replay as an mp4 file."""
    map_config = build_reference_map_config(video_spec)
    solve_result = solve_map_with_astar(map_config)
    frames = collect_reference_frames(map_config, solve_result["primitive_actions"])
    video_path = build_video_path(output_dir, split_label, video_spec["video_tag"])
    write_video_file(video_path, frames)
    return build_reference_row(video_spec, map_config, solve_result, video_path, split_label)


def build_reference_map_config(video_spec):
    """Return the fixed map used for one expert-reference video."""
    if video_spec["kind"] == "fixed":
        return video_spec["map_config"]
    return create_procedural_map_config(
        video_spec["env_id"],
        video_spec["seed"],
        video_spec["episode_index"],
    )


def collect_reference_frames(map_config, primitive_actions):
    """Replay the solved A* actions on the same map and collect rgb frames."""
    env = create_custom_sokoban_env(map_config)
    try:
        frames = [render_reference_frame(env)]
        replay_reference_actions(env, primitive_actions, frames)
    finally:
        env.close()
    return frames


def replay_reference_actions(env, primitive_actions, frames):
    """Append every replayed A* step, or hold the start frame if A* failed."""
    if primitive_actions is None:
        hold_last_frame(frames)
        return
    for primitive_action in primitive_actions:
        env.step(int(primitive_action))
        frames.append(render_reference_frame(env))
    hold_last_frame(frames)


def render_reference_frame(env):
    """Render one normalized expert-reference frame from the custom Sokoban env."""
    return normalize_frame(env.render(mode="rgb_array"))


def hold_last_frame(frames):
    """Repeat the final frame a few times so the saved video pauses at the end."""
    last_frame = frames[-1]
    for _ in range(REFERENCE_HOLD_FRAMES):
        frames.append(last_frame)


def build_reference_row(video_spec, map_config, solve_result, video_path, split_label):
    """Save the small metadata row that describes one expert-reference video."""
    row = {
        "split_label": str(split_label),
        "video_tag": str(video_spec["video_tag"]),
        "kind": str(video_spec["kind"]),
        "video_path": str(video_path),
        "map_name": str(map_config["map_name"]),
        "astar_solved": bool(solve_result["primitive_actions"] is not None),
        "solution_length": int(solve_result["solution_length"]),
        "num_pushes": int(solve_result["num_pushes"]),
        "failure_reason": str(solve_result["failure_reason"]),
    }
    if video_spec["kind"] == "procedural":
        row["env_id"] = str(video_spec["env_id"])
        row["episode_index"] = int(video_spec["episode_index"])
    return row


def write_astar_reference_summary(output_dir, rows):
    """Write the JSON summary that lists every saved expert-reference video."""
    summary_path = os.path.join(output_dir, "astar_reference_videos.json")
    with open(summary_path, "w") as output_file:
        json.dump(rows, output_file, indent=2)
