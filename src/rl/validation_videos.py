"""Helpers for saving validation videos that match the exact evaluated cases."""

import json
import os

import imageio
import numpy as np

from src.rl.evaluate import load_model
from src.rl.train_curriculum_dqn import createCurriculumEvalEnvironment, createEvaluationEnvironment


def save_validation_videos(model_path, fixed_maps, procedural_specs, output_dir, split_label):
    """Save videos for the same fixed maps and procedural episodes used in validation."""
    model = load_model(model_path, "dqn")
    os.makedirs(output_dir, exist_ok=True)
    saved_rows = []
    for video_spec in build_validation_video_specs(fixed_maps, procedural_specs):
        saved_rows.append(save_one_validation_video(model, output_dir, split_label, video_spec))
    write_video_summary(output_dir, saved_rows)
    return saved_rows


def build_validation_video_specs(fixed_maps, procedural_specs):
    """Package the exact fixed and procedural validation cases into video specs."""
    return build_fixed_video_specs(fixed_maps) + build_procedural_video_specs(procedural_specs)


def build_fixed_video_specs(fixed_maps):
    """Build one fixed video spec for each validated fixed map."""
    return [fixed_video_spec(map_config, fixed_video_tag(map_config)) for map_config in fixed_maps]


def fixed_video_spec(map_config, video_tag):
    """Package one fixed validation map into a reusable video request."""
    return {"video_tag": str(video_tag), "kind": "fixed", "map_config": map_config}


def fixed_video_tag(map_config):
    """Build one readable fixed video tag from the validated map name."""
    return f"fixed_{map_config['map_name']}"


def build_procedural_video_specs(procedural_specs):
    """Build one procedural video spec for each validated procedural episode."""
    return [procedural_video_spec(procedural_spec) for procedural_spec in procedural_specs]


def procedural_video_spec(procedural_spec):
    """Package one validated procedural episode into a reusable video request."""
    return {
        "video_tag": procedural_video_tag(procedural_spec),
        "kind": "procedural",
        "env_id": str(procedural_spec["env_id"]),
        "seed": int(procedural_spec["seed"]),
        "episode_index": int(procedural_spec["episode_index"]),
    }


def procedural_video_tag(procedural_spec):
    """Build one readable procedural video tag from the validated env and episode."""
    env_tag = str(procedural_spec["env_id"]).replace("-", "_").lower()
    episode_index = int(procedural_spec["episode_index"]) + 1
    return f"{env_tag}_{episode_index:03d}"


def save_one_validation_video(model, output_dir, split_label, video_spec):
    """Run one deterministic validation case and save its replay as a video."""
    env = build_video_env(video_spec)
    prepare_video_env(env)
    try:
        frames, result = collect_video_frames(model, env, video_seed(video_spec))
    finally:
        env.close()
    video_path = build_video_path(output_dir, split_label, video_spec["video_tag"])
    write_video_file(video_path, frames)
    return build_video_row(video_spec, split_label, video_path, result)


def build_video_env(video_spec):
    """Create the fixed or procedural env used for one saved validation video."""
    if video_spec["kind"] == "fixed":
        return createCurriculumEvalEnvironment([video_spec["map_config"]])
    return createEvaluationEnvironment(env_id=video_spec["env_id"], seed=video_spec["seed"])


def prepare_video_env(env):
    """Turn on primitive-step frame capture so videos match planner-style movement."""
    env.enable_primitive_video_frames()


def video_seed(video_spec):
    """Return the deterministic reset seed used by one saved validation video."""
    if video_spec["kind"] == "fixed":
        return None
    return int(video_spec["seed"])


def collect_video_frames(model, env, reset_seed):
    """Run one evaluation episode while capturing rgb frames for the saved video."""
    observation = env.reset() if reset_seed is None else env.reset(seed=int(reset_seed))
    frames = [normalize_frame(env.render(mode="rgb_array"))]
    done = False
    info = {}
    while not done:
        observation, done, info = play_one_video_macro_step(model, env, observation, frames)
    return frames, dict(info)


def play_one_video_macro_step(model, env, observation, frames):
    """Advance one macro action and append every primitive-step frame it produced."""
    action, _ = model.predict(observation, deterministic=True)
    next_observation, _, done, info = env.step(int(action))
    append_macro_step_frames(env, frames)
    return next_observation, done, info


def append_macro_step_frames(env, frames):
    """Append primitive-step frames, or one fallback frame when none were captured."""
    macro_frames = env.consume_primitive_video_frames()
    if not macro_frames:
        macro_frames = [env.render(mode="rgb_array")]
    for frame in macro_frames:
        frames.append(normalize_frame(frame))


def normalize_frame(frame):
    """Convert one frame into a standard three-channel image array."""
    frame_array = np.array(frame)
    if frame_array.ndim == 2:
        return np.stack([frame_array] * 3, axis=-1)
    if frame_array.ndim == 3 and frame_array.shape[2] == 4:
        return frame_array[:, :, :3]
    return frame_array


def build_video_path(output_dir, split_label, video_tag):
    """Build the full mp4 path for one saved validation video."""
    return os.path.join(output_dir, f"{split_label}_{video_tag}.mp4")


def write_video_file(video_path, frames):
    """Write one mp4 file from the captured validation frames."""
    imageio.mimsave(video_path, frames, fps=3, codec="libx264")


def build_video_row(video_spec, split_label, video_path, result):
    """Build one summary row for a saved validation video."""
    row = {
        "split_label": str(split_label),
        "video_tag": str(video_spec["video_tag"]),
        "kind": str(video_spec["kind"]),
        "video_path": str(video_path),
        "solved": bool(result.get("all_boxes_on_target", False)),
        "termination_reason": str(result.get("termination_reason", "unknown")),
        "boxes_on_target": int(result.get("boxes_on_target", 0)),
    }
    if video_spec["kind"] == "fixed":
        row["map_name"] = str(video_spec["map_config"]["map_name"])
    else:
        row["env_id"] = str(video_spec["env_id"])
        row["episode_index"] = int(video_spec["episode_index"])
    return row


def write_video_summary(output_dir, rows):
    """Save one JSON summary beside the stage validation videos."""
    summary_path = os.path.join(output_dir, "validation_videos.json")
    with open(summary_path, "w") as output_file:
        json.dump(rows, output_file, indent=2)
