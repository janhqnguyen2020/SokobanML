"""Selected-case A* audit videos for imitation-learning curriculum maps."""

import csv
import json
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np

from src.planners.astar import AStarAgent
from src.rl.imitation import create_custom_sokoban_env
from src.utils.show_ui import create_plot, update_plot


class AStarAuditCollector:
    """Collect frames for one A* case and optionally mirror them in a small UI."""

    def __init__(self, env, map_name, show_ui, fps, search_stride, capture_search_frames):
        """Store the rendering settings used while auditing one curriculum map."""
        self.env = env
        self.map_name = map_name
        self.show_ui = bool(show_ui)
        self.fps = max(int(fps), 1)
        self.search_stride = max(int(search_stride), 1)
        self.capture_search_frames = bool(capture_search_frames)
        self.frames = []
        self.figure = None
        self.axis = None
        self.image = None

    def capture_initial_frame(self):
        """Save the starting board before A* begins searching."""
        title_text = f"A* {self.map_name} | initial board"
        self.capture_frame(self.env.render(mode="rgb_array"), title_text)

    def search_callback(self, player_pos, box_positions, nodes_expanded, event_name):
        """Capture a sampled search state while A* explores this map."""
        if not self.capture_search_frames:
            return
        if not should_capture_search_frame(nodes_expanded, event_name, self.search_stride):
            return
        apply_search_state(self.env, player_pos, box_positions)
        title = search_frame_title(self.map_name, nodes_expanded, event_name)
        self.capture_frame(self.env.render(mode="rgb_array"), title)

    def capture_status_frame(self, title_text, hold_seconds=2.0):
        """Save one extra frame that summarizes the final A* search outcome."""
        hold_frames = hold_frame_count(self.fps, hold_seconds)
        self.capture_frame(self.env.render(mode="rgb_array"), title_text, hold_frames)

    def capture_replay_frame(self, replay_env, step_index, total_steps):
        """Save one frame from the primitive-action replay of a solved plan."""
        title = replay_frame_title(self.map_name, step_index, total_steps)
        self.capture_frame(replay_env.render(mode="rgb_array"), title)

    def capture_frame(self, frame, title_text, hold_frames=1):
        """Append one frame and optionally keep it visible for extra time."""
        for _ in range(int(max(hold_frames, 1))):
            self._add_frame(frame, title_text)

    def save_video(self, output_path):
        """Write the collected frames to one mp4 file."""
        if not self.frames:
            return None
        normalized_frames = [normalize_frame(frame) for frame in self.frames]
        imageio.mimsave(output_path, normalized_frames, fps=self.fps, codec="libx264")
        self._close_plot()
        return output_path

    def _add_frame(self, frame, title_text):
        """Append one frame and refresh the optional live preview window."""
        self.frames.append(frame)
        if not self.show_ui:
            return
        self._update_plot(frame, title_text)

    def _update_plot(self, frame, title_text):
        """Create or refresh the small live preview that mirrors saved frames."""
        if self.figure is None:
            self.figure, self.axis, self.image = create_plot(frame, title_text)
            return
        update_plot(self.figure, self.axis, self.image, frame, title_text, 1.0 / self.fps)

    def _close_plot(self):
        """Close the optional preview window after this map finishes."""
        if self.figure is None:
            return
        plt.close(self.figure)


def normalize_frame(frame):
    """Convert one rendered frame into a standard three-channel video frame."""
    frame_array = np.array(frame)
    if frame_array.ndim == 2:
        return np.stack([frame_array] * 3, axis=-1)
    if frame_array.ndim == 3 and frame_array.shape[2] == 4:
        return frame_array[:, :, :3]
    return frame_array


def should_capture_search_frame(nodes_expanded, event_name, search_stride):
    """Keep only a sampled subset of search states so videos stay readable."""
    if event_name == "goal":
        return True
    if event_name == "start":
        return False
    if int(nodes_expanded) <= 10:
        return True
    return int(nodes_expanded) % int(search_stride) == 0


def hold_frame_count(fps, hold_seconds):
    """Convert one hold duration into a repeated-frame count."""
    return max(1, int(round(float(fps) * float(hold_seconds))))


def search_frame_title(map_name, nodes_expanded, event_name):
    """Build the status line used for sampled search snapshots."""
    return f"A* {map_name} | {event_name} | expanded {nodes_expanded}"


def replay_frame_title(map_name, step_index, total_steps):
    """Build the title used while replaying a solved primitive plan."""
    return f"A* {map_name} | replay {step_index}/{total_steps}"


def apply_search_state(env, player_pos, box_positions):
    """Mutate the custom env to mirror one abstract planner state for rendering."""
    room_fixed = env.room_fixed
    env.room_state = build_room_state(room_fixed, player_pos, box_positions)
    env.player_position = np.array([player_pos[0], player_pos[1]])
    env.boxes_on_target = count_boxes_on_targets(room_fixed, box_positions)


def build_room_state(room_fixed, player_pos, box_positions):
    """Construct the full Sokoban board array from fixed tiles and planner state."""
    room_state = np.where(room_fixed == 0, 0, 1).astype(np.uint8)
    room_state[room_fixed == 2] = 2
    add_box_tiles(room_state, room_fixed, box_positions)
    room_state[player_pos] = 6 if room_fixed[player_pos] == 2 else 5
    return room_state


def add_box_tiles(room_state, room_fixed, box_positions):
    """Place all boxes on either floor or goal tiles in the rendered board."""
    for box_pos in box_positions:
        room_state[box_pos] = 3 if room_fixed[box_pos] == 2 else 4


def count_boxes_on_targets(room_fixed, box_positions):
    """Count how many boxes currently sit on goal squares."""
    return sum(1 for box_pos in box_positions if room_fixed[box_pos] == 2)


def select_audit_cases(map_results, max_success_cases):
    """Choose the first few successful demo cases plus every failed case."""
    success_cases = [result for result in sorted(map_results, key=case_sort_key) if result["demo_ready"]]
    failed_cases = [result for result in sorted(map_results, key=case_sort_key) if not result["demo_ready"]]
    return success_cases[: int(max_success_cases)] + failed_cases


def case_sort_key(map_result):
    """Keep the audit selection order stable across repeated runs."""
    return int(map_result["num_boxes"]), str(map_result["map_name"])


def build_map_lookup(map_configs):
    """Index the training maps by name so audit rows can find the real config."""
    return {map_config["map_name"]: map_config for map_config in map_configs}


def save_astar_audit_videos(map_configs, demo_payload, output_dir, max_success_cases, show_ui, fps, search_stride):
    """Save mp4 audits for 5 good cases and every failed case by default."""
    os.makedirs(output_dir, exist_ok=True)
    map_lookup = build_map_lookup(map_configs)
    selected_cases = select_audit_cases(demo_payload["map_results"], max_success_cases)
    saved_rows = []
    for map_result in selected_cases:
        map_config = map_lookup[map_result["map_name"]]
        video_path = save_astar_case_video(map_config, map_result, output_dir, show_ui, fps, search_stride)
        saved_rows.append(build_audit_row(map_result, video_path))
    write_audit_outputs(output_dir, saved_rows)
    return saved_rows


def save_astar_case_video(map_config, map_result, output_dir, show_ui, fps, search_stride):
    """Solve one audit case, sample its search, and save the resulting mp4."""
    search_env = create_custom_sokoban_env(map_config)
    collector = build_audit_collector(search_env, map_result, show_ui, fps, search_stride)
    if collector.capture_search_frames:
        collector.capture_initial_frame()
    solve_result = solve_with_audit_callback(search_env, collector)
    save_case_frames(map_config, map_result, solve_result, search_env, collector)
    search_env.close()
    video_path = os.path.join(output_dir, video_filename(map_result))
    return collector.save_video(video_path)


def build_audit_collector(search_env, map_result, show_ui, fps, search_stride):
    """Choose whether this case should record search states or just the replay."""
    capture_search_frames = not bool(map_result["astar_solved"])
    return AStarAuditCollector(search_env, map_result["map_name"], show_ui, fps, search_stride, capture_search_frames)


def save_case_frames(map_config, map_result, solve_result, search_env, collector):
    """Save either search-failure frames or replay-success frames for one case."""
    if solve_result["primitive_actions"] is None:
        collector.capture_status_frame(status_title(map_result, solve_result))
        return
    final_frame = replay_solution_frames(map_config, solve_result["primitive_actions"], collector)
    hold_frames = hold_frame_count(collector.fps, 2.0)
    collector.capture_frame(final_frame, status_title(map_result, solve_result), hold_frames)


def solve_with_audit_callback(search_env, collector):
    """Run A* on the given env so sampled search frames match the real search."""
    agent = AStarAgent(search_env, state_callback=collector.search_callback)
    primitive_actions = agent._solve()
    return {
        "primitive_actions": primitive_actions,
        "failure_reason": agent.failure_reason or "no_solution_found",
    }


def status_title(map_result, solve_result):
    """Summarize the final planner outcome on the last audit frame."""
    failure_reason = str(map_result["failure_reason"])
    if solve_result["primitive_actions"] is None:
        return f"A* {map_result['map_name']} | failed | {failure_reason}"
    return f"A* {map_result['map_name']} | solved | {failure_reason}"


def replay_solution_frames(map_config, primitive_actions, collector):
    """Replay one solved primitive plan so the saved video shows the final route."""
    replay_env = create_custom_sokoban_env(map_config)
    collector.capture_replay_frame(replay_env, 0, max(len(primitive_actions), 1))
    for step_index, action in enumerate(primitive_actions, start=1):
        replay_env.step(int(action))
        collector.capture_replay_frame(replay_env, step_index, len(primitive_actions))
    final_frame = replay_env.render(mode="rgb_array")
    replay_env.close()
    return final_frame


def video_filename(map_result):
    """Use descriptive filenames so saved mp4s are easy to scan later."""
    video_tag = "success" if map_result["demo_ready"] else "failed"
    return f"{video_tag}_{map_result['map_name']}.mp4"


def build_audit_row(map_result, video_path):
    """Flatten one saved case so the audit folder includes a quick summary table."""
    return {
        "map_name": map_result["map_name"],
        "num_boxes": int(map_result["num_boxes"]),
        "astar_solved": bool(map_result["astar_solved"]),
        "demo_ready": bool(map_result["demo_ready"]),
        "failure_reason": str(map_result["failure_reason"]),
        "expert_pairs": int(map_result["expert_pairs"]),
        "nodes_expanded": int(map_result["nodes_expanded"]),
        "deadlocks_pruned": int(map_result["deadlocks_pruned"]),
        "solution_length": int(map_result["solution_length"]),
        "num_pushes": int(map_result["num_pushes"]),
        "video_path": video_path or "",
    }


def write_audit_outputs(output_dir, rows):
    """Save both CSV and JSON summaries beside the audit videos."""
    write_audit_csv(os.path.join(output_dir, "astar_audit_summary.csv"), rows)
    write_audit_json(os.path.join(output_dir, "astar_audit_summary.json"), rows)


def write_audit_csv(path, rows):
    """Write one flat CSV summary of all saved audit cases."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_json(path, rows):
    """Write one JSON summary of the saved audit cases."""
    with open(path, "w") as output_file:
        json.dump(rows, output_file, indent=2)
