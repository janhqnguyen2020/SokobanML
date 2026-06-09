# run_experiments.py
"""Batch benchmark runner for custom and original Sokoban environments."""
# python run_experiments.py --sources curriculum --algorithms all

import argparse
import csv
import time

from src.env.custom_env import SimpleCustomSokobanEnv
from src.env.sokoban_env import initialize_env
from src.utils.benchmark_registry import PLANNER_REGISTRY
from src.utils.custom_maps import select_custom_maps
from src.utils.result_paths import build_algorithm_runs_path
from src.utils.result_paths import build_algorithm_summary_path
from src.utils.result_paths import build_board_csv_path
from src.utils.result_paths import ensure_parent_folder
from src.utils.show_ui import build_title
from src.utils.show_ui import create_plot
from src.utils.show_ui import finish_plot
from src.utils.show_ui import print_step
from src.utils.show_ui import update_plot
from src.utils.show_ui import GifRecorder

GROUP_DISPLAY_NAMES = {
    "custom_core": "custom_shizuka",
    "canvas":      "canvas_dqn_compatible",
    "dqn_custom":  "dqn_compatible",
    "curriculum":  "curriculum",
    "original_v0": "v0",
    "original_v1": "v1",
    "original_v2": "v2",
    "original_small_v0": "small_v0",
    "original_small_v1": "small_v1",
    "original_large_v0": "large_v0",
    "original_large_v1": "large_v1",
    "original_large_v2": "large_v2",
    "original_huge_v0": "huge_v0",
    # archived — only used when --sources archived is passed explicitly
    "archived":    "archived_joseph",
}


RUN_FIELDS = [
    "source",
    "group",
    "env_version",
    "map_name",
    "difficulty",
    "episode",
    "algorithm",
    "seed",
    "solved",
    "num_steps",
    "num_pushes",
    "runtime_ms",
    "nodes_expanded",
    "failure_reason",
    "deadlocks_pruned",
    "dead_squares",
    # "map_code",
    "height",
    "width",
    "num_boxes",
    "total_reward"
]

SUMMARY_FIELDS = [
    "algorithm",
    "group",
    "total_runs",
    "solved_runs",
    "success_rate",
    "avg_num_steps",
    "avg_num_pushes",
    "avg_runtime_ms",
    "avg_nodes_expanded",
    "avg_deadlocks_pruned",
]


def parse_args():
    """Read CLI options so one runner can handle full or partial benchmarks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=["all"])
    parser.add_argument("--algorithms", nargs="+", default=["all"])
    parser.add_argument("--envs", nargs="+", default=["v0", "v1", "v2"])
    parser.add_argument("--maps", nargs="+", default=[])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=100)
    parser.add_argument("--show-ui", action="store_true")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--save-gif", type=str, default=None, help="Path to save gif (e.g. results/gifs/bfs.gif)")
    return parser.parse_args()


def expand_sources(source_names):
    """Replace all with every active benchmark source (archived excluded)."""
    if "all" in source_names:
        return ["custom_core", "canvas", "original"]
    return source_names


def expand_algorithms(algorithm_names):
    """Replace all with every registered planner key."""
    if "all" in algorithm_names:
        return list(PLANNER_REGISTRY.keys())
    return algorithm_names


def expand_envs(env_names):
    env_map = {
        "v0": "Sokoban-v0",
        "v1": "Sokoban-v1",
        "v2": "Sokoban-v2",
        "small_v0": "Sokoban-small-v0",
        "small_v1": "Sokoban-small-v1",
        "large_v0": "Sokoban-large-v0",
        "large_v1": "Sokoban-large-v1",
        "large_v2": "Sokoban-large-v2",
        "huge_v0": "Sokoban-huge-v0",
    }
    return [env_map[name] for name in env_names]


def ensure_csv_header(file_path, fieldnames):
    """Create a CSV file with header when it does not exist yet."""
    ensure_parent_folder(file_path)
    try:
        with open(file_path, "x", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
    except FileExistsError:
        return


def append_csv_row(file_path, fieldnames, row):
    """Append one finished result row to an existing CSV file."""
    ensure_csv_header(file_path, fieldnames)
    with open(file_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(row)


def run_selected_benchmarks(args):
    """Run every requested benchmark group and collect detailed rows."""
    results = []
    sources = expand_sources(args.sources)
    algorithms = expand_algorithms(args.algorithms)

    custom_sources = {"custom_core", "canvas", "dqn_custom", "curriculum", "additional", "archived"}
    if custom_sources.intersection(sources):
        results.extend(run_custom_groups(args, sources, algorithms))
    if "original" in sources:
        results.extend(run_original_groups(args, algorithms))

    return results


def run_custom_groups(args, source_names, algorithm_names):
    """Run requested planners on the requested custom map groups."""
    results = []
    maps = select_custom_maps(source_names, args.maps)

    for config in maps:
        group_name = choose_custom_group(config)
        for algorithm_name in algorithm_names:
            row = run_custom_case(config, group_name, algorithm_name, args)
            save_run_row(row, group_name, config["map_name"], algorithm_name)
            results.append(row)

    return results

def choose_custom_group(config):
    """Return the source-group label for one custom map config."""
    if config["group_name"]:
        return config["group_name"]
    map_name = config["map_name"]
    if map_name.startswith("dqn_map_") or map_name.startswith("canvas_"):
        return "canvas"
    if map_name.startswith("map_"):
        return "custom_core"
    if map_name.startswith("curr_"):
        return "curriculum"
    return "archived"


def run_custom_case(config, group_name, algorithm_name, args):
    """Run one planner on one fixed custom map."""
    planner_label, planner_class = PLANNER_REGISTRY[algorithm_name]
    map_type = GROUP_DISPLAY_NAMES.get(group_name, group_name)
    env = create_custom_env(config)
    row = run_board(env, planner_class, planner_label, map_type, config["map_name"], args.show_ui, args.delay, save_gif=(args.save_gif is not None), gif_path=args.save_gif)
    env.close()
    return add_custom_metadata(row, config, group_name, algorithm_name)


def create_custom_env(config):
    """Build one fixed custom Sokoban environment from map settings."""
    return SimpleCustomSokobanEnv(
        height=config["height"],
        width=config["width"],
        player_position=config["player"],
        box_positions=config["boxes"],
        goal_positions=config["goals"],
        max_steps=config["max_steps"],
        wall_positions=config["walls"],
    )


def add_custom_metadata(row, config, group_name, algorithm_name):
    """Attach custom benchmark labels to a finished planner row."""
    row["source"] = "custom"
    row["group"] = group_name
    row["env_version"] = "custom_env"
    row["map_name"] = config["map_name"]
    row["difficulty"] = config["difficulty"]
    row["episode"] = 0
    row["algorithm"] = algorithm_name
    row["seed"] = ""
    row["height"] = config["height"]
    row["width"] = config["width"]
    row["num_boxes"] = len(config["boxes"])
    return row


def run_original_groups(args, algorithm_names):
    """Run requested planners on seeded original Sokoban episodes."""
    results = []
    env_ids = expand_envs(args.envs)

    for env_id in env_ids:
        group_name = build_original_group_name(env_id)
        for episode in range(args.episodes):
            seed = args.seed_base + episode
            for algorithm_name in algorithm_names:
                row = run_original_case(env_id, group_name, episode, seed, algorithm_name, args)
                board_name = build_episode_name(episode)
                save_run_row(row, group_name, board_name, algorithm_name)
                results.append(row)

    return results


def build_original_group_name(env_id):
    """Turn a Gym Sokoban env id into a folder name like original_small_v0."""
    suffix_parts = env_id.split("-")[1:]
    suffix = "_".join(part.lower() for part in suffix_parts)
    return f"original_{suffix}"


def build_episode_name(episode):
    """Create a stable episode folder name from the episode index."""
    return f"episode_{episode:02d}"


def run_original_case(env_id, group_name, episode, seed, algorithm_name, args):
    """Run one planner on one seeded original Sokoban episode."""
    planner_label, planner_class = PLANNER_REGISTRY[algorithm_name]
    map_type = GROUP_DISPLAY_NAMES.get(group_name, group_name)
    map_name = build_episode_name(episode)
    env = initialize_env(env_id=env_id, seed=seed)
    row = run_board(env, planner_class, planner_label, map_type, map_name, args.show_ui, args.delay, save_gif=(args.save_gif is not None), gif_path=args.save_gif)
    env.close()
    return add_original_metadata(row, env_id, group_name, episode, seed, algorithm_name)


def add_original_metadata(row, env_id, group_name, episode, seed, algorithm_name):
    """Attach original benchmark labels to a finished planner row."""
    row["source"] = "original"
    row["group"] = group_name
    row["env_version"] = env_id
    row["map_name"] = build_episode_name(episode)
    row["difficulty"] = env_id
    row["episode"] = episode
    row["algorithm"] = algorithm_name
    row["seed"] = seed
    row["height"] = ""
    row["width"] = ""
    row["num_boxes"] = ""
    return row


def run_board(env, planner_class, planner_label, map_type, map_name, show_ui, delay, save_gif=False, gif_path=None):
    """Run one planner on one board and return the measured row fields."""
    planner = planner_class(env)
    observation = env.reset()
    planner.reset()
    map_code = encode_map(env)
    fig, ax, image = start_ui(observation, show_ui, planner_label, map_type, map_name)
   
    gif_recorder = None
    if save_gif and gif_path:
        fps = max(1, int(1 / delay)) if delay > 0 else 5
        gif_recorder = GifRecorder(gif_path, fps=fps)
    if gif_recorder:
        gif_recorder.add_frame(fig)

    reward_sum, num_steps, solved = play_episode(
        env, planner, fig, ax, image, planner_label, map_type, map_name, show_ui, delay, gif_recorder
    )
    finish_ui(show_ui)
    return build_run_row(planner, reward_sum, num_steps, solved, map_code)


def start_ui(observation, show_ui, planner_label, map_type, map_name):
    """Create a matplotlib window only when the user asks for it."""
    if not show_ui:
        return None, None, None
    title = build_title(planner_label, map_type, map_name, 0, 0.0)
    return create_plot(observation, title)


def play_episode(env, planner, fig, ax, image, planner_label, map_type, map_name, show_ui, delay, gif_recorder=None):
    """Step through the environment until the planner finishes."""
    done = False
    num_steps = 0
    reward_sum = 0
    solved = False
    planner.start_time = time.time()

    while not done:
        observation, reward, done, info = run_step(env, planner)
        update_ui(
            fig, ax, image, observation, num_steps, reward_sum + reward, done, info,
            planner_label, map_type, map_name, show_ui, delay, gif_recorder
        )
        if gif_recorder:
            fig.canvas.draw()
            fig.canvas.flush_events()
            gif_recorder.add_frame(fig)

        reward_sum += reward
        num_steps += 1
        solved = info.get("all_boxes_on_target", False)

    if gif_recorder:
        print("GIF frames:", len(gif_recorder.frames))
        gif_recorder.save()
    planner.runtime_ms = (time.time() - planner.start_time) * 1000

    return reward_sum, num_steps, solved


def run_step(env, planner):
    """Run one planner action and return the new environment state."""
    action = planner(env.render(mode="tiny_rgb_array"))
    return env.step(action)


def update_ui(fig, ax, image, observation, num_steps, reward_sum, done, info, planner_label, map_type, map_name, show_ui, delay, gif_recorder=None):
    """Refresh the board window after one primitive planner step."""
    if not show_ui:
        return
    action_name = info.get("action.name", "unknown")
    print_step(num_steps, 0, action_name, reward_sum, done)
    status = "SOLVED!" if info.get("all_boxes_on_target") else ("FAILED" if done else "")
    title = build_title(planner_label, map_type, map_name, num_steps + 1, reward_sum, status)
    update_plot(fig, ax, image, observation, title, delay)

    if gif_recorder:
        gif_recorder.add_frame(fig)


def finish_ui(show_ui):
    """Close plotting mode after a run only when UI was enabled."""
    if show_ui:
        finish_plot()


def build_run_row(planner, reward_sum, num_steps, solved, map_code):
    """Store one finished planner run with explicit step and push counts."""
    return {
        "solved": solved,
        "failure_reason": planner.failure_reason,
        # "map_code": map_code,
        "total_reward": round(reward_sum, 4),
        "num_steps": num_steps,
        "num_pushes": planner.num_pushes,
        "runtime_ms": round(planner.runtime_ms, 4),
        "nodes_expanded": planner.nodes_expanded,
        "deadlocks_pruned": planner.deadlocks_pruned,
        "dead_squares": planner.dead_squares_count,
    }


def encode_map(env):
    """Turn the current Sokoban board into one compact text string."""
    room_state = env.unwrapped.room_state
    room_fixed = env.unwrapped.room_fixed
    rows = []

    for row_index in range(room_state.shape[0]):
        rows.append(encode_row(room_state, room_fixed, row_index))

    return "|".join(rows)


def encode_row(room_state, room_fixed, row_index):
    """Convert one board row into Sokoban text symbols."""
    chars = []
    width = room_state.shape[1]

    for col_index in range(width):
        chars.append(tile_to_char(room_state, room_fixed, row_index, col_index))

    return "".join(chars)


def tile_to_char(room_state, room_fixed, row_index, col_index):
    """Convert one tile into a readable Sokoban character."""
    state = room_state[row_index, col_index]
    fixed = room_fixed[row_index, col_index]

    if fixed == 0:
        return "#"
    if state == 5:
        return "@"
    if state == 4:
        return "$"
    if state == 3:
        return "*"
    if fixed == 2:
        return "."
    return " "


def save_run_row(row, group_name, board_name, algorithm_name):
    """Write one row to both the board file and the algorithm file."""
    board_path = build_board_csv_path(group_name, board_name)
    runs_path = build_algorithm_runs_path(algorithm_name)
    append_csv_row(board_path, RUN_FIELDS, row)
    append_csv_row(runs_path, RUN_FIELDS, row)


def build_algorithm_summaries(rows, algorithm_names):
    """Create one summary table per algorithm from detailed run rows."""
    summaries = {}

    for algorithm_name in algorithm_names:
        runs = [row for row in rows if row["algorithm"] == algorithm_name]
        summaries[algorithm_name] = summarize_algorithm_runs(algorithm_name, runs)

    return summaries


def summarize_algorithm_runs(algorithm_name, rows):
    """Build per-group and overall summary rows for one algorithm."""
    groups = collect_summary_groups(rows)
    summaries = []

    for group_name in groups:
        group_rows = [row for row in rows if row["group"] == group_name]
        summaries.append(build_summary_row(algorithm_name, group_name, group_rows))
    return summaries


def collect_summary_groups(rows):
    """Return each distinct group name in sorted order."""
    group_names = {row["group"] for row in rows}
    return sorted(group_names)


def build_summary_row(algorithm_name, group_name, rows):
    """Aggregate success and average metrics for one algorithm group."""
    total_runs = len(rows)
    solved_runs = count_solved_runs(rows)
    success_rate = round(solved_runs / total_runs, 4)
    avg_num_steps = round(average_value(rows, "num_steps"), 4)
    avg_num_pushes = round(average_value(rows, "num_pushes"), 4)
    avg_time = round(average_value(rows, "runtime_ms"), 4)
    avg_nodes = round(average_value(rows, "nodes_expanded"), 4)
    avg_pruned = round(average_value(rows, "deadlocks_pruned"), 4)

    return {
        "algorithm": algorithm_name,
        "group": group_name,
        "total_runs": total_runs,
        "solved_runs": solved_runs,
        "success_rate": success_rate,
        "avg_num_steps": avg_num_steps,
        "avg_num_pushes": avg_num_pushes,
        "avg_runtime_ms": avg_time,
        "avg_nodes_expanded": avg_nodes,
        "avg_deadlocks_pruned": avg_pruned,
    }


def count_solved_runs(rows):
    """Count how many detailed rows were marked as solved."""
    return sum(1 for row in rows if row["solved"])


def average_value(rows, field_name):
    """Compute the simple average for one numeric result field."""
    total = sum(row[field_name] for row in rows)
    return total / len(rows)


def save_algorithm_summaries(summary_rows_by_algorithm):
    """Rewrite each summary CSV so the header always matches current fields."""
    for algorithm_name, rows in summary_rows_by_algorithm.items():
        file_path = build_algorithm_summary_path(algorithm_name)
        rewrite_csv(file_path, SUMMARY_FIELDS, rows)


def rewrite_csv(file_path, fieldnames, rows):
    """Rewrite a CSV file from scratch with a fresh summary table."""
    ensure_parent_folder(file_path)
    with open(file_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows_by_algorithm):
    """Print compact summary rows so the benchmark is easy to inspect."""
    for algorithm_name, rows in summary_rows_by_algorithm.items():
        print("\n" + "=" * 72)
        print(f"{algorithm_name.upper()} SUMMARY")
        print("=" * 72)
        print(f"{'Group':<22} {'Solved':>8} {'Runs':>8} {'Rate':>10} {'Avg steps':>10} {'Avg pushes':>12} {'Avg ms':>10}")

        for row in rows:
            print(
                f"{row['group']:<22} "
                f"{row['solved_runs']:>8} "
                f"{row['total_runs']:>8} "
                f"{row['success_rate']:>10} "
                f"{row['avg_num_steps']:>10} "
                f"{row['avg_num_pushes']:>12} "
                f"{row['avg_runtime_ms']:>10}"
            )


def main():
    """Run the selected benchmarks, append detailed rows, and save summaries."""
    args = parse_args()
    rows = run_selected_benchmarks(args)
    algorithm_names = expand_algorithms(args.algorithms)
    summaries = build_algorithm_summaries(rows, algorithm_names)
    save_algorithm_summaries(summaries)
    print_summary(summaries)


if __name__ == "__main__":
    main()
