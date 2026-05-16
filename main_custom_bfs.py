import argparse
import csv
import os
import time

import matplotlib.pyplot as plt

from src.env.custom_env import SimpleCustomSokobanEnv
from src.planners.bfs import BFSAgent


PLANNER_FIELDS = [
    "method",
    "env_version",
    "episode",
    "map_name",
    "solved",
    "total_reward",
    "steps",
    "runtime_ms",
    "avg_runtime_ms",
    "nodes_expanded",
    "deadlocks_pruned",
    "dead_squares",
    "pruning_rate",
]


def build_custom_maps():
    return [
        build_map("map_01", 5, 12, (2, 1), [(2, 4)], [(2, 10)]),
        build_map("map_02", 5, 12, (2, 10), [(2, 7)], [(2, 2)]),

        build_map("map_03", 6, 14, (2, 1), [(1, 4), (4, 4)], [(1, 11), (4, 11)]),
        build_map("map_04", 6, 14, (2, 6), [(2, 3), (3, 9)], [(1, 7), (4, 7)]),

        build_map("map_05", 7, 15, (3, 1), [(1, 4), (3, 5), (5, 4)], [(1, 12), (3, 12), (5, 12)]),
        build_map("map_06", 7, 15, (3, 7), [(2, 3), (3, 9), (4, 3)], [(1, 7), (3, 12), (5, 7)]),

        build_map("map_07", 8, 16, (4, 1), [(1, 4), (3, 5), (5, 5), (6, 4)], [(1, 13), (3, 13), (5, 13), (6, 13)]),
        build_map("map_08", 8, 16, (4, 7), [(2, 3), (2, 10), (5, 3), (5, 10)], [(1, 7), (3, 13), (4, 7), (6, 13)]),

        build_map("map_09", 9, 18, (4, 1), [(1, 5), (3, 6), (4, 8), (6, 6), (7, 5)], [(1, 15), (3, 15), (4, 15), (6, 15), (7, 15)]),
        build_map("map_10", 9, 18, (4, 9), [(2, 3), (2, 12), (4, 5), (6, 3), (6, 12)], [(1, 9), (3, 15), (4, 14), (5, 9), (7, 15)]),
    ]


def build_map(map_name, height, width, player, boxes, goals):
    return {
        "map_name": map_name,
        "height": height,
        "width": width,
        "player": player,
        "boxes": boxes,
        "goals": goals,
        "max_steps": 100,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="results/custom_env/top10/planner_results.csv",)
    return parser.parse_args()


def ensure_output_file(output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        return
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PLANNER_FIELDS)
        writer.writeheader()


def append_row(output_path, row):
    with open(output_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PLANNER_FIELDS)
        writer.writerow(row)


def create_env(config):
    return SimpleCustomSokobanEnv(
        height=config["height"],
        width=config["width"],
        player_position=config["player"],
        box_positions=config["boxes"],
        goal_positions=config["goals"],
        max_steps=config["max_steps"],
    )


def print_map_header(index, config, output_path):
    print("\n" + "=" * 60)
    print("Episode:", index)
    print("Map:", config["map_name"])
    print("CSV:", output_path)
    print("height:", config["height"], "width:", config["width"])
    print("player:", config["player"])
    print("boxes:", config["boxes"])
    print("goals:", config["goals"])
    print("=" * 60)


def create_plot(obs, map_name):
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 3))
    image = ax.imshow(obs)
    ax.axis("off")
    ax.set_title(f"{map_name} - Initial Board")
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, image


def update_plot(fig, ax, image, obs, title, delay):
    """Refresh the UI window with the newest board frame"""
    image.set_data(obs)
    ax.set_title(title)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(delay)


def print_step(step, action, action_name, reward, done):
    """Print one executed BFS action"""
    print(
        f"Step {step + 1}: "
        f"action={action}, "
        f"name={action_name}, "
        f"reward={reward}, "
        f"done={done}"
    )


def run_plan(env, plan, fig, ax, image, delay):
    """Execute the BFS plan and animate each step"""
    total_reward = 0
    done = False
    for step, action in enumerate(plan):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        action_name = info.get("action.name", "unknown")
        print_step(step, action, action_name, reward, done)
        update_plot(fig, ax, image, obs, f"Step {step + 1}: {action_name}", delay)
        if done:
            break
    return total_reward, done


def build_result_row(index, config, bfs, total_reward, steps, runtime_ms, solved, avg_runtime_ms):
    pruning_rate = compute_pruning_rate(bfs.deadlocks_pruned, bfs.nodes_expanded)
    return {
        "method": "BFSAgent",
        "env_version": "custom_env",
        "episode": index,
        "map_name": config["map_name"],
        "solved": solved,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "runtime_ms": round(runtime_ms, 4),
        "avg_runtime_ms": round(avg_runtime_ms, 4),
        "nodes_expanded": bfs.nodes_expanded,
        "deadlocks_pruned": bfs.deadlocks_pruned,
        "dead_squares": bfs.dead_squares_count,
        "pruning_rate": pruning_rate,
    }


def compute_pruning_rate(deadlocks_pruned, nodes_expanded):
    if nodes_expanded == 0:
        return 0.0
    return round(deadlocks_pruned / nodes_expanded, 4)


def print_result(row):
    """Print the final result row in a readable way."""
    print("\nFinished:", row["map_name"])
    print("solved:", row["solved"])
    print("total_reward:", row["total_reward"])
    print("steps:", row["steps"])
    print("runtime_ms:", row["runtime_ms"])
    print("avg_runtime_ms:", row["avg_runtime_ms"])
    print("nodes_expanded:", row["nodes_expanded"])
    print("deadlocks_pruned:", row["deadlocks_pruned"])
    print("dead_squares:", row["dead_squares"])
    print("pruning_rate:", row["pruning_rate"])


def run_one_map(index, config, delay, avg_runtime_ms):
    """Run BFS on one custom map and return one result row."""
    env = create_env(config)
    obs = env.reset()
    print("RGB observation shape:", obs.shape)
    print("RGB observation dtype:", obs.dtype)

    fig, ax, image = create_plot(obs, config["map_name"])
    plt.pause(delay)

    bfs = BFSAgent(env)
    bfs.reset()

    start_time = time.time()
    plan = bfs._solve()
    runtime_ms = (time.time() - start_time) * 1000

    row = finish_map_run(index, config, bfs, plan, env, fig, ax, image, delay, runtime_ms, avg_runtime_ms)
    env.close()
    plt.ioff()
    plt.show()
    return row


def finish_map_run(index, config, bfs, plan, env, fig, ax, image, delay, runtime_ms, avg_runtime_ms):
    """Finish the run by either logging failure or animating the plan."""
    if plan is None:
        print("BFS could not find a solution.")
        row = build_result_row(index, config, bfs, 0, 0, runtime_ms, False, avg_runtime_ms)
        print_result(row)
        return row

    print("BFS found a solution.")
    print("Plan length:", len(plan))
    print("Nodes expanded:", bfs.nodes_expanded)
    print("Plan:", plan)

    total_reward, solved = run_plan(env, plan, fig, ax, image, delay)
    row = build_result_row(index, config, bfs, total_reward, len(plan), runtime_ms, solved, avg_runtime_ms)
    print_result(row)
    return row


def choose_maps(all_maps, count):
    """Return the first count maps from the full custom list."""
    return all_maps[:count]


def main():
    """Run custom BFS evaluation with UI and CSV logging."""
    args = parse_args()
    all_maps = build_custom_maps()
    selected_maps = choose_maps(all_maps, args.count)
    ensure_output_file(args.output)

    cumulative_runtime_ms = 0.0
    for index, config in enumerate(selected_maps):
        print_map_header(index, config, args.output)
        runtime_so_far = cumulative_runtime_ms / (index + 1) if index > 0 else 0.0
        row = run_one_map(index, config, args.delay, runtime_so_far)
        cumulative_runtime_ms += row["runtime_ms"]
        row["avg_runtime_ms"] = round(cumulative_runtime_ms / (index + 1), 4)
        print("Saved to CSV:", args.output)
        append_row(args.output, row)


if __name__ == "__main__":
    main()