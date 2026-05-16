import argparse
import csv
import os
import time

import matplotlib.pyplot as plt

from src.env.custom_env import SimpleCustomSokobanEnv
from src.utils.custom_maps import build_custom_maps


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


def parse_args():
    """Read command line options for custom planner evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.5)
    return parser.parse_args()


def build_output_path(map_name):
    """Return the per-map CSV path for custom planner runs."""
    folder = os.path.join("results", "custom_env", map_name)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "planner_runs.csv")


def ensure_output_file(output_path):
    """Create the output CSV and header if it does not exist yet."""
    if os.path.exists(output_path):
        return
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PLANNER_FIELDS)
        writer.writeheader()


def append_row(output_path, row):
    """Append one planner result row to the CSV file."""
    with open(output_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PLANNER_FIELDS)
        writer.writerow(row)


def create_env(config):
    """Create one fixed custom Sokoban environment from a config."""
    return SimpleCustomSokobanEnv(
        height=config["height"],
        width=config["width"],
        player_position=config["player"],
        box_positions=config["boxes"],
        goal_positions=config["goals"],
        max_steps=config["max_steps"],
    )


def choose_maps(count):
    """Return the first count custom maps from the benchmark list."""
    return build_custom_maps()[:count]


def print_map_header(index, config, output_path, method_name):
    """Print the current map and run settings in the terminal."""
    print("\n" + "=" * 60)
    print("Method:", method_name)
    print("Episode:", index)
    print("Map:", config["map_name"])
    print("CSV:", output_path)
    print("height:", config["height"], "width:", config["width"])
    print("player:", config["player"])
    print("boxes:", config["boxes"])
    print("goals:", config["goals"])
    print("=" * 60)


def create_plot(obs, map_name):
    """Create one matplotlib window for the current map."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 3))
    image = ax.imshow(obs)
    ax.axis("off")
    ax.set_title(f"{map_name} - Initial Board")
    fig.canvas.draw()
    fig.canvas.flush_events()
    return fig, ax, image


def update_plot(fig, ax, image, obs, title, delay):
    """Refresh the current board image in the existing window."""
    image.set_data(obs)
    ax.set_title(title)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(delay)


def print_step(step, action, action_name, reward, done):
    """Print one executed planner action."""
    print(
        f"Step {step + 1}: "
        f"action={action}, "
        f"name={action_name}, "
        f"reward={reward}, "
        f"done={done}"
    )


def run_plan(env, plan, fig, ax, image, delay):
    """Execute one solved plan and animate each action."""
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


def compute_pruning_rate(deadlocks_pruned, nodes_expanded):
    """Return the fraction of expanded nodes that were pruned."""
    if nodes_expanded == 0:
        return 0.0
    return round(deadlocks_pruned / nodes_expanded, 4)


def build_result_row(index, config, planner, total_reward, steps, runtime_ms, solved, avg_runtime_ms, method_name):
    """Build one CSV row using the planner evaluation format."""
    return {
        "method": method_name,
        "env_version": "custom_env",
        "episode": index,
        "map_name": config["map_name"],
        "solved": solved,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "runtime_ms": round(runtime_ms, 4),
        "avg_runtime_ms": round(avg_runtime_ms, 4),
        "nodes_expanded": planner.nodes_expanded,
        "deadlocks_pruned": planner.deadlocks_pruned,
        "dead_squares": planner.dead_squares_count,
        "pruning_rate": compute_pruning_rate(
            planner.deadlocks_pruned,
            planner.nodes_expanded,
        ),
    }


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


def solve_with_planner(env, planner_class):
    """Create one planner and ask it for a full solution plan."""
    planner = planner_class(env)
    planner.reset()
    plan = planner._solve()
    return planner, plan


def run_one_map(index, config, planner_class, method_name, delay, avg_runtime_ms):
    """Run one planner on one map and return one result row."""
    env = create_env(config)
    obs = env.reset()
    print("RGB observation shape:", obs.shape)
    print("RGB observation dtype:", obs.dtype)

    fig, ax, image = create_plot(obs, config["map_name"])
    plt.pause(delay)

    start_time = time.time()
    planner, plan = solve_with_planner(env, planner_class)
    runtime_ms = (time.time() - start_time) * 1000

    if plan is None:
        print(f"{method_name} could not find a solution.")
        row = build_result_row(
            index,
            config,
            planner,
            0,
            0,
            runtime_ms,
            False,
            avg_runtime_ms,
            method_name,
        )
        print_result(row)
        env.close()
        plt.ioff()
        plt.show()
        return row

    print(f"{method_name} found a solution.")
    print("Plan length:", len(plan))
    print("Nodes expanded:", planner.nodes_expanded)
    print("Plan:", plan)

    total_reward, solved = run_plan(env, plan, fig, ax, image, delay)
    row = build_result_row(
        index,
        config,
        planner,
        total_reward,
        len(plan),
        runtime_ms,
        solved,
        avg_runtime_ms,
        method_name,
    )
    print_result(row)

    env.close()
    plt.ioff()
    plt.show()
    return row


def run_custom_evaluation(planner_class, method_name):
    """Run one planner across the fixed custom benchmark maps."""
    args = parse_args()
    cumulative_runtime_ms = 0.0
    selected_maps = choose_maps(args.count)

    for index, config in enumerate(selected_maps):
        output_path = build_output_path(config["map_name"])
        ensure_output_file(output_path)

        avg_runtime_ms = 0.0 if index == 0 else cumulative_runtime_ms / index
        print_map_header(index, config, output_path, method_name)

        row = run_one_map(
            index,
            config,
            planner_class,
            method_name,
            args.delay,
            avg_runtime_ms,
        )

        cumulative_runtime_ms += row["runtime_ms"]
        row["avg_runtime_ms"] = round(cumulative_runtime_ms / (index + 1), 4)

        append_row(output_path, row)
        print("Saved to CSV:", output_path)