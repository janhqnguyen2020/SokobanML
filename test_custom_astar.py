import argparse
import time
import matplotlib.pyplot as plt

from src.env.custom_env import SimpleCustomSokobanEnv
from src.planners.astar import AStarAgent
from src.utils.logger import log_run

LOG_PATH = "results/planner_log.csv"


def parse_position(text):
    row, col = text.split(",")
    return int(row), int(col)


def parse_positions(text):
    if text.strip() == "":
        return []
    return [parse_position(part) for part in text.split(";")]


def print_metrics(agent, args, solve_time_ms, total_reward, done):
    mode = agent.mode
    print("\n" + "-" * 38)
    print(f"  A*  |  {mode}")
    print("-" * 38)
    print(f"  Solved            {done}")
    print(f"  Total reward      {total_reward:.3f}")
    print(f"  Box pushes        {agent.pushes}")
    print(f"  Primitive steps   {agent.solution_length}")
    print(f"  Nodes expanded    {agent.nodes_expanded}")
    print(f"  Deadlocks pruned  {agent.deadlocks_pruned}")
    print(f"  Dead squares      {agent.dead_squares_count}")
    print(f"  Solve time        {solve_time_ms:.1f} ms")
    print("-" * 38)
    log_run({
        "planner":          "astar",
        "mode":             mode,
        "map_height":       args.height,
        "map_width":        args.width,
        "num_boxes":        len(args.boxes),
        "solved":           done,
        "total_reward":     round(total_reward, 4),
        "primitive_steps":  agent.solution_length,
        "box_pushes":       agent.pushes,
        "nodes_expanded":   agent.nodes_expanded,
        "deadlocks_pruned": agent.deadlocks_pruned,
        "dead_squares":     agent.dead_squares_count,
        "solve_time_ms":    round(solve_time_ms, 2),
    }, filepath=LOG_PATH)
    print(f"  Logged -> {LOG_PATH}")
    print("-" * 38 + "\n")


def make_stepper(fig):
    advance = [False]
    def on_key(event):
        if event.key in ("right", " ", "enter"):
            advance[0] = True
    fig.canvas.mpl_connect("key_press_event", on_key)
    def wait(delay, manual):
        if manual:
            advance[0] = False
            while not advance[0]:
                plt.pause(0.05)
        else:
            plt.pause(delay)
    return wait


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height",    type=int,            default=3)
    parser.add_argument("--width",     type=int,            default=12)
    parser.add_argument("--player",    type=parse_position, default=(1, 1))
    parser.add_argument("--boxes",     type=parse_positions, default=[(1, 4)])
    parser.add_argument("--goals",     type=parse_positions, default=[(1, 10)])
    parser.add_argument("--max-steps", type=int,            default=100)
   
    parser.add_argument("--manual",    action="store_true",
                        help="step manually with Space / Enter / right-arrow")
    parser.add_argument("--no-vis",    action="store_true",
                        help="skip visualization entirely (fast batch mode)")
    args = parser.parse_args()

    env = SimpleCustomSokobanEnv(
        height=args.height,
        width=args.width,
        player_position=args.player,
        box_positions=args.boxes,
        goal_positions=args.goals,
        max_steps=args.max_steps,
    )

    obs = env.reset()

    # --- visualization setup (skipped in --no-vis mode) ---
    if not args.no_vis:
        if args.manual:
            print("Manual mode: press Space / Enter / right-arrow to advance each step.")
        fig_h = max(3, args.height * 0.4)
        fig_w = max(6, args.width * 0.5)
        plt.ion()
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        image = ax.imshow(obs)
        ax.axis("off")
        ax.set_title("Initial Board  --  A* solving...")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.1)
        wait = make_stepper(fig)

    # --- solve ---
    agent = AStarAgent(env)
    agent.reset()

    start = time.perf_counter()
    plan = agent._solve()
    solve_time_ms = (time.perf_counter() - start) * 1000

    total_reward = 0
    done = False

    if plan is None:
        print("A* could not find a solution.")
        if not args.no_vis:
            ax.set_title("A*: no solution found")
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(args.hold)
            plt.ioff()
            plt.close()
        print_metrics(agent, args, solve_time_ms, total_reward, done)
        return

    print(f"A* found a solution. {len(plan)} steps, {agent.pushes} pushes.")

    if not args.no_vis:
        title = f"A* ready  --  {len(plan)} steps  [Space / -> to begin]" if args.manual \
                else f"A*: solution found  --  {len(plan)} steps, {agent.pushes} pushes"
        ax.set_title(title)
        fig.canvas.draw()
        fig.canvas.flush_events()
        wait(args.delay, args.manual)

    # --- execute plan ---
    for step, action in enumerate(plan):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        action_name = info.get("action.name", "unknown")
        print(f"Step {step + 1:3d}: {action_name:<12}  reward={reward:+.3f}  done={done}")

        if not args.no_vis:
            remaining = len(plan) - step - 1
            if done:
                ax.set_title("Solved!")
            elif args.manual:
                ax.set_title(f"Step {step + 1}/{len(plan)}: {action_name}  [{remaining} left — Space / ->]")
            else:
                ax.set_title(f"Step {step + 1}/{len(plan)}: {action_name}")
            image.set_data(obs)
            fig.canvas.draw()
            fig.canvas.flush_events()

        if done:
            break

        if not args.no_vis:
            wait(args.delay, args.manual)

    if not args.no_vis:
        plt.pause(args.hold)
        plt.ioff()
        plt.close()

    print_metrics(agent, args, solve_time_ms, total_reward, done)


if __name__ == "__main__":
    main()
