import argparse
import matplotlib.pyplot as plt

from src.env.custom_env import SimpleCustomSokobanEnv
from src.planners.bfs import BFSAgent


def parse_position(text):
    row, col = text.split(",")
    return int(row), int(col)


def parse_positions(text):
    if text.strip() == "":
        return []
    return [parse_position(part) for part in text.split(";")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=3)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--player", type=parse_position, default=(1, 1))
    parser.add_argument("--boxes", type=parse_positions, default=[(1, 4)])
    parser.add_argument("--goals", type=parse_positions, default=[(1, 10)])
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    # Create custom Sokoban environment
    env = SimpleCustomSokobanEnv(
        height=args.height,
        width=args.width,
        player_position=args.player,
        box_positions=args.boxes,
        goal_positions=args.goals,
        max_steps=args.max_steps,
    )

    # Reset gives the first RGB observation
    obs = env.reset()
    print("RGB observation shape:", obs.shape)
    print("RGB observation dtype:", obs.dtype)

    # Create BFS planner
    bfs = BFSAgent(env)
    bfs.reset()

    # Compute full BFS solution plan
    plan = bfs._solve()
    if plan is None:
        print("BFS could not find a solution.")
        print("Nodes expanded:", bfs.nodes_expanded)
        return
    print("BFS found a solution.")
    print("Plan length:", len(plan))
    print("Nodes expanded:", bfs.nodes_expanded)
    print("Plan:", plan)

    # Generate UI
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 3))
    image = ax.imshow(obs)
    ax.axis("off")
    ax.set_title("Initial Board")
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(args.delay)

    # Execute BFS actions one by one
    total_reward = 0
    done = False
    for step, action in enumerate(plan):
        obs, reward, done, info = env.step(action)
        total_reward += reward
        action_name = info.get("action.name", "unknown")
        print(
            f"Step {step + 1}: "
            f"action={action}, "
            f"name={action_name}, "
            f"reward={reward}, "
            f"done={done}"
        )

        # Update the existing image 
        image.set_data(obs)
        ax.set_title(f"Step {step + 1}: {action_name}")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(args.delay)
        if done:
            break

    plt.ioff()
    plt.show()
    print("Done:", done)
    print("Total reward:", total_reward)


if __name__ == "__main__":
    main()