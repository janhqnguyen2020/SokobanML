import argparse
import matplotlib.pyplot as plt
from src.env.custom_env import SimpleCustomSokobanEnv


def parse_position(text):
    row, col = text.split(",")
    return int(row), int(col)


def parse_positions(text):
    if text.strip() == "":
        return []
    return [parse_position(part) for part in text.split(";")]


def show_rgb(obs, title="Sokoban Board"):
    plt.figure(figsize=(6, 4))
    plt.imshow(obs)
    plt.axis("off")
    plt.title(title)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=3)
    parser.add_argument("--width", type=int, default=12)
    parser.add_argument("--player", type=parse_position, default=(1, 1))
    parser.add_argument("--boxes", type=parse_positions, default=[(1, 4)])
    parser.add_argument("--goals", type=parse_positions, default=[(1, 10)])
    parser.add_argument("--max-steps", type=int, default=100)
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

    print("RGB observation shape:", obs.shape)
    print("RGB observation dtype:", obs.dtype)
    show_rgb(obs, title="Custom Sokoban Board")


if __name__ == "__main__":
    main()