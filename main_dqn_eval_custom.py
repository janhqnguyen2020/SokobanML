"""Evaluate the trained DQN on all canvas-compatible maps (fit the 10x10 observation canvas)."""

import argparse
import glob
import os

from src.env.custom_env import SimpleCustomSokobanEnv
from src.rl.high_level_env import HighLevelSokobanEnv
from src.rl.masked_dqn import MaskedDQN
from src.utils.config import HIGH_LEVEL_OBS_CANVAS_SHAPE, HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES
from src.utils.custom_maps import build_all_canvas_maps
from src.utils.show_ui import build_title, create_plot, finish_plot, update_plot

DELAY = 0.3

def _to_eval_config(m):
    return {
        "name": m["map_name"],
        "height": m["height"],
        "width": m["width"],
        "player": m["player"],
        "boxes": m["boxes"],
        "goals": m["goals"],
        "max_steps": 150,
    }

# DQN was trained on 3-box environments (obs size 412). Only evaluate on maps
# that match this box count — 1-box and 2-box canvas maps produce obs size 404/408
# and are incompatible with the saved model weights.
DQN_CUSTOM_MAPS = [_to_eval_config(m) for m in build_all_canvas_maps() if len(m["boxes"]) == 3]


def find_latest_model():
    pattern = os.path.join(
        "results", "rl_tests", "high_level_dqn", "*", "high_level_dqn_best.zip"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        pattern = os.path.join(
            "results", "rl_tests", "high_level_dqn", "*", "high_level_dqn_final.zip"
        )
        matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError("No trained model found. Run python main_dqn.py first.")
    return matches[-1]


def make_custom_hl_env(map_config):
    raw_env = SimpleCustomSokobanEnv(
        height=map_config["height"],
        width=map_config["width"],
        player_position=map_config["player"],
        box_positions=map_config["boxes"],
        goal_positions=map_config["goals"],
        max_steps=map_config["max_steps"],
    )
    return HighLevelSokobanEnv(
        env=raw_env,
        observation_board_shape=HIGH_LEVEL_OBS_CANVAS_SHAPE,
        use_extra_scalar_features=HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES,
        use_shaped_reward=False,
    )


def run_episode(model, env, map_name, show_ui, episode_num):
    obs = env.reset()
    done = False
    steps = 0
    total_reward = 0.0
    info = {}
    fig = ax = image = None

    if show_ui:
        frame = env.env.render(mode="rgb_array")
        ep_label = f"{map_name}-ep{episode_num}"
        fig, ax, image = create_plot(frame, build_title("DQN", "dqn_compatible", ep_label, 0, 0.0))

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
        steps += 1

        if show_ui:
            frame = env.env.render(mode="rgb_array")
            solved = info.get("all_boxes_on_target", False)
            reason = info.get("termination_reason", "")
            status = "SOLVED!" if (done and solved) else (reason if done else "")
            title = build_title("DQN", "dqn_compatible", ep_label, steps, total_reward, status)
            update_plot(fig, ax, image, frame, title, DELAY)

    if show_ui:
        finish_plot()

    return {
        "map": map_name,
        "solved": bool(info.get("all_boxes_on_target", False)),
        "steps": steps,
        "total_reward": round(total_reward, 3),
        "reason": str(info.get("termination_reason", "unknown")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-ui", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    model_path = find_latest_model()
    print(f"Model: {model_path}\n")
    model = MaskedDQN.load(model_path)

    print(f"{'Map':<15} {'Ep':<5} {'Solved':<8} {'Steps':<8} {'Reward':<10} Reason")
    print("-" * 62)

    for map_config in DQN_CUSTOM_MAPS:
        for ep in range(args.episodes):
            env = make_custom_hl_env(map_config)
            result = run_episode(model, env, map_config["name"], args.show_ui, ep + 1)
            env.close()
            print(
                f"{result['map']:<15} {ep + 1:<5} {str(result['solved']):<8} "
                f"{result['steps']:<8} {result['total_reward']:<10} {result['reason']}"
            )


if __name__ == "__main__":
    main()
