import glob
import os

from src.rl.masked_dqn import MaskedDQN
from src.rl.train_dqn import make_eval_env
from src.utils.show_ui import build_title, create_plot, finish_plot, update_plot

DELAY = 5
NUM_EPISODES = 10

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
        raise FileNotFoundError(f"No model found. Run Python main_dqn.py first.")
    return matches[-1]

def run_episode(model, env, episode_num):
    obs = env.reset()
    done = False
    steps = 0
    total_reward =0.0
    info = {}

    frame = env.env.render(mode="rgb_array")
    fig, ax, image = create_plot(frame, build_title("DQN", "v1", f"ep{episode_num}", 0, 0.0))

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
        steps += 1

        frame = env.env.render(mode="rgb_array")
        solved = info.get("all_boxes_on_target", False)
        reason = info.get("termination_reason", "unknown")
        status = "SOLVED!" if (done and solved) else (reason if done else "")
        title = build_title("DQN", "v1", f"ep{episode_num}", steps, total_reward, status)
        update_plot(fig, ax, image, frame, title, DELAY)

    finish_plot()

    return {
        "solved": bool(info.get("all_boxes_on_target", False)),
        "steps": steps,
        "total_reward": round(total_reward, 3),
        "reason": str(info.get("termination_reason", "unknown")),
    }

def main():
    model_path = find_latest_model()
    print(f"Model: {model_path}\n")
    model = MaskedDQN.load(model_path)

    base_seed = 5000
    for ep in range(NUM_EPISODES):
        env = make_eval_env()
        env.seed(base_seed + ep)  # different seed = different randomly generated map
        result = run_episode(model, env, ep + 1)
        env.close()
        print(
            f"Episode {ep + 1}: "
            f"solved={result['solved']}  "
            f"steps={result['steps']}  "
            f"reward={result['total_reward']}  "
            f"reason={result['reason']}"
        )


if __name__ == "__main__":
    main()