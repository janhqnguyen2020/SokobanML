# eval_dqn.py
"""Evaluate a trained curriculum DQN model.

Modes
-----
procedural  -- Sokoban-small-v1 procedural maps (generalization test)
currTrain   -- All curriculum training maps: generated 1/2/3-box + walled 1/2/3-box
               + hand-crafted curr_01-10. NOT a held-out test set.
finalEval   -- Held-out benchmark: 150 maps (50 one-box + 50 two-box + 50 three-box),
               seed=999, never seen during training.

Examples
--------
python eval_dqn.py
python eval_dqn.py --mode procedural --episodes 50
python eval_dqn.py --mode currTrain
python eval_dqn.py --mode currTrain --boxes 3
python eval_dqn.py --mode currTrain --save --note "500k_seed42"
python eval_dqn.py --mode finalEval
python eval_dqn.py --mode finalEval --boxes 3 --episodes 5 --save --note "v2_final"
python eval_dqn.py --model results/rl_tests/curriculum_dqn/<run>/curriculum_dqn_best.zip
python eval_dqn.py --show-ui --episodes 1
"""

import argparse
import glob
import json
import os
import time
from datetime import datetime

from src.rl.evaluate import evaluate_episode, summarize_results
from src.rl.masked_dqn import MaskedDQN
from src.rl.train_curriculum_dqn import (
    createCurriculumEvalEnvironment,
    createEvaluationEnvironment,
)
from src.utils.config import CURRICULUM_DQN_CANVAS_SHAPE, CURRICULUM_DQN_MAX_BOXES, MAX_STEPS
from src.utils.custom_maps import build_curriculum_maps
from src.utils.final_eval_maps import (
    build_final_eval_1box_maps, build_final_eval_2box_maps, build_final_eval_3box_maps,
)
from src.utils.generated_maps import (
    build_generated_1box_maps, build_generated_2box_maps, build_generated_3box_maps,
    build_walled_1box_maps, build_walled_2box_maps, build_walled_3box_maps,
)
from src.utils.show_ui import build_title, create_plot, finish_plot, update_plot

DELAY = 0.3


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _find_latest_model():
    for pattern in (
        os.path.join("results", "rl_tests", "curriculum_dqn", "*", "curriculum_dqn_best.zip"),
        os.path.join("results", "rl_tests", "curriculum_dqn", "*", "curriculum_dqn_final.zip"),
    ):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    raise FileNotFoundError(
        "No curriculum DQN model found. "
        "Run python main_curriculum_dqn.py first, or pass --model <path>."
    )


# ---------------------------------------------------------------------------
# Map group builders
# ---------------------------------------------------------------------------

def _build_final_eval_groups(boxes_filter):
    """Return [(label, [maps])] for the held-out FinalEval set (seed=999)."""
    groups_all = {
        "1-box": build_final_eval_1box_maps(),
        "2-box": build_final_eval_2box_maps(),
        "3-box": build_final_eval_3box_maps(),
    }
    if boxes_filter == "all":
        return list(groups_all.items())
    key = f"{boxes_filter}-box"
    return [(key, groups_all.get(key, []))]


def _build_curr_train_groups(boxes_filter):
    """Return [(label, [maps])] for the full currTrain pool, optionally filtered by box count."""
    groups_all = {
        "1-box": (
            build_generated_1box_maps() +
            build_walled_1box_maps() +
            [m for m in build_curriculum_maps() if len(m["boxes"]) == 1]
        ),
        "2-box": (
            build_generated_2box_maps() +
            build_walled_2box_maps() +
            [m for m in build_curriculum_maps() if len(m["boxes"]) == 2]
        ),
        "3-box": (
            build_generated_3box_maps() +
            build_walled_3box_maps() +
            [m for m in build_curriculum_maps() if len(m["boxes"]) == 3]
        ),
    }
    if boxes_filter == "all":
        return list(groups_all.items())
    key = f"{boxes_filter}-box"
    return [(key, groups_all.get(key, []))]


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------

def _run_episode_ui(model, env, map_name, episode_num):
    """Run one episode with matplotlib display."""
    obs = env.reset()
    done = False
    num_steps = 0
    num_pushes = 0
    invalid_macro_actions = 0
    best_boxes_on_target = 0
    total_reward = 0.0
    info = {}
    start_time = time.time()

    frame = env.render(mode="rgb_array")
    fig, ax, image = create_plot(frame, build_title("DQN", "eval", map_name, 0, 0.0))

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        total_reward += reward
        num_steps += 1
        if info.get("executed_push"):
            num_pushes += 1
        if info.get("invalid_macro_action"):
            invalid_macro_actions += 1
        best_boxes_on_target = max(
            best_boxes_on_target,
            int(info.get("best_boxes_on_target", info.get("boxes_on_target", 0))),
        )
        if num_steps >= MAX_STEPS and not done:
            done = True
            info["truncated"] = True
            info.setdefault("termination_reason", "max_steps")
        frame = env.render(mode="rgb_array")
        solved = info.get("all_boxes_on_target", False)
        reason = info.get("termination_reason", "")
        status = "SOLVED!" if (done and solved) else (reason if done else "")
        title = build_title("DQN", "eval", map_name, num_steps, total_reward, status)
        update_plot(fig, ax, image, frame, title, DELAY)

    finish_plot()
    boxes_on_target = int(info.get("boxes_on_target", 0))
    return {
        "solved": bool(info.get("all_boxes_on_target", False)),
        "num_steps": num_steps,
        "num_pushes": num_pushes,
        "runtime_ms": (time.time() - start_time) * 1000.0,
        "total_reward": round(float(total_reward), 3),
        "invalid_macro_actions": invalid_macro_actions,
        "truncated": bool(info.get("truncated", False)),
        "termination_reason": str(info.get("termination_reason", "unknown")),
        "boxes_on_target": boxes_on_target,
        "best_boxes_on_target": best_boxes_on_target,
    }


def _run_episode_silent(model, env):
    """Run one episode without display."""
    return evaluate_episode(model, env, log_progress=False)


def _run_map_episodes(model, map_config, n_episodes, show_ui):
    env = createCurriculumEvalEnvironment([map_config])
    results = []
    try:
        for ep in range(n_episodes):
            if show_ui:
                results.append(_run_episode_ui(model, env, map_config["map_name"], ep + 1))
            else:
                results.append(_run_episode_silent(model, env))
    finally:
        env.close()
    return results


def _run_procedural_episodes(model, n_episodes, show_ui):
    env = createEvaluationEnvironment()
    results = []
    try:
        for ep in range(n_episodes):
            if show_ui:
                results.append(_run_episode_ui(model, env, f"procedural-ep{ep+1}", ep + 1))
                env.reset()
            else:
                results.append(_run_episode_silent(model, env))
    finally:
        env.close()
    return results


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_table_header():
    """Print the table header used by the DQN evaluation script."""
    print(f"  {'Map':<22} {'Ep':<4}  {'Solved':<6}  {'Steps':<6}  {'Pushes':<6}  {'Reward':<8}  Reason")
    print("  " + "-" * 76)


def _print_row(map_name, ep_idx, result):
    """Print one DQN evaluation result row."""
    solved_str = "YES" if result.get("solved") else "no"
    reason = result.get("termination_reason", result.get("reason", ""))
    print(
        f"  {map_name:<22} {ep_idx + 1:<4}  {solved_str:<6}  "
        f"{result.get('num_steps', 0):<6}  {result.get('num_pushes', 0):<6}  "
        f"{result.get('total_reward', 0.0):<8.2f}  {reason}"
    )


def _print_group_summary(label, results):
    """Print one compact group summary for DQN evaluation results."""
    if not results:
        return
    s = summarize_results(results)
    filled = int(s["success_rate"] * 20)
    bar = f"[{'=' * filled}{' ' * (20 - filled)}] {s['success_rate'] * 100:.0f}%"
    print(
        f"\n  {label}: {s['solved_count']}/{s['total_episodes']} solved  {bar}"
        f"  avg_steps={s['avg_num_steps']:.1f}  avg_pushes={s['avg_num_pushes']:.1f}"
        f"  avg_reward={s['avg_reward']:.2f}"
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_results(model_path, mode, boxes_filter, episodes, note, per_map_results, summary, group_summaries=None):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("results", "rl_tests", "curriculum_dqn", "evals", ts)
    os.makedirs(out_dir, exist_ok=True)

    tag = f"{mode}_boxes{boxes_filter}"
    checkpoint_type = "best" if "best" in os.path.basename(model_path) else "final"
    run_dir = os.path.dirname(os.path.normpath(model_path))

    metadata = {
        "eval_timestamp":  ts,
        "model_path":      os.path.normpath(model_path),
        "training_run_id": os.path.basename(run_dir),
        "checkpoint_type": checkpoint_type,
        "training_note":   note or "unset",
        "eval_mode":       mode,
        "eval_boxes":      boxes_filter,
        "episodes":        episodes,
        "canvas_shape":    list(CURRICULUM_DQN_CANVAS_SHAPE),
        "max_boxes":       CURRICULUM_DQN_MAX_BOXES,
    }

    selected_json = os.path.join(run_dir, "selected_model.json")
    if os.path.exists(selected_json):
        with open(selected_json) as f:
            sel = json.load(f)
        metadata["training_procedural_eval"] = sel.get("procedural_eval_summary")

    rows = [
        {"map": name, "episode": ep, **r}
        for name, ep_results in per_map_results.items()
        for ep, r in enumerate(ep_results)
    ]
    with open(os.path.join(out_dir, f"eval_{tag}.json"), "w") as f:
        json.dump({"metadata": metadata, "episodes": rows}, f, indent=2)
    with open(os.path.join(out_dir, f"eval_{tag}_summary.json"), "w") as f:
        json.dump({
            "metadata": metadata,
            "summary_by_group": group_summaries or {},
            "summary": summary,
        }, f, indent=2)
    print(f"\n  Saved to: {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the curriculum DQN")
    parser.add_argument(
        "--mode",
        choices=["procedural", "currTrain", "finalEval"],
        default="currTrain",
        help="Evaluation suite (default: currTrain)",
    )
    parser.add_argument(
        "--boxes",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Box count filter for currTrain / finalEval (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodes per map (currTrain) or total (procedural) (default: 3)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model .zip (default: auto-find latest curriculum_dqn_best.zip)",
    )
    parser.add_argument(
        "--show-ui",
        action="store_true",
        help="Display board in a matplotlib window during evaluation",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save result JSONs to results/rl_tests/curriculum_dqn/evals/",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Label for this eval run, e.g. 'phase2_500k'",
    )
    args = parser.parse_args()

    model_path = args.model or _find_latest_model()
    model = MaskedDQN.load(model_path)

    print(f"\n{'=' * 60}")
    print("CURRICULUM DQN EVALUATION")
    print(f"{'=' * 60}")
    print(f"  model    : {model_path}")
    print(f"  canvas   : {CURRICULUM_DQN_CANVAS_SHAPE}  max_boxes={CURRICULUM_DQN_MAX_BOXES}")
    print(f"  mode     : {args.mode}  |  boxes: {args.boxes}  |  episodes: {args.episodes}")
    print(f"  note     : {args.note or '(none)'}")
    print()

    per_map_results = {}
    group_summaries = {}
    all_results = []

    if args.mode == "procedural":
        print(f"  Running {args.episodes} procedural Sokoban-small-v1 episodes...\n")
        _print_table_header()
        results = _run_procedural_episodes(model, args.episodes, args.show_ui)
        for ep, r in enumerate(results):
            _print_row("procedural", ep, r)
        per_map_results["procedural"] = results
        all_results = results
        _print_group_summary("Sokoban-small-v1", results)
        group_summaries["procedural"] = summarize_results(results)

    else:  # currTrain or finalEval — same loop, different map source
        if args.mode == "finalEval":
            groups = _build_final_eval_groups(args.boxes)
            print("  Source: held-out FinalEval benchmark (seed=999, never seen during training)")
        else:
            groups = _build_curr_train_groups(args.boxes)
        total_maps = sum(len(maps) for _, maps in groups)
        print(f"  {total_maps} maps  x  {args.episodes} episodes each\n")
        _print_table_header()

        for group_label, maps in groups:
            if not maps:
                print(f"\n  (no {group_label} maps)")
                continue
            print(f"\n  ── {group_label.upper()}  ({len(maps)} maps) ──")
            group_results = []
            for m in maps:
                ep_results = _run_map_episodes(model, m, args.episodes, args.show_ui)
                per_map_results[m["map_name"]] = ep_results
                for ep, r in enumerate(ep_results):
                    _print_row(m["map_name"], ep, r)
                group_results.extend(ep_results)
            _print_group_summary(group_label, group_results)
            group_summaries[group_label] = summarize_results(group_results)
            all_results.extend(group_results)

    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    summary = summarize_results(all_results) if all_results else {}
    for k in ["success_rate", "solved_count", "total_episodes", "avg_reward",
              "avg_num_steps", "avg_num_pushes", "avg_boxes_on_target", "termination_counts"]:
        if k not in summary:
            continue
        v = summary[k]
        print(f"  {k:<34} {v:.4f}" if isinstance(v, float) else f"  {k:<34} {v}")

    if args.save:
        _save_results(model_path, args.mode, args.boxes, args.episodes,
                      args.note, per_map_results, summary, group_summaries)


if __name__ == "__main__":
    main()
