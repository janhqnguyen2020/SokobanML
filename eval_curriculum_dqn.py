"""Evaluate the curriculum DQN on custom, generated, or procedural maps.

The env always uses CURRICULUM_DQN_CANVAS_SHAPE / CURRICULUM_DQN_MAX_BOXES from
config.py, so observation and action dimensions always match the trained model.
Safe to run mid-training — pass --model <path> to point at a partial checkpoint.

Modes
-----
generated   -- generated BFS-verified maps  (default; safe at any training stage)
curriculum  -- 10 hand-crafted curriculum maps  (curr_01 .. curr_10)
procedural  -- procedural Sokoban-small-v1  (generalization / transfer test)

Box filter  (generated / curriculum only)
-----------------------------------------
--boxes 1   -- 1-box maps only
--boxes 2   -- 2-box maps only
--boxes 3   -- 3-box maps only
--boxes all -- all difficulties  (default)

Examples
--------
python eval_curriculum_dqn.py
python eval_curriculum_dqn.py --boxes 1
python eval_curriculum_dqn.py --mode procedural --episodes 50
python eval_curriculum_dqn.py --mode curriculum --boxes 2 --episodes 5
python eval_curriculum_dqn.py --model results/rl_tests/curriculum_dqn/<run>/curriculum_dqn_best.zip
python eval_curriculum_dqn.py --save
"""

import argparse
import glob
import json
import os
from datetime import datetime

from src.rl.evaluate import evaluate_episode, summarize_results
from src.rl.masked_dqn import MaskedDQN
from src.rl.train_curriculum_dqn import (
    createCurriculumEvalEnvironment,
    createEvaluationEnvironment,
)
from src.utils.config import CURRICULUM_DQN_CANVAS_SHAPE, CURRICULUM_DQN_MAX_BOXES
from src.utils.custom_maps import build_curriculum_maps
from src.utils.generated_maps import (
    build_generated_1box_maps,
    build_generated_2box_maps,
    build_generated_3box_maps,
)


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
# Episode runners
# ---------------------------------------------------------------------------

def _run_map_episodes(model, map_config, n_episodes):
    """Run N episodes on a single fixed map. Reuses one env (reset each time)."""
    env = createCurriculumEvalEnvironment([map_config])
    results = []
    try:
        for _ in range(n_episodes):
            results.append(evaluate_episode(model, env, log_progress=False))
    finally:
        env.close()
    return results


def _run_procedural_episodes(model, n_episodes):
    """Run N episodes on the procedural env. Reuses one env (reset each time)."""
    env = createEvaluationEnvironment()
    results = []
    try:
        for _ in range(n_episodes):
            results.append(evaluate_episode(model, env, log_progress=False))
    finally:
        env.close()
    return results


# ---------------------------------------------------------------------------
# Map group selection
# ---------------------------------------------------------------------------

def _build_groups(mode, boxes_filter):
    """Return [(label, [map_configs])] based on mode and box filter."""
    if mode == "generated":
        pool = {
            "1-box": build_generated_1box_maps(),
            "2-box": build_generated_2box_maps(),
            "3-box": build_generated_3box_maps(),
        }
    else:  # curriculum
        all_maps = build_curriculum_maps()
        pool = {
            "1-box": [m for m in all_maps if len(m["boxes"]) == 1],
            "2-box": [m for m in all_maps if len(m["boxes"]) == 2],
            "3-box": [m for m in all_maps if len(m["boxes"]) == 3],
        }

    if boxes_filter == "all":
        return list(pool.items())
    key = f"{boxes_filter}-box"
    return [(key, pool.get(key, []))]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_table_header():
    print(f"  {'Map':<20} {'Ep':<4}  {'Solved':<6}  {'Steps':<6}  {'Reward':<8}  Reason")
    print("  " + "-" * 64)


def _print_row(map_name, ep_idx, result):
    solved_str = "YES" if result["solved"] else "no"
    print(
        f"  {map_name:<20} {ep_idx + 1:<4}  {solved_str:<6}  "
        f"{result['steps']:<6}  {result['total_reward']:<8.2f}  {result['termination_reason']}"
    )


def _print_group_summary(label, results):
    if not results:
        return
    s = summarize_results(results)
    filled = int(s["success_rate"] * 20)
    bar = f"[{'=' * filled}{' ' * (20 - filled)}] {s['success_rate'] * 100:.0f}%"
    print(
        f"\n  {label}: {s['solved_count']}/{s['total_episodes']} solved  {bar}"
        f"  avg_steps={s['avg_steps']:.1f}  avg_reward={s['avg_reward']:.2f}"
    )


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _build_metadata(model_path, mode, boxes_filter, episodes, note):
    """Collect everything needed to identify this eval run later."""
    model_path = os.path.normpath(model_path)
    run_dir = os.path.dirname(model_path)
    run_id = os.path.basename(run_dir)
    checkpoint_type = "best" if "best" in os.path.basename(model_path) else "final"

    meta = {
        "eval_timestamp":  datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "model_path":      model_path,
        "training_run_id": run_id,
        "checkpoint_type": checkpoint_type,
        "training_note":   note or "unset — re-run with --note to label training phase",
        "eval_mode":       mode,
        "eval_boxes":      boxes_filter,
        "episodes":        episodes,
        "canvas_shape":    list(CURRICULUM_DQN_CANVAS_SHAPE),
        "max_boxes":       CURRICULUM_DQN_MAX_BOXES,
        "action_space":    CURRICULUM_DQN_MAX_BOXES * 4,
    }

    # Pull training summary from the run's selected_model.json if it exists
    selected_json = os.path.join(run_dir, "selected_model.json")
    if os.path.exists(selected_json):
        with open(selected_json) as f:
            sel = json.load(f)
        meta["training_procedural_eval"] = sel.get("procedural_eval_summary")
        meta["training_selected_label"]  = sel.get("selected_label")

    return meta


def _save_results(model_path, mode, boxes_filter, episodes, note, per_map_results, summary):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("results", "rl_tests", "curriculum_dqn", "evals", ts)
    os.makedirs(out_dir, exist_ok=True)

    tag = f"{mode}_boxes{boxes_filter}"
    metadata = _build_metadata(model_path, mode, boxes_filter, episodes, note)

    rows = [
        {"map": name, "episode": ep, **r}
        for name, ep_results in per_map_results.items()
        for ep, r in enumerate(ep_results)
    ]
    with open(os.path.join(out_dir, f"eval_{tag}.json"), "w") as f:
        json.dump({"metadata": metadata, "episodes": rows}, f, indent=2)
    with open(os.path.join(out_dir, f"eval_{tag}_summary.json"), "w") as f:
        json.dump({"metadata": metadata, "summary": summary}, f, indent=2)
    print(f"\n  Saved to: {out_dir}/eval_{tag}*.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate the curriculum DQN")
    parser.add_argument(
        "--mode",
        choices=["generated", "curriculum", "procedural"],
        default="generated",
        help="Evaluation suite (default: generated)",
    )
    parser.add_argument(
        "--boxes",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Box count filter, ignored for procedural (default: all)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Episodes per map (generated/curriculum) or total (procedural) (default: 3)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model .zip (default: auto-find latest curriculum_dqn_best.zip)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save result JSONs to results/rl_tests/curriculum_dqn/evals/",
    )
    parser.add_argument(
        "--note",
        default=None,
        help="Label what training phase produced this model, e.g. 'phase2_1box+2box'",
    )
    args = parser.parse_args()

    model_path = args.model or _find_latest_model()
    model = MaskedDQN.load(model_path)

    print(f"\n{'=' * 60}")
    print("CURRICULUM DQN EVALUATION")
    print(f"{'=' * 60}")
    print(f"  model    : {model_path}")
    print(
        f"  canvas   : {CURRICULUM_DQN_CANVAS_SHAPE}  "
        f"max_boxes={CURRICULUM_DQN_MAX_BOXES}  "
        f"action_space={CURRICULUM_DQN_MAX_BOXES * 4}"
    )
    print(f"  mode     : {args.mode}  |  boxes: {args.boxes}  |  episodes: {args.episodes}")
    print(f"  note     : {args.note or '(none — use --note to label training phase)'}")
    print()

    per_map_results = {}
    all_results = []

    if args.mode == "procedural":
        print(f"  Running {args.episodes} procedural Sokoban-small-v1 episodes...\n")
        _print_table_header()
        results = _run_procedural_episodes(model, args.episodes)
        for ep, r in enumerate(results):
            _print_row("procedural", ep, r)
        per_map_results["procedural"] = results
        all_results = results
        _print_group_summary("Sokoban-small-v1 (procedural)", results)

    else:
        groups = _build_groups(args.mode, args.boxes)
        total_maps = sum(len(maps) for _, maps in groups)
        print(f"  {total_maps} maps  x  {args.episodes} episodes each\n")
        _print_table_header()

        for group_label, maps in groups:
            if not maps:
                print(f"\n  (no {group_label} maps for mode={args.mode})")
                continue
            print(f"\n  ── {group_label.upper()}  ({len(maps)} maps) ──")
            group_results = []
            for m in maps:
                ep_results = _run_map_episodes(model, m, args.episodes)
                per_map_results[m["map_name"]] = ep_results
                for ep, r in enumerate(ep_results):
                    _print_row(m["map_name"], ep, r)
                group_results.extend(ep_results)
            _print_group_summary(group_label, group_results)
            all_results.extend(group_results)

    print(f"\n{'=' * 60}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 60}")
    summary = summarize_results(all_results) if all_results else {}
    key_order = [
        "success_rate", "solved_count", "total_episodes",
        "avg_reward", "avg_steps",
        "avg_boxes_on_target", "avg_best_boxes_on_target",
        "termination_counts",
    ]
    for k in key_order:
        if k not in summary:
            continue
        v = summary[k]
        print(f"  {k:<34} {v:.4f}" if isinstance(v, float) else f"  {k:<34} {v}")

    if args.save:
        _save_results(model_path, args.mode, args.boxes, args.episodes, args.note, per_map_results, summary)


if __name__ == "__main__":
    main()
