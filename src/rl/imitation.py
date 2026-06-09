"""Imitation-learning helpers for high-level Sokoban DQN training."""

import hashlib
import json
import os
import pickle
import time

import numpy as np

from src.env.custom_env import SimpleCustomSokobanEnv
from src.env.sokoban_env import initialize_env
from src.planners.astar import AStarAgent
from src.rl.high_level_env import HighLevelSokobanEnv
from src.rl.high_level_env_parts.action_profile import (
    board_profile,
    build_physical_action_data,
    split_safe_actions,
)
from src.rl.high_level_env_parts.constants import DIRECTION_DELTAS
from src.rl.high_level_env_parts.state import read_board
from src.utils.config import (
    CURRICULUM_DQN_CANVAS_SHAPE,
    CURRICULUM_DQN_MAX_BOXES,
    CURRICULUM_PROCEDURAL_DEMO_TARGET_PER_ENV,
    HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES,
    SEED,
)
from src.utils.custom_maps import build_curriculum_maps
from src.utils.generated_maps import (
    build_generated_1box_maps,
    build_generated_2box_maps,
    build_generated_3box_maps,
    build_walled_1box_maps,
    build_walled_2box_maps,
    build_walled_3box_maps,
)

DEFAULT_DEMO_CACHE_PATH = os.path.join("data", "demos", "astar_curriculum_demos.pkl")
DEFAULT_HARD_CASE_CACHE_PATH = os.path.join("data", "demos", "astar_failed_curriculum_demos.pkl")
DEFAULT_PROCEDURAL_DEMO_CACHE_PATH = os.path.join("data", "demos", "astar_procedural_demos.pkl")
DEMO_PIPELINE_VERSION = 3


def build_imitation_curriculum_maps(include_handmade_curriculum=False):
    """Return the fixed-map curriculum pool used for imitation learning."""
    training_maps = _generated_and_walled_curriculum_maps()
    if include_handmade_curriculum:
        training_maps.extend(build_curriculum_maps())
    return training_maps


def _generated_and_walled_curriculum_maps():
    """Build the same generated and walled map pool used by baseline training."""
    return (
        build_generated_1box_maps()
        + build_walled_1box_maps()
        + build_generated_2box_maps()
        + build_walled_2box_maps()
        + build_generated_3box_maps()
        + build_walled_3box_maps()
    )


def create_custom_sokoban_env(map_config):
    """Build one fixed primitive Sokoban environment from a map dictionary."""
    return SimpleCustomSokobanEnv(
        height=map_config["height"],
        width=map_config["width"],
        player_position=map_config["player"],
        box_positions=map_config["boxes"],
        goal_positions=map_config["goals"],
        wall_positions=map_config.get("walls", []),
        max_steps=map_config.get("max_steps", 200),
    )


def create_fixed_high_level_env(map_config, max_boxes, observation_board_shape, use_extra_scalar_features):
    """Build one fixed high-level environment for macro-action replay."""
    return HighLevelSokobanEnv(
        observation_board_shape=observation_board_shape,
        use_extra_scalar_features=use_extra_scalar_features,
        use_shaped_reward=False,
        max_boxes=max_boxes,
        map_sampler=lambda: map_config,
    )


def create_procedural_map_config(env_id, reset_seed, sample_index):
    """Build one fixed map dictionary from a procedurally generated Sokoban room."""
    env = initialize_env(env_id=env_id, seed=reset_seed)
    try:
        return procedural_env_to_map_config(env, env_id, sample_index)
    finally:
        env.close()


def procedural_env_to_map_config(env, env_id, sample_index):
    """Convert the current procedural room into the fixed-map format used by A*."""
    player_pos, box_positions, goal_positions, wall_positions = read_board(env)
    room_state = env.unwrapped.room_state
    return {
        "map_name": procedural_map_name(env_id, sample_index),
        "difficulty": str(env_id),
        "height": int(room_state.shape[0]),
        "width": int(room_state.shape[1]),
        "player": player_pos,
        "boxes": sorted(list(box_positions)),
        "goals": sorted(list(goal_positions)),
        "walls": sorted(list(wall_positions)),
        "max_steps": int(env.unwrapped.max_steps),
    }


def procedural_map_name(env_id, sample_index):
    """Return one stable map name for a procedural A* demo sample."""
    return f"{env_id}_procedural_{int(sample_index):03d}"


def solve_map_with_astar(map_config, state_callback=None):
    """Solve one fixed map with A* and return both actions and search stats."""
    env = create_custom_sokoban_env(map_config)
    planner = AStarAgent(env, state_callback=state_callback)
    start_time = time.time()
    primitive_actions = planner._solve()
    runtime_ms = (time.time() - start_time) * 1000.0
    env.close()
    return {
        "primitive_actions": primitive_actions,
        "runtime_ms": runtime_ms,
        "nodes_expanded": planner.nodes_expanded,
        "deadlocks_pruned": planner.deadlocks_pruned,
        "dead_squares_count": planner.dead_squares_count,
        "num_pushes": planner.num_pushes,
        "solution_length": planner.solution_length,
        "failure_reason": solve_failure_reason(primitive_actions, planner.failure_reason),
    }


def solve_failure_reason(primitive_actions, planner_failure_reason):
    """Return a clear planner status string for solved and unsolved A* runs."""
    if primitive_actions is not None:
        return "solved"
    if planner_failure_reason is not None:
        return str(planner_failure_reason)
    return "no_solution_found"


def extract_push_events(map_config, primitive_actions):
    """Replay primitive A* actions and keep only the steps that move a box."""
    env = create_custom_sokoban_env(map_config)
    push_events = []
    for step_index, primitive_action in enumerate(primitive_actions or []):
        push_event = _replay_primitive_step(env, int(primitive_action), step_index, len(push_events))
        if push_event is not None:
            push_events.append(push_event)
    env.close()
    return push_events


def _replay_primitive_step(env, primitive_action, step_index, push_index):
    """Replay one primitive action and describe it when it is a box push."""
    before_player, before_boxes, goals, _ = read_board(env)
    env.step(int(primitive_action))
    after_player, after_boxes, _, _ = read_board(env)
    if before_boxes == after_boxes:
        return None
    moved_box = _moved_box(before_boxes, after_boxes)
    return {
        "primitive_action": int(primitive_action),
        "primitive_step_index": int(step_index),
        "push_index": int(push_index),
        "before_player": before_player,
        "after_player": after_player,
        "before_boxes": before_boxes,
        "after_boxes": after_boxes,
        "goals": goals,
        "moved_box_from": moved_box["from_box"],
        "moved_box_to": moved_box["to_box"],
    }


def _moved_box(before_boxes, after_boxes):
    """Identify which box moved during one primitive push."""
    before_only = sorted(list(before_boxes - after_boxes))
    after_only = sorted(list(after_boxes - before_boxes))
    return {"from_box": before_only[0], "to_box": after_only[0]}


def push_event_to_macro_action(push_event):
    """Convert one primitive box push into the macro-action id used by DQN."""
    moved_box_from = push_event["moved_box_from"]
    # The high-level policy numbers actions by the sorted pre-push box list.
    # We therefore locate the moved box in that sorted list, then pair it with
    # the push direction from the primitive A* action.
    box_order = sorted(list(push_event["before_boxes"]))
    box_index = int(box_order.index(moved_box_from))
    direction_index = int(push_event["primitive_action"]) - 1
    return {
        "success": True,
        "macro_action": box_index * 4 + direction_index,
        "box_index": box_index,
        "direction_index": direction_index,
    }


def build_demo_samples_for_map(map_config, max_boxes, observation_board_shape, use_extra_scalar_features):
    """Build masked macro-action demonstrations for one fixed curriculum map."""
    solve_result = solve_map_with_astar(map_config)
    if not solve_result["primitive_actions"]:
        return _map_demo_result(map_config, solve_result, [], False, solve_result["failure_reason"])
    push_events = extract_push_events(map_config, solve_result["primitive_actions"])
    replay_result = replay_push_events_as_macro_demos(
        map_config,
        push_events,
        solve_result,
        max_boxes,
        observation_board_shape,
        use_extra_scalar_features,
    )
    return replay_result


def _map_demo_result(map_config, solve_result, samples, solved, failure_reason):
    """Package one map-level demonstration result in a consistent shape."""
    return {
        "map_name": map_config["map_name"],
        "num_boxes": len(map_config["boxes"]),
        "astar_solved": solve_result_succeeded(solve_result),
        "demo_ready": bool(solved),
        "solved": bool(solved),
        "failure_reason": str(failure_reason),
        "samples": samples,
        "expert_pairs": len(samples),
        "astar_runtime_ms": float(solve_result["runtime_ms"]),
        "solution_length": int(solve_result["solution_length"]),
        "num_pushes": int(solve_result["num_pushes"]),
        "nodes_expanded": int(solve_result["nodes_expanded"]),
        "deadlocks_pruned": int(solve_result["deadlocks_pruned"]),
        "dead_squares_count": int(solve_result["dead_squares_count"]),
    }


def solve_result_succeeded(solve_result):
    """Return True when A* found a valid primitive solution for the map."""
    if solve_result["primitive_actions"] is not None:
        return True
    return str(solve_result["failure_reason"]) == "solved"


def replay_push_events_as_macro_demos(map_config, push_events, solve_result, max_boxes, observation_board_shape, use_extra_scalar_features):
    """Replay expert pushes inside the high-level env and collect observations."""
    env = create_fixed_high_level_env(
        map_config,
        max_boxes,
        observation_board_shape,
        use_extra_scalar_features,
    )
    observation = env.reset(seed=SEED)
    samples = []
    failure_reason = "ok"
    for push_event in push_events:
        macro_result = push_event_to_macro_action(push_event)
        if not macro_result["success"]:
            failure_reason = macro_result["reason"]
            break
        sample_result = _collect_macro_demo_step(
            env,
            observation,
            map_config,
            push_event,
            macro_result,
            solve_result["runtime_ms"],
            solve_result["solution_length"],
            len(push_events),
        )
        if not sample_result["success"]:
            failure_reason = sample_result["reason"]
            break
        samples.append(sample_result["sample"])
        observation = sample_result["next_observation"]
    env.close()
    solved = failure_reason == "ok"
    return _map_demo_result(map_config, solve_result, samples, solved, failure_reason)


def _collect_macro_demo_step(env, observation, map_config, push_event, macro_result, astar_runtime_ms, solution_length, total_pushes):
    """Save one expert macro step after checking it is valid under the mask."""
    macro_action = int(macro_result["macro_action"])
    replay_step = _resolve_demo_replay_step(env, observation, macro_action)
    if not replay_step["success"]:
        return replay_step
    next_observation, done = _run_demo_macro_step(env, macro_action, replay_step)
    if not _matches_expected_boxes(env, push_event["after_boxes"]):
        return {"success": False, "reason": "macro_replay_box_mismatch"}
    saved_observation = _observation_with_action_mask(observation, replay_step["action_mask"])
    sample = _demo_sample(
        map_config,
        saved_observation,
        replay_step["action_mask"],
        push_event,
        macro_result,
        astar_runtime_ms,
        solution_length,
        total_pushes,
    )
    if done and push_event["push_index"] + 1 < total_pushes:
        return {"success": False, "reason": "macro_replay_ended_early"}
    return {"success": True, "sample": sample, "next_observation": next_observation}


def _resolve_demo_replay_step(env, observation, macro_action):
    """Choose the action mask and replay path for one expert macro action."""
    selected_mask = _selected_action_mask(env, observation)
    if selected_mask[macro_action] > 0.5:
        return {"success": True, "action_mask": selected_mask, "action_data": None}
    safe_actions = _safe_expert_actions(env)
    if macro_action not in safe_actions:
        return {"success": False, "reason": "expert_action_invalid_under_mask"}
    action_mask = _action_mask_from_actions(env.action_space.n, safe_actions)
    action_data = safe_actions[macro_action]
    return {"success": True, "action_mask": action_mask, "action_data": action_data}


def _selected_action_mask(env, observation):
    """Read the current binary action mask from one saved observation vector."""
    return observation[-env.action_space.n :].copy()


def _safe_expert_actions(env):
    """Return non-deadlock expert actions before heuristic pruning trims them."""
    player_pos, box_positions, goals, walls, _ = board_profile(
        env.env,
        env.dead_squares,
        env.num_boxes,
        DIRECTION_DELTAS,
    )
    physical_actions = build_physical_action_data(
        player_pos,
        box_positions,
        walls,
        DIRECTION_DELTAS,
        goals,
    )
    safe_actions, _ = split_safe_actions(
        physical_actions,
        goals,
        walls,
        env.dead_squares,
        env.num_boxes,
        DIRECTION_DELTAS,
    )
    return safe_actions


def _action_mask_from_actions(action_space_size, action_data_by_id):
    """Build one binary action mask from the chosen macro-action ids."""
    action_mask = np.zeros(int(action_space_size), dtype=np.float32)
    for action in action_data_by_id:
        action_mask[int(action)] = 1.0
    return action_mask


def _run_demo_macro_step(env, macro_action, replay_step):
    """Replay one expert macro action through the normal or forced env path."""
    if replay_step["action_data"] is None:
        next_observation, _, done, _ = env.step(macro_action)
        return next_observation, done
    next_observation, _, done, _ = env.step_with_macro_action_data(
        macro_action,
        replay_step["action_data"],
    )
    return next_observation, done


def _observation_with_action_mask(observation, action_mask):
    """Copy one observation and replace its mask tail with the chosen mask."""
    updated_observation = observation.copy()
    updated_observation[-len(action_mask) :] = action_mask
    return updated_observation


def _matches_expected_boxes(env, expected_boxes):
    """Check that macro replay produced the same box layout as primitive replay."""
    _, actual_boxes, _, _ = read_board(env.env)
    return actual_boxes == expected_boxes


def _demo_sample(map_config, observation, action_mask, push_event, macro_result, astar_runtime_ms, solution_length, total_pushes):
    """Build one saved expert state-action pair for behavior cloning."""
    return {
        "observation": observation.copy(),
        "action": int(macro_result["macro_action"]),
        "action_mask": action_mask.copy(),
        "map_name": map_config["map_name"],
        "num_boxes": len(map_config["boxes"]),
        "push_index": int(push_event["push_index"]),
        "primitive_step_index": int(push_event["primitive_step_index"]),
        "box_index": int(macro_result["box_index"]),
        "direction_index": int(macro_result["direction_index"]),
        "astar_runtime_ms": float(astar_runtime_ms),
        "solution_length": int(solution_length),
        "num_pushes": int(total_pushes),
    }


def generate_demonstration_payload(map_configs, max_boxes=CURRICULUM_DQN_MAX_BOXES, observation_board_shape=CURRICULUM_DQN_CANVAS_SHAPE, use_extra_scalar_features=HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES):
    """Solve the requested maps and collect all valid macro demonstrations."""
    map_results = []
    all_samples = []
    for map_config in map_configs:
        map_result = build_demo_samples_for_map(
            map_config,
            max_boxes,
            observation_board_shape,
            use_extra_scalar_features,
        )
        map_results.append(map_result)
        all_samples.extend(map_result["samples"])
    metadata = _fixed_demo_payload_metadata(
        map_configs,
        max_boxes,
        observation_board_shape,
        use_extra_scalar_features,
    )
    return _demo_payload(all_samples, map_results, metadata)


def load_or_generate_procedural_demonstrations(
    env_ids,
    cache_path=DEFAULT_PROCEDURAL_DEMO_CACHE_PATH,
    regenerate=False,
    target_per_env=CURRICULUM_PROCEDURAL_DEMO_TARGET_PER_ENV,
):
    """Reuse cached procedural A* demos unless the caller requests regeneration."""
    if os.path.exists(cache_path) and not regenerate:
        payload = load_demonstration_payload(cache_path)
        if _procedural_payload_matches_request(payload, env_ids, target_per_env):
            return payload
    payload = generate_procedural_demo_payload(env_ids, target_per_env)
    save_demonstration_payload(cache_path, payload)
    return payload


def generate_procedural_demo_payload(env_ids, target_per_env):
    """Collect solved procedural A* demos until each requested env reaches its target."""
    all_results = []
    all_samples = []
    for env_index, env_id in enumerate(env_ids):
        env_results, env_samples = collect_env_demo_results(env_id, env_index, target_per_env)
        all_results.extend(env_results)
        all_samples.extend(env_samples)
    metadata = _procedural_demo_payload_metadata(env_ids, target_per_env)
    return _demo_payload(all_samples, all_results, metadata)


def collect_env_demo_results(env_id, env_index, target_per_env):
    """Collect solved demos for one procedural env id using deterministic seeds."""
    accepted_count = 0
    attempt_index = 0
    max_attempts = int(target_per_env) * 20
    env_results = []
    env_samples = []
    while accepted_count < int(target_per_env) and attempt_index < max_attempts:
        map_config = create_procedural_map_config(env_id, procedural_demo_seed(env_index, attempt_index), attempt_index)
        map_result = build_demo_samples_for_map(
            map_config,
            CURRICULUM_DQN_MAX_BOXES,
            CURRICULUM_DQN_CANVAS_SHAPE,
            HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES,
        )
        env_results.append(map_result)
        if map_result["demo_ready"]:
            env_samples.extend(map_result["samples"])
            accepted_count += 1
        attempt_index += 1
    if accepted_count < int(target_per_env):
        raise ValueError(f"Only collected {accepted_count} solved procedural demos for {env_id}.")
    return env_results, env_samples


def procedural_demo_seed(env_index, attempt_index):
    """Return one stable seed used when sampling procedural A* demo rooms."""
    return int(SEED + 100_000 + env_index * 1_000 + attempt_index)


def _demo_payload(samples, map_results, metadata):
    """Bundle samples, per-map results, and cache metadata into one payload."""
    return {
        "samples": samples,
        "map_results": map_results,
        "summary": summarize_demo_generation(map_results),
        "metadata": metadata,
    }


def _fixed_demo_payload_metadata(map_configs, max_boxes, observation_board_shape, use_extra_scalar_features):
    """Describe one fixed-map demo request so stale caches can be detected."""
    return {
        "payload_kind": "fixed",
        "demo_pipeline_version": DEMO_PIPELINE_VERSION,
        "request_signature": _fixed_request_signature(
            map_configs,
            max_boxes,
            observation_board_shape,
            use_extra_scalar_features,
        ),
    }


def _procedural_demo_payload_metadata(env_ids, target_per_env):
    """Describe one procedural demo request so stale caches can be detected."""
    return {
        "payload_kind": "procedural",
        "demo_pipeline_version": DEMO_PIPELINE_VERSION,
        "request_signature": _procedural_request_signature(env_ids, target_per_env),
    }


def _fixed_request_signature(map_configs, max_boxes, observation_board_shape, use_extra_scalar_features):
    """Build one stable fingerprint for a fixed-map demo request."""
    request = {
        "map_configs": _canonical_fixed_maps(map_configs),
        "max_boxes": int(max_boxes),
        "observation_board_shape": list(observation_board_shape),
        "use_extra_scalar_features": bool(use_extra_scalar_features),
    }
    return _signature_text(request)


def _procedural_request_signature(env_ids, target_per_env):
    """Build one stable fingerprint for a procedural demo request."""
    request = {
        "env_ids": [str(env_id) for env_id in env_ids],
        "target_per_env": int(target_per_env),
        "max_boxes": int(CURRICULUM_DQN_MAX_BOXES),
        "observation_board_shape": list(CURRICULUM_DQN_CANVAS_SHAPE),
        "use_extra_scalar_features": bool(HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES),
    }
    return _signature_text(request)


def _canonical_fixed_maps(map_configs):
    """Sort map configs into one stable order before hashing them."""
    sorted_maps = sorted(map_configs, key=lambda item: str(item["map_name"]))
    return [_canonical_map_config(map_config) for map_config in sorted_maps]


def _canonical_map_config(map_config):
    """Keep only the stable map fields that affect replay correctness."""
    return {
        "map_name": str(map_config["map_name"]),
        "height": int(map_config["height"]),
        "width": int(map_config["width"]),
        "player": [int(value) for value in map_config["player"]],
        "boxes": _sorted_positions(map_config["boxes"]),
        "goals": _sorted_positions(map_config["goals"]),
        "walls": _sorted_positions(map_config.get("walls", [])),
        "max_steps": int(map_config.get("max_steps", 200)),
    }


def _sorted_positions(positions):
    """Turn board positions into one sorted JSON-friendly list."""
    sorted_positions = sorted(list(positions))
    return [[int(row), int(col)] for row, col in sorted_positions]


def _signature_text(payload_request):
    """Hash one cache request into a compact reusable signature string."""
    request_text = json.dumps(payload_request, sort_keys=True)
    request_bytes = request_text.encode("utf-8")
    return hashlib.sha256(request_bytes).hexdigest()


def _fixed_payload_matches_request(payload, map_configs, max_boxes, observation_board_shape, use_extra_scalar_features):
    """Return True when one cached fixed payload matches the current request."""
    expected_signature = _fixed_request_signature(
        map_configs,
        max_boxes,
        observation_board_shape,
        use_extra_scalar_features,
    )
    return _payload_signature_matches(payload, "fixed", expected_signature)


def _procedural_payload_matches_request(payload, env_ids, target_per_env):
    """Return True when one cached procedural payload matches the request."""
    expected_signature = _procedural_request_signature(env_ids, target_per_env)
    return _payload_signature_matches(payload, "procedural", expected_signature)


def _payload_signature_matches(payload, payload_kind, expected_signature):
    """Compare one cached payload fingerprint against the current request."""
    metadata = payload.get("metadata", {})
    if metadata.get("payload_kind") != payload_kind:
        return False
    if metadata.get("demo_pipeline_version") != DEMO_PIPELINE_VERSION:
        return False
    return metadata.get("request_signature") == expected_signature


def summarize_demo_generation(map_results):
    """Build the high-level summary requested for expert generation runs."""
    attempted = len(map_results)
    astar_solved = sum(1 for result in map_results if result["astar_solved"])
    demo_ready = sum(1 for result in map_results if result["demo_ready"])
    collected = sum(result["expert_pairs"] for result in map_results)
    astar_failed = attempted - astar_solved
    macro_rejected = astar_solved - demo_ready
    return {
        "maps_attempted": attempted,
        "maps_solved_by_astar": astar_solved,
        "maps_converted_to_macro_demos": demo_ready,
        "expert_state_action_pairs": collected,
        "maps_rejected_during_macro_replay": macro_rejected,
        "maps_failed_by_astar": astar_failed,
        "maps_skipped_or_failed": astar_failed,
    }


def print_demo_summary(summary):
    """Print a short human-readable summary of the demonstration run."""
    print("A* demonstration summary")
    print(f"  maps attempted: {summary['maps_attempted']}")
    print(f"  maps solved by A*: {summary['maps_solved_by_astar']}")
    print(f"  maps converted to macro demos: {summary['maps_converted_to_macro_demos']}")
    print(f"  expert state-action pairs: {summary['expert_state_action_pairs']}")
    print(f"  maps rejected during macro replay: {summary['maps_rejected_during_macro_replay']}")
    print(f"  maps failed by A*: {summary['maps_failed_by_astar']}")


def load_demonstration_payload(cache_path):
    """Load a cached demonstration payload from disk."""
    with open(cache_path, "rb") as cache_file:
        return pickle.load(cache_file)


def save_demonstration_payload(cache_path, payload):
    """Save a demonstration payload to disk, creating its folder when needed."""
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "wb") as cache_file:
        pickle.dump(payload, cache_file)


def load_or_generate_demonstrations(map_configs, cache_path=DEFAULT_DEMO_CACHE_PATH, regenerate=False, max_boxes=CURRICULUM_DQN_MAX_BOXES, observation_board_shape=CURRICULUM_DQN_CANVAS_SHAPE, use_extra_scalar_features=HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES):
    """Reuse cached demonstrations unless the caller requests regeneration."""
    if os.path.exists(cache_path) and not regenerate:
        payload = load_demonstration_payload(cache_path)
        if _fixed_payload_matches_request(
            payload,
            map_configs,
            max_boxes,
            observation_board_shape,
            use_extra_scalar_features,
        ):
            return payload
    payload = generate_demonstration_payload(
        map_configs,
        max_boxes=max_boxes,
        observation_board_shape=observation_board_shape,
        use_extra_scalar_features=use_extra_scalar_features,
    )
    save_demonstration_payload(cache_path, payload)
    return payload


def limit_demonstration_samples(payload, max_demonstrations, seed=SEED):
    """Optionally subsample the cached demonstrations for faster experiments."""
    if max_demonstrations is None or len(payload["samples"]) <= max_demonstrations:
        return payload
    rng = np.random.default_rng(seed)
    chosen_indices = rng.permutation(len(payload["samples"]))[: int(max_demonstrations)]
    limited_samples = [payload["samples"][int(index)] for index in sorted(chosen_indices)]
    limited_payload = dict(payload)
    limited_payload["samples"] = limited_samples
    limited_payload["summary"] = dict(payload["summary"])
    limited_payload["summary"]["expert_state_action_pairs"] = len(limited_samples)
    return limited_payload


def validate_demonstration_actions(demonstration_samples):
    """Return True only when every expert action is valid under its saved mask."""
    return all(_sample_action_is_valid(sample) for sample in demonstration_samples)


def _sample_action_is_valid(sample):
    """Check one saved expert action against its saved binary action mask."""
    action = int(sample["action"])
    sample_mask = sample["action_mask"]
    observation_mask = sample["observation"][-len(sample_mask) :]
    return float(sample_mask[action]) > 0.5 and float(observation_mask[action]) > 0.5


def merge_demonstration_payloads(primary_payload, secondary_payload):
    """Append two payloads so hard-case demos can weight hard maps more heavily."""
    merged_results = list(primary_payload["map_results"]) + list(secondary_payload["map_results"])
    merged_samples = list(primary_payload["samples"]) + list(secondary_payload["samples"])
    return {
        "samples": merged_samples,
        "map_results": merged_results,
        "summary": summarize_demo_generation(merged_results),
    }


def failed_map_configs(map_configs, map_results):
    """Return the map configs whose evaluation results were not solved."""
    failed_names = {result["map_name"] for result in map_results if not result["solved"]}
    return [map_config for map_config in map_configs if map_config["map_name"] in failed_names]


def run_macro_replay_sanity_check(map_config, max_boxes=CURRICULUM_DQN_MAX_BOXES, observation_board_shape=CURRICULUM_DQN_CANVAS_SHAPE, use_extra_scalar_features=HIGH_LEVEL_USE_EXTRA_SCALAR_FEATURES):
    """Solve one map, extract macro pushes, replay them, and verify mask validity."""
    payload = generate_demonstration_payload(
        [map_config],
        max_boxes=max_boxes,
        observation_board_shape=observation_board_shape,
        use_extra_scalar_features=use_extra_scalar_features,
    )
    sample_validity = validate_demonstration_actions(payload["samples"])
    map_result = payload["map_results"][0]
    return {
        "success": bool(map_result["solved"] and sample_validity and map_result["expert_pairs"] > 0),
        "map_name": map_result["map_name"],
        "failure_reason": map_result["failure_reason"],
        "expert_pairs": map_result["expert_pairs"],
        "sample_actions_valid": sample_validity,
    }
