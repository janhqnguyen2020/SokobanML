"""Phase scheduling helpers for imitation-pretrained curriculum RL runs."""

from src.utils.config import CURRICULUM_DQN_PHASE_TIMESTEPS


def build_curriculum_phase_configs():
    """Return the seven-phase curriculum schedule used by the current RL pipeline."""
    phase_env_ids = build_phase_env_lists()
    phase_specs = build_phase_specs()
    return merge_phase_specs_with_timesteps(phase_specs, phase_env_ids)


def build_phase_env_lists():
    """Return the procedural envs enabled in each curriculum phase."""
    return [
        [],
        ["Sokoban-small-v0"],
        ["Sokoban-small-v0", "Sokoban-small-v1"],
        ["Sokoban-small-v0", "Sokoban-small-v1", "Sokoban-v0"],
        ["Sokoban-small-v0", "Sokoban-small-v1", "Sokoban-v0"],
        ["Sokoban-small-v0", "Sokoban-small-v1", "Sokoban-v0"],
        ["Sokoban-small-v0", "Sokoban-small-v1", "Sokoban-v0"],
    ]


def build_phase_specs():
    """Return the fixed-map weights and procedural fractions for each phase."""
    return [
        {"phase_id": 1, "phase_name": "fixed_foundations", "fixed_fraction": 1.0, "box_weights": {1: 0.80, 2: 0.20, 3: 0.00}},
        {"phase_id": 2, "phase_name": "two_box_transition", "fixed_fraction": 0.85, "box_weights": {1: 0.25, 2: 0.65, 3: 0.10}},
        {"phase_id": 3, "phase_name": "three_box_core", "fixed_fraction": 0.75, "box_weights": {1: 0.20, 2: 0.35, 3: 0.45}},
        {"phase_id": 4, "phase_name": "generalization_easy", "fixed_fraction": 0.70, "box_weights": {1: 0.15, 2: 0.30, 3: 0.55}},
        {"phase_id": 5, "phase_name": "generalization_mid", "fixed_fraction": 0.65, "box_weights": {1: 0.15, 2: 0.30, 3: 0.55}},
        {"phase_id": 6, "phase_name": "generalization_hard", "fixed_fraction": 0.60, "box_weights": {1: 0.15, 2: 0.30, 3: 0.55}},
        {"phase_id": 7, "phase_name": "generalization_full", "fixed_fraction": 0.55, "box_weights": {1: 0.15, 2: 0.30, 3: 0.55}},
    ]


def merge_phase_specs_with_timesteps(phase_specs, phase_env_ids):
    """Attach timesteps and procedural env lists to the four base phase specs."""
    merged_specs = []
    for phase_spec, env_ids, timesteps in zip(phase_specs, phase_env_ids, CURRICULUM_DQN_PHASE_TIMESTEPS):
        merged_phase = dict(phase_spec)
        merged_phase["timesteps"] = int(timesteps)
        merged_phase["procedural_env_ids"] = list(env_ids)
        merged_specs.append(merged_phase)
    return merged_specs


def find_phase_config(phase_configs, phase_id):
    """Return one phase dictionary by id."""
    for phase_config in phase_configs:
        if int(phase_config["phase_id"]) == int(phase_id):
            return phase_config
    raise ValueError(f"Phase {phase_id} is not defined.")
