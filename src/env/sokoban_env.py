"""Helpers for creating procedural gym_sokoban environments."""

import gym
import gym_sokoban

from src.utils.config import ENV_ID


def initialize_env(env_id=None, seed=None):
    """Create one procedural Sokoban environment and optionally seed it."""
    procedural_env = gym.make(resolve_env_id(env_id), disable_env_checker=True)
    seed_environment(procedural_env, seed)
    procedural_env.reset(render_mode="tiny_rgb_array")
    return procedural_env


def resolve_env_id(env_id):
    """Return the caller-provided env id or the project default when none is given."""
    if env_id is not None:
        return str(env_id)
    return str(ENV_ID)


def seed_environment(procedural_env, seed):
    """Seed one gym_sokoban environment when the caller requested reproducibility."""
    if seed is None:
        return
    procedural_env.seed(int(seed))
