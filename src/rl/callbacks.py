# src/rl/callbacks.py
import json
import logging
import os
from stable_baselines3.common.callbacks import BaseCallback
from src.rl.evaluate import evaluate_episode, summarize_results

LOGGER = logging.getLogger(__name__)

def summary_rank(summary):
    """Rank two validation summaries so better checkpoints sort first."""
    return (
        min_box_family_success(summary),
        average_box_family_success(summary),
        average_fixed_family_success(summary),
        procedural_success_rate(summary),
        float(summary.get("success_rate", 0.0)),
        float(summary.get("avg_best_boxes_on_target", 0.0)),
        float(summary.get("avg_boxes_on_target", 0.0)),
        float(summary.get("avg_reward", float("-inf"))),
        -int(summary.get("one_step_dead_end_count", 0)),
    )


def min_box_family_success(summary):
    """Prefer checkpoints that do not collapse one of the 1/2/3-box validation families."""
    box_scores = box_family_success_values(summary)
    if not box_scores:
        return 0.0
    return float(min(box_scores))


def average_box_family_success(summary):
    """Prefer checkpoints with stronger average success across generated 1/2/3-box maps."""
    return average_values(box_family_success_values(summary))


def average_fixed_family_success(summary):
    """Prefer checkpoints that stay balanced across the fixed validation families."""
    fixed_group = type_summary_group(summary, "fixed")
    return average_values([
        family_success_rate(fixed_group, "generated_1box"),
        family_success_rate(fixed_group, "generated_2box"),
        family_success_rate(fixed_group, "generated_3box"),
        family_success_rate(fixed_group, "walled"),
        family_success_rate(fixed_group, "csp"),
    ])


def procedural_success_rate(summary):
    """Use the aggregate procedural success as a secondary checkpoint signal."""
    if "procedural" not in summary:
        return 0.0
    return float(summary["procedural"].get("success_rate", 0.0))


def box_family_success_values(summary):
    """Return the generated 1/2/3-box success rates used to preserve earlier skills."""
    fixed_group = type_summary_group(summary, "fixed")
    return [
        family_success_rate(fixed_group, "generated_1box"),
        family_success_rate(fixed_group, "generated_2box"),
        family_success_rate(fixed_group, "generated_3box"),
    ]


def type_summary_group(summary, group_name):
    """Read one section of the saved type summary and fall back to an empty group."""
    if "type_summary" not in summary:
        return {}
    return dict(summary["type_summary"].get(str(group_name), {}))


def family_success_rate(summary_group, family_name):
    """Read one family success rate from the nested periodic type summary."""
    if family_name not in summary_group:
        return 0.0
    return float(summary_group[str(family_name)].get("success_rate", 0.0))


def average_values(values):
    """Return the average of a few scalar checkpoint scores."""
    if not values:
        return 0.0
    return float(sum(values) / len(values))

class PeriodicEvalCallback(BaseCallback):
    """Run fixed validation every few timesteps and save the best phase checkpoint."""

    def __init__(
        self,
        best_model_path,
        eval_env_factory,
        eval_seed_base,
        eval_freq,
        n_eval_episodes,
        early_stop_patience_evals,
        early_stop_min_timesteps,
        periodic_history_path=None,
        best_summary_path=None,
        snapshot_base_path=None,
        periodic_type_summary_dir=None,
        phase_elapsed_timesteps=0,
        summary_factory=None,
        summary_consumer=None,
    ):
        super().__init__()
        self.best_model_path = best_model_path
        self.eval_env_factory = eval_env_factory
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.early_stop_patience_evals = early_stop_patience_evals
        self.early_stop_min_timesteps = early_stop_min_timesteps
        self.summaryFactory = summary_factory
        self.best_summary = None
        self.no_improvement_evals = 0
        self.last_eval_step = 0
        self.eval_episode_seeds = build_eval_episode_seeds(eval_seed_base, n_eval_episodes, summary_factory)
        self.periodicHistoryPath = periodic_history_path
        self.bestSummaryPath = best_summary_path
        self.snapshotBasePath = snapshot_base_path
        self.periodicTypeSummaryDir = periodic_type_summary_dir
        self.phaseElapsedTimesteps = int(phase_elapsed_timesteps)
        self.periodicHistory = []
        self.phaseStartTimesteps = 0
        self.summaryConsumer = summary_consumer

    def _init_callback(self):
        """Reset the in-memory periodic history before training starts."""
        self.periodicHistory = []

    def _on_training_start(self) -> None:
        """Anchor periodic evaluation to the real resumed timestep of this phase."""
        self.last_eval_step = int(self.num_timesteps)
        self.phaseStartTimesteps = int(self.num_timesteps)
        self.periodicHistory = []

    def _evaluate_current_model(self):
        """Run the current model on the fixed validation episodes and summarize the result."""
        if self.summaryFactory is not None:
            return summarize_factory_result(self.summaryFactory(self.model), self.num_timesteps)
        results = []
        for episode_seed in self.eval_episode_seeds:
            eval_env = self.eval_env_factory()
            try:
                results.append(
                    evaluate_episode(
                        self.model,
                        eval_env,
                        log_progress=False,
                        reset_seed=episode_seed,
                    )
                )
            finally:
                eval_env.close()
        return summarize_factory_result(summarize_results(results), self.num_timesteps)

    def _log_summary(self, summary):
        """Show the periodic evaluation result in both logs and the terminal."""
        message = periodic_eval_message(self.num_timesteps, summary)
        LOGGER.info(message)
        print(message)

    def _save_best_checkpoint(self, summary):
        """Save the best checkpoint seen so far in this phase and print where it went."""
        self.best_summary = dict(summary)
        self.best_summary["model_saved_to"] = self.best_model_path + ".zip"
        self.model.save(self.best_model_path)
        maybe_write_json(self.bestSummaryPath, self.best_summary)
        self.no_improvement_evals = 0
        message = best_checkpoint_message(self.best_model_path)
        LOGGER.info(message)
        print(message)

    def _save_periodic_history(self, summary, is_best):
        """Append one periodic evaluation row and write the phase history JSON."""
        history_row = dict(summary)
        history_row["is_best"] = bool(is_best)
        self.periodicHistory.append(history_row)
        maybe_write_json(self.periodicHistoryPath, self.periodicHistory)
        self._save_periodic_type_summary(summary)
        self._share_periodic_summary(summary)


    def _share_periodic_summary(self, summary):
        """Send one periodic summary to any live training component that wants it."""
        if self.summaryConsumer is None:
            return
        self.summaryConsumer(summary)

    def _save_periodic_type_summary(self, summary):
        """Save one per-checkpoint type summary so each 10k eval is easy to inspect."""
        if self.periodicTypeSummaryDir is None or "type_summary" not in summary:
            return
        summary_path = periodic_type_summary_path(
            self.periodicTypeSummaryDir,
            self.phaseStartTimesteps,
            self.num_timesteps,
            self.phaseElapsedTimesteps,
        )
        maybe_write_json(summary_path, summary["type_summary"])

    def _save_snapshot_checkpoint(self):
        """Save one resumable phase snapshot before periodic evaluation starts."""
        if self.snapshotBasePath is None:
            return
        snapshot_path = snapshot_model_path(
            self.snapshotBasePath,
            self.phaseStartTimesteps,
            self.num_timesteps,
            self.phaseElapsedTimesteps,
        )
        self.model.save(snapshot_path)
        self.model.save_replay_buffer(snapshot_path + "_replay_buffer.pkl")
        print(snapshot_checkpoint_message(snapshot_path))

    def _should_stop_early(self):
        """Return True when the phase has stalled long enough to stop early."""
        if self.num_timesteps < self.early_stop_min_timesteps:
            return False
        return self.no_improvement_evals >= self.early_stop_patience_evals

    def _on_step(self):
        """Run periodic validation when enough timesteps have passed since the last check."""
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        self._save_snapshot_checkpoint()
        summary = self._evaluate_current_model()
        self._log_summary(summary)
        is_best = self.best_summary is None or summary_rank(summary) > summary_rank(self.best_summary)
        if is_best:
            self._save_best_checkpoint(summary)
        else:
            self.no_improvement_evals += 1
        self._save_periodic_history(summary, is_best)
        self.last_eval_step = self.num_timesteps
        if self._should_stop_early():
            message = early_stop_message(self.no_improvement_evals, self.best_summary)
            LOGGER.info(message)
            print(message)
            return False
        return True


class PhaseProgressCallback(BaseCallback):
    """Print simple curriculum progress so long RL phases are easy to follow."""

    def __init__(self, curriculum_teacher, phase_config, phase_timesteps, completed_run_timesteps, total_run_timesteps, log_every_timesteps, phase_elapsed_timesteps=0):
        super().__init__()
        self.curriculumTeacher = curriculum_teacher
        self.phaseConfig = phase_config
        self.phaseTimesteps = int(phase_timesteps)
        self.phaseElapsedTimesteps = int(phase_elapsed_timesteps)
        self.completedRunTimesteps = int(completed_run_timesteps)
        self.totalRunTimesteps = int(total_run_timesteps)
        self.logEveryTimesteps = max(int(log_every_timesteps), 1)
        self.lastLoggedTimesteps = 0
        self.phaseStartTimesteps = 0

    def _init_callback(self):
        """Reset the phase progress counters before training starts."""
        self.phaseStartTimesteps = 0
        self.lastLoggedTimesteps = 0

    def _on_training_start(self) -> None:
        """Anchor progress logging to the real resumed timestep of this phase."""
        self.phaseStartTimesteps = int(self.num_timesteps)
        self.lastLoggedTimesteps = int(self.num_timesteps)
        print(
            phase_start_message(
                self.phaseConfig,
                self.phaseTimesteps,
                self.completedRunTimesteps,
                self.totalRunTimesteps,
                self.phaseElapsedTimesteps,
            )
        )

    def _on_step(self):
        """Print one progress line every fixed timestep interval."""
        if self.num_timesteps - self.lastLoggedTimesteps < self.logEveryTimesteps:
            return True
        self.lastLoggedTimesteps = int(self.num_timesteps)
        phase_steps_done = absolute_phase_timesteps(self.phaseStartTimesteps, self.num_timesteps, self.phaseElapsedTimesteps)
        print(
            phase_progress_message(
                self.phaseConfig,
                self.curriculumTeacher.phaseProgressSummary(),
                phase_steps_done,
                target_phase_timesteps(self.phaseTimesteps, self.phaseElapsedTimesteps),
                self.completedRunTimesteps,
                self.totalRunTimesteps,
            )
        )
        return True


def phase_start_message(phase_config, phase_timesteps, completed_run_timesteps, total_run_timesteps, phase_elapsed_timesteps):
    """Build the short line shown when one curriculum phase begins."""
    return (
        f"[Curriculum] Start phase {phase_config['phase_id']} ({phase_config['phase_name']}) | "
        f"phase_timesteps={phase_elapsed_timesteps}/{target_phase_timesteps(phase_timesteps, phase_elapsed_timesteps)} | "
        f"run_progress={completed_run_timesteps}/{total_run_timesteps} | "
        f"fixed_fraction={phase_config['fixed_fraction']:.2f} | procedural_envs={phase_config['procedural_env_ids']}"
    )


def phase_progress_message(phase_config, phase_summary, phase_timesteps_done, phase_timesteps_total, completed_run_timesteps, total_run_timesteps):
    """Build one readable progress line for the current curriculum phase."""
    run_timesteps_done = completed_run_timesteps + int(phase_timesteps_done)
    return (
        f"[Curriculum] Phase {phase_config['phase_id']} progress | "
        f"phase_timesteps={phase_timesteps_done}/{phase_timesteps_total} | "
        f"run_timesteps={run_timesteps_done}/{total_run_timesteps} | "
        f"episodes={phase_summary['phase_sampled_episodes']} | "
        f"fixed={phase_summary['fixed_counts']} | "
        f"procedural={phase_summary['procedural_counts']} | "
        f"focused_fixed={phase_summary['focused_fixed_samples']}"
    )


def periodic_eval_message(num_timesteps, summary):
    """Build one readable line for the periodic fixed validation checkpoint check."""
    return (
        f"[Curriculum] Periodic eval | timesteps={int(num_timesteps)} | "
        f"success_rate={summary['success_rate']:.3f} | "
        f"solved={summary['solved_count']}/{summary['total_episodes']} | "
        f"avg_best_boxes={summary['avg_best_boxes_on_target']:.2f} | "
        f"avg_reward={summary['avg_reward']:.2f} | "
        f"one_step_dead_ends={summary['one_step_dead_end_count']}"
        + periodic_component_message(summary)
    )


def best_checkpoint_message(best_model_path):
    """Build the short line shown when a new periodic best checkpoint is saved."""
    return f"[Curriculum] Saved new best checkpoint: {best_model_path}.zip"


def snapshot_checkpoint_message(snapshot_path):
    """Build the short line shown when one periodic snapshot checkpoint is saved."""
    return f"[Curriculum] Saved periodic snapshot: {snapshot_path}.zip"


def early_stop_message(no_improvement_evals, best_summary):
    """Build one readable early-stop message when phase validation has plateaued."""
    best_rate = 0.0 if best_summary is None else best_summary["success_rate"]
    best_step = "unknown" if best_summary is None else best_summary["timesteps"]
    return (
        f"[Curriculum] Early stop | stale_evals={int(no_improvement_evals)} | "
        f"best_success_rate={best_rate:.3f} | best_timesteps={best_step}"
    )


def maybe_write_json(path, payload):
    """Write one small JSON file only when this callback was given a save path."""
    if path is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as output_file:
        json.dump(payload, output_file, indent=2)


def build_eval_episode_seeds(eval_seed_base, n_eval_episodes, summary_factory):
    """Return the episode seeds used by env-based periodic evaluation."""
    if summary_factory is not None:
        return []
    return [eval_seed_base + i for i in range(n_eval_episodes)]


def summarize_factory_result(summary, num_timesteps):
    """Attach the current timestep to one periodic evaluation summary."""
    summary_with_steps = dict(summary)
    summary_with_steps["timesteps"] = int(num_timesteps)
    return summary_with_steps


def periodic_component_message(summary):
    """Add fixed and procedural counts when the periodic summary includes both."""
    if "fixed" not in summary or "procedural" not in summary:
        return ""
    fixed_summary = summary["fixed"]
    procedural_summary = summary["procedural"]
    return (
        f" | fixed={fixed_summary['solved_count']}/{fixed_summary['total_episodes']}"
        f" | procedural={procedural_summary['solved_count']}/{procedural_summary['total_episodes']}"
    )


def absolute_phase_timesteps(phase_start_timesteps, current_timesteps, phase_elapsed_timesteps):
    """Return the total phase progress, including work already done before resume."""
    resumed_steps = int(current_timesteps) - int(phase_start_timesteps)
    return int(phase_elapsed_timesteps) + max(resumed_steps, 0)


def target_phase_timesteps(phase_timesteps, phase_elapsed_timesteps):
    """Return the absolute phase target shown in resumed progress messages."""
    return int(phase_timesteps) + int(phase_elapsed_timesteps)


def snapshot_model_path(snapshot_base_path, phase_start_timesteps, current_timesteps, phase_elapsed_timesteps=0):
    """Build one snapshot path using absolute phase progress across resumed runs."""
    phase_timesteps = absolute_phase_timesteps(phase_start_timesteps, current_timesteps, phase_elapsed_timesteps)
    return f"{snapshot_base_path}_step{phase_timesteps:06d}"


def periodic_type_summary_path(periodic_type_summary_dir, phase_start_timesteps, current_timesteps, phase_elapsed_timesteps=0):
    """Build one type-summary path using absolute phase progress across resumed runs."""
    phase_timesteps = absolute_phase_timesteps(phase_start_timesteps, current_timesteps, phase_elapsed_timesteps)
    return os.path.join(periodic_type_summary_dir, f"step{phase_timesteps:06d}_type_summary.json")
