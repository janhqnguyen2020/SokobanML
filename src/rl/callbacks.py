import logging
from stable_baselines3.common.callbacks import BaseCallback
from src.rl.evaluate import evaluate_episode, summarize_results

LOGGER = logging.getLogger(__name__)

def summary_rank(summary):
    return (
        float(summary.get("success_rate", 0.0)),
        float(summary.get("avg_best_boxes_on_target", 0.0)),
        float(summary.get("avg_boxes_on_target", 0.0)),
        float(summary.get("avg_reward", float("-inf"))),
        -int(summary.get("one_step_dead_end_count", 0)),
    )

class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        best_model_path,
        eval_env_factory,
        eval_seed_base,
        eval_freq,
        n_eval_episodes,
        early_stop_patience_evals,
        early_stop_min_timesteps,
    ):
        super().__init__()
        self.best_model_path = best_model_path
        self.eval_env_factory = eval_env_factory
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.early_stop_patience_evals = early_stop_patience_evals
        self.early_stop_min_timesteps = early_stop_min_timesteps
        self.best_summary = None
        self.no_improvement_evals = 0
        self.last_eval_step = 0
        self.eval_episode_seeds = [eval_seed_base + i for i in range(n_eval_episodes)]

    def _init_callback(self):
        self.last_eval_step = int(self.num_timesteps)

    def _evaluate_current_model(self):
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
        summary = summarize_results(results)
        summary["timesteps"] = int(self.num_timesteps)
        return summary

    def _log_summary(self, summary):
        LOGGER.info(
            "Periodic eval: timesteps=%s success_rate=%.3f solved=%s/%s avg_best_boxes=%.2f avg_reward=%.2f one_step_dead_ends=%s",
            self.num_timesteps,
            summary["success_rate"],
            summary["solved_count"],
            summary["total_episodes"],
            summary["avg_best_boxes_on_target"],
            summary["avg_reward"],
            summary["one_step_dead_end_count"],
        )

    def _save_best_checkpoint(self, summary):
        self.best_summary = dict(summary)
        self.best_summary["model_saved_to"] = self.best_model_path + ".zip"
        self.model.save(self.best_model_path)
        self.no_improvement_evals = 0
        LOGGER.info("Saved new best checkpoint to %s.zip", self.best_model_path)

    def _should_stop_early(self):
        if self.num_timesteps < self.early_stop_min_timesteps:
            return False
        return self.no_improvement_evals >= self.early_stop_patience_evals

    def _on_step(self):
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True
        summary = self._evaluate_current_model()
        self._log_summary(summary)
        if self.best_summary is None or summary_rank(summary) > summary_rank(self.best_summary):
            self._save_best_checkpoint(summary)
        else:
            self.no_improvement_evals += 1
        self.last_eval_step = self.num_timesteps
        if self._should_stop_early():
            LOGGER.info(
                "Early stopping training after %s evaluations without improvement. Best success_rate=%.3f at timesteps=%s",
                self.no_improvement_evals,
                self.best_summary["success_rate"] if self.best_summary else 0.0,
                self.best_summary.get("timesteps") if self.best_summary else "unknown",
            )
            return False
        return True
