"""Masked DQN helpers for high-level Sokoban training and imitation pretraining."""

import numpy as np
import torch as th
from stable_baselines3 import DQN
from torch.nn import functional as F


class MaskedDQN(DQN):
    """
    Filter invalid macro-actions and support simple imitation pretraining.
    """

    def __init__(self, *args, action_mask_size=None, **kwargs):
        self.action_mask_size = action_mask_size
        self.phase_exploration_start_eps = None
        self.phase_exploration_end_eps = None
        self.phase_exploration_total_timesteps = 0
        self.phase_exploration_start_timestep = 0
        self.demo_guidance_samples = []
        self.demo_guidance_batch_size = 0
        self.demo_guidance_loss_weight = 0.0
        super().__init__(*args, **kwargs)


    def _excluded_save_params(self):
        """Keep large cached demo samples out of saved checkpoint files."""
        return super()._excluded_save_params() + ["demo_guidance_samples"]


    def _action_mask_size(self):
        """
        action mask is a list of 0s and 1s
        eg. if there are 12 macro-actions (3 boxes and 4 directions), then action
        mask has length 12. If its a valid action 1, else 0
        This method returns length of action mask 
        """
        return int(self.action_space.n)


    def _extract_action_masks(self, observation):
        """
        action mask is appended at the end of the observation 
        Observation is an array, action mask is stored as the several
        last elements of the array
        """
        obs = np.asarray(observation)
        obs = obs.reshape(1, -1) if obs.ndim == 1 else obs
        return obs[..., -self._action_mask_size() :]


    def _masked_q_values(self, q_values, observation_tensor):
        """
        Replace invalid action values with large negative Q-values
        """
        action_masks = observation_tensor[..., -self._action_mask_size() :] > 0.5 # 1 becomes True, 0 becomes False
        masked_q_values = q_values.clone()  # get original q-values
        masked_q_values[~action_masks] = -1e9   # if action mask is 0 (ie invalid action) then replace Q-value with negative value to avoid being selected for the next move 

        # if all actions are invalid, fall back to unmasked Q-values since replacing all with -1e9 wont be helpful
        empty_rows = action_masks.sum(dim=1) == 0
        if th.any(empty_rows):  # th is alias for PyTorch, each row is one observation. -- During training, DQN does not learn only from te current move. It stores past experiences in memory (replay buffer). Each stores current observation, reward, etc...During training it picks one of the many past experiences and learns from it
            masked_q_values[empty_rows] = q_values[empty_rows]
        return masked_q_values


    def _sample_masked_random_actions(self, masks):
        """
        Picks actions from valid actions allowed by the mask
        Used for exploration (During training it uses epsilon-greedy)
        """
        actions = []
        for mask in masks:
            valid_actions = np.flatnonzero(mask > 0.5)
            if len(valid_actions) == 0:
                valid_actions = np.arange(self.action_space.n)  # if there are no valid actions, fall back to all actions
            actions.append(int(np.random.choice(valid_actions)))
        return np.asarray(actions, dtype=np.int64)


    def _masked_greedy_actions(self, observation):
        """
        Chooses best valid action using Q-network
        Used for exploitation (choosing best known action)
        """
        self.policy.set_training_mode(False)    # set to evaluation mode to select actions
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation) # convert observation to PyTorch tensor
        with th.no_grad():
            masked_q_values = self._masked_q_values(self.q_net(obs_tensor), obs_tensor) # run Q-network on the observation and get Q-values for all actions
        return masked_q_values.argmax(dim=1).cpu().numpy(), vectorized_env  # choose best action


    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """
        Decides whether to use epxloration or exploitation
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            actions = self._sample_masked_random_actions(self._extract_action_masks(observation))
            return (actions[0] if np.asarray(observation).ndim == 1 else actions), state

        actions, vectorized_env = self._masked_greedy_actions(observation)
        return (actions if vectorized_env else actions[0]), state


    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        """
        First fill in replay buffer to have enough initial experiences 
        that it can use for learning. Then switch to normal predict logic 
        of epsilon-greedy behavior. 
        """
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            action = self._sample_masked_random_actions(self._extract_action_masks(self._last_obs))
        else:
            action, _ = self.predict(self._last_obs, deterministic=False)
        action = np.asarray(action)
        action = action.reshape(1) if action.ndim == 0 else action
        return action, action.copy()


    def start_phase_exploration(self, start_eps, phase_timesteps, end_eps=None, elapsed_phase_timesteps=0):
        """Start or resume one simple phase-local epsilon schedule for curriculum training."""
        self.phase_exploration_start_eps = float(start_eps)
        self.phase_exploration_end_eps = self.exploration_final_eps if end_eps is None else float(end_eps)
        self.phase_exploration_total_timesteps = max(int(phase_timesteps), 1)
        self.phase_exploration_start_timestep = int(self.num_timesteps) - self._clamped_phase_elapsed_timesteps(elapsed_phase_timesteps)
        self.exploration_rate = self._phase_exploration_rate()


    def _clamped_phase_elapsed_timesteps(self, elapsed_phase_timesteps):
        """Keep resumed phase progress inside the current phase schedule bounds."""
        elapsed_steps = max(int(elapsed_phase_timesteps), 0)
        return min(elapsed_steps, int(self.phase_exploration_total_timesteps))


    def _on_step(self) -> None:
        """Update DQN internals, then re-log the real phase-local epsilon value."""
        super()._on_step()
        self._apply_phase_exploration_rate()
        self._record_exploration_rate()


    def _apply_phase_exploration_rate(self):
        """Replace the default SB3 epsilon with the active phase-local schedule."""
        if self.phase_exploration_start_eps is None:
            return
        self.exploration_rate = self._phase_exploration_rate()


    def _record_exploration_rate(self):
        """Write the current epsilon so the training log shows the real value."""
        self.logger.record("rollout/exploration_rate", float(self.exploration_rate))


    def _phase_exploration_rate(self):
        """Linearly decay one phase-local epsilon schedule down to its target floor."""
        elapsed_steps = int(self.num_timesteps) - int(self.phase_exploration_start_timestep)
        progress = min(max(float(elapsed_steps) / float(self.phase_exploration_total_timesteps), 0.0), 1.0)
        start_eps = float(self.phase_exploration_start_eps)
        end_eps = float(self.phase_exploration_end_eps)
        return start_eps + (end_eps - start_eps) * progress


    def set_demo_guidance(self, samples, batch_size, loss_weight):
        """Keep expert samples active during RL so the model does not forget imitation."""
        if not samples or float(loss_weight) <= 0.0:
            self.clear_demo_guidance()
            return
        self.demo_guidance_samples = list(samples)
        self.demo_guidance_batch_size = max(int(batch_size), 1)
        self.demo_guidance_loss_weight = float(loss_weight)


    def clear_demo_guidance(self):
        """Turn off the extra imitation regularizer for later training updates."""
        self.demo_guidance_samples = []
        self.demo_guidance_batch_size = 0
        self.demo_guidance_loss_weight = 0.0


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Run DQN updates while mixing in a small expert loss when demos are enabled."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        td_losses, demo_losses, total_losses = self._training_loss_lists()
        for _ in range(gradient_steps):
            self._run_training_step(batch_size, td_losses, demo_losses, total_losses)
        self._record_training_metrics(gradient_steps, td_losses, demo_losses, total_losses)


    def _training_loss_lists(self):
        """Start the running TD, demo, and total loss trackers for one update call."""
        return [], [], []


    def _run_training_step(self, batch_size, td_losses, demo_losses, total_losses):
        """Run one replay update and optionally add one expert mini-batch loss."""
        td_loss = self._td_loss(batch_size)
        loss, demo_loss = self._combined_training_loss(td_loss)
        self._optimize_training_loss(loss)
        self._append_training_losses(td_losses, demo_losses, total_losses, td_loss, demo_loss, loss)


    def _td_loss(self, batch_size):
        """Compute one masked double-DQN TD loss from the replay buffer."""
        replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
        target_q_values = self._target_q_values(replay_data)
        current_q_values = self._current_q_values(replay_data)
        return F.smooth_l1_loss(current_q_values, target_q_values)


    def _target_q_values(self, replay_data):
        """Build one bootstrapped target using masked online and target Q-values."""
        with th.no_grad():
            next_online_q = self._masked_q_values(self.q_net(replay_data.next_observations), replay_data.next_observations)
            next_actions = next_online_q.argmax(dim=1, keepdim=True)
            next_target_q = self._masked_q_values(self.q_net_target(replay_data.next_observations), replay_data.next_observations)
        discounted_q = (1 - replay_data.dones) * self.gamma * th.gather(next_target_q, dim=1, index=next_actions)
        return replay_data.rewards + discounted_q


    def _current_q_values(self, replay_data):
        """Read the current Q-values for the actions stored in one replay batch."""
        current_q = self.q_net(replay_data.observations)
        return th.gather(current_q, dim=1, index=replay_data.actions.long())


    def _combined_training_loss(self, td_loss):
        """Add the optional expert regularizer on top of the TD loss."""
        demo_loss = self._demo_guidance_loss()
        if demo_loss is None:
            return td_loss, 0.0
        return td_loss + demo_loss * self.demo_guidance_loss_weight, float(demo_loss.item())


    def _demo_guidance_loss(self):
        """Return one imitation loss from cached demos, or None when guidance is off."""
        if not self._demo_guidance_enabled():
            return None
        observations, actions = self._demo_guidance_tensors()
        logits = self._masked_q_values(self.q_net(observations), observations)
        return F.cross_entropy(logits, actions)


    def _demo_guidance_enabled(self):
        """Check whether demo samples and a positive weight are available right now."""
        return bool(self.demo_guidance_samples) and self.demo_guidance_loss_weight > 0.0


    def _demo_guidance_tensors(self):
        """Convert one random demo mini-batch into tensors for the auxiliary loss."""
        batch_indices = self._demo_batch_indices()
        return self._pretrain_batch_tensors(self.demo_guidance_samples, batch_indices)


    def _demo_batch_indices(self):
        """Sample one random set of cached demo rows for the auxiliary loss."""
        batch_size = min(int(self.demo_guidance_batch_size), len(self.demo_guidance_samples))
        return np.random.randint(0, len(self.demo_guidance_samples), size=batch_size)


    def _optimize_training_loss(self, loss):
        """Apply one optimizer step to the combined TD plus imitation objective."""
        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()


    def _append_training_losses(self, td_losses, demo_losses, total_losses, td_loss, demo_loss, total_loss):
        """Store one step of TD, demo, and combined losses for later logging."""
        td_losses.append(float(td_loss.item()))
        demo_losses.append(float(demo_loss))
        total_losses.append(float(total_loss.item()))


    def _record_training_metrics(self, gradient_steps, td_losses, demo_losses, total_losses):
        """Write the mean TD, demo, and total losses to the SB3 logger."""
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/td_loss", float(np.mean(td_losses)))
        self.logger.record("train/demo_guidance_loss", float(np.mean(demo_losses)))
        self.logger.record("train/loss", float(np.mean(total_losses)))


    def behavior_clone_pretrain(self, samples, epochs=8, batch_size=64, learning_rate=1e-4):
        """Teach the Q-network to copy expert actions before RL begins."""
        self.policy.set_training_mode(True)
        self._set_pretrain_learning_rate(learning_rate)
        losses = []
        for _ in range(int(epochs)):
            losses.extend(self._run_pretrain_epoch(samples, batch_size))
        summary = {"epochs": int(epochs), "loss": float(np.mean(losses))}
        self._record_pretrain_metrics(summary)
        return summary


    def _record_pretrain_metrics(self, summary):
        """Record imitation metrics only when the SB3 logger already exists."""
        if "_logger" not in vars(self):
            return
        self.logger.record("pretrain/epochs", int(summary["epochs"]))
        self.logger.record("pretrain/loss", float(summary["loss"]))


    def _set_pretrain_learning_rate(self, learning_rate):
        """Update the optimizer so imitation pretraining uses the requested step size."""
        for group in self.policy.optimizer.param_groups:
            group["lr"] = float(learning_rate)


    def _run_pretrain_epoch(self, samples, batch_size):
        """Run one shuffled imitation epoch over the cached expert samples."""
        losses = []
        sample_indices = np.random.permutation(len(samples))
        for start_index in range(0, len(sample_indices), int(batch_size)):
            batch_indices = sample_indices[start_index : start_index + int(batch_size)]
            losses.append(self._run_pretrain_batch(samples, batch_indices))
        return losses


    def _run_pretrain_batch(self, samples, batch_indices):
        """Learn from one mini-batch of expert observations and actions."""
        observations, actions = self._pretrain_batch_tensors(samples, batch_indices)
        logits = self._masked_q_values(self.q_net(observations), observations)
        loss = F.cross_entropy(logits, actions)
        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()
        return float(loss.item())


    def _pretrain_batch_tensors(self, samples, batch_indices):
        """Convert one mini-batch of cached demos into PyTorch tensors."""
        observations = [samples[int(index)]["observation"] for index in batch_indices]
        actions = [samples[int(index)]["action"] for index in batch_indices]
        observation_tensor = th.as_tensor(np.asarray(observations), device=self.device, dtype=th.float32)
        action_tensor = th.as_tensor(np.asarray(actions), device=self.device, dtype=th.long)
        return observation_tensor, action_tensor
