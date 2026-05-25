# src/rl/masked_dqn.py
import numpy as np
import torch as th
from stable_baselines3 import DQN
from torch.nn import functional as F


class MaskedDQN(DQN):
    """
    Filter invalid macro-actions based on the action masks produced by high_level_env
    """

    def __init__(self, *args, action_mask_size=None, **kwargs):
        self.action_mask_size = action_mask_size
        super().__init__(*args, **kwargs)


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


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                next_online_q = self._masked_q_values(self.q_net(replay_data.next_observations), replay_data.next_observations)
                next_actions = next_online_q.argmax(dim=1, keepdim=True)
                next_target_q = self._masked_q_values(self.q_net_target(replay_data.next_observations), replay_data.next_observations)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * th.gather(next_target_q, dim=1, index=next_actions)
            current_q_values = th.gather(self.q_net(replay_data.observations), dim=1, index=replay_data.actions.long())
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
