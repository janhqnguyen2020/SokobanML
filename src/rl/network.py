# src/rl/network.py
# Member B — Shizuka Takao
#
# PyTorch neural network architectures for RL agents.
#
# Defines the policy network (PPO) and Q-network (DQN) used to map
# Sokoban board states to action probabilities or Q-values.
#
# Two architecture options:
#
#   1. CNN-based (recommended):
#      Input: (C, H, W) one-hot board tensor
#      Conv layers extract spatial features (box positions, walls, goals)
#      Flatten → fully connected → output
#
#   2. MLP-based (simpler baseline):
#      Input: flattened board vector
#      Multiple linear layers → output
#      Faster to train, less expressive
#
# Classes to implement:
#
#   class SokobanCNN(nn.Module):
#       __init__(self, obs_shape, n_actions)
#           → conv layers + flatten + linear head
#       forward(self, x) → action logits or Q-values
#
#   class PolicyNetwork(nn.Module):  (for PPO)
#       → actor head: outputs action probability distribution
#       → critic head: outputs state value V(s)
#       → shared CNN backbone
#
#   class QNetwork(nn.Module):  (for DQN)
#       → outputs Q(s, a) for each action
#       → single CNN backbone + linear output
#
# Notes:
#   - Use nn.Conv2d, nn.ReLU, nn.Flatten, nn.Linear
#   - obs_shape from env.observation_space.shape
#   - n_actions = 4 (up, down, left, right)
#   - SB3 supports custom policies via policy_kwargs={"features_extractor_class": ...}
#   - If using SB3, extend BaseFeatureExtractor instead of plain nn.Module
