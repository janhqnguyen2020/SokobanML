# src/rl/network.py
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class HighLevelBoardExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor
    Takes the flat observation vector from the HighLevelEnv and combine
    with CNN

    Notes
    HighLevelSokobanEnv creates observation and builds one flat vector 
    MaskedDQN receives that observation during trainig or prediction -- uses MLPpolicy
    HighLevelBoardExtractor combines the obseration vector and combines with CNN (used inside MlPPolicy)
    
    Flow
    HighLevelEnv observation
        -> flat vector
        -> HighLevelBoardExtractor for feature extraction
        -> CNN on board + concat scalar/mask
        -> MlpPolicy / Q-network
        -> Q-values
        -> MaskedDQN masks invalid actions
        -> choose action
        -> env step
        -> store transition
        -> train from replay buffer
    """

    def __init__(
        self,
        observation_space,
        board_shape,
        action_mask_size,
        scalar_feature_size=0,
        features_dim=256,
    ):
        self.board_shape = tuple(board_shape)
        self.action_mask_size = int(action_mask_size)
        self.scalar_feature_size = int(scalar_feature_size)
        self.num_board_channels = 4
        self.board_feature_size = int(np.prod(self.board_shape) * self.num_board_channels)
        expected_obs_size = self.board_feature_size + self.scalar_feature_size + self.action_mask_size
        actual_obs_size = int(observation_space.shape[0])
        if expected_obs_size != actual_obs_size:
            raise ValueError(
                f"HighLevelBoardExtractor expected observation size {expected_obs_size}, "
                f"got {actual_obs_size}"
            )

        super().__init__(observation_space, features_dim)
        self.board_encoder = nn.Sequential(
            nn.Conv2d(self.num_board_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_board = th.zeros((1, self.num_board_channels, *self.board_shape), dtype=th.float32)
            encoded_board_size = int(self.board_encoder(sample_board).shape[1])
        head_input_size = encoded_board_size + self.scalar_feature_size + self.action_mask_size
        self.projection = nn.Sequential(
            nn.Linear(head_input_size, features_dim),
            nn.ReLU(),
        )
        self._features_dim = int(features_dim)

    def forward(self, observations):
        board_flat = observations[:, : self.board_feature_size]
        board = board_flat.view(-1, self.num_board_channels, *self.board_shape)
        scalar_start = self.board_feature_size
        scalar_end = scalar_start + self.scalar_feature_size
        scalar_features = observations[:, scalar_start:scalar_end]
        action_mask = observations[:, scalar_end:]
        encoded_board = self.board_encoder(board)
        combined = th.cat([encoded_board, scalar_features, action_mask], dim=1)
        return self.projection(combined)
