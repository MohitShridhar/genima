import numpy as np
from typing import Dict

import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, WrapperActType

import os
import json


class JointNormalization(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Normalize action."""

    def __init__(
        self,
        env: gym.Env,
        action_stats: Dict | None = None,
        action_stats_path: str = None,
    ):
        """Init.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)
        action_space = env.action_space
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape=action_space.shape, dtype=action_space.dtype
        )
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.env = env
        self.action_stats = action_stats
        self.action_stats_path = action_stats_path
        assert (
            self.action_stats is not None or self.action_stats_path is not None
        ), print("either provide action stats dictionary or provide a path to it")

        self.init_stats()

    def init_stats(self):
        if self.action_stats is not None:
            if not os.path.exists(self.action_stats_path):
                os.makedirs(self.action_stats_path)

            stats = {
                "mean": self.action_stats["mean"].tolist(),
                "std": self.action_stats["std"].tolist(),
            }
            stats_json = os.path.join(self.action_stats_path, "action_stats.json")
            with open(stats_json, "w") as f:
                json.dump(stats, f)
            low_dim_state_mean, low_dim_state_std = (
                self.action_stats["mean"],
                self.action_stats["std"],
            )
        elif self.action_stats_path is not None:
            stats_json = os.path.join(self.action_stats_path, "action_stats.json")
            print(f"Loading stats from {stats_json}")
            with open(stats_json, "r") as f:
                stats = json.load(f)
            low_dim_state_mean = np.array(stats["mean"])
            low_dim_state_std = np.array(stats["std"])

        print(
            f"Action mean: {low_dim_state_mean}\nAction std: {low_dim_state_std}\n"
            + "Saved to {self.action_stats_path}/action_stats.json"
        )

        self.low_dim_obs_mean, self.low_dim_obs_std = (
            low_dim_state_mean,
            low_dim_state_std,
        )

    @staticmethod
    def transform_from_norm(action, demo_action_mean, demo_action_std):
        action[:-1] = (action[:-1] * demo_action_std[:-1]) + demo_action_mean[:-1]
        return action

    @staticmethod
    def transform_to_norm(action, demo_action_mean, demo_action_std):
        action[:-1] = (action[:-1] - demo_action_mean[:-1]) / demo_action_std[:-1]
        return action

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions
        """
        return JointNormalization.transform_from_norm(
            action, self.low_dim_obs_mean, self.low_dim_obs_std
        )
