"""Todo."""

from typing import Dict
import numpy as np
import os
import json
import gymnasium as gym
from gymnasium import spaces


class NormProprioFromStats(gym.ObservationWrapper, gym.utils.RecordConstructorArgs):
    """Rescale observation to desired range based on provided statistics."""

    def __init__(
        self,
        env: gym.Env,
        proprio_stats: Dict | None = None,
        proprio_stats_path: str = None,
    ):
        """Init.

        Args:
            env: The environment to apply the wrapper
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ObservationWrapper.__init__(self, env)
        action_space = env.action_space
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape=action_space.shape, dtype=action_space.dtype
        )
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.env = env
        self.proprio_stats = proprio_stats
        self.proprio_stats_path = proprio_stats_path
        assert (
            self.proprio_stats is not None or self.proprio_stats_path is not None
        ), print("either provide action stats dictionary or provide a path to it")

        self.init_stats_proprio()

    def init_stats_proprio(self):
        if self.proprio_stats is not None:
            if not os.path.exists(self.proprio_stats_path):
                os.makedirs(self.proprio_stats_path)

            stats = {
                "mean": self.proprio_stats["mean"].tolist(),
                "std": self.proprio_stats["std"].tolist(),
            }
            stats_json = os.path.join(self.proprio_stats_path, "proprio_stats.json")
            with open(stats_json, "w") as f:
                json.dump(stats, f)
            low_dim_state_mean, low_dim_state_std = (
                self.proprio_stats["mean"],
                self.proprio_stats["std"],
            )
        elif self.proprio_stats_path is not None:
            stats_json = os.path.join(self.proprio_stats_path, "proprio_stats.json")
            print(f"Loading stats from {stats_json}")
            with open(stats_json, "r") as f:
                stats = json.load(f)
            low_dim_state_mean = np.array(stats["mean"])
            low_dim_state_std = np.array(stats["std"])

        print(
            f"Proprio mean: {low_dim_state_mean}\nProprio std: {low_dim_state_std}\n"
            + "Saved to {self.proprio_stats_path}/proprio_stats.json"
        )

        self.low_dim_obs_mean, self.low_dim_obs_std = (
            low_dim_state_mean,
            low_dim_state_std,
        )

    @staticmethod
    def transform_to_norm(action, low_dim_state_mean, low_dim_state_std):
        epsilon = 1e-10
        action[1:] = (action[1:] - low_dim_state_mean[1:]) / (
            low_dim_state_std[1:] + epsilon
        )
        return action

    def observation(self, observation):
        """Returns a modified action before :meth:`env.step` is called.

        Args:
            action: The original :meth:`step` actions
        """
        observation["low_dim_state"] = self.transform_to_norm(
            observation["low_dim_state"], self.low_dim_obs_mean, self.low_dim_obs_std
        )
        return observation
