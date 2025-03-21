import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

import numpy as np


class DistractingObsWrapper(ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
        num_distractions=0,
        correlated=False,
        linear=True,
        switching=False,
        pure_noise=False,
    ):
        super().__init__(env)
        self.num_distractions = num_distractions
        self.linear = linear
        self.switching = switching
        self.pure_noise = pure_noise

        self.distractor_obs = np.zeros(num_distractions)

        low = np.concatenate(
            [env.observation_space.low, -np.ones(num_distractions) * np.inf]
        )
        high = np.concatenate(
            [env.observation_space.high, np.ones(num_distractions) * np.inf]
        )

        self.observation_space = Box(
            low=low, high=high, dtype=env.observation_space.dtype
        )

        self.distractor_state = np.random.normal(0, 0.1, size=(num_distractions,))
        self.linear_map = np.random.normal(
            0, 1.0, size=(num_distractions, high.shape[0])
        )
        if not correlated:
            self.linear_map[:, : env.observation_space.shape[0]] = 0.0

        self.random_sin_parameters = (
            np.random.normal(0, 10.0, size=(1, num_distractions, num_distractions)) ** 2
        )
        self.reset_switching = np.random.normal(size=(num_distractions,))

    def step(
        self, action
    ):  # -> tuple[NDArray[float64], SupportsFloat, Any, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_distractor(obs, action)
        return self.observation(obs), reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.distractor_obs = np.random.normal(0, 0.1, size=(self.num_distractions,))
        return self.observation(obs), info

    def _step_distractor(self, obs, action):
        self.distractor_obs = np.matmul(
            self.linear_map, np.concatenate((obs, self.distractor_obs))
        )

        if not self.linear:
            self.distractor_obs += 0.1 * np.sum(
                np.sin(np.matmul(self.random_sin_parameters, self.distractor_obs)), 0
            )

        if self.switching:
            self.distractor_obs = np.where(
                np.abs(self.distractor_obs) > 20.0,
                self.reset_switching,
                self.distractor_obs,
            )

        if self.pure_noise:
            self.distractor_obs = np.random.normal(0, 1.0, size=(self.dimensions,))

    def observation(self, obs):
        return np.concatenate([obs, self.distractor_obs], axis=-1)
