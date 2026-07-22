import gymnasium as gym
import panda_gym
import numpy as np


class PandaEnvironment:
    """
    Wrapper around Panda-Gym environments.

    Handles:
    - environment creation
    - reset
    - stepping
    - observation flattening
    - action clipping
    """

    def __init__(
        self,
        env_name="PandaPickAndPlace-v3",
        seed=42,
        render=False,
    ):

        self.render = render

        render_mode = "human" if render else None

        self.env = gym.make(
            env_name,
            render_mode=render_mode
        )

        self.seed = seed

        self.env.reset(seed=seed)

        # Determine observation size

        obs, _ = self.env.reset()

        obs = self._flatten_obs(obs)

        self.state_dim = obs.shape[0]

        # Continuous action space

        self.action_dim = self.env.action_space.shape[0]

        self.action_low = self.env.action_space.low

        self.action_high = self.env.action_space.high

    ############################################################

    def _flatten_obs(self, obs):
        """
        PandaGym observations are dictionaries.

        Convert them into one flat vector.
        """

        if isinstance(obs, dict):

            pieces = []

            for key in sorted(obs.keys()):
                pieces.append(np.asarray(obs[key]).flatten())

            return np.concatenate(pieces).astype(np.float32)

        return np.asarray(obs, dtype=np.float32)

    ############################################################

    def reset(self):

        obs, info = self.env.reset()

        return self._flatten_obs(obs), info

    ############################################################

    def step(self, action):

        action = np.clip(
            action,
            self.action_low,
            self.action_high
        )

        obs, reward, terminated, truncated, info = self.env.step(action)

        done = terminated or truncated

        return (
            self._flatten_obs(obs),
            float(reward),
            done,
            info,
        )

    ############################################################

    def sample_action(self):

        return self.env.action_space.sample()

    ############################################################

    def close(self):

        self.env.close()

    ############################################################

    def render_frame(self):

        return self.env.render()
