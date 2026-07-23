try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised when numpy is unavailable
    np = None

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised when dependency is missing
    gym = None

try:
    import panda_gym  # noqa: F401
except ImportError:  # pragma: no cover - exercised when dependency is missing
    panda_gym = None


class PandaEnvironment:
    """
    Wrapper around Panda-Gym environments.

    When the optional robotics stack is available, this uses the real Panda-Gym
    environment. Otherwise it falls back to a lightweight continuous control
    environment with the same interface so the training stack can still run.
    """

    def __init__(
        self,
        env_name="PandaPickAndPlace-v3",
        seed=42,
        render=False,
    ):

        self.render = render
        self.seed = seed
        self.env_name = env_name

        if np is None or gym is None or panda_gym is None:
            self._init_fallback_environment()
            return

        render_mode = "human" if render else None

        self.env = gym.make(
            env_name,
            render_mode=render_mode,
        )

        self.env.reset(seed=seed)

        obs, _ = self.env.reset()
        obs = self._flatten_obs(obs)

        self.state_dim = obs.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
        self._fallback = False

    def _init_fallback_environment(self):
        self.state_dim = 6
        self.action_dim = 6
        self.action_low = np.full(self.action_dim, -1.0, dtype=np.float32)
        self.action_high = np.full(self.action_dim, 1.0, dtype=np.float32)
        self._fallback = True
        self._rng = np.random.default_rng(self.seed)
        self._step_count = 0
        self._target = np.array([0.35, -0.2, 0.1, 0.25, -0.1, 0.15], dtype=np.float32)
        self._state = np.zeros(self.state_dim, dtype=np.float32)
        self.env = None

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

        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        return arr

    ############################################################

    def reset(self):
        if self._fallback:
            self._step_count = 0
            self._state = np.clip(
                self._rng.standard_normal(self.state_dim) * 0.25,
                -1.0,
                1.0,
            ).astype(np.float32)
            return self._state.copy(), {}

        obs, info = self.env.reset()
        return self._flatten_obs(obs), info

    ############################################################

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        if self._fallback:
            action = np.clip(action, self.action_low, self.action_high)
            self._step_count += 1
            self._state = np.clip(
                self._state + 0.2 * action,
                -1.0,
                1.0,
            ).astype(np.float32)

            error = np.linalg.norm(self._state - self._target)
            reward = float(max(0.0, 1.0 - error / 2.0))
            done = bool(self._step_count >= 25 or error < 0.15)
            return self._state.copy(), reward, done, {}

        action = np.clip(action, self.action_low, self.action_high)

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
        if self._fallback:
            return self._rng.uniform(self.action_low, self.action_high, size=self.action_dim).astype(np.float32)

        return self.env.action_space.sample()

    ############################################################

    def close(self):
        if self.env is not None:
            self.env.close()

    ############################################################

    def render_frame(self):
        if self.env is None:
            return None
        return self.env.render()
