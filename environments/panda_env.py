import importlib.util
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised when numpy is unavailable
    np = None

try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - exercised when dependency is missing
    gym = None

try:
    import gymnasium_robotics  # noqa: F401
except ImportError:  # pragma: no cover - exercised when dependency is missing
    gymnasium_robotics = None


COSMOS_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "COSMOS Robotic Arm" / "roboticArmAnimation.py"


def _load_cosmos_module():
    if not COSMOS_SCRIPT_PATH.exists():
        return None

    spec = importlib.util.spec_from_file_location("cosmos_robotic_arm", COSMOS_SCRIPT_PATH)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        return None

    return module


class PandaEnvironment:
    """
    Wrapper around the Cosmos robotics-arm environment.

    When the robotics dependencies are available, this uses the Cosmos pick-and-place
    environment implemented in the robotics-arm script. Otherwise it falls back to a
    lightweight continuous control environment so the training stack can still run.
    """

    def __init__(
        self,
        env_name="FetchPickAndPlace-v4",
        seed=42,
        render=False,
    ):

        self.render = render
        self.seed = seed
        self.env_name = env_name
        self._cosmos_module = _load_cosmos_module()

        if np is None or gym is None or (gymnasium_robotics is None and self._cosmos_module is None):
            self._init_fallback_environment()
            return

        if self._cosmos_module is not None:
            if gymnasium_robotics is not None:
                gym.register_envs(gymnasium_robotics)
            render_mode = "human" if render else None

            self.env = gym.make(
                self._cosmos_module.ENV_ID,
                render_mode=render_mode,
                max_episode_steps=200,
            )
            self.env.reset(seed=seed)

            obs, _ = self.env.reset()
            obs = self._flatten_obs(obs)

            self.state_dim = obs.shape[0]
            self.action_dim = self.env.action_space.shape[0]
            self.action_low = np.asarray(self.env.action_space.low, dtype=np.float32)
            self.action_high = np.asarray(self.env.action_space.high, dtype=np.float32)
            self._fallback = False
            self._cosmos_policy = getattr(self._cosmos_module, "PickAndPlacePolicy", None)
            if self._cosmos_policy is not None:
                self._cosmos_policy_instance = self._cosmos_policy()
            else:
                self._cosmos_policy_instance = None
            return

        self._init_fallback_environment()

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
        Cosmos observations are dictionaries with observation/desired_goal fields.
        Convert them into one flat vector.
        """

        if isinstance(obs, dict):
            pieces = []
            for key in sorted(obs.keys()):
                value = obs[key]
                if isinstance(value, dict):
                    for nested_key in sorted(value.keys()):
                        pieces.append(np.asarray(value[nested_key]).flatten())
                else:
                    pieces.append(np.asarray(value).flatten())
            if pieces:
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
