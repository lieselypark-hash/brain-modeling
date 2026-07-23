try:
    from .panda_env import PandaEnvironment
except ImportError:  # pragma: no cover - exercised when optional deps are missing
    PandaEnvironment = None

__all__ = ["PandaEnvironment"]
