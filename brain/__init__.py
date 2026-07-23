try:
    from .actor import Actor
    from .critic import Critic
    from .dopamine_system import DopamineSystem
    from .trainer import Trainer
except ImportError:  # pragma: no cover - exercised when torch is unavailable
    Actor = None
    Critic = None
    DopamineSystem = None
    Trainer = None

__all__ = [
    "Actor",
    "Critic",
    "DopamineSystem",
    "Trainer",
]
