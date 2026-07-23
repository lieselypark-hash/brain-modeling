import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is missing
    torch = None

if torch is not None:
    from brain.actor import Actor
    from brain.critic import Critic


class ActorCriticTests(unittest.TestCase):

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_network_output_shapes(self):
        state_dim = 25
        action_dim = 4

        actor = Actor(state_dim=state_dim, action_dim=action_dim)
        critic = Critic(state_dim=state_dim)

        state = torch.randn(8, state_dim)
        mean, std = actor(state)
        values = critic(state)

        self.assertEqual(mean.shape, (8, action_dim))
        self.assertEqual(std.shape, (8, action_dim))
        self.assertEqual(values.shape, (8, 1))


if __name__ == "__main__":
    unittest.main()
