import unittest

from environments import PandaEnvironment


class PandaEnvironmentTests(unittest.TestCase):

    def test_environment_shapes_and_reset(self):
        env = PandaEnvironment(render=False)

        try:
            state, info = env.reset()

            self.assertEqual(state.ndim, 1)
            self.assertGreater(env.state_dim, 0)
            self.assertGreater(env.action_dim, 0)
            self.assertEqual(state.shape[0], env.state_dim)

            action = env.sample_action()
            self.assertEqual(action.shape, (env.action_dim,))

            next_state, reward, done, info = env.step(action)
            self.assertEqual(next_state.shape[0], env.state_dim)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
