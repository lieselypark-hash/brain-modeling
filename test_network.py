import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised when torch is missing
    torch = None

if torch is not None:
    from brain.networks import MLP


class MLPTests(unittest.TestCase):

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_network_output_shape(self):
        network = MLP(input_dim=25, output_dim=4)
        x = torch.randn(8, 25)
        y = network(x)

        self.assertEqual(y.shape, (8, 4))


if __name__ == "__main__":
    unittest.main()
