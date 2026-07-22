import torch

from brain.networks import MLP


network = MLP(
    input_dim=25,
    output_dim=4
)

x = torch.randn(8, 25)

y = network(x)

print(y.shape)
