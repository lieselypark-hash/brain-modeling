import torch
import torch.nn as nn

from .networks import MLP


class Critic(nn.Module):
    """
    Critic network.

    Maps:
        state -> value estimate V(s)

    Output is a single scalar representing the expected
    future discounted reward.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims=(256, 256),
    ):
        super().__init__()

        self.network = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
        )

    def forward(self, state):

        value = self.network(state)

        return value
