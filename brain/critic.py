import torch
import torch.nn as nn

from .networks import MLP


class Critic(nn.Module):
    """
    Value network for A2C.

    Estimates the value function:

        V(s)

    which represents the expected discounted future reward
    from the current state.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims=(256, 256),
    ):
        super().__init__()

        self.value_network = MLP(
            input_dim=state_dim,
            output_dim=1,
            hidden_dims=hidden_dims,
        )

    ###########################################################

    def forward(self, state):

        return self.value_network(state)

    ###########################################################

    @torch.no_grad()
    def predict(self, state):
        """
        Returns the value estimate for a single state.
        """

        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        value = self.forward(state)

        return value.squeeze(0).cpu().numpy()
