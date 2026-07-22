import torch
import torch.nn as nn

from .networks import MLP


class Actor(nn.Module):
    """
    Actor network.

    Maps:
        state -> action

    Output actions are squashed to [-1, 1] using tanh.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims=(256, 256),
    ):
        super().__init__()

        self.network = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
        )

        self.output_activation = nn.Tanh()

    def forward(self, state):

        action = self.network(state)

        action = self.output_activation(action)

        return action

    @torch.no_grad()
    def act(self, state):
        """
        Returns a NumPy action for interacting with the environment.
        """

        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        action = self.forward(state)

        return action.squeeze(0).cpu().numpy()
