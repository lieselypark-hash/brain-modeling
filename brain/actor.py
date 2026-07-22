import torch
import torch.nn as nn
from torch.distributions import Normal

from .networks import MLP


class Actor(nn.Module):
    """
    Gaussian policy network for continuous-action A2C.

    Outputs:
        mean (μ)
        standard deviation (σ)

    Actions are sampled from:

        a ~ N(μ, σ)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(256, 256),
    ):
        super().__init__()

        self.mean_network = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_dims=hidden_dims,
        )

        # Learn one log standard deviation per action dimension
        self.log_std = nn.Parameter(
            torch.zeros(action_dim)
        )

    ###########################################################

    def forward(self, state):

        mean = torch.tanh(
            self.mean_network(state)
        )

        std = torch.exp(self.log_std)

        return mean, std

    ###########################################################

    def get_distribution(self, state):

        mean, std = self.forward(state)

        return Normal(mean, std)

    ###########################################################

    def sample_action(self, state):

        dist = self.get_distribution(state)

        action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)

        action = torch.tanh(action)

        return action, log_prob, entropy

    ###########################################################

    @torch.no_grad()
    def act(self, state):

        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        action, _, _ = self.sample_action(state)

        return action.squeeze(0).cpu().numpy()
