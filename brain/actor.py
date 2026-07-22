import torch
import torch.nn as nn
from torch.distributions import Normal

from .networks import MLP


class Actor(nn.Module):
    """
    State-dependent Gaussian policy for A2C.

    Outputs:

        μ(s)     -> action mean
        σ(s)     -> action uncertainty

    Actions are sampled:

        a ~ Normal(μ(s), σ(s))

    The environment is responsible for clipping actions.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dims=(256, 256),
    ):
        super().__init__()

        self.action_dim = action_dim

        # Shared feature extractor
        self.feature_network = MLP(
            input_dim=state_dim,
            output_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
        )

        # Mean head
        self.mean_head = nn.Linear(
            hidden_dims[-1],
            action_dim,
        )

        # Log standard deviation head
        self.log_std_head = nn.Linear(
            hidden_dims[-1],
            action_dim,
        )

    ###########################################################

    def forward(self, state):

        features = self.feature_network(state)

        mean = self.mean_head(features)

        log_std = self.log_std_head(features)

        # Prevent exploding variance
        log_std = torch.clamp(
            log_std,
            min=-5,
            max=2,
        )

        std = torch.exp(log_std)

        return mean, std

    ###########################################################

    def get_distribution(self, state):

        mean, std = self.forward(state)

        return Normal(
            mean,
            std,
        )

    ###########################################################

    def sample_action(self, state):
        """
        Samples an action.

        Returns:
            action
            log probability
            entropy
        """

        distribution = self.get_distribution(state)

        action = distribution.rsample()

        log_prob = distribution.log_prob(
            action
        ).sum(dim=-1)

        entropy = distribution.entropy().sum(dim=-1)

        return (
            action,
            log_prob,
            entropy,
        )

    ###########################################################

    @torch.no_grad()
    def act(self, state):
        """
        Used during evaluation.

        Returns the mean action rather than
        a stochastic sample.
        """

        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)

        if state.dim() == 1:
            state = state.unsqueeze(0)

        mean, _ = self.forward(state)

        return (
            mean.squeeze(0)
            .cpu()
            .numpy()
        )
