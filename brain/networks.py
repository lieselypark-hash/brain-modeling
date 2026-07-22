import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Generic Multi-Layer Perceptron.

    Used by:
        - Actor
        - Critic

    Parameters
    ----------
    input_dim : int
        Size of the input vector.

    output_dim : int
        Size of the output vector.

    hidden_dims : tuple[int]
        Hidden layer sizes.

    activation : nn.Module
        Activation function used between layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims=(256, 256),
        activation=nn.ReLU,
    ):
        super().__init__()

        layers = []

        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Xavier initialization.

        Good default for reinforcement learning.
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(module.weight)

                nn.init.zeros_(module.bias)

    def forward(self, x):

        return self.network(x)
