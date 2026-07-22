import torch


class DopamineSystem:
    """
    Healthy dopamine system.

    Computes the Reward Prediction Error (RPE):

        δ = r + γV(s') − V(s)

    This δ is used as the teaching signal for learning.
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma

    def compute_rpe(
        self,
        reward,
        current_value,
        next_value,
        done,
    ):
        """
        Parameters
        ----------
        reward : torch.Tensor
            Immediate reward.

        current_value : torch.Tensor
            V(s)

        next_value : torch.Tensor
            V(s')

        done : torch.Tensor
            Episode termination mask.
            1 = episode ended
            0 = continue
        """

        # No future reward after terminal state
        target = reward + self.gamma * next_value * (1.0 - done)

        rpe = target - current_value

        return rpe

    def value_target(
        self,
        reward,
        next_value,
        done,
    ):
        """
        Critic learning target.
        """

        return reward + self.gamma * next_value * (1.0 - done)
