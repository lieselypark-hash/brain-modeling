import torch
import torch.nn.functional as F
import torch.optim as optim

from .actor import Actor
from .critic import Critic
from .dopamine_system import DopamineSystem


class Trainer:
    """
    Trainer for a continuous-action A2C agent.

    Coordinates:
        - Actor
        - Critic
        - Dopamine (advantage/RPE)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        config,
        device="cpu",
    ):

        self.device = torch.device(device)

        #########################################
        # Networks
        #########################################

        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=(
                config.hidden_dim,
                config.hidden_dim,
            ),
        ).to(self.device)

        self.critic = Critic(
            state_dim=state_dim,
            hidden_dims=(
                config.hidden_dim,
                config.hidden_dim,
            ),
        ).to(self.device)

        #########################################
        # Optimizers
        #########################################

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
        )

        #########################################
        # Dopamine system
        #########################################

        self.dopamine = DopamineSystem(
            gamma=config.gamma
        )

        #########################################

        self.device = device

    ###########################################################

    def select_action(self, state):
    """
    Samples an action from the policy.

    Returns:
        action for environment
        log probability for learning
        entropy for exploration
    """

    state = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=self.device,
    ).unsqueeze(0)

    # Sample from Gaussian policy
    action, log_prob, entropy = self.actor.sample_action(state)

    # Clip ONLY the environment action
    action = torch.clamp(
        action,
        min=-1.0,
        max=1.0,
    )

    return (
        action.squeeze(0).detach().cpu().numpy(),
        log_prob.squeeze(0),
        entropy.squeeze(0),
    )

    ###########################################################

    def train_step(
        self,
        state,
        reward,
        next_state,
        done,
        log_prob,
        entropy,
    ):

        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        next_state = torch.as_tensor(
            next_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        reward = torch.tensor(
            [[reward]],
            dtype=torch.float32,
            device=self.device,
        )

        done = torch.tensor(
            [[float(done)]],
            dtype=torch.float32,
            device=self.device,
        )

        ###################################################
        # Critic forward
        ###################################################

        value = self.critic(state)

        with torch.no_grad():
            next_value = self.critic(next_state)

        ###################################################
        # Dopamine / Advantage
        ###################################################

        advantage = self.dopamine.compute_rpe(
            reward=reward,
            current_value=value,
            next_value=next_value,
            done=done,
        )

        ###################################################
        # Critic update
        ###################################################

        critic_loss = advantage.pow(2).mean()

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        ###################################################
        # Actor update
        ###################################################

        actor_loss = -(
            log_prob * advantage.detach()
        ).mean()

        entropy_bonus = 0.001 * entropy.mean()

        total_actor_loss = actor_loss - entropy_bonus

        self.actor_optimizer.zero_grad()

        total_actor_loss.backward()

        self.actor_optimizer.step()

        ###################################################

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "advantage": advantage.mean().item(),
            "value": value.mean().item(),
            "entropy": entropy.mean().item(),
        }
