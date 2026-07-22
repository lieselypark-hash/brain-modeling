import torch
import torch.nn.functional as F
import torch.optim as optim

from .actor import Actor
from .critic import Critic
from .dopamine import DopamineSystem


class Trainer:

    def __init__(
        self,
        state_dim,
        action_dim,
        config,
        device="cpu",
    ):

        self.device = device

        self.actor = Actor(
            state_dim,
            action_dim,
            hidden_dims=(config.hidden_dim,
                         config.hidden_dim),
        ).to(device)

        self.critic = Critic(
            state_dim,
            hidden_dims=(config.hidden_dim,
                         config.hidden_dim),
        ).to(device)

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.actor_lr,
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.critic_lr,
        )

        self.dopamine = DopamineSystem(
            gamma=config.gamma
        )

    ###########################################################

    def select_action(
        self,
        state,
    ):

        return self.actor.act(state)

    ###########################################################

    def train_step(
        self,
        replay_buffer,
        batch_size,
    ):

        if len(replay_buffer) < batch_size:
            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = replay_buffer.sample(batch_size)

        ####################################################
        # Critic
        ####################################################

        values = self.critic(states)

        with torch.no_grad():

            next_values = self.critic(next_states)

            targets = self.dopamine.value_target(
                rewards,
                next_values,
                dones,
            )

        critic_loss = F.mse_loss(
            values,
            targets,
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        ####################################################
        # Actor
        ####################################################

        predicted_actions = self.actor(states)

        rpe = self.dopamine.compute_rpe(
            rewards,
            values.detach(),
            next_values.detach(),
            dones,
        )

        actor_loss = -(rpe.mean())

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        ####################################################

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "mean_rpe": rpe.mean().item(),
        }
