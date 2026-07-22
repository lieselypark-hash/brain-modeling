import torch

from brain.actor import Actor
from brain.critic import Critic


STATE_DIM = 25
ACTION_DIM = 4

actor = Actor(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM
)

critic = Critic(
    state_dim=STATE_DIM
)

state = torch.randn(8, STATE_DIM)

actions = actor(state)
values = critic(state)

print("State shape :", state.shape)
print("Action shape:", actions.shape)
print("Value shape :", values.shape)
