import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """
    Experience Replay Buffer.

    Stores transitions:

        (state,
         action,
         reward,
         next_state,
         done)
    """

    def __init__(
        self,
        capacity=100000,
        device="cpu"
    ):

        self.buffer = deque(maxlen=capacity)

        self.device = device

    ############################################################

    def __len__(self):

        return len(self.buffer)

    ############################################################

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    ############################################################

    def sample(
        self,
        batch_size,
    ):

        batch = random.sample(
            self.buffer,
            batch_size,
        )

        state, action, reward, next_state, done = zip(*batch)

        state = torch.FloatTensor(
            np.array(state)
        ).to(self.device)

        action = torch.FloatTensor(
            np.array(action)
        ).to(self.device)

        reward = torch.FloatTensor(
            np.array(reward)
        ).unsqueeze(1).to(self.device)

        next_state = torch.FloatTensor(
            np.array(next_state)
        ).to(self.device)

        done = torch.FloatTensor(
            np.array(done)
        ).unsqueeze(1).to(self.device)

        return (
            state,
            action,
            reward,
            next_state,
            done,
        )
