from dataclasses import dataclass

@dataclass
class Config:

    ########################
    # Environment
    ########################

    env_name = "PandaPickAndPlace-v3"

    ########################
    # Training
    ########################

    episodes = 1000
    max_steps = 250

    gamma = 0.99

    actor_lr = 3e-4
    critic_lr = 3e-4

    batch_size = 256

    replay_size = 100000

    tau = 0.005

    ########################
    # Networks
    ########################

    hidden_dim = 256

    ########################
    # Exploration
    ########################

    exploration_noise = 0.1

    ########################
    # Evaluation
    ########################

    eval_episodes = 10

    ########################
    # Saving
    ########################

    checkpoint_path = "checkpoints"

    ########################
    # Randomness
    ########################

    seed = 42
