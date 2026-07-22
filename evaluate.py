import torch
import numpy as np
from tqdm import tqdm

from config import Config
from environments import PandaEnvironment
from brain import Actor, Critic


def evaluate():

    ############################################
    # Configuration
    ############################################

    config = Config()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    ############################################
    # Environment
    ############################################

    env = PandaEnvironment(
        env_name=config.env_name,
        seed=config.seed,
        render=True,
    )


    ############################################
    # Load brain
    ############################################

    actor = Actor(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dims=(
            config.hidden_dim,
            config.hidden_dim,
        ),
    ).to(device)


    critic = Critic(
        state_dim=env.state_dim,
        hidden_dims=(
            config.hidden_dim,
            config.hidden_dim,
        ),
    ).to(device)


    actor.load_state_dict(
        torch.load(
            "checkpoints/healthy_actor.pt",
            map_location=device,
        )
    )


    critic.load_state_dict(
        torch.load(
            "checkpoints/healthy_critic.pt",
            map_location=device,
        )
    )


    actor.eval()
    critic.eval()


    ############################################
    # Evaluation
    ############################################

    rewards = []

    successes = []


    for episode in tqdm(
        range(config.eval_episodes)
    ):

        state, _ = env.reset()

        episode_reward = 0


        for step in range(
            config.max_steps
        ):

            ################################
            # Deterministic action
            ################################

            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)


            with torch.no_grad():

                action_mean, _ = (
                    actor(state_tensor)
                )


            action = (
                action_mean
                .squeeze(0)
                .cpu()
                .numpy()
            )


            ################################
            # Environment step
            ################################

            next_state, reward, done, info = (
                env.step(action)
            )


            state = next_state

            episode_reward += reward


            if done:
                break


        rewards.append(
            episode_reward
        )


        # Panda-Gym environments usually store success
        # information in info["is_success"]
        if "is_success" in info:
            successes.append(
                info["is_success"]
            )


    ############################################
    # Results
    ############################################

    print("\nEvaluation complete")

    print(
        "Average reward:",
        np.mean(rewards)
    )

    print(
        "Reward std:",
        np.std(rewards)
    )


    if len(successes) > 0:

        print(
            "Success rate:",
            np.mean(successes)
        )


    env.close()



if __name__ == "__main__":
    evaluate()
