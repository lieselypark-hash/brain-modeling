import os
import torch
from tqdm import tqdm

from config import Config
from environments import PandaEnvironment
from brain import Trainer
from utils.seed import set_seed
from utils.logger import Logger
from utils.plot import plot_rewards


def train():

    ############################################
    # Configuration
    ############################################

    config = Config()

    set_seed(config.seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Using device:", device)


    ############################################
    # Environment
    ############################################

    env = PandaEnvironment(
        env_name=config.env_name,
        seed=config.seed,
        render=False,
    )

    print(
        "State dimension:",
        env.state_dim
    )

    print(
        "Action dimension:",
        env.action_dim
    )


    ############################################
    # Brain
    ############################################

    trainer = Trainer(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        config=config,
        device=device,
    )


    ############################################
    # Logger
    ############################################

    logger = Logger(
        save_dir="logs/healthy"
    )


    ############################################
    # Training
    ############################################

    rewards_history = []


    for episode in tqdm(
        range(config.episodes)
    ):

        state, _ = env.reset()

        episode_reward = 0

        metrics = None


        for step in range(
            config.max_steps
        ):

            ################################
            # Brain chooses action
            ################################

            action, log_prob, entropy = (
                trainer.select_action(state)
            )


            ################################
            # Environment transition
            ################################

            next_state, reward, done, info = (
                env.step(action)
            )


            ################################
            # Brain learns
            ################################

            metrics = trainer.train_step(
                state=state,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                entropy=entropy,
            )


            ################################
            # Log training metrics
            ################################

            logger.log_training(
                metrics
            )


            ################################
            # Update state
            ################################

            state = next_state

            episode_reward += reward


            if done:
                break


        ####################################
        # End of episode logging
        ####################################

        rewards_history.append(
            episode_reward
        )

        logger.log_episode(
            episode_reward
        )


        if episode % 10 == 0:

            print(
                f"\nEpisode {episode}"
            )

            print(
                "Reward:",
                episode_reward
            )

            if metrics:
                print(metrics)


    ############################################
    # Save model
    ############################################

    os.makedirs(
        config.checkpoint_path,
        exist_ok=True
    )


    torch.save(
        trainer.actor.state_dict(),
        os.path.join(
            config.checkpoint_path,
            "healthy_actor.pt"
        ),
    )


    torch.save(
        trainer.critic.state_dict(),
        os.path.join(
            config.checkpoint_path,
            "healthy_critic.pt"
        ),
    )


    ############################################
    # Save logs
    ############################################

    logger.save(
        "healthy_training.json"
    )


    ############################################
    # Plot results
    ############################################

    plot_rewards(
        rewards_history,
        save_path="results/healthy_rewards.png",
    )


    ############################################

    env.close()

    print("Training complete.")


if __name__ == "__main__":
    train()
