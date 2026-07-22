import os

import matplotlib.pyplot as plt
import numpy as np


def plot_rewards(
    rewards,
    save_path="results/rewards.png",
    window=50,
):
    """
    Plot episode rewards.

    Parameters
    ----------
    rewards : list
        Total reward per episode.

    save_path : str
        Location to save figure.

    window : int
        Moving average window.
    """

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True,
    )

    rewards = np.array(rewards)

    plt.figure(figsize=(10, 5))

    plt.plot(
        rewards,
        alpha=0.4,
        label="Episode reward",
    )

    if len(rewards) >= window:

        moving_average = np.convolve(
            rewards,
            np.ones(window) / window,
            mode="valid",
        )

        plt.plot(
            range(window - 1, len(rewards)),
            moving_average,
            label=f"{window}-episode average",
        )

    plt.xlabel("Episode")
    plt.ylabel("Reward")

    plt.title(
        "Training Reward"
    )

    plt.legend()

    plt.grid(True)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_metrics(
    metrics,
    save_dir="results",
):
    """
    Plot training metrics from trainer.

    Example metrics:

        {
          "actor_loss": [],
          "critic_loss": [],
          "advantage": [],
          "entropy": []
        }
    """

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    for name, values in metrics.items():

        plt.figure(figsize=(8, 4))

        plt.plot(values)

        plt.xlabel(
            "Training step"
        )

        plt.ylabel(
            name
        )

        plt.title(
            name.replace("_", " ").title()
        )

        plt.grid(True)

        plt.savefig(
            os.path.join(
                save_dir,
                f"{name}.png",
            ),
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


def compare_rewards(
    healthy_rewards,
    parkinson_rewards,
    save_path="results/comparison.png",
):
    """
    Compare healthy and Parkinsonian models.
    """

    plt.figure(figsize=(10, 5))

    plt.plot(
        healthy_rewards,
        label="Healthy dopamine",
    )

    plt.plot(
        parkinson_rewards,
        label="Parkinson dopamine",
    )

    plt.xlabel(
        "Episode"
    )

    plt.ylabel(
        "Reward"
    )

    plt.title(
        "Healthy vs Parkinsonian Learning"
    )

    plt.legend()

    plt.grid(True)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
