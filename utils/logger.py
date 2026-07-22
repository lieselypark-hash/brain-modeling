import os
import json
import numpy as np


class Logger:
    """
    Stores training statistics.

    Used for:
        - reward curves
        - A2C losses
        - dopamine/RPE analysis
    """

    def __init__(
        self,
        save_dir="logs",
    ):

        self.save_dir = save_dir

        os.makedirs(
            save_dir,
            exist_ok=True,
        )

        self.data = {
            "episode_rewards": [],
            "actor_loss": [],
            "critic_loss": [],
            "advantage": [],
            "value": [],
            "entropy": [],
        }


    #################################################

    def log_episode(
        self,
        reward,
    ):

        self.data[
            "episode_rewards"
        ].append(reward)


    #################################################

    def log_training(
        self,
        metrics,
    ):

        if metrics is None:
            return

        for key in metrics:

            if key in self.data:

                self.data[key].append(
                    metrics[key]
                )


    #################################################

    def save(
        self,
        filename="training_log.json",
    ):

        path = os.path.join(
            self.save_dir,
            filename,
        )

        with open(
            path,
            "w",
        ) as f:

            json.dump(
                self.data,
                f,
                indent=4,
            )


    #################################################

    def load(
        self,
        filename="training_log.json",
    ):

        path = os.path.join(
            self.save_dir,
            filename,
        )

        with open(
            path,
            "r",
        ) as f:

            self.data = json.load(f)

        return self.data
