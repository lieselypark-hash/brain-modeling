"""
Robotic arm pick-and-place demo using Gymnasium-Robotics' built-in Fetch arm,
simulated in MuJoCo.

Environment: FetchPickAndPlace-v4 (Gymnasium-Robotics)
  - A 7-DoF Fetch robotic arm with a two-finger gripper.
  - Task: pick up a block from a table and place it at a target location.
  - Physics/rendering: MuJoCo.

The environment only defines the task (physics, observations, reward/goal).
It does NOT ship a controller that solves the task, so this file adds a
small scripted "pick and place" policy: a simple finite-state controller
that reads the object/goal positions out of the observation and drives the
gripper through align -> descend -> grasp -> lift -> transport -> release.
"""

import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

ENV_ID = "FetchPickAndPlace-v4"


class PickAndPlacePolicy:
    """Very basic scripted controller for the Fetch pick-and-place task.

    Reads positions out of the 25-dim observation vector:
      obs[0:3]   grip_pos          (end-effector position)
      obs[3:6]   object_pos
      obs[9:11]  gripper_state     (finger joint positions; ~0 closed, ~0.05 open)
    and drives the 4-dim action [dx, dy, dz, gripper]:
      dx, dy, dz in [-1, 1]   end-effector velocity command
      gripper in [-1, 1]      -1 close, +1 open
    """

    ABOVE_OFFSET = 0.05      # height to hover above object/goal before descending
    XY_TOLERANCE = 0.02
    Z_TOLERANCE = 0.01
    GRASP_TOLERANCE = 0.015

    def __init__(self):
        self.state = "align"

    def reset(self):
        self.state = "align"

    def __call__(self, obs, desired_goal):
        grip_pos = obs[0:3]
        object_pos = obs[3:6]
        gripper_state = obs[9:11]  # finger positions
        gripper_open_amount = gripper_state.sum()

        action = np.zeros(4, dtype=np.float32)
        target_above_object = object_pos + np.array([0, 0, self.ABOVE_OFFSET])
        target_above_goal = desired_goal + np.array([0, 0, self.ABOVE_OFFSET])

        if self.state == "align":
            delta = target_above_object - grip_pos
            action[:3] = delta
            action[3] = 1.0  # keep gripper open
            if np.linalg.norm(delta[:2]) < self.XY_TOLERANCE and abs(delta[2]) < 0.02:
                self.state = "descend"

        elif self.state == "descend":
            delta = object_pos - grip_pos
            action[:3] = delta
            action[3] = 1.0
            if np.linalg.norm(delta) < self.GRASP_TOLERANCE:
                self.state = "grasp"

        elif self.state == "grasp":
            action[:3] = 0.0
            action[3] = -1.0  # close gripper
            if gripper_open_amount < 0.045:
                self.state = "lift"

        elif self.state == "lift":
            delta = target_above_object - grip_pos
            action[:3] = delta
            action[3] = -1.0
            if delta[2] > -0.01 and np.linalg.norm(delta[:2]) < self.XY_TOLERANCE:
                self.state = "transport"

        elif self.state == "transport":
            delta = target_above_goal - grip_pos
            action[:3] = delta
            action[3] = -1.0
            if np.linalg.norm(delta) < self.XY_TOLERANCE:
                self.state = "place"

        elif self.state == "place":
            delta = desired_goal - grip_pos
            action[:3] = delta
            action[3] = -1.0
            if np.linalg.norm(delta) < self.Z_TOLERANCE:
                self.state = "release"

        elif self.state == "release":
            action[:3] = 0.0
            action[3] = 1.0  # open gripper, drop the object

        action[:3] = np.clip(action[:3] * 10.0, -1.0, 1.0)
        return action


def run_episode(env, policy, max_steps=200):
    obs_dict, info = env.reset()
    policy.reset()
    for step in range(max_steps):
        action = policy(obs_dict["observation"], obs_dict["desired_goal"])
        obs_dict, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info.get("is_success", 0.0)


def main(episodes=5, render=True):
    env = gym.make(ENV_ID, render_mode="human" if render else None, max_episode_steps=200)
    policy = PickAndPlacePolicy()

    successes = 0
    for ep in range(episodes):
        success = run_episode(env, policy)
        successes += success
        print(f"Episode {ep + 1}/{episodes}: success={bool(success)}")

    print(f"Finished {episodes} episodes, {int(successes)} successful.")
    env.close()


if __name__ == "__main__":
    main()
