# Healthy Dopamine Reinforcement Learning Model

This project simulates a simplified healthy dopamine system for a robotic
pick-and-place task using Panda-Gym.

The healthy model computes reward prediction error (RPE)

δ = r + γV(s') − V(s)

which modulates learning.

Later versions will replace this dopamine system with impaired versions
to simulate Parkinson's disease.
