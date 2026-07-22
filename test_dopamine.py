import torch

from brain.dopamine import DopamineSystem

dopamine = DopamineSystem()

reward = torch.tensor([[1.0]])

current = torch.tensor([[0.4]])

next_value = torch.tensor([[0.9]])

done = torch.tensor([[0.0]])

rpe = dopamine.compute_rpe(
    reward,
    current,
    next_value,
    done,
)

target = dopamine.value_target(
    reward,
    next_value,
    done,
)

print("Target:", target)
print("RPE:", rpe)
