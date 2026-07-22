from environments import PandaEnvironment


env = PandaEnvironment(render=True)

print("Observation size:", env.state_dim)
print("Action size:", env.action_dim)

state, info = env.reset()

print(state.shape)

for i in range(200):

    action = env.sample_action()

    next_state, reward, done, info = env.step(action)

    print(
        f"Step {i:3d}",
        f"Reward = {reward:.3f}"
    )

    if done:
        state, info = env.reset()

env.close()
