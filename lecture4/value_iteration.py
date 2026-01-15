import numpy as np
import matplotlib.pyplot as plt
import grid_world as gw

# Hyperparameters
gamma = 0.9


def value_iteration(env, gamma):
    """Value iteration algorithm: simultaneously updates values and policy."""
    while True:
        old_state_value = env.state_value.copy()

        for i in range(env.m):
            for j in range(env.n):
                best_action_value = -float("inf")
                best_action_index = 0

                for action_index, _ in enumerate(env.ACTIONS):
                    next_state, reward, done = env.step((i, j), action_index)

                    if done:
                        q_values = reward
                    else:
                        q_values = reward + gamma * old_state_value[next_state]

                    if q_values > best_action_value:
                        best_action_value = q_values
                        best_action_index = action_index

                env.state_value[i, j] = best_action_value
                env.policy[i, j] = best_action_index

        if np.max(np.abs(old_state_value - env.state_value)) < 1e-4:
            break


if __name__ == "__main__":
    # Initialize environment
    env = gw.GridWorld(m=5, n=6, forbidden_ratio=0.2, seed=42)
    env.random_policy()

    # Plot initial policy and value
    gw.plot_policy_and_value(
        state_value=env.state_value,
        policy=env.policy,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path="images/inital_value_iteration_policy_and_value.png"
    )

    # Run value iteration
    value_iteration(env, gamma)

    # Plot final policy and value
    gw.plot_policy_and_value(
        state_value=env.state_value,
        policy=env.policy,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path="images/value_iteration_policy_and_value.png"
    )
