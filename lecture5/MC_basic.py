import numpy as np
import matplotlib.pyplot as plt
import grid_world as gw

# Hyperparameter configuration
gamma = 0.9  # Discount factor, used to calculate cumulative returns
NUM_EPISODES = 100  # Number of episodes to evaluate each state-action pair
MAX_EPISODE_LENGTH = 100  # Maximum length of a single episode (prevents infinite loops)


def estimate_action_value(env, state, action):
    """
    Estimate the action value function q(s, a) using Monte Carlo method
    
    Sample multiple episodes starting from a given state-action pair and return the average return
    
    Parameters:
        env: GridWorld, environment object
        state: tuple (i, j), starting state
        action: int, starting action
        
    Returns:
        float: Monte Carlo estimate of the action value
    """
    returns = []

    # Sample multiple episodes and calculate returns
    for _ in range(NUM_EPISODES):
        # Execute initial action
        next_state, immediate_reward, done = env.step(state, action)
        episode_return = immediate_reward

        # If not terminated, continue following current policy
        if not done:
            current_state = next_state
            for step in range(1, MAX_EPISODE_LENGTH):
                next_action = env.sample_action(current_state)
                current_state, reward, done = env.step(current_state, next_action)
                episode_return += (gamma ** step) * reward
                if done:
                    break

        returns.append(episode_return)

    # Return average return as action value estimate
    return np.mean(returns)


def policy_evaluation(env):
    """
    Policy iteration based on Monte Carlo method
    
    Evaluate the value of all actions for each state, then update to a greedy policy
    
    Parameters:
        env: GridWorld, environment object
        
    Notes:
        - Uses estimated action values to improve the policy
        - Converges when the policy no longer changes
    """
    iteration = 0
    max_iterations = 100  # Prevent infinite loops

    while iteration < max_iterations:
        iteration += 1
        new_policy = env.policy.copy()
        q_values = np.zeros((env.m, env.n, 5))  # Action value function, used to update state values

        # Iterate through all states
        for i in range(env.m):
            for j in range(env.n):
                max_q_value = -np.inf
                best_action = None

                # Evaluate all actions for current state
                for action in range(5):
                    # Estimate action value q(s, a)
                    q_values_temp = estimate_action_value(env, (i, j), action)
                    q_values[i, j, action] = q_values_temp

                    # Record best action
                    if q_values_temp > max_q_value:
                        max_q_value = q_values_temp
                        best_action = action

                # Update policy to greedy policy (deterministic policy)
                if best_action is not None:
                    new_policy[i, j, :] = 0.0
                    new_policy[i, j, best_action] = 1.0

        # Check if policy has converged
        if np.array_equal(env.policy, new_policy):
            print(f"Policy converged after {iteration} iterations")
            break

        env.policy = new_policy

        # Update state values based on new policy
        for i in range(env.m):
            for j in range(env.n):
                best_action = np.argmax(env.policy[i, j])
                env.state_value[i, j] = q_values[i, j, best_action]

    if iteration >= max_iterations:
        print(f"Reached maximum iterations ({max_iterations}) without convergence")


if __name__ == "__main__":
    # Initialize environment
    env = gw.GridWorld(m=5, n=6, forbidden_ratio=0.2, seed=42)
    env.random_policy()  # Generate random initial policy

    # Plot initial policy and value
    print("Plotting initial policy and value...")
    gw.plot_policy_and_value(
        state_value=env.state_value,
        policy=env.policy,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path="images/inital_policy_iteration_policy_and_value.png"
    )

    # Execute policy iteration
    print("Starting policy iteration...")
    policy_evaluation(env)

    # Plot final policy and value
    print("Plotting final policy and value...")
    gw.plot_policy_and_value(
        state_value=env.state_value,
        policy=env.policy,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path="images/MC_basic_policy_and_value.png"
    )

    print("\nTraining completed!")
    print(f"Final policy shape: {env.policy.shape}")
    print(f"Final state value shape: {env.state_value.shape}")