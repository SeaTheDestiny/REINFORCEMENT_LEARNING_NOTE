import numpy as np
import matplotlib.pyplot as plt
import grid_world as gw

# Hyperparameters
gamma = 0.9
NUM_EPISODES = 100000
MAX_EPISODE_LENGTH = 1000


def generate_episode(env, start_state, start_action):
    """
    Generate an episode starting from a specific state-action pair,
    following the current policy for subsequent steps.
    
    Returns:
        list: Episode as [(state, action, reward), ...]
    """
    episode = []
    
    # Execute the starting action
    current_state = start_state
    current_action = start_action
    next_state, reward, done = env.step(current_state, current_action)
    episode.append((current_state, current_action, reward))
    
    if done:
        return episode
    
    current_state = next_state
    
    # Follow current policy for subsequent steps
    for t in range(1, MAX_EPISODE_LENGTH):
        current_action = env.sample_action(current_state)
        next_state, reward, done = env.step(current_state, current_action)
        episode.append((current_state, current_action, reward))
        
        if done:
            break
            
        current_state = next_state
    
    return episode


def monte_carlo_epsilon_greedy(env, num_episodes=NUM_EPISODES):
    """
    Monte Carlo Epsilon-Greedy Algorithm

    Algorithm steps:
    1. Initialization:
       - q(s,a) = 0 for all (s,a)
       - Returns(s,a) = 0 for all (s,a)
       - Num(s,a) = 0 for all (s,a)
       - π(a|s) = uniform distribution

    2. For each episode:
       - Randomly select starting state-action pair
       - Generate episode following current policy
       - Calculate cumulative returns backwards
       - Update Returns(s,a) and Num(s,a)
       - q(s,a) = Returns(s,a) / Num(s,a)
       - Policy improvement: epsilon-greedy policy

    Parameters:
        env: GridWorld, environment object
        num_episodes: int, total number of episodes

    Returns:
        np.ndarray: Action value function q_values, shape=(m, n, 5)
    """
    
    # Initialize action value function and return statistics
    q_values = np.zeros((env.m, env.n, 5))  # Action value function q(s,a)
    returns_sum = np.zeros((env.m, env.n, 5))  # Cumulative return sum for each (s,a)
    returns_count = np.zeros((env.m, env.n, 5))  # Visit count for each (s,a)

    # Initialize policy as uniform distribution
    env.policy = np.ones((env.m, env.n, 5), dtype=float) / 5.0

    # Get all valid states (randomly select some states as starting states)
    valid_states = []
    for i in range(env.m):
        for j in range(env.n):
            # Randomly select about half of the states as candidate starting states
            if np.random.rand() > 0.5:
                valid_states.append((i, j))

    print(f"Starting Monte Carlo epsilon-greedy algorithm, total {num_episodes} episodes...")

    # Epsilon value linearly decays from 1 to 0 (epsilon-greedy with decay)
    epsilon_array = np.linspace(1, 0, num_episodes)

    # Main loop: generate multiple episodes
    for episode_num in range(num_episodes):
        if episode_num % 1000 == 0:
            print(f"Episode {episode_num}/{num_episodes}")

        # Randomly select starting state-action pair
        start_state = valid_states[np.random.choice(len(valid_states))]
        start_action = np.random.choice(5)

        # Generate episode following current policy
        episode = generate_episode(env, start_state, start_action)

        # Calculate cumulative returns backwards
        episode_return = 0.0

        # Process episode (every-visit MC: update on every visit)
        for step in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[step]
            episode_return = gamma * episode_return + reward

            # Update visit statistics
            i, j = state
            returns_sum[i, j, action] += episode_return
            returns_count[i, j, action] += 1

            # Policy evaluation: update action value q(s,a)
            q_values[i, j, action] = returns_sum[i, j, action] / returns_count[i, j, action]

            # Policy improvement: epsilon-greedy policy
            best_action = np.argmax(q_values[i, j, :])
            current_epsilon = epsilon_array[episode_num]
            # Non-optimal action probability: epsilon/|A|
            env.policy[i, j, :] = (1 / 5.0) * current_epsilon
            # Optimal action probability: 1-epsilon+epsilon/|A|
            env.policy[i, j, best_action] = 1.0 - current_epsilon + (current_epsilon / 5.0)

    # Update state values based on final policy
    for i in range(env.m):
        for j in range(env.n):
            # Use Q-value of best action as state value
            best_action = np.argmax(env.policy[i, j, :])
            env.state_value[i, j] = q_values[i, j, best_action]

    print(f"Completed {num_episodes} episodes")
    return q_values


if __name__ == "__main__":
    # Initialize environment
    env = gw.GridWorld(m=5, n=6, forbidden_ratio=0.2, seed=42)

    # Plot initial random policy and value
    print("Plotting initial random policy and value...")
    gw.plot_policy_and_value(
        state_value=env.state_value,
        policy=env.policy,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path="images/inital_policy_iteration_policy_and_value.png"
    )

    # Run Monte Carlo epsilon-greedy algorithm
    print("Starting Monte Carlo epsilon-greedy algorithm...")
    q_values = monte_carlo_epsilon_greedy(env, num_episodes=NUM_EPISODES)

    # Plot final policy and value
    print("Plotting final policy and value...")
    gw.plot_policy_and_value(
        state_value=env.state_value,
        policy=env.policy,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path="images/MC_epsilon_greedy_policy_and_value.png"
    )

    print("\nTraining completed!")
    print(f"Final policy shape: {env.policy.shape}")
    print(f"Final state value shape: {env.state_value.shape}")
    print(f"Q-value shape: {q_values.shape}")