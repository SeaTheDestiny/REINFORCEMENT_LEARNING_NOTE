import numpy as np
import matplotlib.pyplot as plt
import grid_world as gw

# Hyperparameters
gamma = 0.9
NUM_EPISODES = 10000
MAX_EPISODE_LENGTH = 100
EPSILON = 0.2


def update_q_value_and_update_policy_from_q(goal, q_values, experience, learning_rate=0.1):
    """
    Update Q-values based on the generated experience using SARSA update rule.
    """
    state = experience[0]
    action = experience[1]
    reward = experience[2]
    next_state = experience[3]

    # update_q_value
    state_idx = state
    next_state_idx = next_state

    q_predict = q_values[state_idx][action]

    if next_state != goal:
        q_target = reward + gamma * max(q_values[next_state_idx])
        #  q_values[next_state_idx][next_action]
    else:
        q_target = reward  # Terminal state

    q_values[state_idx][action] -= learning_rate * (q_predict - q_target)

    # update_policy_from_q
    best_action = np.argmax(q_values[state_idx])
    num_actions = len(env.ACTIONS)
    for act in range(num_actions):
        if act == best_action:
            env.policy[state_idx[0], state_idx[1], act] = 1 - EPSILON + (EPSILON / num_actions)
        else:
            env.policy[state_idx[0], state_idx[1], act] = EPSILON / num_actions

    return next_state, env.policy, q_values



def q_learning(env, num_episodes=NUM_EPISODES):
    """
    Main q learning loop.
    
    Returns:
        np.ndarray: Learned Q-values
    """
    q_values = np.zeros((env.m, env.n, 5))  # Action value function q(s,a)
    
    for episode_num in range(num_episodes):
        if episode_num % 1000 == 0:
            print(f"Episode {episode_num}/{num_episodes}")
        
        start_state = (0,0)

        start_action = env.sample_action(start_state)

        state = start_state
        action = start_action

        for _ in range(MAX_EPISODE_LENGTH):
            next_state, reward, done = env.step(state, action)
            experience = (state, action, reward, next_state)

            state, env.policy, q_values = update_q_value_and_update_policy_from_q(
                env.goal, q_values, experience, learning_rate=0.1
            )
            action = env.sample_action(state)

            
                
    return q_values

if __name__ == "__main__":
    # Initialize environment
    env = gw.GridWorld(m=8, n=8, forbidden_ratio=0.2, seed=42)

    # Plot initial random policy and value
    print("Plotting initial random policy and value...")
    gw.plot_policy_and_value(
        state_value=env.state_value,
        policy=env.policy,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path="images/inital_policy_iteration_policy_and_value.png"
    )


    q_values = q_learning(env, num_episodes=NUM_EPISODES)
    # Calculate final state value function from Q-values
    for i in range(env.m):
        for j in range(env.n):
            env.state_value[i, j] = q_values[(i, j)].max()
    # Plot final policy and value

    print("Plotting final policy and value...")
    gw.plot_policy_and_value(
        state_value=env.state_value,
        policy=env.policy,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path="images/Q_policy_and_value.png"
    )

    print("\nTraining completed!")
    print(f"Final policy shape: {env.policy.shape}")
    print(f"Final state value shape: {env.state_value.shape}")
    print(f"Q-value shape: {q_values.shape}")