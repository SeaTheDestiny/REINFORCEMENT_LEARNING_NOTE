import numpy as np
import torch
import matplotlib.pyplot as plt
import grid_world as gw
import os

# Hyperparameters
gamma = 0.9
NUM_EPISODES = 5000  
MAX_EPISODE_LENGTH = 1000
EPSILON = 0.1
learning_rate = 0.1

def get_feature(state, m, n):
    """
    Convert state to One-Hot feature vector
    """
    feature = torch.zeros(m * n)
    idx = state[0] * n + state[1]
    feature[idx] = 1.0
    return feature

def update_parameter_and_policy(w, experience, env):
    """
    Use PyTorch automatic differentiation to compute gradients and manually update parameters (Q-learning)
    """
    state, action, reward, next_state = experience
    m, n = env.m, env.n

    # 1. Compute current Q(s, a)
    feature_state = get_feature(state, m, n)
    q_state_all = torch.matmul(feature_state, w)
    q_state_action = q_state_all[action]

    # 2. Compute target Q Target (Q-learning: use max_a Q(s', a))
    with torch.no_grad():
        if next_state is None:  # Terminal
            q_target = reward
        else:
            feature_next_state = get_feature(next_state, m, n)
            q_next_state_all = torch.matmul(feature_next_state, w)
            # Q-learning: use maximum Q-value
            q_target = reward + gamma * q_next_state_all.max()

        td_error = q_target - q_state_action.item()

    # 3. Compute gradient using automatic differentiation (without backward)
    grads = torch.autograd.grad(q_state_action, w)[0]

    # 4. Update parameters
    with torch.no_grad():
        w += learning_rate * td_error * grads

    # 5. Update policy (epsilon-greedy)
    with torch.no_grad():
        updated_q_values = torch.matmul(feature_state, w).numpy()
        best_action = np.argmax(updated_q_values)

        num_actions = len(env.ACTIONS)
        for act in range(num_actions):
            if act == best_action:
                env.policy[state[0], state[1], act] = 1 - EPSILON + (EPSILON / num_actions)
            else:
                env.policy[state[0], state[1], act] = EPSILON / num_actions

    return w

def q_learning(env, num_episodes=NUM_EPISODES):
    """
    Q-learning with function approximation (on-policy version)
    Main learning loop with history tracking
    """
    input_dim = env.m * env.n
    num_actions = 5 
    
    # Initialize weights
    w = torch.zeros((input_dim, num_actions), requires_grad=True)

    # Used to record plotting data
    episode_rewards = []
    episode_lengths = []
    
    for episode_num in range(num_episodes):
        if episode_num % 100 == 0:
            print(f"Episode {episode_num}/{num_episodes}")

        state = (0, 0)

        # Statistics for current episode
        current_reward = 0
        steps = 0

        for _ in range(MAX_EPISODE_LENGTH):
            # Select action based on policy (epsilon-greedy)
            action = env.sample_action(state)

            # Interact with environment
            next_state, reward, done = env.step(state, action)

            # Accumulate reward and steps
            current_reward += reward
            steps += 1

            if done:
                next_state = None

            experience = (state, action, reward, next_state)

            # Update parameters and policy
            w = update_parameter_and_policy(w, experience, env)

            if done:
                break

            state = next_state

        # Record data for this episode
        episode_rewards.append(current_reward)
        episode_lengths.append(steps)
            
    return w, episode_rewards, episode_lengths

def plot_training_history(rewards, lengths, save_path="images/Q_learning_training_history.png"):
    """
    Plot Total Reward and Episode Length in two subplots
    """
    plt.figure(figsize=(8, 6))

    # Subplot 1: Total Reward
    plt.subplot(2, 1, 1)
    plt.plot(rewards)
    plt.ylabel('Total reward')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Subplot 2: Episode Length
    plt.subplot(2, 1, 2)
    plt.plot(lengths)
    plt.ylabel('Episode length')
    plt.xlabel('Episode index')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"Training curve saved to: {save_path}")

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
        save_path="images/inital_Q_learning_policy_and_value.png"
    )

    # Start training (receive returned historical data)
    trained_w, rewards, lengths = q_learning(env, num_episodes=NUM_EPISODES)

    # Plot training curve (Total Reward & Episode Length)
    print("Plotting training process curve...")
    plot_training_history(rewards, lengths, save_path="images/Q_learning_training_history.png")

    # Calculate final state value function for GridWorld plotting
    with torch.no_grad():
        for i in range(env.m):
            for j in range(env.n):
                feature = get_feature((i, j), env.m, env.n)
                q_values = torch.matmul(feature, trained_w).numpy()
                env.state_value[i, j] = q_values.max()

    print("Plotting final policy and value...")
    gw.plot_policy_and_value(
        state_value=env.state_value,
        policy=env.policy,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path="images/Q_learning_FunctionApprox_policy_and_value.png"
    )

    print("\nTraining completed!")