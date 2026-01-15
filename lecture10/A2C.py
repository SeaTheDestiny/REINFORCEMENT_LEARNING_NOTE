import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import imageio
import network
from pathlib import Path

# Hyperparameters
NUM_EPISODES = 4800
MAX_EPISODE_LENGTH = 10000
gamma = 0.99
learning_rate_theta = 5e-4
learning_rate_w = 5e-4

def A2C_learning(env, num_episodes=NUM_EPISODES):
    """
    Advantage Actor-Critic learning loop with history tracking
    
    Algorithm:
    1. Actor: Policy network π(a|s) - outputs action probabilities
    2. Critic: State-value network V(s) - outputs state values
    3. Update rule based on TD error = r + γV(s') - V(s)
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize Actor and Critic networks
    actor = network.ActorNetwork(state_size, action_size)
    critic = network.CriticNetworkForStateValue(state_size)

    # Optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate_theta)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate_w)

    # Used to record plotting data
    episode_rewards = []
    episode_lengths = []
    
    for episode_num in range(num_episodes):
        if episode_num % 100 == 0:
            print(f"Episode {episode_num}/{num_episodes}")

        state, _ = env.reset()
        done, truncated = False, False
        
        # Statistics for current episode
        current_reward = 0
        steps = 0

        for _ in range(MAX_EPISODE_LENGTH):
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Actor: Sample action based on policy
            action_probs = actor(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()

            # Execute action
            next_state, reward, done, truncated, _ = env.step(action)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # Accumulate reward and steps
            current_reward += reward
            steps += 1

            # Critic: Compute Q(s, a)
            #action_one_hot = F.one_hot(torch.tensor([action]), num_classes=action_size).float()
            v_s = critic(state_tensor)

            # Compute TD target
            with torch.no_grad():
                if done or truncated:
                    td_target = reward
                else:
                    # Sample next action (SARSA style)
                                        
                    v_snext = critic(next_state_tensor)
                    td_target = reward + gamma * v_snext

            # Compute TD error
            td_error = td_target - v_s

            # Critic update: Minimize squared TD error
            critic_loss = 0.5 * td_error.pow(2)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Actor update: Policy gradient ∇logπ(a|s) * Q(s, a)
            log_pi = torch.log(action_probs.squeeze(0)[action] + 1e-8)  # Add small constant to avoid log(0)
            actor_loss = -log_pi * td_error.detach()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update state
            state = next_state

            if done or truncated:
                break

        # Record data for this episode
        episode_rewards.append(current_reward)
        episode_lengths.append(steps)
    
    return actor, critic, episode_rewards, episode_lengths

def plot_training_history(rewards, lengths, save_path="images/A2C_training_history.png"):
    """
    Plot Total Reward and Episode Length in two subplots
    """
    plt.figure(figsize=(8, 6))  # Set canvas size

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
    plt.close()
    print(f"Training curve saved to: {save_path}")

def visualize_policy(actor, env_name='LunarLander-v2', num_episodes=3, output_path='images/A2C_LunarLander_trained_policy.gif'):
    """
    Visualize trained policy and generate GIF animation
    """
    env = gym.make(env_name, render_mode='rgb_array')
    frames = []
    actor.eval()

    for ep in range(num_episodes):
        state, _ = env.reset()
        done, truncated = False, False
        step = 0

        while not (done or truncated) and step < 500:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = actor(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()

            state, _, done, truncated, _ = env.step(action)
            frames.append(env.render())
            step += 1

    env.close()
    imageio.mimsave(Path(output_path), frames, duration=33)
    print(f"GIF saved to {Path(output_path).absolute()}")

if __name__ == "__main__":
    # Initialize environment
    env = gym.make("LunarLander-v2")

    # Start training
    print("Starting A2C training...")
    trained_actor, trained_critic, rewards, lengths = A2C_learning(env, num_episodes=NUM_EPISODES)

    # Plot training curves
    print("Plotting training curves...")
    plot_training_history(rewards, lengths, save_path="images/A2C_LunarLander_training_history.png")

    # Visualize trained policy
    print("Generating policy visualization GIF...")
    visualize_policy(trained_actor, env_name='LunarLander-v2', num_episodes=3, output_path='images/A2C_LunarLander_trained_policy.gif')

    print("\nTraining completed!")
    

            
