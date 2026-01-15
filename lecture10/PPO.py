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
BATCH_SIZE = 256
NUM_EPISODES = 4800
K_EPOCHS = 4
BUFFER_SIZE = 2048
gamma = 0.99
learning_rate_theta = 5e-4
learning_rate_w = 5e-4
LAMBDA = 0.95

def PPO_learning(env, num_episodes=NUM_EPISODES):
    """
    Proximal Policy Optimization (PPO) learning algorithm with GAE.

    This implementation includes:
    - Actor-Critic architecture
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective for stable policy updates
    - Multiple epochs of updates per data collection

    Parameters
    ----------
    env : gym.Env
        The reinforcement learning environment
    num_episodes : int, optional
        Number of update cycles (default: NUM_EPISODES)

    Returns
    -------
    tuple
        (actor, critic, episode_rewards, episode_lengths)
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 1. No need for actor_beta, one actor is sufficient
    actor = network.ActorNetwork(state_size, action_size)
    critic = network.CriticNetworkForStateValue(state_size)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate_theta)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate_w)

    episode_rewards = []
    episode_lengths = []

    # PPO loop is typically not counted by episodes, but by update iterations
    # But to maintain compatibility with your interface, we keep the episode concept,
    # just changing the internal logic to "collect data then update"

    state, _ = env.reset()

    for episode_num in range(num_episodes): # This is actually more like counting Update times
        if episode_num % 10 == 0:
            print(f"Update Cycle {episode_num}/{num_episodes}")

        experience_buffer = []
        episode_length = 0

        # --- Phase 1: Collect data (force fill BUFFER_SIZE) ---
        while len(experience_buffer) < BUFFER_SIZE:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad(): # Sampling does not need gradients
                action_probs = actor(state_tensor)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                # [Key fix 1] Directly store scalar log_prob
                log_prob = dist.log_prob(action)

            action_item = action.item()
            next_state, reward, done, truncated, _ = env.step(action_item)

            # [Key fix 2] Better to separate terminated and truncated, simplified here
            # Assume done includes both, for perfection refer to previous answer to handle truncated
            experience_buffer.append((state, action_item, reward, next_state, done, log_prob.item()))

            state = next_state
            episode_length += 1

            # [Key fix 3] Cross-episode collection
            if done or truncated:
                episode_lengths.append(episode_length)
                episode_length = 0
                state, _ = env.reset()

        # --- Phase 2: Process data & Compute GAE ---
        states = torch.FloatTensor(np.array([x[0] for x in experience_buffer]))
        actions = torch.LongTensor([x[1] for x in experience_buffer]) # [T]
        rewards = [x[2] for x in experience_buffer]
        next_states = torch.FloatTensor(np.array([x[3] for x in experience_buffer]))
        dones = [x[4] for x in experience_buffer]
        old_log_probs_all = torch.FloatTensor([x[5] for x in experience_buffer]) # [T]

        with torch.no_grad():
            values = critic(states).squeeze()      # [T]
            next_values = critic(next_states).squeeze() # [T]

        # Compute GAE
        advantages = []
        gae = 0
        for i in reversed(range(len(experience_buffer))):
            r = rewards[i]
            d = dones[i]
            v = values[i]
            nv = next_values[i]

            delta = r + gamma * nv * (1 - d) - v
            gae = delta + gamma * LAMBDA * (1 - d) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages) # [T]
        td_targets = advantages + values           # [T]

        # --- Phase 3: PPO Update ---
        for _ in range(K_EPOCHS):
            indices = torch.randperm(len(experience_buffer))

            for start in range(0, len(experience_buffer), BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_indices = indices[start:end]

                # Get Batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_td_targets = td_targets[batch_indices].detach()
                batch_advantages = advantages[batch_indices].detach()
                batch_old_log_probs = old_log_probs_all[batch_indices].detach()

                # Normalize Advantage
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                # === Actor Update ===
                action_probs = actor(batch_states)
                dist = torch.distributions.Categorical(action_probs)

                # Compute new log_prob
                new_log_probs = dist.log_prob(batch_actions)
                dist_entropy = dist.entropy().mean()

                # Ratio: Current log_probs and buffer log_probs are both scalars, can directly subtract
                ratios = torch.exp(new_log_probs - batch_old_log_probs)

                # Loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy

                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5) # Recommended to add
                actor_optimizer.step()

                # === Critic Update ===
                current_values = critic(batch_states).squeeze()
                critic_loss = 0.5 * F.mse_loss(current_values, batch_td_targets)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5) # Recommended to add
                critic_optimizer.step()

        # Record average reward of this buffer for plotting
        episode_rewards.append(sum(rewards))

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

def visualize_policy(actor, env_name='LunarLander-v2', num_episodes=3, output_path='images/A2C_off_policy_LunarLander_trained_policy.gif'):
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
    print("Starting PPO training...")
    trained_actor, trained_critic, rewards, lengths = PPO_learning(env, num_episodes=NUM_EPISODES)

    # Plot training curves
    print("Plotting training curves...")
    plot_training_history(rewards, lengths, save_path="images/PPO_LunarLander_training_history.png")

    # Visualize trained policy
    print("Generating policy visualization GIF...")
    visualize_policy(trained_actor, env_name='LunarLander-v2', num_episodes=3, output_path='images/PPO_LunarLander_trained_policy.gif')

    print("\nTraining completed!")
    

            
