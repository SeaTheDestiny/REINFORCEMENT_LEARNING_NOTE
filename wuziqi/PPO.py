import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import imageio
import network
from pathlib import Path
from gomoku_env import GomokuEnv

# Hyperparameters
BATCH_SIZE = 256
NUM_EPISODES = 2400
K_EPOCHS = 10
BUFFER_SIZE = 4096
gamma = 0.99
learning_rate_theta = 3e-4
learning_rate_w = 3e-4
LAMBDA = 0.95

def PPO_learning(env, num_episodes=NUM_EPISODES):
    """
    Proximal Policy Optimization (PPO) learning algorithm with GAE for Gomoku.

    This implementation includes:
    - Actor-Critic architecture with perspective normalization
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective for stable policy updates
    - Multiple epochs of updates per data collection
    - Splitting episodes into black and white perspectives

    Parameters
    ----------
    env : gym.Env
        The reinforcement learning environment (GomokuEnv)
    num_episodes : int, optional
        Number of update cycles (default: NUM_EPISODES)

    Returns
    -------
    tuple
        (actor, critic, episode_rewards, episode_lengths)
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Create actor and critic networks
    actor = network.ActorNetwork(state_size, action_size)
    critic = network.CriticNetworkForStateValue(state_size)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate_theta)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate_w)

    # Set to eval mode for data collection (BatchNorm issue)
    actor.eval()
    critic.eval()

    episode_rewards = []
    episode_lengths = []

    # PPO loop
    state, _ = env.reset()

    for episode_num in range(num_episodes):
        if episode_num % 10 == 0:
            print(f"Update Cycle {episode_num}/{num_episodes}")

        # --- Phase 1: Collect complete games ---
        game_buffer = []  # Store complete games
        total_steps = 0

        while total_steps < BUFFER_SIZE:
            game_states = []
            game_actions = []
            game_rewards = []
            game_next_states = []
            game_dones = []
            game_players = []  # Track which player made each move
            game_log_probs = []
            game_masks = []  # Track valid action masks

            state, _ = env.reset()
            done = False
            step = 0

            # Play a complete game
            while not done and step < 200:  # Max 200 steps per game
                # Get normalized state from current player's perspective
                player = env.current_player
                normalized_state = env.get_normalized_state(player)

                state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)

                # 获取合法动作掩码（1为合法，0为非法）
                valid_mask = torch.FloatTensor([1 if env._is_valid_move(i // 15, i % 15) else 0 for i in range(225)])

                with torch.no_grad():
                    # Actor现在返回logits
                    logits = actor(state_tensor)
                    # 应用动作屏蔽：将非法动作的logits设为负无穷大
                    masked_logits = logits + (1 - valid_mask) * -1e9
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                action_item = action.item()
                next_state, reward, done, truncated, _ = env.step(action_item)

                # Store experience with player information and mask
                game_states.append(normalized_state.copy())
                game_actions.append(action_item)
                game_rewards.append(reward)
                game_next_states.append(env.get_normalized_state(env.current_player).copy())
                game_dones.append(done or truncated)
                game_players.append(player)
                game_log_probs.append(log_prob.item())
                game_masks.append(valid_mask.numpy())

                state = next_state
                step += 1

            # Store complete game
            if len(game_states) > 0:
                game_buffer.append({
                    'states': game_states,
                    'actions': game_actions,
                    'rewards': game_rewards,
                    'next_states': game_next_states,
                    'dones': game_dones,
                    'players': game_players,
                    'log_probs': game_log_probs,
                    'masks': game_masks,
                    'winner': env.winner
                })
                episode_lengths.append(step)
                total_steps += step

        # --- Phase 2: Split games into black and white perspectives ---
        black_buffer = []
        white_buffer = []

        for game in game_buffer:
            # Split into alternating moves
            for i in range(len(game['states'])):
                player = game['players'][i]
                exp = {
                    'state': game['states'][i],
                    'action': game['actions'][i],
                    'reward': game['rewards'][i],
                    'next_state': game['next_states'][i],
                    'done': game['dones'][i],
                    'log_prob': game['log_probs'][i],
                    'mask': game['masks'][i],
                    'winner': game['winner'],
                    'player': player
                }

                if player == 1:  # Black player
                    black_buffer.append(exp)
                else:  # White player
                    white_buffer.append(exp)

        # --- Phase 3: Process and update ---
        all_buffers = [black_buffer, white_buffer]
        total_reward = 0

        for buffer in all_buffers:
            if len(buffer) == 0:
                continue

            states = torch.FloatTensor(np.array([x['state'] for x in buffer]))
            actions = torch.LongTensor([x['action'] for x in buffer])
            rewards = [x['reward'] for x in buffer]
            next_states = torch.FloatTensor(np.array([x['next_state'] for x in buffer]))
            dones = [x['done'] for x in buffer]
            old_log_probs = torch.FloatTensor([x['log_prob'] for x in buffer])
            masks = torch.FloatTensor(np.array([x['mask'] for x in buffer]))

            # Rewards are already calculated correctly in the environment
            # No need to adjust them here
            total_reward += sum(rewards)

            # Compute GAE
            with torch.no_grad():
                values = critic(states).squeeze()
                next_values = critic(next_states).squeeze()

            advantages = []
            gae = 0
            for i in reversed(range(len(buffer))):
                r = rewards[i]
                d = dones[i]
                v = values[i]
                nv = next_values[i]

                delta = r + gamma * nv * (1 - d) - v
                gae = delta + gamma * LAMBDA * (1 - d) * gae
                advantages.insert(0, gae)

            advantages = torch.FloatTensor(advantages)
            td_targets = advantages + values

            # PPO Update
            # Set to train mode for updates
            actor.train()
            critic.train()

            for _ in range(K_EPOCHS):
                indices = torch.randperm(len(buffer))

                for start in range(0, len(buffer), BATCH_SIZE):
                    end = start + BATCH_SIZE
                    batch_indices = indices[start:end]

                    # Skip batches with fewer than 2 samples to avoid BatchNorm error
                    if len(batch_indices) < 2:
                        continue

                    batch_states = states[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_td_targets = td_targets[batch_indices].detach()
                    batch_advantages = advantages[batch_indices].detach()
                    batch_old_log_probs = old_log_probs[batch_indices].detach()
                    batch_masks = masks[batch_indices].detach()

                    # Normalize Advantage
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                    # Actor Update
                    logits = actor(batch_states)
                    # 应用动作屏蔽
                    masked_logits = logits + (1 - batch_masks) * -1e9
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    dist_entropy = dist.entropy().mean()

                    ratios = torch.exp(new_log_probs - batch_old_log_probs)
                    surr1 = ratios * batch_advantages
                    surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                    actor_optimizer.step()

                    # Critic Update
                    current_values = critic(batch_states).squeeze()
                    critic_loss = 0.5 * F.mse_loss(current_values, batch_td_targets)

                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                    critic_optimizer.step()

            # Set back to eval mode for data collection
            actor.eval()
            critic.eval()

        episode_rewards.append(total_reward)

    # Save trained models
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
    }, 'model/gomoku_ppo_model.pth')
    print(f"Models saved to model/gomoku_ppo_model.pth")

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

def visualize_policy(actor, num_episodes=3, output_path='images/PPO_Gomoku_trained_policy.gif'):
    """
    Visualize trained policy and generate GIF animation for Gomoku
    Uses perspective normalization
    """
    env = GomokuEnv(render_mode='rgb_array', record_frames=True)
    actor.eval()

    for ep in range(num_episodes):
        state, _ = env.reset()
        done, truncated = False, False
        step = 0

        while not (done or truncated) and step < 200:  # 五子棋通常在100步以内结束
            # Get normalized state from current player's perspective
            player = env.current_player
            normalized_state = env.get_normalized_state(player)

            with torch.no_grad():
                state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)
                # 获取合法动作掩码
                valid_mask = torch.FloatTensor([1 if env._is_valid_move(i // 15, i % 15) else 0 for i in range(225)])
                # Actor现在返回logits
                logits = actor(state_tensor)
                # 应用动作屏蔽
                masked_logits = logits + (1 - valid_mask) * -1e9
                dist = torch.distributions.Categorical(logits=masked_logits)
                action = dist.sample().item()

            # 执行动作（由于已应用mask，动作应该是合法的）
            state, _, done, truncated, _ = env.step(action)
            step += 1

    # 使用环境自带的save_gif方法保存GIF
    env.save_gif(output_path, duration=1000)
    env.close()
    print(f"GIF saved to {Path(output_path).absolute()}")

if __name__ == "__main__":
    # Initialize environment
    env = GomokuEnv()

    # Start training
    print("Starting PPO training on Gomoku...")
    trained_actor, trained_critic, rewards, lengths = PPO_learning(env, num_episodes=NUM_EPISODES)

    # Plot training curves
    print("Plotting training curves...")
    plot_training_history(rewards, lengths, save_path="images/PPO_Gomoku_training_history.png")

    # Visualize trained policy
    print("Generating policy visualization GIF...")
    visualize_policy(trained_actor, num_episodes=3, output_path='images/PPO_Gomoku_trained_policy.gif')

    print("\nTraining completed!")
    

            
