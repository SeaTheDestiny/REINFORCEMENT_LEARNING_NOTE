import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import matplotlib.pyplot as plt
import gymnasium as gym
import imageio
import network
import utils
from pathlib import Path
from gomoku_env import GomokuEnv

# Hyperparameters
BATCH_SIZE = 512  # 增大Batch Size以提高GPU效率
NUM_EPISODES = 4800*2
K_EPOCHS = 4      # 减少Epoch数，防止过拟合
BUFFER_SIZE = 16384 # 增大Buffer，提升样本多样性
gamma = 0.99
learning_rate_theta = 3e-4
learning_rate_w = 3e-4
LAMBDA = 0.95
NUM_ENVS = 16      #并行环境数量

def get_batch_tensor(states, device):
    """Convert list of states to batch tensor"""
    return torch.FloatTensor(np.array(states)).to(device)

def PPO_learning(env_prototype, num_episodes=NUM_EPISODES):
    """
    Optimized PPO with Parallel Environments and Vectorized Data Augmentation
    """
    # Initialize parallel environments
    envs = [GomokuEnv() for _ in range(NUM_ENVS)]
    state_size = int(np.prod(envs[0].observation_space.shape)) # 15x15 = 225
    action_size = envs[0].action_space.n
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device} with {NUM_ENVS} parallel environments")

    # Networks
    # 增加网络容量：hidden_dim 64->128, num_blocks 4->6
    # 15x15棋盘需要更深更宽的网络来捕捉复杂的棋型
    actor = network.ActorNetwork(state_size, action_size, num_blocks=6, hidden_dim=128).to(device)
    critic = network.CriticNetworkForStateValue(state_size, num_blocks=6, hidden_dim=128).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate_theta)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate_w)
    
    # 学习率调度器：每500轮衰减一次，gamma=0.9
    actor_scheduler = torch.optim.lr_scheduler.StepLR(actor_optimizer, step_size=500, gamma=0.9)
    critic_scheduler = torch.optim.lr_scheduler.StepLR(critic_optimizer, step_size=500, gamma=0.9)

    # Opponent Pool
    opponent_pool = []
    MAX_POOL_SIZE = 50  # Limit usage of memory
    SAVE_POOL_INTERVAL = 50
    USE_OPPONENT_PROB = 0.5 # 提高对抗旧模型的概率，不仅仅是自我对弈

    # Initialize environment states and histories
    # histories[i] stores the current game trajectory for env[i]
    histories = [[] for _ in range(NUM_ENVS)]
    
    # Track which opponent model each env is using (None for self-play)
    env_opponent_actors = [None] * NUM_ENVS
    
    # Initial Reset
    current_states = []
    current_players = []
    
    for i in range(NUM_ENVS):
        s, _ = envs[i].reset()
        current_states.append(envs[i].get_normalized_state(envs[i].current_player))
        current_players.append(envs[i].current_player)
        
        # Determine opponent for this new game
        if len(opponent_pool) > 0 and np.random.random() < USE_OPPONENT_PROB:
            opp_idx = np.random.randint(len(opponent_pool))
            opp_actor = network.ActorNetwork(state_size, action_size, num_blocks=6, hidden_dim=128).to(device)
            opp_actor.load_state_dict(opponent_pool[opp_idx])
            opp_actor.eval()
            env_opponent_actors[i] = opp_actor
        else:
            env_opponent_actors[i] = None

    actor.eval()
    critic.eval()

    episode_rewards = []
    episode_lengths = []

    total_steps_collected = 0
    total_episodes_completed = 0

    for update_cycle in range(num_episodes):
        # Initialize temporary lists for current update cycle
        cycle_rewards = []
        cycle_lengths = []
        if update_cycle % 10 == 0:
            print(f"Update Cycle {update_cycle}/{num_episodes}")

        # --- Phase 1: Parallel Data Collection ---
        game_buffer = [] # Store COMPLETED games
        step_count = 0
        
        while step_count < BUFFER_SIZE:
            # Prepare batch inputs
            states_tensor = torch.FloatTensor(np.array(current_states)).to(device) # [N, 1, 15, 15] or [N, 225]
            
            # 1. Select Actions using Main Actor (for all envs)
            with torch.no_grad():
                logits = actor(states_tensor)
                values = critic(states_tensor).squeeze(-1) # [N]
            
            # 2. Handle Opponents & Invalid Masks
            actions = []
            log_probs = []
            values_list = values.cpu().numpy().tolist()
            
            # We need to process each env because some might use opponent model
            # and masks are individual.
            # Optimization: Can be vectorized if no opponent models, but hybrid is tricky.
            # Let's do a semi-vectorized approach:
            # Majority is self-play.
            
            # For simplicity in this logic, we use the main logits but substitute actions 
            # for opponent turns if needed.
            
            actions_to_step = []
            
            for i in range(NUM_ENVS):
                env = envs[i]
                player = env.current_player
                
                # Check for Valid Moves Mask (Optimized)
                # 使用numpy直接生成，比循环快得多
                valid_mask = torch.from_numpy((env.board.flatten() == 0).astype(np.float32)).to(device)
                
                # Check if we should use opponent model for this step
                # Opponent plays White (2) if opponent model is set
                is_opponent_turn = (env_opponent_actors[i] is not None) and (player == 2)
                
                if is_opponent_turn:
                    # Specific inference for this opponent
                    with torch.no_grad():
                        # Single state inference for opponent
                         opp_s = torch.FloatTensor(current_states[i]).unsqueeze(0).to(device)
                         opp_logits = env_opponent_actors[i](opp_s)
                         opp_masked = opp_logits + (1 - valid_mask) * -1e9
                         opp_dist = torch.distributions.Categorical(logits=opp_masked)
                         action = opp_dist.sample().item()
                         # We don't care about log_prob for opponent moves in buffer (usually)
                         # but we store it for consistency
                         log_prob = opp_dist.log_prob(torch.tensor(action).to(device)).item()
                else:
                    # Use main actor logits computed in batch
                    # Apply mask
                    env_logits = logits[i]
                    masked_logits = env_logits + (1 - valid_mask) * -1e9
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    action = dist.sample().item()
                    log_prob = dist.log_prob(torch.tensor(action).to(device)).item()

                actions_to_step.append(action)
                
                # Store step info in history
                histories[i].append({
                    'state': current_states[i], # numpy array
                    'action': action,
                    'reward': 0, # Placeholder
                    'value': values_list[i],
                    'log_prob': log_prob,
                    'mask': valid_mask.cpu().numpy(),
                    'player': player,
                    'next_state': None, # Filled later
                    'done': False
                })

            # 3. Step Environments
            for i in range(NUM_ENVS):
                action = actions_to_step[i]
                next_state_raw, reward, done, truncated, _ = envs[i].step(action)
                
                # Fill next state in history
                # Note: next_state for storage should be normalized from CURRENT player perspective before switch?
                # Actually, standard PPO stores s_t, a_t, r_t, s_{t+1}.
                # s_{t+1} is usually the state observed by the agent at t+1.
                # In self-play, s_{t+1} is opponent's state? 
                
                # Get normalized state for the NEXT player (whoever that is)
                next_player = envs[i].current_player
                norm_next_state = envs[i].get_normalized_state(next_player)
                
                # Update history
                histories[i][-1]['next_state'] = norm_next_state
                histories[i][-1]['done'] = done or truncated
                
                # Update current state for next loop
                current_states[i] = norm_next_state
                current_players[i] = next_player
                
                step_count += 1
                
                if done or truncated:
                    # --- Game Finished ---
                    # 1. Calculate Rewards & GAE for this game
                    full_game = histories[i]
                    
                    winner = envs[i].winner
                    
                    # Compute Final Rewards
                    rewards = np.zeros(len(full_game))
                    players = [step['player'] for step in full_game]
                    
                    # 基础步骤奖励 (dense rewards 已经在 step 中返回并存储在 full_game['reward'])
                    # 这里我们先把 env 返回的 immediate reward 提取出来
                    raw_rewards = np.array([step['reward'] for step in full_game])
                    rewards += raw_rewards # 累加中间奖励 (连3/连4)

                    if winner != 0:
                        black_indices = [idx for idx, p in enumerate(players) if p == 1]
                        white_indices = [idx for idx, p in enumerate(players) if p == 2]
                        
                        if winner == 1: # Black wins
                            if black_indices: rewards[black_indices[-1]] += 1.0 # 叠加胜利奖励
                            if white_indices: rewards[white_indices[-1]] -= 1.0 # 叠加失败惩罚
                        else: # White wins
                            if white_indices: rewards[white_indices[-1]] += 1.0
                            if black_indices: rewards[black_indices[-1]] -= 1.0
                    
                    # Compute GAE
                    gae = 0
                    advantages = np.zeros(len(full_game))
                    
                    # Need next values. Since we only have values for s_t, we need v(s_{t+1})
                    # We can approximate or run a quick inference. 
                    # For terminal state, value is 0.
                    # For non-terminal, we need V(s'). 
                    # Optimization: We could have stored values in history, but we need V(next_state).
                    # Let's simply run batch inference for the whole trajectory next_states?
                    # Or better: V(s_{t+1}) is simply V(s) of the next step in history!
                    
                    # Extract values from history
                    traj_values = np.array([step['value'] for step in full_game])
                    # We need value for the state AFTER the last step (terminal). It is 0.
                    traj_values = np.append(traj_values, 0.0)
                    
                    for t in reversed(range(len(full_game))):
                        r = rewards[t]
                        v = traj_values[t]
                        
                        # Logic:
                        # If I played and game ended -> Terminal.
                        # If I played and game continues -> Next state is Opponent's turn.
                        # V(next) is Opponent's advantage.
                        # My Advantage = r - gamma * V_opponent(next) - V_mine(current)
                        
                        is_loser_last_move = (r == -1.0)
                        
                        if full_game[t]['done'] or is_loser_last_move:
                             delta = r - v
                             gae = delta
                        else:
                             # Next value is traj_values[t+1]
                             nv = traj_values[t+1]
                             delta = r - gamma * nv - v
                             gae = delta - gamma * LAMBDA * gae
                        
                        advantages[t] = gae
                        
                        # Store in step
                        full_game[t]['reward'] = r
                        full_game[t]['advantage'] = gae
                        # We also need 'return' or 'value_target' for Critic loss
                        # Target = Advantage + Value
                        full_game[t]['value_target'] = gae + v

                    # 2. Filter data (Remove opponent steps if fixed opponent)
                    # And unpack to flat list
                    is_fixed_opp = (env_opponent_actors[i] is not None)

                    for step in full_game:
                        if is_fixed_opp and step['player'] == 2:
                            continue # Skip fixed opponent data
                        game_buffer.append(step)

                    # Collect stats for current cycle
                    cycle_rewards.append(rewards.sum())
                    cycle_lengths.append(len(full_game))
                    
                    # 3. Reset Env
                    histories[i] = []
                    s, _ = envs[i].reset()
                    current_states[i] = envs[i].get_normalized_state(envs[i].current_player)
                    current_players[i] = envs[i].current_player
                    
                    # Re-roll opponent
                    if len(opponent_pool) > 0 and np.random.random() < USE_OPPONENT_PROB:
                        opp_idx = np.random.randint(len(opponent_pool))
                        opp_actor = network.ActorNetwork(state_size, action_size, num_blocks=6, hidden_dim=128).to(device)
                        opp_actor.load_state_dict(opponent_pool[opp_idx])
                        opp_actor.eval()
                        env_opponent_actors[i] = opp_actor
                    else:
                        env_opponent_actors[i] = None

        # --- Phase 2: Vectorized Data Augmentation (GPU/Tensor) ---
        if len(game_buffer) == 0: continue
        
        # Convert buffer to tensors
        # Keys: state, action, mask, value_target, advantage, log_prob
        
        b_states = torch.FloatTensor(np.array([x['state'] for x in game_buffer])).to(device)
        b_actions = torch.LongTensor([x['action'] for x in game_buffer]).to(device)
        b_masks = torch.FloatTensor(np.array([x['mask'] for x in game_buffer])).to(device)
        b_targets = torch.FloatTensor([x['value_target'] for x in game_buffer]).to(device)
        b_advantages = torch.FloatTensor([x['advantage'] for x in game_buffer]).to(device)
        b_old_log_probs = torch.FloatTensor([x['log_prob'] for x in game_buffer]).to(device)
        
        # Reshape for rotation [N, 3, 15, 15]
        # Network expects [N, 3, 15, 15] usually
        if b_states.dim() == 2:
             b_states = b_states.view(-1, 3, 15, 15)
        # elif b_states.dim() == 3:
             # b_states = b_states.unsqueeze(1) # OLD One Channel Code
             # pass 

        # Create Action Maps for rotation [N, 1, 15, 15]
        b_action_maps = torch.zeros(len(game_buffer), 225).to(device)
        b_action_maps.scatter_(1, b_actions.unsqueeze(1), 1.0)
        b_action_maps = b_action_maps.view(-1, 1, 15, 15)
        
        b_masks = b_masks.view(-1, 1, 15, 15)
        
        # Start Augmentation List
        aug_states = [b_states]
        aug_act_maps = [b_action_maps]
        aug_masks = [b_masks]
        # These are invariant to rotation (scalar)
        aug_targets = [b_targets] 
        aug_advs = [b_advantages]
        aug_probs = [b_old_log_probs]

        # Rotate 90, 180, 270
        for k in [1, 2, 3]:
            # rot90 dims is (2,3) for N,C,H,W
            aug_states.append(torch.rot90(b_states, k, [2, 3]))
            aug_act_maps.append(torch.rot90(b_action_maps, k, [2, 3]))
            aug_masks.append(torch.rot90(b_masks, k, [2, 3]))
            aug_targets.append(b_targets)
            aug_advs.append(b_advantages)
            aug_probs.append(b_old_log_probs)

        # Flip
        flip_states = torch.flip(b_states, [3]) # Flip width
        flip_act_maps = torch.flip(b_action_maps, [3])
        flip_masks = torch.flip(b_masks, [3])
        
        aug_states.append(flip_states)
        aug_act_maps.append(flip_act_maps)
        aug_masks.append(flip_masks)
        aug_targets.append(b_targets)
        aug_advs.append(b_advantages)
        aug_probs.append(b_old_log_probs)
        
        # Rotate Flipped
        for k in [1, 2, 3]:
            aug_states.append(torch.rot90(flip_states, k, [2, 3]))
            aug_act_maps.append(torch.rot90(flip_act_maps, k, [2, 3]))
            aug_masks.append(torch.rot90(flip_masks, k, [2, 3]))
            aug_targets.append(b_targets)
            aug_advs.append(b_advantages)
            aug_probs.append(b_old_log_probs)
            
        # Concatenate all
        all_states = torch.cat(aug_states, dim=0)
        all_act_maps = torch.cat(aug_act_maps, dim=0)
        all_masks = torch.cat(aug_masks, dim=0)
        all_targets = torch.cat(aug_targets, dim=0)
        all_advs = torch.cat(aug_advs, dim=0)
        all_probs = torch.cat(aug_probs, dim=0)
        
        # Convert action maps back to indices
        all_act_maps_flat = all_act_maps.view(all_act_maps.size(0), -1)
        all_actions = torch.argmax(all_act_maps_flat, dim=1)
        
        all_masks_flat = all_masks.view(all_masks.size(0), -1)
        
        # Normalize Advantages
        all_advs = (all_advs - all_advs.mean()) / (all_advs.std() + 1e-8)

        # --- Phase 3: PPO Update ---
        actor.train()
        critic.train()

        dataset_size = all_states.size(0)
        indices = torch.arange(dataset_size).to(device)

        for _ in range(K_EPOCHS):
            indices = indices[torch.randperm(dataset_size)]

            for start in range(0, dataset_size, BATCH_SIZE):
                end = start + BATCH_SIZE
                batch_idx = indices[start:end]

                if len(batch_idx) < 2: continue

                b_s = all_states[batch_idx]
                b_a = all_actions[batch_idx]
                b_t = all_targets[batch_idx]
                b_adv = all_advs[batch_idx]
                b_old_p = all_probs[batch_idx]
                b_m = all_masks_flat[batch_idx]

                # Logic
                logits = actor(b_s)
                masked_logits = logits + (1 - b_m) * -1e9
                dist = torch.distributions.Categorical(logits=masked_logits)
                new_log_probs = dist.log_prob(b_a)
                dist_entropy = dist.entropy().mean()

                ratios = torch.exp(new_log_probs - b_old_p)
                surr1 = ratios * b_adv
                surr2 = torch.clamp(ratios, 1.0 - 0.2, 1.0 + 0.2) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * dist_entropy

                # Critic
                current_values = critic(b_s).squeeze()
                critic_loss = 0.5 * F.mse_loss(current_values, b_t)

                # Step
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()

        # Record average stats for this update cycle
        if len(cycle_rewards) > 0:
            avg_reward = np.mean(cycle_rewards)
            avg_length = np.mean(cycle_lengths)
            episode_rewards.append(avg_reward)
            episode_lengths.append(avg_length)
        else:
            # No games completed in this cycle, use NaN to indicate no data
            episode_rewards.append(np.nan)
            episode_lengths.append(np.nan)

        # Update opponent pool
        if update_cycle > 0 and update_cycle % SAVE_POOL_INTERVAL == 0:
            # Save to CPU to avoid VRAM explosion
            cpu_state_dict = {k: v.cpu() for k, v in actor.state_dict().items()}
            opponent_pool.append(cpu_state_dict)
            
            if len(opponent_pool) > MAX_POOL_SIZE:
                opponent_pool.pop(0)
                
            print(f"Opponent pool updated, size: {len(opponent_pool)} (Stored on CPU)")

        # Update learning rate
        actor_scheduler.step()
        critic_scheduler.step()

        # Periodic Save (Save every 100 cycles)
        if update_cycle % 100 == 0:
            print(f"Saving checkpoint at cycle {update_cycle}...")
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
            }, 'model/gomoku_ppo_model.pth')
            
    # Final Save
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic_state_dict': critic.state_dict(),
    }, 'model/gomoku_ppo_model.pth')
    
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
    
    # 检测actor所在的device
    device = next(actor.parameters()).device
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
                state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0).to(device)
                # 获取合法动作掩码
                valid_mask = torch.from_numpy((env.board.flatten() == 0).astype(np.float32)).to(device)
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
    

            
