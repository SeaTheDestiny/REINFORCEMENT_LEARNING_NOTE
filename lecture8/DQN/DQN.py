import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import imageio
import random
from pathlib import Path
from collections import deque

gamma = 0.99
C = 100                    # target network update frequency (steps)
BUFFER_SIZE = 100_000
BATCH_SIZE = 64
learning_rate = 5e-4
NUM_EPISODES = 500

# ======================
# Q Network
# ======================
class QNetwork(nn.Module):
    """
    Q-Network for Deep Q-Learning.

    This network approximates the Q-value function Q(s, a) using a multi-layer
    perceptron with ReLU activations and Xavier initialization.

    Parameters
    ----------
    state_size : int
        Dimension of the state space
    action_size : int
        Dimension of the action space
    hidden_size : int, optional
        Number of neurons in hidden layers (default: 128)
    """

    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        """
        Forward pass through the Q-network.

        Parameters
        ----------
        state : torch.Tensor
            Input state tensor

        Returns
        -------
        torch.Tensor
            Q-values for all actions
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# ======================
# ε-greedy behavior policy π_b
# ======================
def select_action(q_network, state, epsilon, action_space):
    """
    Select action using epsilon-greedy policy.

    With probability epsilon, select a random action (exploration).
    Otherwise, select the action with the highest Q-value (exploitation).

    Parameters
    ----------
    q_network : QNetwork
        The Q-value network
    state : np.ndarray
        Current state
    epsilon : float
        Exploration rate (0.0 = greedy, 1.0 = random)
    action_space : gym.Space
        Action space of the environment

    Returns
    -------
    int
        Selected action
    """
    if np.random.rand() < epsilon:
        return action_space.sample()
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            return torch.argmax(q_network(state)).item()

# ======================
# Standard DQN main loop
# ======================
def DQN_learning(env):
    """
    Deep Q-Network (DQN) learning algorithm with experience replay and target network.

    This implementation includes:
    - Experience replay buffer for stable training
    - Target network for stable Q-value estimation
    - Epsilon-greedy exploration strategy

    Parameters
    ----------
    env : gym.Env
        The reinforcement learning environment

    Returns
    -------
    tuple
        (q_network, episode_rewards, episode_lengths)
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    q_network = QNetwork(state_size, action_size)
    target_network = QNetwork(state_size, action_size)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = torch.optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = deque(maxlen=BUFFER_SIZE)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    global_step = 0

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        done, truncated = False, False
        episode_reward = 0

        while not (done or truncated):
            action = select_action(q_network, state, epsilon, env.action_space)
            next_state, reward, done, truncated, _ = env.step(action)

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward

            # ======================
            # Training
            # ======================           
            if len(replay_buffer) >= BATCH_SIZE:
                batch = random.sample(replay_buffer, BATCH_SIZE)

                states = torch.FloatTensor([b[0] for b in batch])
                actions = torch.LongTensor([b[1] for b in batch]).unsqueeze(1)
                rewards = torch.FloatTensor([b[2] for b in batch]).unsqueeze(1)
                next_states = torch.FloatTensor([b[3] for b in batch])
                dones = torch.FloatTensor([b[4] for b in batch]).unsqueeze(1)

                current_q = q_network(states).gather(1, actions)

                with torch.no_grad():
                    max_next_q = target_network(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + (1 - dones) * gamma * max_next_q

                loss = F.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1
                if global_step % C == 0:
                    target_network.load_state_dict(q_network.state_dict())

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode:4d} | Reward: {episode_reward:7.2f} | Epsilon: {epsilon:.3f}")

    return q_network

# ======================
# Visualization
# ======================
def visualize_policy(q_network, env_name='LunarLander-v2', num_episodes=3, output_path='trained_policy.gif'):
    env = gym.make(env_name, render_mode='rgb_array')
    frames = []
    q_network.eval()

    for ep in range(num_episodes):
        state, _ = env.reset()
        done, truncated = False, False
        step = 0

        while not (done or truncated) and step < 500:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = torch.argmax(q_network(state_tensor)).item()

            state, _, done, truncated, _ = env.step(action)
            frames.append(env.render())
            step += 1

    env.close()
    imageio.mimsave(Path(output_path), frames, duration=33)
    print(f"GIF saved to {Path(output_path).absolute()}")

# ======================
# Main
# ======================
if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    trained_q = DQN_learning(env)
    visualize_policy(trained_q)

    