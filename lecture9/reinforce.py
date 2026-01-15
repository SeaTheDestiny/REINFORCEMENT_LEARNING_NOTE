import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from grid_world import GridWorld, plot_state_value, plot_policy_and_value

# Hyperparameters
gamma = 0.99
NUM_EPISODES = 2000
learning_rate = 1e-3
MAX_EPISODE_LENGTH = 1000
HIDDEN_DIM = 128


class PolicyNetwork(nn.Module):
    """
    Policy network: maps states to action probability distributions

    Network architecture:
    - Input: state coordinates (i, j) -> 2 dimensions
    - Hidden layer: HIDDEN_DIM neurons with ReLU activation
    - Output: probability distribution over 5 actions (after softmax)
    """

    def __init__(self, state_dim=2, action_dim=5, hidden_dim=HIDDEN_DIM):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Forward pass

        Parameters
        ----------
        state : torch.Tensor
            Input state tensor, shape=(batch_size, 2) or (2,)

        Returns
        -------
        action_probs : torch.Tensor
            Action probabilities, shape=(batch_size, 5) or (5,)
        """
        x = torch.relu(self.fc1(state))
        logits = self.fc2(x)
        action_probs = torch.softmax(logits, dim=-1)
        return action_probs


class REINFORCEAgent:
    """
    REINFORCE algorithm implementation

    Core idea:
    1. Generate complete trajectory using current policy
    2. Compute discounted cumulative returns for each timestep
    3. Update policy parameters using policy gradient theorem
    """

    def __init__(self, env, learning_rate=1e-3, gamma=0.99):
        """
        Initialize REINFORCE agent

        Parameters
        ----------
        env : GridWorld
            GridWorld environment
        learning_rate : float
            Learning rate
        gamma : float
            Discount factor
        """
        self.env = env
        self.gamma = gamma

        # Policy network
        self.policy_net = PolicyNetwork(state_dim=2, action_dim=5)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Store trajectory for one episode
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        Select action based on current policy

        Parameters
        ----------
        state : tuple (i, j)
            Current state coordinates

        Returns
        -------
        action : int
            Selected action (0-4)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)

        # Create categorical distribution and sample
        m = Categorical(action_probs)
        action = m.sample()

        # Save log probability (for gradient computation)
        self.saved_log_probs.append(m.log_prob(action))

        return action.item()

    def compute_returns(self):
        """
        Compute discounted cumulative returns for each timestep

        Formula: G_t = r_{t+1} + γ * r_{t+2} + γ^2 * r_{t+3} + ...

        Returns
        -------
        returns : list
            Returns for each timestep
        """
        returns = []
        R = 0

        # Compute from back to front (reverse traversal)
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)

        # Normalize returns (reduce variance, improve training stability)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def update_policy(self):
        """
        Update policy parameters using REINFORCE algorithm

        Core formula:
        θ ← θ + α * Σ_t ∇_θ ln π_θ(a_t|s_t) * G_t

        Where:
        - ∇_θ ln π_θ(a_t|s_t) is the gradient of policy log probability
        - G_t is the discounted cumulative return from timestep t
        """
        # Compute returns
        returns = self.compute_returns()

        # Compute policy gradient loss
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            # REINFORCE loss function: -log_prob * R
            # Note: We use negative sign because PyTorch uses gradient descent by default
            policy_loss.append(-log_prob * R)

        policy_loss = torch.stack(policy_loss).sum()

        # Gradient descent update
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Clear trajectory cache
        self.saved_log_probs = []
        self.rewards = []

        return policy_loss.item()

    def train_episode(self):
        """
        Execute one training episode

        Returns
        -------
        total_reward : float
            Total reward for this episode
        steps : int
            Number of steps in this episode
        """
        state = (0, 0)  # Start from top-left corner
        total_reward = 0
        steps = 0
        done = False

        while not done and steps < MAX_EPISODE_LENGTH:  # Prevent infinite loop
            # Select action
            action = self.select_action(state)

            # Execute action
            next_state, reward, done = self.env.step(state, action)

            # Store reward
            self.rewards.append(reward)

            # Update state and reward
            state = next_state
            total_reward += reward
            steps += 1

        # Update policy
        loss = self.update_policy()
        
        return total_reward, steps, loss
    
    def get_policy_matrix(self):
        """
        Get current policy probability distribution matrix

        Returns
        -------
        policy_matrix : np.ndarray
            Policy matrix with shape (m, n, 5)
        """
        policy_matrix = np.zeros((self.env.m, self.env.n, 5))

        for i in range(self.env.m):
            for j in range(self.env.n):
                state = (i, j)
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs = self.policy_net(state_tensor)
                policy_matrix[i, j] = action_probs.detach().numpy()

        return policy_matrix

    def get_state_value(self, num_episodes=100):
        """
        Estimate state value function through Monte Carlo sampling

        Parameters
        ----------
        num_episodes : int
            Number of sampling episodes

        Returns
        -------
        state_value : np.ndarray
            State value matrix with shape (m, n)
        """
        state_value = np.zeros((self.env.m, self.env.n))
        visit_counts = np.zeros((self.env.m, self.env.n))

        for _ in range(num_episodes):
            state = (0, 0)
            done = False
            episode_rewards = []
            episode_states = []

            while not done:
                episode_states.append(state)
                action = self.select_action(state)
                next_state, reward, done = self.env.step(state, action)
                episode_rewards.append(reward)
                state = next_state

            # Compute returns for each state
            returns = []
            R = 0
            for r in reversed(episode_rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            # Accumulate returns
            for i, s in enumerate(episode_states):
                state_value[s] += returns[i]
                visit_counts[s] += 1

        # Compute average
        visit_counts[visit_counts == 0] = 1  # Avoid division by zero
        state_value = state_value / visit_counts
        
        return state_value


def train_reinforce(env, num_episodes=NUM_EPISODES, learning_rate=learning_rate, gamma=gamma):
    """
    Train REINFORCE agent

    Parameters
    ----------
    env : GridWorld
        GridWorld environment
    num_episodes : int
        Number of training episodes (default: global constant NUM_EPISODES)
    learning_rate : float
        Learning rate (default: global constant learning_rate)
    gamma : float
        Discount factor (default: global constant gamma)

    Returns
    -------
    agent : REINFORCEAgent
        Trained agent
    rewards_history : list
        Total reward history for each episode
    """
    agent = REINFORCEAgent(env, learning_rate=learning_rate, gamma=gamma)
    rewards_history = []

    print(f"Starting REINFORCE training, {num_episodes} episodes...")
    print("=" * 60)

    for episode in range(num_episodes):
        total_reward, steps, loss = agent.train_episode()
        rewards_history.append(total_reward)

        # Print statistics every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Average reward: {avg_reward:.2f} | "
                  f"Current reward: {total_reward:.2f} | "
                  f"Steps: {steps}")

    print("=" * 60)
    print("Training completed!")

    return agent, rewards_history


def plot_training_curve(rewards_history, save_path="training_curve.png"):
    """
    Plot training curve

    Parameters
    ----------
    rewards_history : list
        Total reward history for each episode
    save_path : str
        Save path for the plot
    """
    plt.figure(figsize=(10, 5))

    # Original reward curve
    plt.plot(rewards_history, alpha=0.3, color='blue', label='Original Reward')

    # Smoothed curve (moving average)
    window_size = 50
    if len(rewards_history) >= window_size:
        smoothed = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards_history)), smoothed, 
                 color='red', linewidth=2, label=f'Moving Average (window={window_size})')
    
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Training curve saved to: {save_path}")


def visualize_results(agent, env, save_prefix="reinforce"):
    """
    Visualize training results

    Parameters
    ----------
    agent : REINFORCEAgent
        Trained agent
    env : GridWorld
        GridWorld environment
    save_prefix : str
        Save file prefix
    """
    # Get policy matrix
    policy_matrix = agent.get_policy_matrix()

    # Estimate state value function
    print("Estimating state value function...")
    state_value = agent.get_state_value(num_episodes=200)

    # Plot policy and state value
    plot_policy_and_value(
        state_value=state_value,
        policy=policy_matrix,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path=f"{save_prefix}_policy_and_value.png"
    )

    # Plot state value heatmap
    plot_state_value(
        state_value=state_value,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path=f"{save_prefix}_state_value.png"
    )

    print(f"Visualization results saved: {save_prefix}_policy_and_value.png, {save_prefix}_state_value.png")


def test_agent(agent, env, num_episodes=10):
    """
    Test trained agent

    Parameters
    ----------
    agent : REINFORCEAgent
        Trained agent
    env : GridWorld
        GridWorld environment
    num_episodes : int
        Number of test episodes
    """
    print("\n" + "=" * 60)
    print("Testing agent performance...")
    print("=" * 60)
    
    total_rewards = []
    total_steps = []
    success_count = 0
    
    for episode in range(num_episodes):
        state = (0, 0)
        total_reward = 0
        steps = 0
        done = False
        trajectory = [state]

        while not done and steps < MAX_EPISODE_LENGTH:
            action = agent.select_action(state)
            next_state, reward, done = env.step(state, action)
            trajectory.append(next_state)
            state = next_state
            total_reward += reward
            steps += 1
        
        total_rewards.append(total_reward)
        total_steps.append(steps)
        
        if state == env.goal:
            success_count += 1
            print(f"Episode {episode+1}: Successfully reached goal! | Reward: {total_reward:.2f} | Steps: {steps}")
        else:
            print(f"Episode {episode+1}: Did not reach goal | Reward: {total_reward:.2f} | Steps: {steps}")
    
    print("=" * 60)
    print(f"Success rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    # Create images directory
    import os
    os.makedirs("./images", exist_ok=True)

    # Create environment
    env = GridWorld(m=5, n=6, forbidden_ratio=0.25, seed=42)

    print("Environment Information:")
    print(f"Grid size: {env.m} x {env.n}")
    print(f"Number of forbidden areas: {env.forbidden.sum()}")
    print(f"Goal position: {env.goal}")
    print(f"Action space: 5 actions (stop, up, down, left, right)")
    print()
    print("Hyperparameters:")
    print(f"gamma: {gamma}")
    print(f"NUM_EPISODES: {NUM_EPISODES}")
    print(f"LEARNING_RATE: {learning_rate}")
    print(f"MAX_EPISODE_LENGTH: {MAX_EPISODE_LENGTH}")
    print(f"HIDDEN_DIM: {HIDDEN_DIM}")
    print()

    # Train REINFORCE agent (using global hyperparameters)
    agent, rewards_history = train_reinforce(env)

    # Plot training curve
    plot_training_curve(rewards_history, save_path="./images/reinforce_training_curve.png")

    # Visualize training results
    visualize_results(agent, env, save_prefix="./images/reinforce")

    # Test agent
    test_agent(agent, env, num_episodes=10)

    print("\nAll tasks completed! Generated files:")
    print("- ./images/reinforce_training_curve.png: Training curve")
    print("- ./images/reinforce_policy_and_value.png: Policy probability distribution and state value")
    print("- ./images/reinforce_state_value.png: State value heatmap")