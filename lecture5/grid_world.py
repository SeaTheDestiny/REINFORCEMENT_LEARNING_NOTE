import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    """
    Grid World Environment Class

    Environment features:
    - Contains forbidden areas that cannot be entered
    - Bottom-right corner is the terminal goal state
    - Each state has a reward value (reward matrix)
    - Action space: 5 discrete actions (stop, up, down, left, right)
    """

    # Action definitions: 0=stop, 1=up, 2=down, 3=left, 4=right
    ACTIONS = {
        0: (0, 0),   # stop
        1: (-1, 0),  # up
        2: (1, 0),   # down
        3: (0, -1),  # left
        4: (0, 1),   # right
    }

    def __init__(self, m, n, forbidden_ratio=0.2, seed=None):
        """
        Initialize grid world environment

        Parameters:
            m: int, number of grid rows
            n: int, number of grid columns
            forbidden_ratio: float, proportion of forbidden areas
            seed: int, random seed (for reproducibility)
        """
        self.m = m  # Number of grid rows
        self.n = n  # Number of grid columns

        if seed is not None:
            np.random.seed(seed)

        # Policy: action probability distribution for each state, shape=(m, n, 5)
        # Initialize as uniform random policy
        self.policy = np.ones((m, n, 5), dtype=float) / 5.0

        # State value function: value of each state, shape=(m, n)
        self.state_value = np.zeros((m, n), dtype=float)

        # Terminal state (goal state): bottom-right corner
        self.goal = (m - 1, n - 1)

        # Forbidden area marker: True means the state cannot be entered
        self.forbidden = np.zeros((m, n), dtype=bool)
        self._generate_forbidden(forbidden_ratio)

        # Reward matrix: immediate reward value for each state
        self.reward = np.zeros((m, n), dtype=float)
        self._init_reward()

    def _generate_forbidden(self, ratio):
        """
        Randomly generate forbidden areas (excluding goal state)

        Parameters:
            ratio: float, proportion of forbidden areas
        """
        num_cells = self.m * self.n
        num_forbidden = int(num_cells * ratio)

        # Candidate positions: all non-goal state positions
        candidates = [
            (i, j)
            for i in range(self.m)
            for j in range(self.n)
            if (i, j) != self.goal
        ]

        # Randomly select forbidden areas
        chosen = np.random.choice(len(candidates), num_forbidden, replace=False)
        for idx in chosen:
            i, j = candidates[idx]
            self.forbidden[i, j] = True

    def _init_reward(self):
        """
        Initialize reward matrix

        Reward rules:
        - Normal states: 0
        - Forbidden states: -1
        - Goal state: 10
        """
        self.reward[:, :] = 0.0
        self.reward[self.forbidden] = -1.0
        self.reward[self.goal] = 10.0

    def random_policy(self):
        """
        Generate random policy (random action probability distribution for each state)
        """
        self.policy = np.random.rand(self.m, self.n, 5)
        # Normalize to ensure action probabilities sum to 1 for each state
        self.policy = self.policy / self.policy.sum(axis=2, keepdims=True)

    def sample_action(self, state):
        """
        Sample action from given state according to policy distribution

        Parameters:
            state: tuple (i, j), current state coordinates

        Returns:
            int: sampled action (0-4)
        """
        i, j = state
        action_probs = self.policy[i, j]
        return np.random.choice(5, p=action_probs)

    def step(self, state, action):
        """
        Execute single step state transition

        Parameters:
            state: tuple (i, j), current state
            action: int, action to execute (0-4)

        Returns:
            tuple: (next_state, reward, done)
                - next_state: tuple, next state
                - reward: float, immediate reward
                - done: bool, whether terminal state is reached
        """
        # If already at goal state, stay in place and receive reward
        if state == self.goal:
            return state, 10.0, True

        # Calculate next state position
        dx, dy = self.ACTIONS[action]
        x, y = state
        nx, ny = x + dx, y + dy

        # Check if out of bounds
        if nx < 0 or nx >= self.m or ny < 0 or ny >= self.n:
            return state, -1.0, False

        next_state = (nx, ny)

        # Check if entering forbidden area
        if self.forbidden[nx, ny]:
            return next_state, -1.0, False

        # Check if reached goal
        if next_state == self.goal:
            return next_state, 10.0, True

        # Normal state transition
        return next_state, 0.0, False


def plot_state_value(state_value, forbidden, goal, save_path="state_value.png"):
    """
    Plot state value function heatmap

    Parameters:
        state_value: np.ndarray, state value matrix, shape=(m, n)
        forbidden: np.ndarray, forbidden area marker, shape=(m, n)
        goal: tuple, goal state coordinates (i, j)
        save_path: str, save path
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(state_value, cmap="viridis")
    plt.colorbar(im)

    m, n = state_value.shape
    for i in range(m):
        for j in range(n):
            # Forbidden areas show red X
            if forbidden[i, j]:
                plt.text(j, i, "X", ha="center", va="center", color="red", fontsize=15)
            # Goal state shows white G
            elif (i, j) == goal:
                plt.text(j, i, "G", ha="center", va="center", color="white", fontsize=15)
            # Other states show value numbers
            else:
                plt.text(j, i, f"{state_value[i, j]:.1f}",
                         ha="center", va="center", color="white", fontsize=8)

    plt.title("State Value Function")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_policy_and_value(
    state_value,
    policy,
    forbidden,
    goal,
    save_path="policy_and_value.png"
):
    """
    Plot policy probability distribution and state value (no heatmap)

    Display for each state:
    - Center: state value (black large text)
    - Surroundings: probability of each action (blue small text with arrow direction)
    - Forbidden areas: light red background
    - Goal state: light green background

    Parameters:
        state_value: np.ndarray, state value matrix, shape=(m, n)
        policy: np.ndarray, policy probability distribution, shape=(m, n, 5)
        forbidden: np.ndarray, forbidden area marker, shape=(m, n)
        goal: tuple, goal state coordinates (i, j)
        save_path: str, save path
    """
    m, n = state_value.shape

    plt.figure(figsize=(10, 8))
    
    # Set white background
    plt.gca().set_facecolor('white')

    # Draw grid lines
    for i in range(m + 1):
        plt.axhline(y=i - 0.5, color='gray', linewidth=0.5)
    for j in range(n + 1):
        plt.axvline(x=j - 0.5, color='gray', linewidth=0.5)

    for i in range(m):
        for j in range(n):
            # State value (black large text, centered)
            plt.text(
                j, i,
                f"{state_value[i, j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=13,
                fontweight="bold"
            )

            # Policy probability distribution (arranged by direction)
            action_probs = policy[i, j]

            # Up - position above center
            plt.text(j, i - 0.25, f"↑\n{action_probs[1]:.2f}",
                     ha="center", va="center", color="blue", fontsize=9, fontweight="bold")

            # Down - position below center
            plt.text(j, i + 0.25, f"↓\n{action_probs[2]:.2f}",
                     ha="center", va="center", color="blue", fontsize=9, fontweight="bold")

            # Left - position to the left of center
            plt.text(j - 0.25, i, f"←\n{action_probs[3]:.2f}",
                     ha="center", va="center", color="blue", fontsize=9, fontweight="bold")

            # Right - position to the right of center
            plt.text(j + 0.25, i, f"→\n{action_probs[4]:.2f}",
                     ha="center", va="center", color="blue", fontsize=9, fontweight="bold")

            # Stop - bottom-left position
            plt.text(j - 0.25, i + 0.25, f"○\n{action_probs[0]:.2f}",
                     ha="center", va="center", color="orange", fontsize=9, fontweight="bold")

            # Forbidden areas - fill with light red background
            if forbidden[i, j]:
                plt.gca().add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1, 1,
                        fill=True,
                        facecolor="mistyrose",
                        edgecolor="red",
                        linewidth=2
                    )
                )

            # Goal state - fill with light green background
            if (i, j) == goal:
                plt.gca().add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1, 1,
                        fill=True,
                        facecolor="lightgreen",
                        edgecolor="green",
                        linewidth=2
                    )
                )
                plt.text(
                    j, i,
                    "G",
                    ha="center",
                    va="center",
                    color="darkgreen",
                    fontsize=14,
                    fontweight="bold"
                )

    plt.title("Policy Probability Distribution")
    plt.xticks(range(n))
    plt.yticks(range(m))
    plt.xlim(-0.5, n - 0.5)
    plt.ylim(m - 0.5, -0.5)
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()



if __name__ == "__main__":
    env = GridWorld(m=5, n=6, forbidden_ratio=0.25, seed=42)

    env.random_policy()
    env.state_value = np.random.uniform(-2, 5, size=(5, 6))

    print("Forbidden area (True = forbidden):")
    print(env.forbidden)

    print("\nReward matrix:")
    print(env.reward)

    print("\nPolicy shape (states x actions):", env.policy.shape)
    print("\nPolicy probability distribution for state (0, 0):")
    print(f"  Stop: {env.policy[0, 0, 0]:.3f}")
    print(f"  Up:   {env.policy[0, 0, 1]:.3f}")
    print(f"  Down: {env.policy[0, 0, 2]:.3f}")
    print(f"  Left: {env.policy[0, 0, 3]:.3f}")
    print(f"  Right: {env.policy[0, 0, 4]:.3f}")
    print(f"  Sum:  {env.policy[0, 0].sum():.3f} (should be 1.0)")

    # Demonstrate probability sampling
    print("\nSampling actions from state (0, 0):")
    samples = []
    for _ in range(20):
        action = env.sample_action((0, 0))
        samples.append(action)
    print("Sampled actions:", samples)
    print("Action counts:", {a: samples.count(a) for a in range(5)})

    # Original state value visualization
    plot_state_value(
        env.state_value,
        env.forbidden,
        env.goal,
        save_path="state_value.png"
    )

    # New policy visualization (show probability distribution)
    plot_policy_and_value(
        state_value=env.state_value,
        policy=env.policy,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path="policy_and_value.png"
    )

    print("\nGenerated files:")
    print("- state_value.png: state value heatmap")
    print("- policy_and_value.png: policy probability distribution (no heatmap)")
