import numpy as np
import matplotlib.pyplot as plt


class GridWorld:
    """
    Grid World with:
    - forbidden areas
    - terminal goal at bottom-right
    - reward matrix
    """

    # actions: 0=stop, 1=up, 2=down, 3=left, 4=right
    ACTIONS = {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

    def __init__(self, m, n, forbidden_ratio=0.2, seed=None):
        self.m = m
        self.n = n

        if seed is not None:
            np.random.seed(seed)

        # policy & state value
        self.policy = np.zeros((m, n), dtype=int)
        self.state_value = np.zeros((m, n), dtype=float)

        # terminal state
        self.goal = (m - 1, n - 1)

        # forbidden areas
        self.forbidden = np.zeros((m, n), dtype=bool)
        self._generate_forbidden(forbidden_ratio)

        # reward matrix
        self.reward = np.zeros((m, n), dtype=float)
        self._init_reward()

    def _generate_forbidden(self, ratio):
        """
        Randomly generate forbidden areas,
        excluding the goal.
        """
        num_cells = self.m * self.n
        num_forbidden = int(num_cells * ratio)

        candidates = [
            (i, j)
            for i in range(self.m)
            for j in range(self.n)
            if (i, j) != self.goal
        ]

        chosen = np.random.choice(len(candidates), num_forbidden, replace=False)
        for idx in chosen:
            i, j = candidates[idx]
            self.forbidden[i, j] = True

    def _init_reward(self):
        """
        Initialize reward matrix.
        """
        self.reward[:, :] = 0.0
        self.reward[self.forbidden] = -1.0
        self.reward[self.goal] = 10.0

    def random_policy(self):
        self.policy = np.random.randint(0, 5, size=(self.m, self.n))

    def step(self, state, action):
        """
        One-step transition:
        return next_state, reward, done
        """
        if state == self.goal:
            return state, 10.0, True

        dx, dy = self.ACTIONS[action]
        x, y = state
        nx, ny = x + dx, y + dy

        # out of bounds
        if nx < 0 or nx >= self.m or ny < 0 or ny >= self.n:
            return state, -1.0, False

        next_state = (nx, ny)

        # forbidden
        if self.forbidden[nx, ny]:
            return next_state, -1.0, False

        # goal
        if next_state == self.goal:
            return next_state, 10.0, True

        # normal cell
        return next_state, 0.0, False


def plot_state_value(state_value, forbidden, goal, save_path="state_value.png"):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(state_value, cmap="viridis")
    plt.colorbar(im)

    m, n = state_value.shape
    for i in range(m):
        for j in range(n):
            if forbidden[i, j]:
                plt.text(j, i, "X", ha="center", va="center", color="red", fontsize=15)
            elif (i, j) == goal:
                plt.text(j, i, "G", ha="center", va="center", color="white", fontsize=15)
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
    Plot policy arrows + state value heatmap.
    Forbidden areas are NOT terminal and still have values.
    """
    action_symbol = {
        0: "○",   # stop
        1: "↑",
        2: "↓",
        3: "←",
        4: "→",
    }

    m, n = state_value.shape

    plt.figure(figsize=(7, 6))
    im = plt.imshow(state_value, cmap="viridis")
    plt.colorbar(im)

    for i in range(m):
        for j in range(n):
            # state value
            plt.text(
                j, i,
                f"{state_value[i, j]:.1f}",
                ha="center",
                va="center",
                color="white",
                fontsize=10
            )

            # policy arrow
            symbol = action_symbol[policy[i, j]]
            plt.text(
                j, i + 0.28,
                symbol,
                ha="center",
                va="center",
                color="cyan",
                fontsize=14,
                fontweight="bold"
            )

            # forbidden area marker (still valid state)
            if forbidden[i, j]:
                plt.gca().add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1, 1,
                        fill=False,
                        edgecolor="red",
                        linewidth=2
                    )
                )

            # goal marker
            if (i, j) == goal:
                plt.text(
                    j, i - 0.28,
                    "G",
                    ha="center",
                    va="center",
                    color="yellow",
                    fontsize=14,
                    fontweight="bold"
                )

    plt.title("Policy and State Value")
    plt.xticks(range(n))
    plt.yticks(range(m))
    plt.grid(False)
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

    print("\nPolicy matrix:")
    print(env.policy)

    # Original state value visualization
    plot_state_value(
        env.state_value,
        env.forbidden,
        env.goal,
        save_path="state_value.png"
    )

    # New policy + value visualization
    plot_policy_and_value(
        state_value=env.state_value,
        policy=env.policy,
        forbidden=env.forbidden,
        goal=env.goal,
        save_path="policy_and_value.png"
    )

    print("\nGenerated files:")
    print("- state_value.png: display only state values")
    print("- policy_and_value.png: display policy arrows + state values")
