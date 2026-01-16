import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional


class GomokuEnv(gym.Env):
    """
    五子棋环境 (Gomoku/Gobang Environment)

    棋盘大小：15x15
    玩家：1（黑棋）和2（白棋）
    胜利条件：任意方向连成5个或更多同色棋子
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, board_size: int = 15, render_mode: Optional[str] = None, record_frames: bool = False):
        super().__init__()
        self.board_size = board_size
        self.render_mode = render_mode
        self.record_frames = record_frames

        # 观察空间：15x15的棋盘，每个位置有3种状态（0=空，1=黑棋，2=白棋）
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(board_size, board_size), dtype=np.int8
        )

        # 动作空间：0到224（15*15-1），表示在哪个位置落子
        self.action_space = spaces.Discrete(board_size * board_size)

        # 游戏状态
        self.board = None
        self.current_player = None
        self.last_move = None
        self.done = False
        self.winner = None

        # 记录帧用于生成GIF
        self.frames = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        重置环境
        """
        super().reset(seed=seed)

        # 初始化棋盘
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 黑棋先手
        self.last_move = None
        self.done = False
        self.winner = None

        # 清空帧记录
        self.frames = []

        # 如果启用了记录，保存初始状态
        if self.record_frames:
            self.frames.append(self._render_rgb_array())

        return self.board.copy(), {"current_player": self.current_player}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一步动作

        Returns:
            observation: 新的棋盘状态
            reward: 奖励值（基于结果的奖励：中间步0，胜利+1，失败-1，平局0）
            terminated: 游戏是否结束（胜利/失败）
            truncated: 是否被截断（平局）
            info: 额外信息
        """
        if self.done:
            return self.board.copy(), 0.0, True, False, {}

        # 将动作转换为坐标
        row = action // self.board_size
        col = action % self.board_size

        # 检查动作是否合法
        if not self._is_valid_move(row, col):
            # 非法动作，返回当前状态和0奖励（让PPO处理非法动作）
            return self.board.copy(), 0.0, False, False, {}

        # 执行落子
        self.board[row, col] = self.current_player
        self.last_move = (row, col)

        # 检查是否胜利
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            reward = 1.0  # 胜利奖励
            return self.board.copy(), reward, True, False, {"winner": self.winner}

        # 检查是否平局
        if self._is_board_full():
            self.done = True
            self.winner = 0  # 平局
            reward = 0.0  # 平局奖励
            return self.board.copy(), reward, False, True, {"winner": self.winner}

        # 中间步奖励为0
        reward = 0.0

        # 切换玩家
        self.current_player = 3 - self.current_player  # 1->2, 2->1

        # 如果启用了记录，保存当前状态
        if self.record_frames:
            self.frames.append(self._render_rgb_array())

        return self.board.copy(), reward, False, False, {"current_player": self.current_player}

    def _is_valid_move(self, row: int, col: int) -> bool:
        """检查落子是否合法"""
        return 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row, col] == 0

    def _check_win(self, row: int, col: int) -> bool:
        """检查是否胜利"""
        player = self.board[row, col]
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 水平、垂直、对角线、反对角线

        for dr, dc in directions:
            count = 1  # 当前落子算1个

            # 正方向计数
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc

            # 反方向计数
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return True

        return False

    def _is_board_full(self) -> bool:
        """检查棋盘是否已满"""
        return np.all(self.board != 0)

    def render(self):
        """渲染环境"""
        if self.render_mode == "human":
            self._render_text()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _render_text(self):
        """文本渲染"""
        print("\n  ", end="")
        for i in range(self.board_size):
            print(f"{i:2}", end="")
        print()

        for i in range(self.board_size):
            print(f"{i:2}", end="")
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    print(" .", end="")
                elif self.board[i, j] == 1:
                    print(" X", end="")
                else:
                    print(" O", end="")
            print()

        if self.done:
            if self.winner == 0:
                print("\n平局！")
            else:
                winner_text = "黑棋" if self.winner == 1 else "白棋"
                print(f"\n{winner_text}获胜！")

    def _render_rgb_array(self):
        """RGB数组渲染"""
        import matplotlib.pyplot as plt
        import io

        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制棋盘背景
        ax.set_facecolor('#DEB887')  # 木质颜色
        ax.set_xlim(-0.5, self.board_size - 0.5)
        ax.set_ylim(-0.5, self.board_size - 0.5)
        ax.set_aspect('equal')

        # 绘制网格线
        for i in range(self.board_size):
            ax.axhline(i, color='black', linewidth=1)
            ax.axvline(i, color='black', linewidth=1)

        # 绘制棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 1:  # 黑棋
                    circle = plt.Circle((j, i), 0.4, color='black', ec='gray', linewidth=0.5)
                    ax.add_patch(circle)
                elif self.board[i, j] == 2:  # 白棋
                    circle = plt.Circle((j, i), 0.4, color='white', ec='gray', linewidth=0.5)
                    ax.add_patch(circle)

        # 标记最后一步
        if self.last_move is not None:
            row, col = self.last_move
            ax.plot(col, row, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1)

        ax.set_title(f"Gomoku - Current Player: {'Black' if self.current_player == 1 else 'White'}")
        ax.set_xticks([])
        ax.set_yticks([])

        # 转换为RGB数组
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        buf.seek(0)

        from PIL import Image
        img = Image.open(buf)
        return np.array(img)

    def save_gif(self, filepath: str, duration: float = 1000):
        """
        保存游戏过程为GIF

        Parameters:
        -----------
        filepath : str
            保存路径（如 'game.gif'）
        duration : float
            每帧显示时间（毫秒），默认1000ms（1秒）
        """
        if not self.frames:
            print("没有可用的帧记录，无法生成GIF")
            return

        try:
            import imageio
            from pathlib import Path

            # 确保目录存在
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            # 保存GIF，duration单位是毫秒
            imageio.mimsave(filepath, self.frames, duration=duration)
            print(f"GIF已保存到: {Path(filepath).absolute()}")
        except ImportError:
            print("需要安装imageio库: pip install imageio")
        except Exception as e:
            print(f"保存GIF时出错: {e}")

    def get_normalized_state(self, player):
        """
        获取视角归一化后的状态

        Args:
            player: 当前玩家（1或2）

        Returns:
            normalized_state: np.ndarray
                - 1: 当前玩家的棋子
                - -1: 对手的棋子
                - 0: 空格
        """
        normalized = np.zeros_like(self.board, dtype=np.float32)

        # 当前玩家的棋子标记为1
        normalized[self.board == player] = 1

        # 对手的棋子标记为-1
        opponent = 3 - player
        normalized[self.board == opponent] = -1

        # 空格保持为0
        normalized[self.board == 0] = 0

        return normalized

    def close(self):
        """关闭环境"""
        pass


# 测试环境
if __name__ == "__main__":
    # 测试1: 基本功能测试
    print("=" * 60)
    print("测试1: 基本功能测试")
    print("=" * 60)

    env = GomokuEnv(render_mode="human")

    print("测试五子棋环境...")
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")

    state, info = env.reset()
    env.render()

    # 模拟几步
    print("\n模拟对局...")
    done = False
    step_count = 0

    while not done and step_count < 10:
        # 获取合法动作
        legal_actions = [i for i in range(env.action_space.n) if env._is_valid_move(i // env.board_size, i % env.board_size)]

        if not legal_actions:
            break

        # 随机选择一个合法动作
        action = np.random.choice(legal_actions)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"\n第{step_count + 1}步:")
        print(f"动作: {action} -> 位置 ({action // env.board_size}, {action % env.board_size})")
        print(f"奖励: {reward:.2f}")
        env.render()

        step_count += 1

    env.close()
    print("\n测试完成！")

    # 测试2: GIF生成测试
    print("\n" + "=" * 60)
    print("测试2: GIF生成测试")
    print("=" * 60)

    env = GomokuEnv(render_mode="rgb_array", record_frames=True)

    print("\n开始记录对局...")
    state, info = env.reset()
    done = False
    step_count = 0

    while not done and step_count < 20:
        # 获取合法动作
        legal_actions = [i for i in range(env.action_space.n) if env._is_valid_move(i // env.board_size, i % env.board_size)]

        if not legal_actions:
            break

        # 随机选择一个合法动作
        action = np.random.choice(legal_actions)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        if step_count % 5 == 0:
            print(f"已记录 {step_count} 步...")

    print(f"对局结束，共 {step_count} 步")
    print(f"共记录 {len(env.frames)} 帧")

    # 保存GIF
    env.save_gif("images/gomoku_game.gif", duration=1000)

    env.close()
    print("\nGIF测试完成！")
