import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    Actor网络（策略网络）

    用于五子棋环境，输入棋盘状态（视角归一化后），输出每个动作的概率分布
    输入范围：[-1, 1]（-1=对手棋子，0=空格，1=我方棋子）
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 256):
        super(ActorNetwork, self).__init__()

        # 五子棋的棋盘是15x15，state_size=225
        # 使用卷积层提取空间特征
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 全连接层
        self.fc1 = nn.Linear(128 * 15 * 15, hidden_size)
        self.bn_fc1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn_fc2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入状态 [batch_size, state_size] 或 [batch_size, 15, 15]
               输入范围：[-1, 1]

        Returns:
            logits: 动作logits [batch_size, action_size]（未经过softmax）
        """
        # 如果输入是扁平的，转换为2D
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 15, 15).unsqueeze(1)  # [batch_size, 1, 15, 15]
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # [batch_size, 1, 15, 15]

        # 卷积层 + BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层 + BatchNorm
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        # 直接返回logits，不经过softmax（支持动作屏蔽）
        return x


class CriticNetworkForStateValue(nn.Module):
    """
    Critic网络（价值网络）

    用于五子棋环境，输入棋盘状态（视角归一化后），输出状态价值V(s)
    输入范围：[-1, 1]（-1=对手棋子，0=空格，1=我方棋子）
    """

    def __init__(self, state_size: int, hidden_size: int = 256):
        super(CriticNetworkForStateValue, self).__init__()

        # 卷积层提取空间特征
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 全连接层
        self.fc1 = nn.Linear(128 * 15 * 15, hidden_size)
        self.bn_fc1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn_fc2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # Dropout层
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入状态 [batch_size, state_size] 或 [batch_size, 15, 15]
               输入范围：[-1, 1]

        Returns:
            value: 状态价值 [batch_size, 1]
        """
        # 如果输入是扁平的，转换为2D
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 15, 15).unsqueeze(1)  # [batch_size, 1, 15, 15]
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # [batch_size, 1, 15, 15]

        # 卷积层 + BatchNorm
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层 + BatchNorm
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


# 测试网络
if __name__ == "__main__":
    print("测试Actor和Critic网络...")

    # 创建网络
    state_size = 225  # 15x15棋盘
    action_size = 225  # 225个位置

    actor = ActorNetwork(state_size, action_size)
    critic = CriticNetworkForStateValue(state_size)

    # 测试输入（范围：[-1, 1]）
    batch_size = 4
    state = torch.randn(batch_size, 15, 15)  # 随机噪声，范围大致在[-1, 1]

    # 测试Actor
    print("\n测试Actor网络:")
    logits = actor(state)
    action_probs = F.softmax(logits, dim=-1)  # 手动应用softmax
    print(f"输入形状: {state.shape}")
    print(f"输入范围: [{state.min():.2f}, {state.max():.2f}]")
    print(f"输出形状: {logits.shape}")
    print(f"概率和: {action_probs.sum(dim=1)}")  # 应该是1

    # 测试Critic
    print("\n测试Critic网络:")
    value = critic(state)
    print(f"输入形状: {state.shape}")
    print(f"输出形状: {value.shape}")

    # 测试扁平输入
    print("\n测试扁平输入:")
    state_flat = torch.randn(batch_size, state_size)
    action_probs_flat = actor(state_flat)
    value_flat = critic(state_flat)
    print(f"Actor输出形状: {action_probs_flat.shape}")
    print(f"Critic输出形状: {value_flat.shape}")

    # 计算参数数量
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"\nActor网络参数数量: {actor_params:,}")
    print(f"Critic网络参数数量: {critic_params:,}")
    print(f"总参数数量: {actor_params + critic_params:,}")

    print("\n网络测试完成！")