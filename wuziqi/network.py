import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    残差块 (ResNet Basic Block)
    """
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += x  # 残差连接
        return F.relu(out)


class ActorNetwork(nn.Module):
    """
    Actor网络（策略网络）- 基于ResNet

    用于五子棋环境，输入棋盘状态（视角归一化后），输出每个动作的概率分布
    输入范围：[-1, 1]（-1=对手棋子，0=空格，1=我方棋子）
    """

    def __init__(self, state_size: int = 225, action_size: int = 225, num_blocks: int = 4, hidden_dim: int = 64):
        super(ActorNetwork, self).__init__()

        self.in_planes = hidden_dim
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # 残差块层
        self.layer1 = self._make_layer(hidden_dim, num_blocks, stride=1)

        # Policy Head: 1x1 卷积压缩通道数，大幅减少参数量
        # 128通道 -> 4通道
        self.policy_conv = nn.Conv2d(hidden_dim, 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.policy_bn = nn.BatchNorm2d(4)
        
        # 全连接层: 4 * 15 * 15 -> action_size
        self.fc = nn.Linear(4 * 15 * 15, action_size)

    def _make_layer(self, planes, num_blocks, stride):
        """创建残差块层"""
        layers = []
        for _ in range(num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入状态 [batch_size, 3, 15, 15]

        Returns:
            logits: 动作logits [batch_size, action_size]（未经过softmax）
        """
        # 如果输入是扁平的，或者通道不正确，进行调整
        # 这里假设输入通常是正确的维度 (N, 3, 15, 15)
        # 如果是 (N, C, H, W) 且 C=1 (以前的代码), 这会报错，因为 conv1 期望 3 通道。
        # 上游 PPO.py 应该保证传入的是 (N, 3, 15, 15)
        
        # 兼容性处理：如果只有2维 (Batch, Flattened)
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 3, 15, 15)
        elif x.dim() == 3:
             # 单个样本 (3, 15, 15) -> (1, 3, 15, 15)
             x = x.unsqueeze(0)

        # 初始卷积
        out = F.relu(self.bn1(self.conv1(x)))

        # 残差块
        out = self.layer1(out)

        # Policy Head
        out = F.relu(self.policy_bn(self.policy_conv(out)))
        
        # 展平
        out = out.view(out.size(0), -1)

        # 全连接层输出logits
        return self.fc(out)


class CriticNetworkForStateValue(nn.Module):
    """
    Critic网络（价值网络）- 基于ResNet

    用于五子棋环境，输入棋盘状态（视角归一化后），输出状态价值V(s)
    输入范围：[-1, 1]（-1=对手棋子，0=空格，1=我方棋子）
    """

    def __init__(self, state_size: int = 225, num_blocks: int = 4, hidden_dim: int = 64):
        super(CriticNetworkForStateValue, self).__init__()

        self.in_planes = hidden_dim
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)

        # 残差块层
        self.layer1 = self._make_layer(hidden_dim, num_blocks, stride=1)

        # Value Head: 1x1 卷积压缩通道
        # 128通道 -> 2通道
        self.value_conv = nn.Conv2d(hidden_dim, 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.value_bn = nn.BatchNorm2d(2)
        
        # 全连接层
        self.fc1 = nn.Linear(2 * 15 * 15, 64)
        self.fc2 = nn.Linear(64, 1)

    def _make_layer(self, planes, num_blocks, stride):
        """创建残差块层"""
        layers = []
        for _ in range(num_blocks):
            layers.append(BasicBlock(self.in_planes, planes, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入状态 [batch_size, 3, 15, 15]

        Returns:
            value: 状态价值 [batch_size, 1]
        """
        # 如果输入是扁平的，转换为2D
        if x.dim() == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 3, 15, 15)
        elif x.dim() == 3:
             # 单个样本 (3, 15, 15) -> (1, 3, 15, 15)
             x = x.unsqueeze(0)
        
        # 初始卷积
        out = F.relu(self.bn1(self.conv1(x)))

        # 残差块
        out = self.layer1(out)

        # Value Head
        out = F.relu(self.value_bn(self.value_conv(out)))
        
        # 展平
        out = out.view(out.size(0), -1)

        # 全连接层
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        return torch.tanh(out)


# 测试网络
if __name__ == "__main__":
    print("测试ResNet Actor和Critic网络...")

    # 创建网络
    state_size = 225  # 15x15棋盘
    action_size = 225  # 225个位置

    actor = ActorNetwork(state_size, action_size)
    critic = CriticNetworkForStateValue(state_size)

    # 测试输入（3通道）
    batch_size = 4
    state = torch.randn(batch_size, 3, 15, 15)  # 随机噪声

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
    print("\n测试扁平输入 (3*15*15):")
    state_flat = torch.randn(batch_size, 3 * 15 * 15)
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