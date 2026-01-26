import numpy as np
import torch


def get_symmetries(state, logits, reward):
    """
    通过旋转和翻转生成数据增强的变体

    该函数将一个棋盘状态生成8种变体（旋转0, 90, 180, 270度，以及这4种旋转后的水平翻转）。
    支持3通道输入 (3, 15, 15)

    Args:
        state: (3, 15, 15) numpy array - 3通道棋盘状态
        logits: (225,) numpy array or tensor - 动作logits
        reward: float - 奖励值

    Returns:
        list: 包含所有变体的列表，每个元素是 (state_flat, logits_flat, reward) 的元组
    """
    data = []
    state_board = state.copy() # (3, 15, 15) or (15, 15)

    # 检查维度以支持旧代码的单纯2D输入
    if state_board.ndim == 2:
        state_board = state_board[np.newaxis, :, :] # (1, 15, 15)

    # 将 logits 还原为 15x15 以便旋转
    if isinstance(logits, torch.Tensor):
        logit_board = logits.detach().cpu().numpy().reshape(15, 15)
    else:
        logit_board = np.array(logits).reshape(15, 15)

    for i in range(4):
        # 旋转 i * 90 度
        # 对于 (C, H, W)，旋转轴为 (1, 2)
        s_rot = np.rot90(state_board, i, axes=(1, 2))
        l_rot = np.rot90(logit_board, i)  # (15, 15) 默认 axes=(0,1)

        data.append((s_rot.flatten(), l_rot.flatten(), reward))

        # 翻转 (Flip Width - Axis 2)
        # 对于 (C, H, W)，翻转 Axis 2
        s_flip = np.flip(s_rot, axis=2)
        l_flip = np.fliplr(l_rot) # (15, 15) fliplr 翻转 Axis 1 (Width)
        
        data.append((s_flip.flatten(), l_flip.flatten(), reward))

    return data


def get_symmetries_with_mask(state, logits, reward, mask):
    """
    通过旋转和翻转生成数据增强的变体（包含mask）

    Args:
        state: (15, 15) numpy array - 棋盘状态
        logits: (225,) numpy array or tensor - 动作logits
        reward: float - 奖励值
        mask: (225,) numpy array or tensor - 动作掩码（1为合法，0为非法）

    Returns:
        list: 包含所有变体的列表，每个元素是 (state_flat, logits_flat, reward, mask_flat) 的元组
    """
    data = []
    state_board = state.copy()

    # 将 logits 和 mask 还原为 15x15 以便旋转
    if isinstance(logits, torch.Tensor):
        logit_board = logits.detach().cpu().numpy().reshape(15, 15)
    else:
        logit_board = np.array(logits).reshape(15, 15)

    if isinstance(mask, torch.Tensor):
        mask_board = mask.detach().cpu().numpy().reshape(15, 15)
    else:
        mask_board = np.array(mask).reshape(15, 15)

    for i in range(4):
        # 旋转 i * 90 度
        s_rot = np.rot90(state_board, i)
        l_rot = np.rot90(logit_board, i)
        m_rot = np.rot90(mask_board, i)
        data.append((s_rot.flatten(), l_rot.flatten(), reward, m_rot.flatten()))

        # 翻转
        s_flip = np.fliplr(s_rot)
        l_flip = np.fliplr(l_rot)
        m_flip = np.fliplr(m_rot)
        data.append((s_flip.flatten(), l_flip.flatten(), reward, m_flip.flatten()))

    return data


# 测试函数
if __name__ == "__main__":
    print("测试数据增强函数...")

    # 创建一个简单的测试棋盘
    test_state = np.zeros((15, 15))
    test_state[7, 7] = 1  # 中心位置
    test_state[7, 8] = -1  # 中心右侧

    # 创建测试logits
    test_logits = np.random.randn(225)

    # 测试奖励
    test_reward = 1.0

    # 测试mask
    test_mask = np.ones(225)
    test_mask[:100] = 0  # 前100个位置非法

    print(f"\n原始状态形状: {test_state.shape}")
    print(f"原始logits形状: {test_logits.shape}")
    print(f"原始mask形状: {test_mask.shape}")

    # 测试 get_symmetries
    print("\n测试 get_symmetries:")
    symmetries = get_symmetries(test_state, test_logits, test_reward)
    print(f"生成了 {len(symmetries)} 个变体")

    for i, (s, l, r) in enumerate(symmetries):
        print(f"变体 {i+1}: state={s.shape}, logits={l.shape}, reward={r}")

    # 测试 get_symmetries_with_mask
    print("\n测试 get_symmetries_with_mask:")
    symmetries_with_mask = get_symmetries_with_mask(test_state, test_logits, test_reward, test_mask)
    print(f"生成了 {len(symmetries_with_mask)} 个变体")

    for i, (s, l, r, m) in enumerate(symmetries_with_mask):
        print(f"变体 {i+1}: state={s.shape}, logits={l.shape}, reward={r}, mask={m.shape}")

    # 验证旋转后的位置是否正确
    print("\n验证旋转逻辑:")
    # 原始位置 (7, 7) 对应的索引
    original_idx = 7 * 15 + 7
    print(f"原始位置 (7, 7) 对应索引: {original_idx}")

    # 旋转90度后，(7, 7) 应该变成 (7, 7)（因为中心点旋转后不变）
    # 但 (7, 8) 应该变成 (6, 7)
    s_rot = np.rot90(test_state, 1)
    print(f"旋转90度后，(7, 7) 的值: {s_rot[7, 7]}")
    print(f"旋转90度后，(6, 7) 的值: {s_rot[6, 7]} (应该是1)")

    print("\n数据增强测试完成！")