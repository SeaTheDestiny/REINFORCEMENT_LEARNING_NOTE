from flask import Flask, render_template, jsonify, request
import torch
import numpy as np
from gomoku_env import GomokuEnv
import network

app = Flask(__name__)

# 全局变量存储游戏状态
env = None
actor = None
user_player = None  # 1 = 黑子，2 = 白子（GomokuEnv 格式）
ai_player = None    # 1 = 黑子，2 = 白子（GomokuEnv 格式）

# 调试和推理配置
DEBUG_MODE = True  # 是否打印调试信息
USE_GREEDY = False  # 是否使用贪心采样（False=采样，True=选择最高概率）

def convert_board_for_frontend(board):
    """
    将棋盘值从 GomokuEnv 格式（1=黑，2=白）转换为前端格式（1=黑，-1=白）
    """
    converted = np.copy(board)
    # 2（白棋）转换为 -1
    converted[board == 2] = -1
    return converted

def load_model():
    """加载训练好的PPO模型"""
    global actor
    try:
        checkpoint = torch.load('model/gomoku_ppo_model.pth', map_location='cpu')
        state_size = 225  # 15x15
        action_size = 225
        actor = network.ActorNetwork(state_size, action_size)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()
        return True
    except FileNotFoundError:
        return False

@app.route('/')
def index():
    """渲染主页面"""
    return render_template('index.html')

@app.route('/api/start_game', methods=['POST'])
def start_game():
    """开始新游戏"""
    global env, user_player, ai_player

    data = request.json
    player_choice = data.get('player')  # 1 = 黑子（先手），-1 = 白子（后手）
    
    # 转换前端的选择为 GomokuEnv 中的玩家值
    # 前端：1 = 黑子，-1 = 白子
    # GomokuEnv：1 = 黑子，2 = 白子
    if player_choice == 1:
        user_player = 1  # 黑子
        ai_player = 2    # AI 是白子
    else:  # player_choice == -1
        user_player = 2  # 白子
        ai_player = 1    # AI 是黑子

    env = GomokuEnv()
    state, _ = env.reset()

    # 如果用户选择白子（后手），AI先走
    if user_player == 2:
        ai_move()

    return jsonify({
        'success': True,
        'board': convert_board_for_frontend(env.board).tolist(),
        'current_player': env.current_player,
        'user_player': player_choice,  # 返回前端格式的玩家值
        'game_over': False,
        'winner': None
    })

@app.route('/api/make_move', methods=['POST'])
def make_move():
    """用户下子"""
    global env, user_player, ai_player

    data = request.json
    row = data.get('row')
    col = data.get('col')

    # 检查游戏是否结束
    if env.winner is not None or env._is_board_full():
        return jsonify({
            'success': False,
            'message': '游戏已结束'
        })

    # 检查是否轮到用户
    if env.current_player != user_player:
        return jsonify({
            'success': False,
            'message': '不是你的回合'
        })

    # 检查位置是否合法
    if not env._is_valid_move(row, col):
        return jsonify({
            'success': False,
            'message': '非法位置'
        })

    # 用户下子
    action = row * 15 + col
    next_state, reward, done, truncated, _ = env.step(action)

    # 检查游戏是否结束
    if env.winner is not None or env._is_board_full():
        winner = env.winner if env.winner is not None else 0
        return jsonify({
            'success': True,
            'board': convert_board_for_frontend(env.board).tolist(),
            'current_player': env.current_player,
            'game_over': True,
            'winner': winner,
            'message': get_winner_message(winner)
        })

    # AI下子
    ai_move()

    # 检查AI下子后游戏是否结束
    if env.winner is not None or env._is_board_full():
        winner = env.winner if env.winner is not None else 0
        return jsonify({
            'success': True,
            'board': convert_board_for_frontend(env.board).tolist(),
            'current_player': env.current_player,
            'game_over': True,
            'winner': winner,
            'message': get_winner_message(winner)
        })

    return jsonify({
        'success': True,
        'board': convert_board_for_frontend(env.board).tolist(),
        'current_player': env.current_player,
        'game_over': False,
        'winner': None
    })

def ai_move():
    """AI下子"""
    global env, actor, ai_player

    # 验证 ai_player 的值是否正确
    if DEBUG_MODE:
        print(f"[DEBUG] AI下子 - ai_player={ai_player}, env.current_player={env.current_player}")
    
    # 检查 ai_player 是否与当前玩家一致
    if env.current_player != ai_player:
        print(f"[WARNING] AI玩家不匹配！ai_player={ai_player}, 当前玩家={env.current_player}")
        return

    # 获取归一化状态（从AI视角）
    normalized_state = env.get_normalized_state(ai_player)
    
    # 调试：打印状态统计
    if DEBUG_MODE:
        print(f"[DEBUG] 归一化状态 - 我方数量={np.sum(normalized_state == 1)}, 对方数量={np.sum(normalized_state == -1)}")

    # 使用模型预测动作
    with torch.no_grad():
        state_tensor = torch.FloatTensor(normalized_state).unsqueeze(0)
        action_probs = actor(state_tensor)
        
        if DEBUG_MODE:
            # 获取 top-5 概率的动作
            top_probs, top_actions = torch.topk(action_probs, 5)
            print(f"[DEBUG] Top-5 概率: {top_probs[0].cpu().numpy()}")
        
        # 选择动作：贪心或采样
        if USE_GREEDY:
            # 贪心：选择概率最高的合法动作
            action = get_greedy_action(action_probs, env)
        else:
            # 采样：从概率分布中采样
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample().item()

    # 转换为行列
    row = action // 15
    col = action % 15
    
    if DEBUG_MODE:
        print(f"[DEBUG] 模型选择的动作: action={action}, 位置=({row},{col}), 是否合法={env._is_valid_move(row, col)}")

    # 如果AI选择了非法位置，随机选择一个合法位置
    if not env._is_valid_move(row, col):
        legal_actions = []
        for r in range(15):
            for c in range(15):
                if env._is_valid_move(r, c):
                    legal_actions.append(r * 15 + c)
        
        print(f"[WARNING] 模型选择非法位置，合法位置数={len(legal_actions)}")
        
        if legal_actions:
            action = np.random.choice(legal_actions)
            row = action // 15
            col = action % 15
            if DEBUG_MODE:
                print(f"[DEBUG] 改用随机合法位置: action={action}, 位置=({row},{col})")
        else:
            # 棋盘已满，不能下子
            print(f"[ERROR] 棋盘已满，无法下子")
            return

    # 执行AI下子
    if DEBUG_MODE:
        print(f"[DEBUG] 执行下子: action={action}, 位置=({row},{col})")
    env.step(action)
    if DEBUG_MODE:
        print(f"[DEBUG] 下子后 current_player={env.current_player}")


def get_greedy_action(action_probs, env):
    """
    选择概率最高的合法动作
    
    Args:
        action_probs: 模型输出的动作概率 (batch_size=1, action_size=225)
        env: 游戏环境
    
    Returns:
        合法动作的索引
    """
    # 将概率转换为 numpy
    probs = action_probs[0].cpu().numpy()
    
    # 获取所有合法动作
    legal_actions = []
    for r in range(15):
        for c in range(15):
            if env._is_valid_move(r, c):
                legal_actions.append(r * 15 + c)
    
    if not legal_actions:
        return 0  # 不应该发生
    
    # 找到合法动作中概率最高的
    best_action = legal_actions[0]
    best_prob = probs[best_action]
    
    for action in legal_actions[1:]:
        if probs[action] > best_prob:
            best_action = action
            best_prob = probs[action]
    
    if DEBUG_MODE:
        print(f"[DEBUG] 贪心选择: action={best_action}, prob={best_prob:.4f}")
    
    return best_action

def get_winner_message(winner):
    """获取获胜信息"""
    if winner == 0:
        return "平局！"
    elif winner == user_player:
        return "恭喜你赢了！"
    else:
        return "AI赢了！"

@app.route('/api/reset', methods=['POST'])
def reset():
    """重置游戏"""
    global env
    if env is not None:
        env.reset()
    return jsonify({'success': True})

@app.route('/api/model_status')
def model_status():
    """检查模型状态"""
    model_loaded = actor is not None
    return jsonify({
        'model_loaded': model_loaded,
        'message': '模型已加载' if model_loaded else '模型未找到，请先训练模型'
    })

if __name__ == '__main__':
    # 尝试加载模型
    if load_model():
        print("PPO模型加载成功！")
    else:
        print("警告：未找到训练好的模型 (model/gomoku_ppo_model.pth)")
        print("请先运行 PPO.py 训练模型")

    # 启动Flask应用
    print("服务器启动中...")
    print("请在浏览器中访问: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)