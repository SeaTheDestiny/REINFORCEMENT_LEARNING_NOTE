# Reinforcement Learning Notes

A comprehensive collection of reinforcement learning algorithm implementations, from classic dynamic programming to modern deep reinforcement learning methods.

## 🎯 Overview

This repository contains implementations of various reinforcement learning algorithms, organized by lecture topics. Each algorithm is implemented in Python with clean, well-documented code and includes visualizations of the learning process.

## 📚 Course Source

These implementations are based on the course **"Mathematical Foundation of Reinforcement Learning"** available at:

https://github.com/MathFoundationRL/Book-Mathmatical-Foundation-of-Reinforcement-Learning

The code follows the mathematical formulations and algorithmic approaches presented in the course, with additional documentation and visualizations for better understanding.

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/reinforcement-learning-notes.git
cd reinforcement-learning-notes
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Run All Algorithms

To run all algorithms in sequence:

```bash
./run_all.sh
```

### Run Individual Lectures

Navigate to the lecture directory and run the specific algorithm:

```bash
cd lecture4
python policy_iteration.py
```

## 📖 Lectures

### Lecture 4: Dynamic Programming

- **Policy Iteration**: Iterative policy evaluation and improvement
- **Value Iteration**: Simultaneous value and policy updates

### Lecture 5: Monte Carlo Methods

- **MC Basic**: First-visit Monte Carlo with policy iteration
- **MC Epsilon-Greedy**: Monte Carlo with ε-greedy exploration
- **MC Exploring Starts**: Monte Carlo with exploring starts assumption

### Lecture 7: Temporal Difference Learning

- **SARSA**: On-policy TD control
- **Q-Learning (On-Policy)**: Q-learning with ε-greedy policy
- **Q-Learning (Off-Policy)**: Off-policy Q-learning with experience replay

### Lecture 8: Function Approximation

- **Q-Learning with Function Approximation**: Linear function approximation for Q-learning
- **SARSA with Function Approximation**: Linear function approximation for SARSA
- **DQN**: Deep Q-Network with experience replay and target network

### Lecture 9: Policy Gradient Methods

- **REINFORCE**: Monte Carlo policy gradient method

### Lecture 10: Actor-Critic Methods

- **QAC**: Q-Actor-Critic with Q-value critic
- **A2C**: Advantage Actor-Critic with state-value critic
- **A2C Off-Policy**: Off-policy A2C with importance sampling
- **PPO**: Proximal Policy Optimization with GAE

## 🧠 Algorithms

### Dynamic Programming

| Algorithm | Environment | Key Features |
|-----------|-------------|--------------|
| Policy Iteration | GridWorld | Iterative policy evaluation and improvement |
| Value Iteration | GridWorld | Simultaneous value and policy updates |

### Monte Carlo Methods

| Algorithm | Environment | Exploration | Key Features |
|-----------|-------------|-------------|--------------|
| MC Basic | GridWorld | Policy-based | First-visit MC, policy iteration |
| MC Epsilon-Greedy | GridWorld | ε-greedy | Decaying ε, every-visit MC |
| MC Exploring Starts | GridWorld | Exploring starts | Greedy policy improvement |

### Temporal Difference Learning

| Algorithm | Environment | On/Off-Policy | Key Features |
|-----------|-------------|---------------|--------------|
| SARSA | GridWorld | On-policy | ε-greedy, TD(0) |
| Q-Learning (On-Policy) | GridWorld | On-policy | ε-greedy, max Q-value |
| Q-Learning (Off-Policy) | GridWorld | Off-policy | Experience replay |

### Function Approximation

| Algorithm | Environment | Function Approximator | Key Features |
|-----------|-------------|----------------------|--------------|
| Q-Learning (FA) | GridWorld | Linear | One-hot features |
| SARSA (FA) | GridWorld | Linear | One-hot features |
| DQN | LunarLander-v2 | Neural Network | Experience replay, target network |

### Policy Gradient Methods

| Algorithm | Environment | Key Features |
|-----------|-------------|--------------|
| REINFORCE | GridWorld | Monte Carlo policy gradient, return normalization |

### Actor-Critic Methods

| Algorithm | Environment | Critic Type | Key Features |
|-----------|-------------|-------------|--------------|
| QAC | LunarLander-v2 | Q-value | SARSA-style TD error |
| A2C | LunarLander-v2 | State-value | Advantage estimation |
| A2C Off-Policy | LunarLander-v2 | State-value | Importance sampling |
| PPO | LunarLander-v2 | State-value | Clipped objective, GAE |

## 🎓 Learning Path

Recommended order for learning:

1. **Lecture 4**: Start with Dynamic Programming (Policy Iteration → Value Iteration)
2. **Lecture 5**: Learn Monte Carlo methods (Basic → ε-Greedy → Exploring Starts)
3. **Lecture 7**: Understand Temporal Difference learning (SARSA → Q-Learning)
4. **Lecture 8**: Explore function approximation (Linear → Deep with DQN)
5. **Lecture 9**: Study policy gradient methods (REINFORCE)
6. **Lecture 10**: Master Actor-Critic methods (QAC → A2C → PPO)

## 🙏 Acknowledgments

- **Course**: Mathematical Foundation of Reinforcement Learning
  - Repository: https://github.com/MathFoundationRL/Book-Mathmatical-Foundation-of-Reinforcement-Learning




