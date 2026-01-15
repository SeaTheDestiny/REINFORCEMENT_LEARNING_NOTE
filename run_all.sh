#!/bin/bash

# Reinforcement Learning Notes - Run All Scripts
# This script runs all the reinforcement learning algorithms in the project

set -e  # Exit on error

echo "=========================================="
echo "Reinforcement Learning Notes - Run All"
echo "=========================================="
echo ""

# Create images directories if they don't exist
mkdir -p lecture4/images
mkdir -p lecture5/images
mkdir -p lecture7/images
mkdir -p lecture8/images
mkdir -p lecture9/images
mkdir -p lecture10/images

echo "=========================================="
echo "Lecture 4: Dynamic Programming"
echo "=========================================="
echo "Running Policy Iteration..."
cd lecture4 && python policy_iteration.py
cd ..
echo "Policy Iteration completed!"
echo ""

echo "Running Value Iteration..."
cd lecture4 && python value_iteration.py
cd ..
echo "Value Iteration completed!"
echo ""

echo "=========================================="
echo "Lecture 5: Monte Carlo Methods"
echo "=========================================="
echo "Running MC Basic..."
cd lecture5 && python MC_basic.py
cd ..
echo "MC Basic completed!"
echo ""

echo "Running MC Epsilon-Greedy..."
cd lecture5 && python MC_epsilon_greedy.py
cd ..
echo "MC Epsilon-Greedy completed!"
echo ""

echo "Running MC Exploring Starts..."
cd lecture5 && python MC_exploring_starts.py
cd ..
echo "MC Exploring Starts completed!"
echo ""

echo "=========================================="
echo "Lecture 7: Temporal Difference Learning"
echo "=========================================="
echo "Running SARSA..."
cd lecture7 && python SARSA.py
cd ..
echo "SARSA completed!"
echo ""

echo "Running Q-Learning (On-Policy)..."
cd lecture7 && python Q-learning_on_policy.py
cd ..
echo "Q-Learning (On-Policy) completed!"
echo ""

echo "Running Q-Learning (Off-Policy)..."
cd lecture7 && python Q-learning_off_policy.py
cd ..
echo "Q-Learning (Off-Policy) completed!"
echo ""

echo "=========================================="
echo "Lecture 8: Function Approximation"
echo "=========================================="
echo "Running Q-Learning with Function Approximation..."
cd lecture8 && python Q_learning.py
cd ..
echo "Q-Learning with Function Approximation completed!"
echo ""

echo "Running SARSA with Function Approximation..."
cd lecture8 && python SARSA.py
cd ..
echo "SARSA with Function Approximation completed!"
echo ""

echo "Running DQN (Deep Q-Network)..."
cd lecture8/DQN && python DQN.py
cd ../..
echo "DQN completed!"
echo ""

echo "=========================================="
echo "Lecture 9: Policy Gradient Methods"
echo "=========================================="
echo "Running REINFORCE..."
cd lecture9 && python reinforce.py
cd ..
echo "REINFORCE completed!"
echo ""

echo "=========================================="
echo "Lecture 10: Actor-Critic Methods"
echo "=========================================="
echo "Running QAC (Q-Actor-Critic)..."
cd lecture10 && python QAC.py
cd ..
echo "QAC completed!"
echo ""

echo "Running A2C (Advantage Actor-Critic)..."
cd lecture10 && python A2C.py
cd ..
echo "A2C completed!"
echo ""

echo "Running A2C Off-Policy..."
cd lecture10 && python A2C_off_policy.py
cd ..
echo "A2C Off-Policy completed!"
echo ""

echo "Running PPO (Proximal Policy Optimization)..."
cd lecture10 && python PPO.py
cd ..
echo "PPO completed!"
echo ""

echo "=========================================="
echo "All algorithms completed successfully!"
echo "=========================================="
echo ""
echo "Generated images and results are saved in each lecture's 'images' directory."
echo ""