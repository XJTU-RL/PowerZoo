#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单智能体PowerZoo环境训练示例

这个脚本展示了如何使用SingleAgentPowerZooEnv进行强化学习训练
支持DQN、PPO等主流算法
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../envs/power_envs'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from powerzoo_llm.single_agent import SingleAgentPowerZooEnv, SingleAgentConfig

class DQNNetwork(nn.Module):
    """多头DQN网络，适用于MultiDiscrete动作空间"""
    
    def __init__(self, obs_dim, action_dims, hidden_dim=256):
        super().__init__()
        self.action_dims = action_dims
        
        # 共享特征提取层
        self.feature_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 为每个设备创建独立的Q值头
        self.q_heads = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim) 
            for action_dim in action_dims
        ])
    
    def forward(self, x):
        features = self.feature_net(x)
        q_values = [head(features) for head in self.q_heads]
        return q_values

class SimpleDQNAgent:
    """简化的DQN智能体，用于演示"""
    
    def __init__(self, obs_dim, action_dims, lr=1e-3, epsilon=0.1):
        self.action_dims = action_dims
        self.epsilon = epsilon
        
        # 创建网络
        self.q_net = DQNNetwork(obs_dim, action_dims)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        
    def select_action(self, obs, training=True):
        """选择动作"""
        if training and random.random() < self.epsilon:
            # 随机探索
            return [random.randint(0, dim-1) for dim in self.action_dims]
        
        # 贪婪策略
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            q_values = self.q_net(obs_tensor)
            actions = [q_vals.argmax().item() for q_vals in q_values]
            return actions
    
    def store_transition(self, obs, action, reward, next_obs, done):
        """存储经验"""
        self.memory.append((obs, action, reward, next_obs, done))
    
    def update(self, batch_size=32):
        """更新网络"""
        if len(self.memory) < batch_size:
            return 0.0
        
        # 采样批次
        batch = random.sample(self.memory, batch_size)
        obs_batch = torch.FloatTensor([t[0] for t in batch])
        action_batch = [t[1] for t in batch]
        reward_batch = torch.FloatTensor([t[2] for t in batch])
        next_obs_batch = torch.FloatTensor([t[3] for t in batch])
        done_batch = torch.BoolTensor([t[4] for t in batch])
        
        # 计算当前Q值
        current_q_values = self.q_net(obs_batch)
        current_q_selected = []
        for i, q_vals in enumerate(current_q_values):
            actions_for_head = torch.LongTensor([action_batch[j][i] for j in range(batch_size)])
            q_selected = q_vals.gather(1, actions_for_head.unsqueeze(1)).squeeze(1)
            current_q_selected.append(q_selected)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.q_net(next_obs_batch)
            next_q_max = torch.stack([q_vals.max(1)[0] for q_vals in next_q_values]).sum(0)
            target_q = reward_batch + 0.99 * next_q_max * (~done_batch)
        
        # 计算损失
        total_loss = 0
        for q_selected in current_q_selected:
            loss = nn.MSELoss()(q_selected, target_q)
            total_loss += loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

def train_dqn_example():
    """DQN训练示例"""
    print("=== DQN训练示例 ===")
    
    # 创建环境
    info = {
        'system_name': '13Bus',
        'dss_file': 'IEEE13Nodeckt_daily.dss',
        'max_episode_steps': 24,
        'reg_act_num': 33,
        'bat_act_num': 33,
        'power_w': 10.0,
        'cap_w': 1.0/33,
        'reg_w': 1.0/33,
        'soc_w': 0.0/33,
        'dis_w': 6.0/33,
        'for_LLM': True,
        'wrap_observation': True
    }
    
    try:
        env = SingleAgentPowerZooEnv(
            folder_path='/home/zhengxiaodong/exps/PowerZoo/node_systems',
            info=info,
            dss_act='discrete'
        )
        
        print(f"环境创建成功")
        print(f"观测空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")
        
        # 创建智能体
        obs_dim = env.observation_space.shape[0]
        action_dims = env.action_space.nvec.tolist()
        agent = SimpleDQNAgent(obs_dim, action_dims)
        
        print(f"智能体创建成功，动作维度: {action_dims}")
        
        # 训练循环
        num_episodes = 10
        max_steps = 50
        
        for episode in range(num_episodes):
            obs = env.reset(load_profile_idx=0)
            episode_reward = 0
            
            for step in range(max_steps):
                # 选择动作
                action = agent.select_action(obs)
                
                try:
                    # 执行动作
                    next_obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    
                    # 存储经验
                    agent.store_transition(obs, action, reward, next_obs, done)
                    
                    # 更新网络
                    loss = agent.update()
                    
                    obs = next_obs
                    
                    if done:
                        break
                        
                except Exception as e:
                    print(f"步进失败: {e}")
                    break
            
            print(f"Episode {episode+1:2d}: 奖励={episode_reward:8.3f}, 步数={step+1:2d}")
        
        print("\n训练完成！")
        
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()

def simple_random_test():
    """简单随机测试"""
    print("\n=== 简单随机测试 ===")
    
    # 使用新的配置系统
    config = SingleAgentConfig(
        circuit_name="13Bus",
        max_episode_steps=24,
        voltage_penalty_weight=1.0,
        power_loss_weight=0.1,
        discharge_penalty_weight=0.5,
        log_level="INFO"
    )
    
    try:
        env = SingleAgentPowerZooEnv(config=config)
        
        obs = env.reset(load_profile_idx=0)
        print(f"初始观测: {obs[:5]}... (显示前5个值)")
        
        total_reward = 0
        for step in range(10):
            action = env.action_space.sample()
            print(f"步骤 {step+1}: 动作={action}")
            
            try:
                obs, reward, done, info = env.step(action)
                total_reward += reward
                print(f"  奖励: {reward:.3f}, 累计: {total_reward:.3f}")
                
                if done:
                    print("  环境结束")
                    break
                    
            except Exception as e:
                print(f"  步进失败: {e}")
                break
        
        print(f"\n测试完成，总奖励: {total_reward:.3f}")
        
    except Exception as e:
        print(f"测试失败: {e}")

if __name__ == "__main__":
    print("PowerZoo单智能体环境训练示例")
    print("=" * 50)
    
    # 简单测试
    simple_random_test()
    
    # DQN训练示例
    train_dqn_example()
    
    print("\n所有测试完成！")