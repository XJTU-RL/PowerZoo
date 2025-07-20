#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""简化的训练测试脚本"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.power_envs.powerzoo.powerzoo_env import PowerZooEnv

def test_basic_env():
    """测试基本环境功能"""
    print("=== 测试PowerZoo环境 ===")
    
    # 创建环境参数
    env_args = {
        'env_name': '13Bus',
        'seed': 123456,
        'num_steps': 24,
        'num_workers': None,
        'use_plot': False,
        'do_testing': False,
        'mode': 'single',
        'useS': False,
        'use_render': False,
        'record_node': False
    }
    
    try:
        # 创建环境
        print("创建PowerZoo环境...")
        env = PowerZooEnv(env_args, rank=0)
        print(f"环境创建成功!")
        print(f"代理数量: {env.n_agents}")
        print(f"观察空间: {env.observation_space}")
        print(f"动作空间: {env.action_space}")
        
        # 重置环境
        print("\n重置环境...")
        obs = env.reset()
        print(f"观察类型: {type(obs)}")
        if isinstance(obs, list) and len(obs) > 0:
            print(f"第一个观察类型: {type(obs[0])}")
            if isinstance(obs[0], list) and len(obs[0]) > 0:
                print(f"第一个代理的观察形状: {obs[0][0].shape}")
        
        # 执行一步
        print("\n执行随机动作...")
        actions = [[env.action_space[i].sample() for i in range(env.n_agents)]]
        obs, rewards, dones, infos = env.step(actions)
        print(f"奖励: {rewards}")
        print(f"完成状态: {dones}")
        
        print("\n✅ 环境测试成功!")
        
    except Exception as e:
        print(f"\n❌ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_env()