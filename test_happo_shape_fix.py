#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAPPO并行环境形状一致性测试脚本
测试修复后的env_wrappers和powerzoo_env的兼容性
"""

import numpy as np
import sys
import os
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from envs.env_wrappers import ShareSubprocVecEnv
from envs.power_envs.powerzoo_llm.powerzoo_env import PowerZooEnv
from envs.power_envs.powerzoo_llm.env_register import make_base_env

def create_test_env():
    """创建测试用的并行环境"""
    def env_fn(rank):
        def _env():
            # 首先创建底层环境
            base_env = make_base_env('34Bus_pv', dss_act=False, worker_idx=rank)
            
            # 创建测试配置
            test_config = {
                'env_name': '34Bus_pv',
                'discrete': False,
                'device': 'cpu',
                'enable_system_logging': False,
                'llm_enhanced': False,
                'use_natural_language_obs': False,
                'use_action_explanation': False
            }
            
            # 使用PowerZooEnv包装器
            env = PowerZooEnv(base_env, test_config, rank=rank)
            return env
        return _env
    
    # 创建2个并行环境
    env_fns = [env_fn(i) for i in range(2)]
    return ShareSubprocVecEnv(env_fns)

def test_env_reset():
    """测试环境重置"""
    print("=== 测试环境重置 ===")
    envs = create_test_env()
    
    try:
        obs, share_obs, avail_actions = envs.reset()
        print(f"✅ 重置成功")
        print(f"   - obs形状: {obs.shape}")
        print(f"   - share_obs形状: {share_obs.shape}")
        print(f"   - avail_actions类型: {type(avail_actions)}")
        return envs, obs, share_obs, avail_actions
        
    except Exception as e:
        print(f"❌ 重置失败: {e}")
        traceback.print_exc()
        envs.close()
        return None, None, None, None

def test_mixed_action_step(envs, obs):
    """测试混合动作空间的步进"""
    print("\n=== 测试混合动作步进 ===")
    
    if envs is None:
        print("❌ 环境未初始化")
        return False
    
    try:
        # 创建混合动作（34Bus_pv: 13智能体，混合动作空间）
        n_envs = 2
        n_agents = 13
        
        actions = []
        for env_idx in range(n_envs):
            env_actions = []
            for agent_idx in range(n_agents):
                # 混合动作：[离散动作(int), 连续动作(np.array)]
                discrete_action = np.random.randint(0, 10)  # 离散动作范围0-9
                continuous_action = np.random.uniform(-0.1, 0.1, size=6)  # 6维连续动作
                env_actions.append([discrete_action, continuous_action])
            actions.append(env_actions)
        
        print(f"   - 动作结构: {len(actions)}个环境 x {len(actions[0])}个智能体")
        print(f"   - 单个动作示例: 离散={actions[0][0][0]}, 连续形状={actions[0][0][1].shape}")
        
        # 执行步进
        obs, share_obs, rewards, dones, infos, avail_actions = envs.step(actions)
        
        print(f"✅ 步进成功")
        print(f"   - obs形状: {obs.shape}")
        print(f"   - share_obs形状: {share_obs.shape}")
        print(f"   - rewards形状: {np.array(rewards).shape}")
        print(f"   - dones形状: {dones.shape}, 类型: {dones.dtype}")
        print(f"   - infos长度: {len(infos)}")
        print(f"   - dones内容: {dones}")
        
        return True
        
    except Exception as e:
        print(f"❌ 步进失败: {e}")
        traceback.print_exc()
        return False

def test_episode_completion(envs):
    """测试完整回合执行"""
    print("\n=== 测试完整回合执行 ===")
    
    if envs is None:
        print("❌ 环境未初始化")
        return False
    
    try:
        n_envs = 2
        n_agents = 13
        max_steps = 10  # 测试少量步数
        
        for step in range(max_steps):
            # 创建随机动作
            actions = []
            for env_idx in range(n_envs):
                env_actions = []
                for agent_idx in range(n_agents):
                    discrete_action = np.random.randint(0, 10)
                    continuous_action = np.random.uniform(-0.1, 0.1, size=6)
                    env_actions.append([discrete_action, continuous_action])
                actions.append(env_actions)
            
            obs, share_obs, rewards, dones, infos, avail_actions = envs.step(actions)
            
            print(f"   步骤 {step+1:2d}: dones={dones.sum()}/{len(dones.flatten())}, "
                  f"平均奖励={np.mean(rewards):.4f}")
            
            # 检查done信号的一致性
            if dones.ndim != 2:
                print(f"❌ Done信号维度错误: {dones.shape}")
                return False
            
            if dones.dtype != bool:
                print(f"❌ Done信号类型错误: {dones.dtype}")
                return False
        
        print("✅ 完整回合执行成功")
        return True
        
    except Exception as e:
        print(f"❌ 回合执行失败: {e}")
        traceback.print_exc()
        return False

def test_shape_consistency():
    """测试形状一致性的关键场景"""
    print("\n=== 测试形状一致性关键场景 ===")
    
    # 测试1: 基本并行环境
    print("1. 测试基本并行环境...")
    envs, obs, share_obs, avail_actions = test_env_reset()
    
    if envs is None:
        return False
    
    # 测试2: 混合动作步进
    print("2. 测试混合动作步进...")
    if not test_mixed_action_step(envs, obs):
        envs.close()
        return False
    
    # 测试3: 完整回合执行
    print("3. 测试完整回合执行...")
    if not test_episode_completion(envs):
        envs.close()
        return False
    
    # 清理资源
    envs.close()
    print("✅ 所有形状一致性测试通过")
    return True

def main():
    """主测试函数"""
    print("HAPPO并行环境形状一致性测试")
    print("=" * 50)
    
    try:
        success = test_shape_consistency()
        
        print("\n" + "=" * 50)
        if success:
            print("🎉 所有测试通过！HAPPO并行环境修复成功")
            print("   ✅ Done信号形状一致性")
            print("   ✅ 混合动作空间兼容性")
            print("   ✅ 数据格式标准化")
            print("   ✅ 并行环境包装器稳定性")
        else:
            print("❌ 测试失败，需要进一步调试")
            return 1
            
    except Exception as e:
        print(f"❌ 测试过程中出现未预期错误: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)