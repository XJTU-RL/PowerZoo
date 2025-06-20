#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试DSR环境的负荷聚合功能
Test DSR environment load aggregation functionality
"""

import os
import sys
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.dsr.core.config import DSRConfig
from envs.dsr.core.dsr_core import DSRCoreEnv
from envs.dsr.dsr_env import DSREnv


def test_load_aggregation():
    """测试不同系统规模下的负荷聚合"""
    
    print("=" * 80)
    print("测试DSR环境负荷聚合功能")
    print("=" * 80)
    
    # 测试不同系统配置
    test_configs = [
        {
            'name': '13Bus系统（无聚合）',
            'system_name': '13Bus',
            'use_load_aggregation': False,
            'n_load_agents': None,
            'load_aggregation_method': 'zone'
        },
        {
            'name': '34Bus系统（自动聚合）',
            'system_name': '34Bus',
            'use_load_aggregation': True,
            'n_load_agents': None,  # 自动计算
            'load_aggregation_method': 'zone'
        },
        {
            'name': '123Bus系统（指定30个智能体）',
            'system_name': '123Bus',
            'use_load_aggregation': True,
            'n_load_agents': 30,
            'load_aggregation_method': 'priority'
        },
        {
            'name': '8500-Node系统（聚合到50个智能体）',
            'system_name': '8500-Node',
            'use_load_aggregation': True,
            'n_load_agents': 50,
            'load_aggregation_method': 'zone'
        }
    ]
    
    for test_config in test_configs:
        print(f"\n测试配置: {test_config['name']}")
        print("-" * 60)
        
        try:
            # 创建配置
            config = DSRConfig(
                system_name=test_config['system_name'],
                use_load_aggregation=test_config['use_load_aggregation'],
                n_load_agents=test_config['n_load_agents'],
                load_aggregation_method=test_config['load_aggregation_method'],
                max_episode_steps=5  # 快速测试
            )
            
            # 获取智能体配置
            agent_config = config.get_agent_config()
            
            print(f"系统: {config.system_name}")
            print(f"实际负荷数量: {agent_config['actual_load_count']}")
            print(f"负荷智能体数量: {agent_config['n_load_agents']}")
            print(f"聚合比例: {agent_config['aggregation_ratio']:.2f}")
            print(f"使用聚合: {agent_config['use_aggregation']}")
            print(f"总智能体数量: {agent_config['total_agents']}")
            
            # 创建核心环境
            core_env = DSRCoreEnv(config)
            
            # 重置环境
            obs, state = core_env.reset()
            
            # 显示智能体详情
            print(f"\n智能体详情:")
            print(f"- 开关智能体: {core_env.n_switch_agents}个")
            print(f"- PV智能体: {core_env.n_pv_agents}个")
            print(f"- 负荷智能体: {core_env.n_load_agents}个")
            
            # 显示负荷聚合情况
            if core_env.use_aggregation and core_env.aggregation_ratio > 1:
                print(f"\n负荷聚合情况:")
                for i, load_agent in enumerate(core_env.load_agents[:5]):  # 显示前5个
                    managed_loads = load_agent.get('managed_loads', [])
                    print(f"  智能体{i}: 管理{len(managed_loads)}个负荷, "
                          f"总功率={load_agent['kw']:.1f}kW, "
                          f"优先级={load_agent['priority']}")
            
            # 执行一步测试
            actions = [0] * core_env.n_agents  # 所有智能体不动作
            obs, state, rewards, done, info = core_env.step(actions)
            
            print(f"\n环境步进测试: 成功")
            print(f"潮流收敛: {info['converged']}")
            print(f"通电母线比例: {info['energized_buses']}/{info['total_buses']}")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("负荷聚合功能测试完成!")
    print("=" * 80)


def test_dsr_env_wrapper():
    """测试DSR环境包装器"""
    print("\n" + "=" * 80)
    print("测试DSR环境包装器（PowerZoo接口）")
    print("=" * 80)
    
    # 测试参数
    args = {
        'env_name': 'dsr',
        'system_name': '123Bus',
        'use_load_aggregation': True,
        'n_load_agents': 30,
        'load_aggregation_method': 'zone',
        'max_episode_steps': 5,
        'seed': 12345,
        'use_render': False,
        'load_noise': False
    }
    
    try:
        # 创建环境
        env = DSREnv(args)
        
        print(f"环境创建成功!")
        print(f"智能体数量: {env.n_agents}")
        print(f"观测空间维度: {[space.shape for space in env.observation_space]}")
        print(f"动作空间: {[space.n for space in env.action_space]}")
        
        # 重置环境
        obs, states, avail_actions = env.reset()
        
        print(f"\n初始状态:")
        print(f"观测形状: {[o.shape for o in obs]}")
        print(f"可用动作: {[len(a) for a in avail_actions[:5]]}...")  # 显示前5个
        
        # 执行随机动作
        actions = []
        for i in range(env.n_agents):
            avail = avail_actions[i]
            valid_actions = [j for j, a in enumerate(avail) if a == 1]
            if valid_actions:
                actions.append(np.random.choice(valid_actions))
            else:
                actions.append(0)
        
        # 步进
        obs, states, rewards, dones, infos, avail_actions = env.step(actions)
        
        print(f"\n步进后:")
        print(f"奖励: {[r[0] for r in rewards[:5]]}...")  # 显示前5个
        print(f"完成标志: {dones[0]}")
        print(f"信息: {infos[0]}")
        
        env.close()
        print(f"\n环境包装器测试成功!")
        
    except Exception as e:
        print(f"环境包装器测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 测试负荷聚合功能
    test_load_aggregation()
    
    # 测试环境包装器
    test_dsr_env_wrapper()