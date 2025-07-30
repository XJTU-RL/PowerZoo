#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PowerZoo系统监控使用示例
演示如何在HAPPO训练过程中使用系统参数记录和分析功能

用法:
    python examples/system_monitoring_example.py --config configs/envs_cfgs/powerzoo.yaml
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # 导入环境和记录器
    from envs.power_envs.powerzoo_llm.env_register import make_base_env
    from envs.power_envs.powerzoo_llm.powerzoo_env import PowerZooEnv
    from envs.power_envs.powerzoo_llm.system_logger import get_system_logger, close_system_logger
    from envs.power_envs.powerzoo_llm.system_analyzer import analyze_training_session, AnalysisConfig
    from utils.config import get_config
    
    print("✅ 成功导入所有必需模块")
    
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保在PowerZoo项目根目录下运行此脚本")
    sys.exit(1)


class MockConfig:
    """模拟配置类，用于测试"""
    def __init__(self, **kwargs):
        # 基础环境配置
        self.env_name = kwargs.get('env_name', '13Bus')
        self.seed = kwargs.get('seed', 123456)
        self.num_env = kwargs.get('num_env', 1)
        self.useS = kwargs.get('useS', False)
        
        # 系统记录配置
        self.enable_system_logging = kwargs.get('enable_system_logging', True)
        self.system_log_dir = kwargs.get('system_log_dir', "./logs/system_params")
        self.log_buffer_size = kwargs.get('log_buffer_size', 1000)  # 减小用于演示
        self.log_save_interval = kwargs.get('log_save_interval', 10)  # 减小用于演示
        self.enable_realtime_log = kwargs.get('enable_realtime_log', True)


def create_demo_environment(config):
    """创建演示环境"""
    print("🔧 创建PowerZoo环境...")
    
    try:
        # 创建基础环境
        base_env = make_base_env(config.env_name)
        
        # 创建包装环境
        env = PowerZooEnv(base_env, config, rank=0)
        
        print(f"✅ 环境创建成功 - {config.env_name}")
        print(f"   智能体数量: {env.n_agents}")
        print(f"   电容器数量: {env.cap_num}")
        print(f"   调压器数量: {env.reg_num}")
        print(f"   电池数量: {env.bat_num}")
        print(f"   系统记录: {'启用' if config.enable_system_logging else '禁用'}")
        
        return env
        
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return None


def run_demo_training(env, num_episodes=5, steps_per_episode=20):
    """运行演示训练"""
    print(f"\n🎯 开始演示训练 - {num_episodes}个回合，每回合{steps_per_episode}步")
    
    total_reward = 0
    episode_rewards = []
    
    for episode in range(num_episodes):
        print(f"\n📊 回合 {episode + 1}/{num_episodes}")
        
        # 重置环境
        obs, state, avail_actions = env.reset()
        episode_reward = 0
        
        for step in range(steps_per_episode):
            # 随机动作（实际训练中这里应该是智能体决策）
            if hasattr(env.env, 'action_space'):
                if hasattr(env.env.action_space, 'sample'):
                    action = env.env.action_space.sample()
                else:
                    # 为多智能体环境生成随机动作
                    action = []
                    for agent_id in range(env.n_agents):
                        if hasattr(env.action_space[agent_id], 'sample'):
                            action.append(env.action_space[agent_id].sample())
                        else:
                            action.append(0)  # 默认动作
                    action = action
            else:
                action = [0] * env.n_agents  # 默认动作
            
            # 执行动作
            obs, state, rewards, dones, infos, avail_actions = env.step(action)
            
            # 累计奖励
            reward = rewards[0][0] if rewards and rewards[0] else 0
            episode_reward += reward
            
            # 打印步骤信息
            if step % 5 == 0 or step == steps_per_episode - 1:
                print(f"  Step {step+1:2d}: Reward = {reward:8.4f}, "
                      f"Done = {dones[0] if dones else False}, "
                      f"Info = {len(infos[0]) if infos else 0} 项")
            
            # 检查是否终止
            if dones and any(dones):
                print(f"  回合在第{step+1}步提前结束")
                break
        
        episode_rewards.append(episode_reward)
        total_reward += episode_reward
        
        print(f"  回合奖励: {episode_reward:8.4f}")
        print(f"  平均奖励: {total_reward/(episode+1):8.4f}")
    
    print(f"\n🏆 训练完成!")
    print(f"总奖励: {total_reward:8.4f}")
    print(f"平均回合奖励: {total_reward/num_episodes:8.4f}")
    print(f"奖励范围: [{min(episode_rewards):6.4f}, {max(episode_rewards):6.4f}]")
    
    return episode_rewards


def analyze_training_results(log_dir):
    """分析训练结果"""
    print(f"\n📈 开始分析训练结果...")
    
    # 等待一下确保数据写入完成
    time.sleep(2)
    
    try:
        # 配置分析参数
        analysis_config = AnalysisConfig(
            smooth_window=20,  # 减小平滑窗口用于演示
            figure_size=(10, 6),
            generate_summary_only=False,
            include_detailed_plots=True
        )
        
        # 运行分析
        results = analyze_training_session(
            log_dir=log_dir,
            session_id=None,  # 使用最新会话
            output_dir=f"{log_dir}/analysis",
            config=analysis_config
        )
        
        if results:
            print("✅ 分析完成!")
            print(f"   输出目录: {results.get('output_directory', 'Unknown')}")
            print(f"   报告文件: {results.get('report_path', 'None')}")
            print(f"   生成图表: {len(results.get('visualizations', {}))}")
            
            # 显示可用的图表
            visualizations = results.get('visualizations', {})
            if visualizations:
                print("   📊 生成的图表:")
                for name, path in visualizations.items():
                    print(f"     - {name}: {Path(path).name}")
        else:
            print("⚠️ 分析未产生结果")
            
    except Exception as e:
        print(f"❌ 分析失败: {e}")


def cleanup_resources(env):
    """清理资源"""
    print(f"\n🧹 清理资源...")
    
    try:
        if env:
            env.close()
            print("✅ 环境已关闭")
        
        # 关闭全局系统记录器
        close_system_logger()
        print("✅ 系统记录器已关闭")
        
    except Exception as e:
        print(f"⚠️ 清理资源时出现警告: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PowerZoo系统监控演示")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--env_name", type=str, default="13Bus", help="环境名称")
    parser.add_argument("--episodes", type=int, default=3, help="演示回合数")
    parser.add_argument("--steps", type=int, default=15, help="每回合步数")
    parser.add_argument("--log_dir", type=str, default="./logs/demo_system_params", help="日志目录")
    parser.add_argument("--skip_analysis", action="store_true", help="跳过结果分析")
    
    args = parser.parse_args()
    
    print("🚀 PowerZoo系统监控演示")
    print("=" * 50)
    
    # 创建配置
    if args.config and os.path.exists(args.config):
        try:
            config = get_config(args.config)
            print(f"✅ 加载配置文件: {args.config}")
        except:
            print(f"⚠️ 配置文件加载失败，使用默认配置")
            config = MockConfig(env_name=args.env_name, system_log_dir=args.log_dir)
    else:
        print("🔧 使用模拟配置")
        config = MockConfig(env_name=args.env_name, system_log_dir=args.log_dir)
    
    # 更新日志目录
    config.system_log_dir = args.log_dir
    
    env = None
    try:
        # 1. 创建环境
        env = create_demo_environment(config)
        if not env:
            print("❌ 无法继续，环境创建失败")
            return
        
        # 2. 运行演示训练
        episode_rewards = run_demo_training(env, args.episodes, args.steps)
        
        # 3. 分析结果（如果未跳过）
        if not args.skip_analysis:
            analyze_training_results(args.log_dir)
        else:
            print("\n⏭️ 跳过结果分析")
        
        print(f"\n🎉 演示完成! 检查日志目录获取详细数据: {args.log_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断演示")
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 4. 清理资源
        cleanup_resources(env)


if __name__ == "__main__":
    main()