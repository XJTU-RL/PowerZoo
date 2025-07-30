#!/usr/bin/env python3
"""
PowerZoo增强日志系统使用示例

演示如何在训练过程中使用新的日志功能进行详细监控
"""

import numpy as np
from typing import Dict, Any
from utils import (
    setup_training_logger,
    create_training_debug_logger,
    log_training_step,
    log_reward_components,
    log_device_actions,
    log_training_summary
)

def training_example_with_enhanced_logging():
    """
    演示增强日志系统在训练中的使用
    """
    # 设置不同级别的日志记录器
    training_logger = setup_training_logger("powerzoo_demo", "logs")
    debug_logger = create_training_debug_logger("powerzoo_demo")
    
    training_logger.info("开始PowerZoo训练演示")
    
    # 模拟训练循环
    for episode in range(1, 4):  # 演示3个回合
        training_logger.train_info(f"开始Episode {episode}")
        episode_reward = 0.0
        
        for step in range(1, 11):  # 每回合10步
            # 模拟动作执行
            action = np.random.randint(0, 2, size=5)  # 5个智能体的随机动作
            
            # 模拟奖励计算
            power_loss = np.random.uniform(-0.05, -0.01)
            voltage_reward = np.random.uniform(-2.0, 0.5)
            control_reward = np.random.uniform(-0.5, 0.0)
            total_reward = power_loss + voltage_reward + control_reward
            episode_reward += total_reward
            
            # 记录训练步骤
            log_training_step(
                training_logger, step, episode, 
                str(action), total_reward, False, 
                {"power_loss": power_loss, "voltage_violations": np.random.randint(0, 3)}
            )
            
            # 记录奖励分解
            reward_components = {
                'power_loss': power_loss,
                'voltage': voltage_reward,
                'control': control_reward,
                'total': total_reward
            }
            log_reward_components(training_logger, reward_components)
            
            # 记录设备动作（模拟）
            device_names = ["Cap1", "Reg1", "Bat1"]
            device_types = ["Capacitor", "Regulator", "Battery"]
            for i, (dev_type, dev_name) in enumerate(zip(device_types, device_names)):
                old_state = np.random.random()
                new_state = np.random.random()
                diff = abs(new_state - old_state)
                log_device_actions(training_logger, dev_type, dev_name, old_state, new_state, diff)
            
            # 系统状态记录（调试级别）
            system_state = {
                "avg_voltage": np.random.uniform(0.95, 1.05),
                "power_loss_ratio": abs(power_loss),
                "dss_converged": np.random.choice([True, False], p=[0.9, 0.1])
            }
            debug_logger.system_state(system_state)
            
            # DSS收敛性检查
            converged = system_state["dss_converged"]
            debug_logger.convergence_check(converged, np.random.randint(3, 15))
        
        # 记录回合总结
        final_info = {
            'power_loss_ratio': abs(power_loss),
            'voltage_violations': np.random.randint(0, 5),
            'voltage_compliance_rate': np.random.uniform(0.85, 1.0)
        }
        log_training_summary(training_logger, episode, episode_reward, 10, final_info)
    
    training_logger.info("PowerZoo训练演示完成")

def log_level_demonstration():
    """
    演示不同日志级别的输出效果
    """
    import logging
    from utils import get_logger
    
    # 创建不同级别的日志记录器
    info_logger = get_logger("demo_info", level=logging.INFO)
    debug_logger = get_logger("demo_debug", level=logging.DEBUG)
    
    print("\n=== 日志级别演示 ===")
    
    # INFO级别日志记录器
    print("\n1. INFO级别日志记录器输出：")
    info_logger.debug("这条DEBUG信息不会显示")
    info_logger.info("这条INFO信息会显示")
    info_logger.train_info("这条TRAIN信息会显示")
    info_logger.warning("这条WARNING信息会显示")
    
    # DEBUG级别日志记录器
    print("\n2. DEBUG级别日志记录器输出：")
    debug_logger.debug("这条DEBUG信息会显示")
    debug_logger.action_debug("这条ACTION DEBUG信息会显示")
    debug_logger.reward_debug("这条REWARD DEBUG信息会显示")
    debug_logger.info("这条INFO信息会显示")
    debug_logger.train_info("这条TRAIN信息会显示")

def performance_monitoring_example():
    """
    演示性能监控日志的使用
    """
    from utils import get_logger
    import time
    import logging
    
    logger = get_logger("performance_demo", level=logging.DEBUG)
    
    print("\n=== 性能监控演示 ===")
    
    # 模拟不同的操作耗时
    operations = [
        ("环境重置", 0.1),
        ("动作执行", 0.05),
        ("DSS求解", 0.15),
        ("奖励计算", 0.02),
        ("状态更新", 0.03)
    ]
    
    for op_name, duration in operations:
        start_time = time.time()
        time.sleep(duration)  # 模拟操作耗时
        elapsed = time.time() - start_time
        
        if elapsed > 0.1:
            logger.warning(f"{op_name} 耗时较长: {elapsed:.3f}s")
        else:
            logger.debug(f"{op_name} 完成: {elapsed:.3f}s")

if __name__ == "__main__":
    print("PowerZoo增强日志系统演示")
    print("=" * 50)
    
    # 演示1: 完整的训练日志
    print("\n1. 训练过程日志演示")
    training_example_with_enhanced_logging()
    
    # 演示2: 不同日志级别
    log_level_demonstration()
    
    # 演示3: 性能监控
    performance_monitoring_example()
    
    print("\n演示完成! 请检查生成的日志文件在 'logs/' 目录下")