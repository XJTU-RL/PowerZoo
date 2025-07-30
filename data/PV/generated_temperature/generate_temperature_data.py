#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
温度数据扩展脚本
将24小时的温度数据（每小时一个点）扩展为8640个点（10秒间隔）
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# 原始24小时温度数据（每小时一个点）
original_temps = [25, 25, 25, 25, 25, 25, 25, 25, 35, 40, 45, 50, 60, 60, 55, 40, 35, 30, 25, 25, 25, 25, 25, 25]

def method1_simple_repeat(hourly_temps: List[float]) -> List[float]:
    """
    方案1：简单重复方案
    每小时的温度值重复360次（3600秒/10秒）
    """
    expanded_temps = []
    for temp in hourly_temps:
        # 每小时重复360次（10秒间隔，共3600秒）
        expanded_temps.extend([temp] * 360)
    return expanded_temps

def method2_linear_interpolation(hourly_temps: List[float]) -> List[float]:
    """
    方案2：线性插值方案
    在相邻小时之间进行线性插值，使温度变化更平滑
    """
    # 创建时间轴（小时）
    hours = np.arange(24)
    # 创建新的时间轴（10秒间隔）
    new_time = np.arange(0, 24, 10/3600)  # 10秒 = 10/3600小时
    
    # 线性插值
    expanded_temps = np.interp(new_time, hours, hourly_temps)
    return expanded_temps.tolist()

def method3_realistic_fluctuation(hourly_temps: List[float]) -> List[float]:
    """
    方案3：添加真实波动方案
    在每小时基础温度上添加合理的温度波动
    """
    expanded_temps = []
    
    for hour, base_temp in enumerate(hourly_temps):
        hour_temps = []
        
        for minute_10s in range(360):  # 每小时360个10秒间隔
            # 计算当前时间（秒）
            current_second = minute_10s * 10
            
            # 添加短期周期性波动（模拟温度的自然波动）
            # 使用正弦波，周期约15分钟，幅度±1°C
            periodic_fluctuation = 1.0 * np.sin(2 * np.pi * current_second / 900)  # 900秒=15分钟
            
            # 添加随机噪声（模拟测量误差和微环境影响）
            # 幅度±0.5°C
            random_noise = np.random.normal(0, 0.3)  # 标准差0.3，约±0.5°C范围
            
            # 添加更细致的小时内温度变化
            # 在小时开始和结束时向相邻小时的温度靠近
            if hour < 23:
                next_temp = hourly_temps[hour + 1]
                # 线性过渡，在小时的后半段逐渐向下一小时温度靠近
                transition_factor = max(0, (current_second - 1800) / 1800)  # 后半小时开始过渡
                transition_adjustment = transition_factor * (next_temp - base_temp) * 0.3  # 30%的过渡
            else:
                transition_adjustment = 0
            
            # 计算最终温度
            final_temp = base_temp + periodic_fluctuation + random_noise + transition_adjustment
            
            # 确保温度在合理范围内（不低于0°C，不高于80°C）
            final_temp = max(0, min(80, final_temp))
            
            hour_temps.append(final_temp)
        
        expanded_temps.extend(hour_temps)
    
    return expanded_temps

def generate_dss_tshape(temps: List[float], name: str = "MyTemp_Expanded") -> str:
    """
    生成OpenDSS的Tshape定义字符串
    """
    # 将温度值格式化为字符串，保留1位小数
    temp_str = ", ".join([f"{temp:.1f}" for temp in temps])
    
    # 生成DSS命令
    dss_command = f"New Tshape.{name} npts={len(temps)} interval=10s temp=[{temp_str}]"
    
    return dss_command

def save_to_csv(temps: List[float], filename: str):
    """
    将温度数据保存为CSV文件
    """
    with open(filename, 'w') as f:
        for temp in temps:
            f.write(f"{temp:.2f}\n")

def plot_comparison(original: List[float], method1: List[float], 
                   method2: List[float], method3: List[float]):
    """
    绘制不同方案的对比图
    """
    # 创建时间轴
    original_time = np.arange(24)  # 小时
    expanded_time = np.arange(0, 24, 10/3600)  # 10秒间隔转换为小时
    
    plt.figure(figsize=(15, 10))
    
    # 原始数据
    plt.subplot(2, 2, 1)
    plt.plot(original_time, original, 'ro-', linewidth=2, markersize=6)
    plt.title('Original Temperature Data (24 points)', fontsize=12)
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, alpha=0.3)
    
    # 方案1：简单重复
    plt.subplot(2, 2, 2)
    plt.plot(expanded_time, method1, 'b-', linewidth=0.5)
    plt.title('Method 1: Simple Repeat (8640 points)', fontsize=12)
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, alpha=0.3)
    
    # 方案2：线性插值
    plt.subplot(2, 2, 3)
    plt.plot(expanded_time, method2, 'g-', linewidth=0.5)
    plt.title('Method 2: Linear Interpolation (8640 points)', fontsize=12)
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, alpha=0.3)
    
    # 方案3：真实波动
    plt.subplot(2, 2, 4)
    plt.plot(expanded_time, method3, 'r-', linewidth=0.5)
    plt.title('Method 3: Realistic Fluctuation (8640 points)', fontsize=12)
    plt.xlabel('Time (hours)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('temperature_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    主函数：生成所有方案的温度数据
    """
    print("正在生成温度数据...")
    
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 生成三种方案的数据
    temps_method1 = method1_simple_repeat(original_temps)
    temps_method2 = method2_linear_interpolation(original_temps)
    temps_method3 = method3_realistic_fluctuation(original_temps)
    
    print(f"原始数据点数: {len(original_temps)}")
    print(f"方案1数据点数: {len(temps_method1)}")
    print(f"方案2数据点数: {len(temps_method2)}")
    print(f"方案3数据点数: {len(temps_method3)}")
    
    # 生成DSS命令
    print("\n=== OpenDSS Tshape 定义 ===")
    print("\n方案1（简单重复）:")
    dss_cmd1 = generate_dss_tshape(temps_method1, "MyTemp_Method1")
    print(f"文件长度过长，建议保存为CSV文件引用")
    
    print("\n方案2（线性插值）:")
    dss_cmd2 = generate_dss_tshape(temps_method2, "MyTemp_Method2")
    print(f"文件长度过长，建议保存为CSV文件引用")
    
    print("\n方案3（真实波动）:")
    dss_cmd3 = generate_dss_tshape(temps_method3, "MyTemp_Method3")
    print(f"文件长度过长，建议保存为CSV文件引用")
    
    # 保存为CSV文件
    save_to_csv(temps_method1, "temperature_method1.csv")
    save_to_csv(temps_method2, "temperature_method2.csv")
    save_to_csv(temps_method3, "temperature_method3.csv")
    
    print("\n=== 推荐的DSS文件修改 ===")
    print("将原来的:")
    print("New Tshape.MyTemp npts=24 interval=1 temp=[25, 25, 25, ...]")
    print("\n替换为（推荐方案3）:")
    print("New Tshape.MyTemp npts=8640 interval=10s temp=(file=./temperature_method3.csv)")
    
    # 绘制对比图
    plot_comparison(original_temps, temps_method1, temps_method2, temps_method3)
    
    # 显示统计信息
    print("\n=== 统计信息 ===")
    print(f"方案1 - 温度范围: {min(temps_method1):.1f}°C 到 {max(temps_method1):.1f}°C")
    print(f"方案2 - 温度范围: {min(temps_method2):.1f}°C 到 {max(temps_method2):.1f}°C")
    print(f"方案3 - 温度范围: {min(temps_method3):.1f}°C 到 {max(temps_method3):.1f}°C")
    
    print(f"\n方案2 - 标准差: {np.std(temps_method2):.2f}°C")
    print(f"方案3 - 标准差: {np.std(temps_method3):.2f}°C")

if __name__ == "__main__":
    main()