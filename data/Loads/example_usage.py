#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
负荷数据分割工具使用示例
"""

from load_data_splitter import LoadDataSplitter
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def example_basic_usage():
    """
    基本使用示例
    """
    print("=== 基本使用示例 ===")
    
    # 输入文件路径
    input_file = "minute_level/LoadShape1_minute_level.csv"
    
    # 创建分割器
    splitter = LoadDataSplitter(input_file)
    
    # 加载数据
    splitter.load_data()
    
    # 获取并显示统计信息
    stats = splitter.get_data_statistics()
    print(f"总数据点: {stats['total_points']:,}")
    print(f"时间范围: {stats['start_time']} 到 {stats['end_time']}")
    
    # 提取1月份早上5点到下午17点的数据
    jan_data = splitter.split_by_month_and_time(
        month=1, 
        start_hour=5, 
        end_hour=17,
        start_minute=0,
        end_minute=59
    )
    
    print(f"1月份5:00-17:59数据点数: {len(jan_data)}")
    print("\n注意: 保存的CSV文件只包含纯数据值，无表头和时间戳")
    
    return jan_data

def example_all_months():
    """
    处理所有月份的示例
    """
    print("\n=== 处理所有月份示例 ===")
    
    input_file = "minute_level/LoadShape1_minute_level.csv"
    splitter = LoadDataSplitter(input_file)
    splitter.load_data()
    
    # 提取所有月份的工作时间数据（8:00-18:00）
    all_data = splitter.split_all_months(
        start_hour=8,
        end_hour=18,
        start_minute=0,
        end_minute=0  # 18:00整点结束
    )
    
    print(f"成功处理了 {len(all_data)} 个月份")
    
    # 显示每个月的数据量
    for month, data in all_data.items():
        print(f"{month:2d}月: {len(data):6,} 个数据点")
    
    return all_data

def example_visualization_with_stats(data_dict, enable_stats=True):
    """
    数据可视化和统计分析示例（集成版本）
    
    Args:
        data_dict (dict): 月度数据字典
        enable_stats (bool): 是否启用统计分析功能
    """
    print("\n=== 数据可视化和统计分析示例 ===")
    
    # 使用LoadDataSplitter的可视化功能
    # 注意：这里需要一个LoadDataSplitter实例
    # 在实际使用中，应该从已有的splitter实例调用
    print("提示：此功能已集成到LoadDataSplitter类中")
    print("使用方法：splitter.visualize_data(data_dict, enable_stats=True)")
    print("或通过命令行：python load_data_splitter.py [文件] --visualize")
    
    # 显示数据概览
    print(f"\n数据概览：")
    print(f"包含月份: {sorted(data_dict.keys())}")
    for month, data in data_dict.items():
        print(f"{month:2d}月: {len(data):6,} 个数据点, 平均值: {data.iloc[:, 0].mean():.4f}")

def example_visualization(data_dict):
    """
    数据可视化示例（原版本，保持不变）
    """
    print("\n=== 数据可视化示例 ===")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Load Data Analysis', fontsize=16)
    
    # 1. 月度平均负荷对比
    monthly_means = [data.mean().iloc[0] for data in data_dict.values()]
    months = list(data_dict.keys())
    
    axes[0, 0].bar(months, monthly_means, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Monthly Average Load')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Load Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 选择几个月份的日内变化模式
    selected_months = [1, 4, 7, 10]  # 选择四个季节代表月份
    colors = ['red', 'green', 'blue', 'orange']
    
    for i, month in enumerate(selected_months):
        if month in data_dict:
            # 计算每小时的平均负荷
            hourly_avg = data_dict[month].groupby(data_dict[month].index.hour).mean()
            axes[0, 1].plot(hourly_avg.index, hourly_avg.iloc[:, 0],
                           color=colors[i], label=f'{month}月', linewidth=2)
    
    axes[0, 1].set_title('Hourly Load Pattern by Month')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Load Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 负荷分布直方图（以1月为例）
    if 1 in data_dict:
        axes[1, 0].hist(data_dict[1].iloc[:, 0], bins=50, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('Load Distribution (January)')
        axes[1, 0].set_xlabel('Load Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 月度负荷变异系数
    monthly_cv = [data.std().iloc[0] / data.mean().iloc[0] for data in data_dict.values()]
    
    axes[1, 1].plot(months, monthly_cv, marker='o', linewidth=2, markersize=6, color='purple')
    axes[1, 1].set_title('Monthly Load Coefficient of Variation')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('CV (std/mean)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path("monthly_splits")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "load_analysis.png", dpi=300, bbox_inches='tight')
    print(f"可视化图表已保存到: {output_dir / 'load_analysis.png'}")
    
    plt.show()

def example_custom_time_ranges():
    """
    自定义时间段示例
    """
    print("\n=== 自定义时间段示例 ===")
    
    input_file = "minute_level/LoadShape1_minute_level.csv"
    splitter = LoadDataSplitter(input_file)
    splitter.load_data()
    
    # 定义不同的时间段
    time_periods = {
        "早高峰": {"start_hour": 7, "end_hour": 9, "start_minute": 0, "end_minute": 59},
        "午间": {"start_hour": 11, "end_hour": 13, "start_minute": 0, "end_minute": 59},
        "晚高峰": {"start_hour": 17, "end_hour": 19, "start_minute": 0, "end_minute": 59},
        "夜间": {"start_hour": 22, "end_hour": 6, "start_minute": 0, "end_minute": 59}
    }
    
    # 为每个时间段创建数据
    for period_name, time_config in time_periods.items():
        print(f"\n处理{period_name}时段...")
        
        # 为每个月份提取该时间段的数据
        for month in range(1, 13):
            try:
                data = splitter.split_by_month_and_time(
                    month=month,
                    output_dir=f"monthly_splits/{period_name}",
                    **time_config
                )
                if not data.empty:
                    print(f"  {month:2d}月 {period_name}: {len(data):5,} 个数据点")
            except Exception as e:
                print(f"  {month:2d}月 {period_name}: 处理失败 - {e}")

def main():
    """
    主函数 - 运行所有示例
    """
    try:
        # 基本使用
        jan_data = example_basic_usage()
        
        # 处理所有月份
        all_data = example_all_months()
        
        # 数据可视化
        if all_data:
            example_visualization(all_data)
        
        # 集成的可视化和统计分析功能
        print("\n" + "="*50)
        print("集成可视化和统计分析功能")
        print("="*50)
        
        # 演示新的集成功能
        example_visualization_with_stats(all_data, enable_stats=True)
        
        # 自定义时间段
        example_custom_time_ranges()
        
        print("\n=== 所有示例运行完成 ===")
        
    except FileNotFoundError:
        print("错误: 找不到输入文件 'minute_level/LoadShape1_minute_level.csv'")
        print("请确保文件路径正确，或修改 input_file 变量")
    except Exception as e:
        print(f"运行过程中出现错误: {e}")

if __name__ == "__main__":
    main()