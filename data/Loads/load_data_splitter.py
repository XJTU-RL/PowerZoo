#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
负荷数据分割脚本
功能：将一年的分钟级负荷数据按月份和时间段进行分割
作者：Data Analysis Agent
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
# 设置matplotlib后端，避免在无GUI环境下出错
matplotlib.use('Agg')

class LoadDataSplitter:
    def __init__(self, input_file):
        """
        初始化负荷数据分割器
        
        Args:
            input_file (str): 输入的CSV文件路径
        """
        self.input_file = input_file
        self.data = None
        self.start_date = datetime(2024, 1, 1)  # 默认从2024年1月1日开始
        
    def load_data(self):
        """
        加载负荷数据
        """
        print(f"正在加载数据文件: {self.input_file}")
        
        # 读取数据
        with open(self.input_file, 'r') as f:
            values = [float(line.strip()) for line in f if line.strip()]
        
        # 创建时间索引（分钟级别）
        total_minutes = len(values)
        time_index = pd.date_range(
            start=self.start_date, 
            periods=total_minutes, 
            freq='T'  # T表示分钟
        )
        
        # 创建DataFrame
        self.data = pd.DataFrame({
            'load_value': values
        }, index=time_index)
        
        print(f"数据加载完成，共 {len(self.data)} 个数据点")
        print(f"时间范围: {self.data.index[0]} 到 {self.data.index[-1]}")
        
    def split_by_month_and_time(self, month, start_hour=0, end_hour=23, 
                               start_minute=0, end_minute=59, output_dir=None):
        """
        按月份和时间段分割数据
        
        Args:
            month (int): 月份 (1-12)
            start_hour (int): 开始小时 (0-23)
            end_hour (int): 结束小时 (0-23)
            start_minute (int): 开始分钟 (0-59)
            end_minute (int): 结束分钟 (0-59)
            output_dir (str): 输出目录，默认为输入文件同目录下的monthly_splits文件夹
        
        Returns:
            pd.DataFrame: 分割后的数据
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 方法加载数据")
            
        if not (1 <= month <= 12):
            raise ValueError("月份必须在1-12之间")
            
        if not (0 <= start_hour <= 23) or not (0 <= end_hour <= 23):
            raise ValueError("小时必须在0-23之间")
            
        if not (0 <= start_minute <= 59) or not (0 <= end_minute <= 59):
            raise ValueError("分钟必须在0-59之间")
        
        print(f"正在提取 {month} 月份 {start_hour:02d}:{start_minute:02d} 到 {end_hour:02d}:{end_minute:02d} 的数据...")
        
        # 筛选指定月份的数据
        monthly_data = self.data[self.data.index.month == month].copy()
        
        if monthly_data.empty:
            print(f"警告: {month} 月份没有数据")
            return pd.DataFrame()
        
        # 优化的时间筛选方法 - 使用向量化操作
        # 创建时间掩码
        hour_mask = (monthly_data.index.hour >= start_hour) & (monthly_data.index.hour <= end_hour)
        minute_mask = True  # 默认所有分钟都符合
        
        # 如果指定了具体的分钟范围，则添加分钟筛选
        if start_minute != 0 or end_minute != 59:
            if start_hour == end_hour:
                # 同一小时内的分钟范围
                minute_mask = (monthly_data.index.minute >= start_minute) & (monthly_data.index.minute <= end_minute)
            else:
                # 跨小时的情况，需要特殊处理
                start_hour_mask = (monthly_data.index.hour == start_hour) & (monthly_data.index.minute >= start_minute)
                end_hour_mask = (monthly_data.index.hour == end_hour) & (monthly_data.index.minute <= end_minute)
                middle_hour_mask = (monthly_data.index.hour > start_hour) & (monthly_data.index.hour < end_hour)
                minute_mask = start_hour_mask | end_hour_mask | middle_hour_mask
        
        # 处理跨天情况
        if end_hour < start_hour:
            # 跨天：从start_hour到23:59 + 从00:00到end_hour
            night_mask = monthly_data.index.hour >= start_hour
            morning_mask = monthly_data.index.hour <= end_hour
            time_mask = night_mask | morning_mask
        else:
            # 同一天内
            time_mask = hour_mask
        
        # 应用时间和分钟掩码
        result_data = monthly_data[time_mask & minute_mask]
        
        if result_data.empty:
            print(f"警告: {month} 月份在指定时间段内没有数据")
            return pd.DataFrame()
        
        print(f"提取完成，共 {len(result_data)} 个数据点")
        
        # 保存数据
        if output_dir is None:
            output_dir = Path(self.input_file).parent / "monthly_splits"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 生成输出文件名
        time_suffix = f"{start_hour:02d}{start_minute:02d}_to_{end_hour:02d}{end_minute:02d}"
        output_file = output_dir / f"month_{month:02d}_{time_suffix}.csv"
        
        # 保存为CSV文件，只保留纯数据，不要表头和索引
        result_data['load_value'].to_csv(output_file, header=False, index=False)
        print(f"数据已保存到: {output_file}")
        
        return result_data
    
    def split_all_months(self, start_hour=0, end_hour=23, 
                        start_minute=0, end_minute=59, output_dir=None):
        """
        分割所有月份的数据
        
        Args:
            start_hour (int): 开始小时
            end_hour (int): 结束小时
            start_minute (int): 开始分钟
            end_minute (int): 结束分钟
            output_dir (str): 输出目录
        
        Returns:
            dict: 包含所有月份数据的字典
        """
        all_data = {}
        
        for month in range(1, 13):
            try:
                monthly_data = self.split_by_month_and_time(
                    month, start_hour, end_hour, start_minute, end_minute, output_dir
                )
                if not monthly_data.empty:
                    all_data[month] = monthly_data
            except Exception as e:
                print(f"处理 {month} 月份时出错: {e}")
                
        return all_data
    
    def get_data_statistics(self):
        """
        获取数据统计信息
        
        Returns:
            dict: 统计信息
        """
        if self.data is None:
            raise ValueError("请先调用 load_data() 方法加载数据")
            
        stats = {
            'total_points': len(self.data),
            'start_time': self.data.index[0],
            'end_time': self.data.index[-1],
            'mean_value': self.data['load_value'].mean(),
            'max_value': self.data['load_value'].max(),
            'min_value': self.data['load_value'].min(),
            'std_value': self.data['load_value'].std()
        }
        
        # 按月份统计
        monthly_stats = {}
        for month in range(1, 13):
            month_data = self.data[self.data.index.month == month]
            if not month_data.empty:
                monthly_stats[month] = {
                    'points': len(month_data),
                    'mean': month_data['load_value'].mean(),
                    'max': month_data['load_value'].max(),
                    'min': month_data['load_value'].min()
                }
        
        stats['monthly_stats'] = monthly_stats
        return stats
    
    def visualize_data(self, data_dict=None, enable_stats=True, output_dir=None):
        """
        数据可视化和统计分析
        
        Args:
            data_dict (dict): 月度数据字典，如果为None则使用全部数据按月分组
            enable_stats (bool): 是否启用统计分析功能
            output_dir (str): 输出目录
        """
        print("\n=== 数据可视化示例 ===")
        
        # 如果没有提供data_dict，则按月分组数据
        if data_dict is None:
            if self.data is None:
                print("错误：数据未加载")
                return
            
            data_dict = {}
            for month in range(1, 13):
                month_data = self.data[self.data.index.month == month]
                if not month_data.empty:
                    data_dict[month] = month_data
        
        if not data_dict:
            print("错误：没有可用的数据进行可视化")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans']
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
                               color=colors[i], label=f'month: {month}', linewidth=2)
        
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
        if output_dir is None:
            output_dir = Path(self.input_file).parent / "monthly_splits"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        plt.savefig(output_dir / "load_analysis.png", dpi=300)
        print(f"可视化图表已保存到: {output_dir / 'load_analysis.png'}")
        
        # 统计分析功能
        if enable_stats:
            self._perform_statistical_analysis(data_dict, output_dir)
        
        plt.close()  # 关闭图形以释放内存
    
    def _perform_statistical_analysis(self, data_dict, output_dir):
        """
        执行统计分析
        
        Args:
            data_dict (dict): 月度数据字典
            output_dir (Path): 输出目录
        """
        print("\n=== 统计分析结果 ===")
        
        # 1. 基本统计信息
        stats_summary = []
        for month, data in data_dict.items():
            load_values = data.iloc[:, 0]
            stats = {
                '月份': month,
                '数据点数': len(load_values),
                '平均值': load_values.mean(),
                '中位数': load_values.median(),
                '标准差': load_values.std(),
                '最大值': load_values.max(),
                '最小值': load_values.min(),
                '变异系数': load_values.std() / load_values.mean(),
                '偏度': load_values.skew(),
                '峰度': load_values.kurtosis()
            }
            stats_summary.append(stats)
        
        # 转换为DataFrame并保存
        stats_df = pd.DataFrame(stats_summary)
        stats_file = output_dir / "statistical_analysis.csv"
        stats_df.to_csv(stats_file, index=False, encoding='utf-8-sig')
        print(f"统计分析结果已保存到: {stats_file}")
        
        # 打印关键统计信息
        print("\n月度负荷统计摘要:")
        print(f"{'月份':<4} {'平均值':<10} {'标准差':<10} {'变异系数':<10} {'偏度':<10}")
        print("-" * 50)
        for _, row in stats_df.iterrows():
            print(f"{row['月份']:<4} {row['平均值']:<10.4f} {row['标准差']:<10.4f} {row['变异系数']:<10.4f} {row['偏度']:<10.4f}")
        
        # 2. 相关性分析（如果有多个月份）
        if len(data_dict) > 1:
            print("\n=== 月度负荷相关性分析 ===")
            # 计算月度平均负荷的相关性
            monthly_means = {month: data.iloc[:, 0].mean() for month, data in data_dict.items()}
            monthly_stds = {month: data.iloc[:, 0].std() for month, data in data_dict.items()}
            
            correlation = np.corrcoef(list(monthly_means.values()), list(monthly_stds.values()))[0, 1]
            print(f"月度平均负荷与标准差的相关系数: {correlation:.4f}")
        
        # 3. 季节性分析
        print("\n=== 季节性分析 ===")
        seasons = {
            '春季': [3, 4, 5],
            '夏季': [6, 7, 8], 
            '秋季': [9, 10, 11],
            '冬季': [12, 1, 2]
        }
        
        seasonal_stats = []
        for season_name, season_months in seasons.items():
            season_data = []
            for month in season_months:
                if month in data_dict:
                    season_data.extend(data_dict[month].iloc[:, 0].tolist())
            
            if season_data:
                season_series = pd.Series(season_data)
                seasonal_stats.append({
                    '季节': season_name,
                    '平均负荷': season_series.mean(),
                    '负荷标准差': season_series.std(),
                    '最大负荷': season_series.max(),
                    '最小负荷': season_series.min()
                })
        
        if seasonal_stats:
            seasonal_df = pd.DataFrame(seasonal_stats)
            seasonal_file = output_dir / "seasonal_analysis.csv"
            seasonal_df.to_csv(seasonal_file, index=False, encoding='utf-8-sig')
            print(f"季节性分析结果已保存到: {seasonal_file}")
            
            print("\n季节负荷统计:")
            print(f"{'季节':<6} {'平均负荷':<12} {'负荷标准差':<12} {'最大负荷':<12} {'最小负荷':<12}")
            print("-" * 60)
            for _, row in seasonal_df.iterrows():
                print(f"{row['季节']:<6} {row['平均负荷']:<12.4f} {row['负荷标准差']:<12.4f} {row['最大负荷']:<12.4f} {row['最小负荷']:<12.4f}")

def main():
    parser = argparse.ArgumentParser(description='负荷数据分割工具')
    parser.add_argument('input_file', help='输入的CSV文件路径')
    parser.add_argument('--month', type=int, choices=range(1, 13), 
                       help='指定月份 (1-12)，不指定则处理所有月份')
    parser.add_argument('--start-hour', type=int, default=0, choices=range(0, 24),
                       help='开始小时 (0-23)，默认为0')
    parser.add_argument('--end-hour', type=int, default=23, choices=range(0, 24),
                       help='结束小时 (0-23)，默认为23')
    parser.add_argument('--start-minute', type=int, default=0, choices=range(0, 60),
                       help='开始分钟 (0-59)，默认为0')
    parser.add_argument('--end-minute', type=int, default=59, choices=range(0, 60),
                       help='结束分钟 (0-59)，默认为59')
    parser.add_argument('--output-dir', help='输出目录，默认为输入文件同目录下的monthly_splits')
    parser.add_argument('--stats', action='store_true', help='显示数据统计信息')
    parser.add_argument('--visualize', action='store_true', help='生成数据可视化图表和统计分析')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"错误: 输入文件 {args.input_file} 不存在")
        return
    
    # 创建分割器
    splitter = LoadDataSplitter(args.input_file)
    
    try:
        # 加载数据
        splitter.load_data()
        
        # 显示统计信息
        if args.stats:
            stats = splitter.get_data_statistics()
            print("\n=== 数据统计信息 ===")
            print(f"总数据点: {stats['total_points']:,}")
            print(f"时间范围: {stats['start_time']} 到 {stats['end_time']}")
            print(f"平均值: {stats['mean_value']:.6f}")
            print(f"最大值: {stats['max_value']:.6f}")
            print(f"最小值: {stats['min_value']:.6f}")
            print(f"标准差: {stats['std_value']:.6f}")
            
            print("\n=== 月度统计 ===")
            for month, month_stats in stats['monthly_stats'].items():
                print(f"{month:2d}月: {month_stats['points']:6,} 点, "
                      f"均值: {month_stats['mean']:.4f}, "
                      f"最大: {month_stats['max']:.4f}, "
                      f"最小: {month_stats['min']:.4f}")
        
        # 分割数据
        if args.month:
            # 处理单个月份
            splitter.split_by_month_and_time(
                args.month, args.start_hour, args.end_hour,
                args.start_minute, args.end_minute, args.output_dir
            )
        else:
            # 处理所有月份
            print("\n=== 开始分割所有月份数据 ===")
            all_data = splitter.split_all_months(
                args.start_hour, args.end_hour,
                args.start_minute, args.end_minute, args.output_dir
            )
            print(f"\n分割完成，共处理了 {len(all_data)} 个月份的数据")
        
        # 可视化分析
        if args.visualize:
            print("\n=== 开始数据可视化分析 ===")
            try:
                # 如果处理了单个月份，只可视化该月份
                if args.month:
                    month_data = splitter.data[splitter.data.index.month == args.month]
                    if not month_data.empty:
                        # 应用时间筛选
                        if args.start_hour != 0 or args.end_hour != 23 or args.start_minute != 0 or args.end_minute != 59:
                            # 创建时间掩码
                            time_mask = pd.Series(False, index=month_data.index)
                            
                            if args.start_hour <= args.end_hour:
                                # 同一天内的时间段
                                time_mask = (
                                    (month_data.index.hour > args.start_hour) |
                                    ((month_data.index.hour == args.start_hour) & (month_data.index.minute >= args.start_minute))
                                ) & (
                                    (month_data.index.hour < args.end_hour) |
                                    ((month_data.index.hour == args.end_hour) & (month_data.index.minute <= args.end_minute))
                                )
                            else:
                                # 跨天的时间段
                                time_mask = (
                                    (month_data.index.hour > args.start_hour) |
                                    ((month_data.index.hour == args.start_hour) & (month_data.index.minute >= args.start_minute)) |
                                    (month_data.index.hour < args.end_hour) |
                                    ((month_data.index.hour == args.end_hour) & (month_data.index.minute <= args.end_minute))
                                )
                            
                            month_data = month_data[time_mask]
                        
                        data_dict = {args.month: month_data}
                        splitter.visualize_data(data_dict, enable_stats=True, output_dir=args.output_dir)
                    else:
                        print(f"警告：{args.month}月没有数据")
                else:
                    # 处理所有月份，使用分割后的数据进行可视化
                    if 'all_data' in locals():
                        splitter.visualize_data(all_data, enable_stats=True, output_dir=args.output_dir)
                    else:
                        # 如果没有分割数据，使用全部数据
                        splitter.visualize_data(enable_stats=True, output_dir=args.output_dir)
                        
            except Exception as e:
                print(f"可视化过程中出现错误: {e}")
                print("提示：可视化功能需要matplotlib库，请确保已安装")
            
    except Exception as e:
        print(f"错误: {e}")
        return

if __name__ == "__main__":
    main()