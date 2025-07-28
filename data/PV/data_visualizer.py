#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合PV数据可视化分析脚本
分析太阳能数据的多个维度，包括辐射、温度、湿度、风速等
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from datetime import datetime
import os
import argparse
import warnings
from pathlib import Path
import matplotlib.dates as mdates
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

warnings.filterwarnings('ignore')

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
if HAS_SEABORN:
    sns.set_style("whitegrid")
    sns.set_palette("husl")
else:
    plt.style.use('default')

class PVDataVisualizer:
    def __init__(self, output_dir="visualization_results"):
        self.output_dir = Path(output_dir)
        self.setup_directories()
        
        # 定义关键字段映射（基于field_definition.md）
        self.field_mapping = {
            # 时间字段
            'datetime_cols': [0, 1, 2, 3, 4],  # Year, Month, Day, Hour, Minute
            
            # 太阳辐射字段（W/m²）
            'solar_radiation': {
                'Global SR20': 5,
                'Direct DR20': 7,
                'Global LI-200': 9,
                'Global CMP22': 11,
                'Global PSP': 13,
                'Global CMP11': 15,
                'Diffuse 8-48': 75,
                'Net Radiation': 153
            },
            
            # 温度字段（°C）
            'temperature': {
                'Wet Bulb (Tower)': 73,
                'Dry Bulb (Deck)': 93,
                'Dry Bulb (Tower)': 97,
                'Wind Chill (Tower)': 101,
                'Wind Chill (Deck)': 207,
                'Dew Point (Tower)': 181
            },
            
            # 湿度字段（%）
            'humidity': {
                'Relative Humidity (Deck)': 95,
                'Relative Humidity (Tower)': 99,
                'Relative Humidity (SE)': 205
            },
            
            # 风速风向字段
            'wind': {
                'Wind Speed @ 6ft': 105,
                'Wind Direction @ 6ft': 107,
                'Wind Speed @ 22ft': 109,
                'Wind Direction @ 22ft': 111,
                'Wind Speed @ 33ft': 115,
                'Wind Direction @ 33ft': 117
            },
            
            # 云量字段（%）
            'cloud': {
                'Cloud Cover (Total)': 69,
                'Cloud Cover (Opaque)': 71
            },
            
            # 气压字段（mBar）
            'pressure': {
                'Station Pressure': 121,
                'Sea-Level Pressure': 123
            },
            
            # 其他环境参数
            'environmental': {
                'Precipitation': 125,  # mm
                'Snow Depth': 145,     # cm
                'Zenith Angle': 161,   # degrees
                'Azimuth Angle': 163,  # degrees
                'Airmass': 165         # -
            }
        }
    
    def setup_directories(self):
        """创建输出目录结构"""
        directories = [
            'solar_radiation',
            'temperature_analysis', 
            'weather_conditions',
            'correlation_analysis',
            'time_series_analysis',
            'statistical_summary',
            'comparative_analysis',
            'quality_assessment'
        ]
        
        for dir_name in directories:
            (self.output_dir / dir_name).mkdir(parents=True, exist_ok=True)
        
        print(f"输出目录已创建: {self.output_dir}")
    
    def load_and_preprocess_data(self, file_path):
        """加载和预处理数据"""
        print(f"正在加载数据文件: {file_path}")
        
        # 读取数据
        data = pd.read_csv(file_path, header=None, sep=',')
        print(f"数据形状: {data.shape}")
        
        # 创建时间戳
        data['datetime'] = pd.to_datetime(
            data[0].astype(str) + '-' + 
            data[1].astype(str).str.zfill(2) + '-' + 
            data[2].astype(str).str.zfill(2) + ' ' + 
            data[3].astype(str).str.zfill(2) + ':' + 
            data[4].astype(str).str.zfill(2)
        )
        
        return data
    
    def plot_solar_radiation_analysis(self, data):
        """太阳辐射分析图表"""
        print("生成太阳辐射分析图表...")
        
        # 1. 主要辐射参数时间序列
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Solar Radiation Analysis', fontsize=16, fontweight='bold')
        
        # 全球水平辐射
        axes[0,0].plot(data['datetime'], data[5], alpha=0.7, linewidth=0.5)
        axes[0,0].set_title('Global Horizontal Irradiance (SR20)')
        axes[0,0].set_ylabel('Irradiance (W/m²)')
        axes[0,0].grid(True, alpha=0.3)
        
        # 直射辐射
        axes[0,1].plot(data['datetime'], data[7], alpha=0.7, linewidth=0.5, color='orange')
        axes[0,1].set_title('Direct Normal Irradiance (DR20)')
        axes[0,1].set_ylabel('Irradiance (W/m²)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 散射辐射
        axes[1,0].plot(data['datetime'], data[75], alpha=0.7, linewidth=0.5, color='green')
        axes[1,0].set_title('Diffuse Horizontal Irradiance')
        axes[1,0].set_ylabel('Irradiance (W/m²)')
        axes[1,0].set_xlabel('Time')
        axes[1,0].grid(True, alpha=0.3)
        
        # 净辐射
        axes[1,1].plot(data['datetime'], data[153], alpha=0.7, linewidth=0.5, color='red')
        axes[1,1].set_title('Net Radiation')
        axes[1,1].set_ylabel('Irradiance (W/m²)')
        axes[1,1].set_xlabel('Time')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'solar_radiation' / 'radiation_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 辐射日变化模式
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 按小时分组计算平均值
        hourly_radiation = data.groupby(data[3]).agg({
            5: 'mean',   # Global SR20
            7: 'mean',   # Direct DR20
            75: 'mean',  # Diffuse
            153: 'mean'  # Net Radiation
        })
        
        hours = hourly_radiation.index
        ax.plot(hours, hourly_radiation[5], label='Global Horizontal', linewidth=2, marker='o')
        ax.plot(hours, hourly_radiation[7], label='Direct Normal', linewidth=2, marker='s')
        ax.plot(hours, hourly_radiation[75], label='Diffuse Horizontal', linewidth=2, marker='^')
        ax.plot(hours, hourly_radiation[153], label='Net Radiation', linewidth=2, marker='d')
        
        ax.set_title('Daily Solar Radiation Patterns', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hour of Day')
        ax.set_ylabel('Average Irradiance (W/m²)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 23)
        
        plt.savefig(self.output_dir / 'solar_radiation' / 'daily_radiation_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 辐射分布直方图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Solar Radiation Distribution Analysis', fontsize=16, fontweight='bold')
        
        radiation_data = {
            'Global Horizontal (SR20)': data[5],
            'Direct Normal (DR20)': data[7],
            'Diffuse Horizontal': data[75],
            'Net Radiation': data[153]
        }
        
        for i, (title, values) in enumerate(radiation_data.items()):
            ax = axes[i//2, i%2]
            # 过滤有效数据
            valid_data = values[(values >= 0) & (values <= 1500)]
            ax.hist(valid_data, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title(title)
            ax.set_xlabel('Irradiance (W/m²)')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.1f}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'solar_radiation' / 'radiation_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_temperature_analysis(self, data):
        """温度分析图表"""
        print("生成温度分析图表...")
        
        # 1. 多点温度对比
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Temperature Analysis', fontsize=16, fontweight='bold')
        
        # 温度时间序列
        temp_cols = [73, 93, 97, 101, 181]  # 不同位置的温度
        temp_labels = ['Wet Bulb (Tower)', 'Dry Bulb (Deck)', 'Dry Bulb (Tower)', 
                      'Wind Chill (Tower)', 'Dew Point (Tower)']
        
        for col, label in zip(temp_cols, temp_labels):
            if col < data.shape[1]:
                valid_data = data[col][(data[col] >= -50) & (data[col] <= 50)]
                valid_datetime = data['datetime'][(data[col] >= -50) & (data[col] <= 50)]
                axes[0].plot(valid_datetime, valid_data, label=label, alpha=0.7, linewidth=0.8)
        
        axes[0].set_title('Temperature Time Series')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 温度日变化模式
        hourly_temp = data.groupby(data[3]).agg({
            73: 'mean',   # Wet Bulb
            93: 'mean',   # Dry Bulb Deck
            97: 'mean',   # Dry Bulb Tower
            181: 'mean'   # Dew Point
        })
        
        hours = hourly_temp.index
        axes[1].plot(hours, hourly_temp[73], label='Wet Bulb (Tower)', linewidth=2, marker='o')
        axes[1].plot(hours, hourly_temp[93], label='Dry Bulb (Deck)', linewidth=2, marker='s')
        axes[1].plot(hours, hourly_temp[97], label='Dry Bulb (Tower)', linewidth=2, marker='^')
        axes[1].plot(hours, hourly_temp[181], label='Dew Point (Tower)', linewidth=2, marker='d')
        
        axes[1].set_title('Daily Temperature Patterns')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Average Temperature (°C)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 23)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temperature_analysis' / 'temperature_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 温度与辐射关系
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 选择有效数据
        valid_mask = (data[5] >= 0) & (data[5] <= 1500) & (data[93] >= -30) & (data[93] <= 50)
        radiation_valid = data[5][valid_mask]
        temp_valid = data[93][valid_mask]
        
        # 散点图
        scatter = ax.scatter(radiation_valid, temp_valid, alpha=0.5, s=1)
        
        # 添加趋势线
        z = np.polyfit(radiation_valid, temp_valid, 1)
        p = np.poly1d(z)
        ax.plot(radiation_valid.sort_values(), p(radiation_valid.sort_values()), "r--", alpha=0.8, linewidth=2)
        
        # 计算相关系数
        correlation = np.corrcoef(radiation_valid, temp_valid)[0, 1]
        
        ax.set_title(f'Solar Radiation vs Temperature\nCorrelation: {correlation:.3f}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Global Horizontal Irradiance (W/m²)')
        ax.set_ylabel('Dry Bulb Temperature (°C)')
        ax.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / 'temperature_analysis' / 'radiation_temperature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_weather_conditions(self, data):
        """天气条件分析"""
        print("生成天气条件分析图表...")
        
        # 1. 风速风向分析
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Weather Conditions Analysis', fontsize=16, fontweight='bold')
        
        # 风速时间序列
        wind_speeds = [105, 109, 115]  # 6ft, 22ft, 33ft
        wind_labels = ['6ft', '22ft', '33ft']
        
        for ws, label in zip(wind_speeds, wind_labels):
            if ws < data.shape[1]:
                valid_data = data[ws][(data[ws] >= 0) & (data[ws] <= 30)]
                valid_datetime = data['datetime'][(data[ws] >= 0) & (data[ws] <= 30)]
                axes[0,0].plot(valid_datetime, valid_data, label=f'Wind Speed @ {label}', alpha=0.7)
        
        axes[0,0].set_title('Wind Speed Time Series')
        axes[0,0].set_ylabel('Wind Speed (m/s)')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 风向玫瑰图（简化版）
        if 107 < data.shape[1] and 105 < data.shape[1]:
            wind_dir = data[107][(data[107] >= 0) & (data[107] <= 360)]
            wind_speed = data[105][(data[107] >= 0) & (data[107] <= 360)]
            
            # 风向分布
            axes[0,1].hist(wind_dir, bins=36, alpha=0.7, edgecolor='black')
            axes[0,1].set_title('Wind Direction Distribution')
            axes[0,1].set_xlabel('Wind Direction (degrees from N)')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].grid(True, alpha=0.3)
        
        # 云量分析
        if 69 < data.shape[1]:
            cloud_cover = data[69][(data[69] >= 0) & (data[69] <= 100)]
            axes[1,0].hist(cloud_cover, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
            axes[1,0].set_title('Cloud Cover Distribution')
            axes[1,0].set_xlabel('Cloud Cover (%)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].grid(True, alpha=0.3)
        
        # 湿度分析
        if 95 < data.shape[1]:
            humidity = data[95][(data[95] >= 0) & (data[95] <= 100)]
            axes[1,1].hist(humidity, bins=20, alpha=0.7, edgecolor='black', color='lightgreen')
            axes[1,1].set_title('Relative Humidity Distribution')
            axes[1,1].set_xlabel('Relative Humidity (%)')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weather_conditions' / 'weather_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_correlation_analysis(self, data):
        """相关性分析"""
        print("生成相关性分析图表...")
        
        # 选择关键变量进行相关性分析
        key_variables = {
            'Global_Radiation': 5,
            'Direct_Radiation': 7,
            'Diffuse_Radiation': 75,
            'Temperature_Deck': 93,
            'Temperature_Tower': 97,
            'Humidity_Deck': 95,
            'Wind_Speed_6ft': 105,
            'Cloud_Cover': 69,
            'Pressure': 121
        }
        
        # 创建数据框
        corr_data = pd.DataFrame()
        for name, col_idx in key_variables.items():
            if col_idx < data.shape[1]:
                # 数据清洗
                if 'Radiation' in name:
                    valid_data = data[col_idx][(data[col_idx] >= 0) & (data[col_idx] <= 1500)]
                elif 'Temperature' in name:
                    valid_data = data[col_idx][(data[col_idx] >= -50) & (data[col_idx] <= 50)]
                elif 'Humidity' in name or 'Cloud' in name:
                    valid_data = data[col_idx][(data[col_idx] >= 0) & (data[col_idx] <= 100)]
                elif 'Wind' in name:
                    valid_data = data[col_idx][(data[col_idx] >= 0) & (data[col_idx] <= 30)]
                else:
                    valid_data = data[col_idx]
                
                # 重新索引以匹配长度
                if len(valid_data) > 0:
                    corr_data[name] = data[col_idx]
        
        # 计算相关矩阵
        correlation_matrix = corr_data.corr()
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        if HAS_SEABORN:
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        else:
            # 使用matplotlib实现热力图
            masked_corr = correlation_matrix.copy()
            masked_corr[mask] = np.nan
            
            im = ax.imshow(masked_corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='equal')
            
            # 添加文本注释
            for i in range(len(correlation_matrix)):
                for j in range(len(correlation_matrix)):
                    if not mask[i, j]:
                        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=8)
            
            # 设置坐标轴
            ax.set_xticks(range(len(correlation_matrix.columns)))
            ax.set_yticks(range(len(correlation_matrix.index)))
            ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            ax.set_yticklabels(correlation_matrix.index)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        ax.set_title('Correlation Matrix of Key Variables', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_analysis' / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_statistical_summary(self, data):
        """统计摘要"""
        print("生成统计摘要...")
        
        # 关键变量统计
        key_vars = {
            'Global Radiation (W/m²)': 5,
            'Direct Radiation (W/m²)': 7,
            'Temperature (°C)': 93,
            'Humidity (%)': 95,
            'Wind Speed (m/s)': 105,
            'Cloud Cover (%)': 69
        }
        
        stats_summary = []
        for name, col_idx in key_vars.items():
            if col_idx < data.shape[1]:
                col_data = data[col_idx]
                
                # 数据清洗
                if 'Radiation' in name:
                    valid_data = col_data[(col_data >= 0) & (col_data <= 1500)]
                elif 'Temperature' in name:
                    valid_data = col_data[(col_data >= -50) & (col_data <= 50)]
                elif 'Humidity' in name or 'Cloud' in name:
                    valid_data = col_data[(col_data >= 0) & (col_data <= 100)]
                elif 'Wind' in name:
                    valid_data = col_data[(col_data >= 0) & (col_data <= 30)]
                else:
                    valid_data = col_data
                
                if len(valid_data) > 0:
                    stats_summary.append({
                        'Variable': name,
                        'Count': len(valid_data),
                        'Mean': valid_data.mean(),
                        'Std': valid_data.std(),
                        'Min': valid_data.min(),
                        'Max': valid_data.max(),
                        'Median': valid_data.median()
                    })
        
        # 创建统计表格
        stats_df = pd.DataFrame(stats_summary)
        
        # 保存统计摘要
        stats_df.to_csv(self.output_dir / 'statistical_summary' / 'data_statistics.csv', index=False)
        
        # 绘制箱线图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Summary - Box Plots', fontsize=16, fontweight='bold')
        
        for i, (name, col_idx) in enumerate(key_vars.items()):
            if col_idx < data.shape[1] and i < 6:
                ax = axes[i//3, i%3]
                col_data = data[col_idx]
                
                # 数据清洗
                if 'Radiation' in name:
                    valid_data = col_data[(col_data >= 0) & (col_data <= 1500)]
                elif 'Temperature' in name:
                    valid_data = col_data[(col_data >= -50) & (col_data <= 50)]
                elif 'Humidity' in name or 'Cloud' in name:
                    valid_data = col_data[(col_data >= 0) & (col_data <= 100)]
                elif 'Wind' in name:
                    valid_data = col_data[(col_data >= 0) & (col_data <= 30)]
                else:
                    valid_data = col_data
                
                if len(valid_data) > 0:
                    ax.boxplot(valid_data)
                    ax.set_title(name)
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_summary' / 'box_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"统计摘要已保存到: {self.output_dir / 'statistical_summary' / 'data_statistics.csv'}")
    
    def generate_comprehensive_report(self, file_path):
        """生成综合分析报告"""
        print("开始生成综合分析报告...")
        
        # 加载数据
        data = self.load_and_preprocess_data(file_path)
        
        # 生成各类分析图表
        self.plot_solar_radiation_analysis(data)
        self.plot_temperature_analysis(data)
        self.plot_weather_conditions(data)
        self.plot_correlation_analysis(data)
        self.plot_statistical_summary(data)
        
        # 生成总结报告
        report_content = f"""
# PV数据综合分析报告

## 数据概览
- 数据文件: {file_path}
- 数据形状: {data.shape}
- 时间范围: {data['datetime'].min()} 至 {data['datetime'].max()}
- 总数据点: {len(data)}

## 分析内容

### 1. 太阳辐射分析
- 全球水平辐射时间序列
- 直射辐射变化模式
- 散射辐射分布
- 日变化规律分析

### 2. 温度分析
- 多点温度对比
- 温度日变化模式
- 温度与辐射相关性

### 3. 天气条件分析
- 风速风向分布
- 云量统计
- 湿度变化

### 4. 相关性分析
- 关键变量相关矩阵
- 变量间关系热力图

### 5. 统计摘要
- 描述性统计
- 数据分布箱线图

## 输出文件结构
```
{self.output_dir}/
├── solar_radiation/          # 太阳辐射分析
├── temperature_analysis/     # 温度分析
├── weather_conditions/       # 天气条件
├── correlation_analysis/     # 相关性分析
├── statistical_summary/      # 统计摘要
└── README.md                # 本报告
```

分析完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        # 保存报告
        with open(self.output_dir / 'README.md', 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n=== 分析完成 ===")
        print(f"所有图表和报告已保存到: {self.output_dir}")
        print(f"请查看 {self.output_dir / 'README.md'} 获取详细说明")

def main():
    parser = argparse.ArgumentParser(description='PV数据综合可视化分析')
    parser.add_argument('input_file', help='输入数据文件路径')
    parser.add_argument('-o', '--output', default='visualization_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = PVDataVisualizer(args.output)
    
    # 生成综合报告
    visualizer.generate_comprehensive_report(args.input_file)

if __name__ == "__main__":
    main()