#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按天处理PV数据脚本
将原始数据按月份和日期分别存储到不同文件夹中
月份文件夹格式：2025-01
日期文件夹格式：01, 02, 03...
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DailyDataProcessor:
    def __init__(self):
        # 字段索引定义
        self.field_indices = {
            'datetime': 0,
            'ghi_tower': 7, 'ghi_tower_flag': 8,
            'ghi_deck': 15, 'ghi_deck_flag': 16,
            'ghi_se': 23, 'ghi_se_flag': 24,
            'temp_tower': 93, 'temp_tower_flag': 94,
            'temp_deck': 97, 'temp_deck_flag': 98,
            'temp_se': 203, 'temp_se_flag': 204
        }
        self.quality_threshold = 3
    
    def select_best_irradiance(self, row):
        """选择最佳辐照度数据"""
        sources = [('ghi_tower', 'ghi_tower_flag'), ('ghi_deck', 'ghi_deck_flag'), ('ghi_se', 'ghi_se_flag')]
        
        for irrad_field, flag_field in sources:
            value = row[self.field_indices[irrad_field]]
            flag = row[self.field_indices[flag_field]]
            if 0 <= value <= 1500 and flag <= self.quality_threshold:
                return value
        return 0
    
    def select_best_temperature(self, row):
        """选择最佳温度数据"""
        sources = [('temp_tower', 'temp_tower_flag'), ('temp_deck', 'temp_deck_flag'), ('temp_se', 'temp_se_flag')]
        
        for temp_field, flag_field in sources:
            value = row[self.field_indices[temp_field]]
            flag = row[self.field_indices[flag_field]]
            if -50 <= value <= 70 and flag <= self.quality_threshold:
                return value
        return 25
    
    def process_file_by_days(self, input_file, output_base_dir):
        """按天处理文件"""
        logger.info(f"处理文件: {input_file}")
        
        # 读取数据文件，使用逗号分隔符
        data = pd.read_csv(input_file, header=None, sep=',')
        print(f"读取了 {len(data)} 行数据")
        
        daily_data = {}
        
        # 按行处理数据
        for index, row in data.iterrows():
            try:
                # 解析时间戳 - 前5列为年、月、日、时、分
                year, month, day, hour_int, minute = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])
                timestamp = pd.Timestamp(year=year, month=month, day=day, hour=hour_int, minute=minute)
                date_str = timestamp.strftime('%Y-%m-%d')
                hour = hour_int + minute / 60.0
                
                if date_str not in daily_data:
                    daily_data[date_str] = []
                
                # 处理数据
                irradiance = self.select_best_irradiance(row)
                temperature = self.select_best_temperature(row)
                
                daily_data[date_str].append({
                    'datetime': timestamp,
                    'hour': hour,
                    'irradiance': irradiance,
                    'irradiance_mult': irradiance / 1000.0,
                    'temperature': temperature
                })
            except Exception as e:
                if index < 10:  # 只打印前几个错误
                    logger.warning(f"处理第 {index} 行数据时出错: {e}")
                continue
        
        logger.info(f"按日期分组完成，共找到 {len(daily_data)} 个日期")
        
        # 保存每日数据
        for date_str, day_data in daily_data.items():
            if not day_data:
                continue
                
            df = pd.DataFrame(day_data)
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # 创建目录结构：output_base_dir/2025-01/01/
            year_month = date_obj.strftime('%Y-%m')
            day = date_obj.strftime('%d')
            output_dir = os.path.join(output_base_dir, year_month, day)
            os.makedirs(output_dir, exist_ok=True)
            
            # 文件前缀
            file_prefix = date_str.replace('-', '')
            
            # 保存文件
            self.save_daily_files(df, output_dir, file_prefix)
            logger.info(f"保存日期 {date_str} 数据到 {output_dir}")
    
    def save_daily_files(self, df, output_dir, file_prefix):
        """保存单日文件"""
        # 1. 完整数据
        df.to_csv(os.path.join(output_dir, f"{file_prefix}_complete_data.csv"), index=False)
        
        # 2. 辐照度时序
        df[['hour', 'irradiance_mult']].rename(columns={'irradiance_mult': 'Mult'}).to_csv(
            os.path.join(output_dir, f"{file_prefix}_irradiance_timeseries.csv"), index=False)
        
        # 3. 温度时序
        df[['hour', 'temperature']].rename(columns={'hour': 'Hour', 'temperature': 'Temp'}).to_csv(
            os.path.join(output_dir, f"{file_prefix}_temperature_timeseries.csv"), index=False)
        
        # 4. OpenDSS文件
        npts = len(df)
        avg_irrad = df['irradiance_mult'].mean()
        avg_temp = df['temperature'].mean()
        
        dss_content = f"""! OpenDSS PV System Commands - {file_prefix}

New LoadShape.MyIrrad npts={npts} interval=1 mult=(file={file_prefix}_irradiance_timeseries.csv)
New LoadShape.MyTemp npts={npts} interval=1 mult=(file={file_prefix}_temperature_timeseries.csv)

New XYCurve.Myeff npts=4 xarray=[.1 .2 .4 1.0] yarray=[.86 .9 .93 .97]
New XYCurve.MyPvsT npts=4 xarray=[0 25 75 100] yarray=[1.2 1.0 0.8 0.6]

New PVSystem.PV834 phases=3 bus1=trafo_pv kV=0.48 kVA=200 irrad={avg_irrad:.3f} Pmpp=180 temperature={avg_temp:.1f} PF=1 %cutin=0.1 %cutout=0.1 effcurve=Myeff P-TCurve=MyPvsT Duty=MyIrrad TDuty=MyTemp
"""
        
        with open(os.path.join(output_dir, f"{file_prefix}_opendss_commands.dss"), 'w') as f:
            f.write(dss_content)
        
        # 5. 统计报告
        stats = f"""数据统计报告 - {file_prefix}
==================================================

数据时间范围: {df['datetime'].min()} 至 {df['datetime'].max()}
数据点数: {len(df)}

辐照度统计 (W/m²):
  平均值: {df['irradiance'].mean():.2f}
  最大值: {df['irradiance'].max():.2f}
  最小值: {df['irradiance'].min():.2f}

温度统计 (°C):
  平均值: {df['temperature'].mean():.2f}
  最大值: {df['temperature'].max():.2f}
  最小值: {df['temperature'].min():.2f}
"""
        
        with open(os.path.join(output_dir, f"{file_prefix}_data_statistics.txt"), 'w') as f:
            f.write(stats)
    
    def process_batch(self, input_dir, output_dir):
        """批量处理"""
        txt_files = [f for f in os.listdir(input_dir) if f.endswith('.txt')]
        logger.info(f"找到 {len(txt_files)} 个文件")
        
        for txt_file in sorted(txt_files):
            input_path = os.path.join(input_dir, txt_file)
            self.process_file_by_days(input_path, output_dir)
        
        logger.info("批量处理完成")

def main():
    parser = argparse.ArgumentParser(description='按天处理PV数据')
    parser.add_argument('input_path', help='输入文件或目录路径')
    parser.add_argument('-o', '--output_dir', default='daily_data', help='输出目录')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    
    args = parser.parse_args()
    processor = DailyDataProcessor()
    
    if args.batch:
        processor.process_batch(args.input_path, args.output_dir)
    else:
        processor.process_file_by_days(args.input_path, args.output_dir)

if __name__ == '__main__':
    main()