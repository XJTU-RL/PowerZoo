#!/usr/bin/env python3
"""
气象数据转换脚本 - 将MIDC格式数据转换为OpenDSS PV系统可用格式
支持批量处理txt目录下的所有数据文件
作者: AI Assistant
日期: 2025-01-27
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import logging
import glob
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MIDCtoOpenDSSConverter:
    """MIDC气象数据到OpenDSS PV数据转换器"""
    
    def __init__(self):
        # 字段索引定义（从0开始，根据filed_definition.md更新）
        self.field_indices = {
            'year': 0,
            'month': 1,
            'day': 2,
            'hour': 3,
            'minute': 4,
            'global_sr20': 5,           # Global SR20 (vent./corrected)
            'global_sr20_flag': 6,
            'direct_dr20': 7,           # Direct DR20 (corrected)
            'direct_dr20_flag': 8,
            'global_li200': 9,          # Global LI-200
            'global_li200_flag': 10,
            'global_cmp22': 11,         # Global CMP22 (vent./corrected)
            'global_cmp22_flag': 12,
            'global_psp': 137,          # Global PSP (corrected)
            'global_psp_flag': 138,
            'temp_deck': 93,            # Dry Bulb Temperature (Deck)
            'temp_deck_flag': 94,
            'temp_tower': 97,           # Dry Bulb Temperature (Tower)
            'temp_tower_flag': 98,
            'temp_se': 203,             # Dry Bulb Temperature (SE)
            'temp_se_flag': 204
        }
        
        # 数据质量阈值
        self.quality_threshold = 3  # Flag <= 3 认为是可用数据（根据实际数据调整）
        
    def read_midc_data(self, filepath):
        """读取MIDC格式的数据文件"""
        logger.info(f"读取文件: {filepath}")
        
        try:
            # 读取数据，假设没有表头
            data = []
            with open(filepath, 'r') as f:
                for line in f:
                    if line.strip():  # 跳过空行
                        values = line.strip().split(',')
                        data.append([float(v) for v in values])
            
            df = pd.DataFrame(data)
            logger.info(f"成功读取 {len(df)} 行数据")
            return df
            
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            raise
    
    def select_best_irradiance(self, row):
        """选择最佳的辐照度数据"""
        # 优先级顺序：PSP > CMP22 > SR20 > Direct DR20 > LI200
        irrad_sources = [
            ('global_psp', 'global_psp_flag'),
            ('global_cmp22', 'global_cmp22_flag'),
            ('global_sr20', 'global_sr20_flag'),
            ('direct_dr20', 'direct_dr20_flag'),
            ('global_li200', 'global_li200_flag')
        ]
        
        for irrad_field, flag_field in irrad_sources:
            irrad_idx = self.field_indices[irrad_field]
            flag_idx = self.field_indices[flag_field]
            
            if row[flag_idx] <= self.quality_threshold:
                value = row[irrad_idx]
                # 将负值或异常值设为0
                if value < 0 or value > 1500:  # 太阳常数约1367 W/m²
                    return 0
                return value
        
        return 0  # 如果所有数据都不可用，返回0
    
    def select_best_temperature(self, row):
        """选择最佳的温度数据"""
        # 优先级顺序：Tower > Deck > SE
        temp_sources = [
            ('temp_tower', 'temp_tower_flag'),
            ('temp_deck', 'temp_deck_flag'),
            ('temp_se', 'temp_se_flag')
        ]
        
        valid_temps = []  # 存储所有有效温度值
        
        for temp_field, flag_field in temp_sources:
            temp_idx = self.field_indices[temp_field]
            flag_idx = self.field_indices[flag_field]
            
            value = row[temp_idx]
            # 温度合理性检查（-50°C 到 70°C，扩大范围以包含更多真实数据）
            if -50 <= value <= 70:
                # 优先使用质量标志好的数据
                if row[flag_idx] <= self.quality_threshold:
                    return value
                else:
                    # 质量标志不佳但数值合理的数据作为备选
                    valid_temps.append(value)
        
        # 如果没有质量好的数据，使用第一个合理的温度值
        if valid_temps:
            return valid_temps[0]
        
        return 25  # 如果所有数据都不可用，返回标准温度25°C
    
    def process_data(self, df):
        """处理数据并创建时序数据"""
        logger.info("处理数据...")
        
        processed_data = []
        
        for idx, row in df.iterrows():
            try:
                # 创建时间戳
                year = int(row[self.field_indices['year']])
                month = int(row[self.field_indices['month']])
                day = int(row[self.field_indices['day']])
                hour = int(row[self.field_indices['hour']])
                minute = int(row[self.field_indices['minute']])
                
                timestamp = datetime(year, month, day, hour, minute)
                
                # 获取辐照度和温度
                irradiance = self.select_best_irradiance(row)
                temperature = self.select_best_temperature(row)
                
                # 将辐照度从W/m²转换为标幺值（基准1000 W/m²）
                irrad_pu = irradiance / 1000.0
                
                processed_data.append({
                    'timestamp': timestamp,
                    'hour': hour + minute/60.0,  # 小时格式（带小数）
                    'irradiance_wm2': irradiance,
                    'irradiance_pu': irrad_pu,
                    'temperature_c': temperature
                })
                
            except Exception as e:
                logger.warning(f"处理第{idx}行数据时出错: {e}")
                continue
        
        result_df = pd.DataFrame(processed_data)
        logger.info(f"成功处理 {len(result_df)} 条有效数据")
        
        return result_df
    
    def save_opendss_files(self, df, output_dir, base_name):
        """保存OpenDSS所需的文件"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 保存完整数据CSV
        full_csv_path = output_path / f"{base_name}_complete_data.csv"
        df.to_csv(full_csv_path, index=False)
        logger.info(f"保存完整数据到: {full_csv_path}")
        
        # 2. 保存辐照度时序文件（OpenDSS Duty）- 无表头格式
        irrad_csv_path = output_path / f"{base_name}_irradiance.csv"
        irrad_data = df['irradiance_pu'].copy()
        # 保存为无表头的单列数据
        irrad_data.to_csv(irrad_csv_path, index=False, header=False)
        logger.info(f"保存辐照度数据到: {irrad_csv_path}")
        
        # 3. 保存温度时序文件（OpenDSS TDuty）- 无表头格式
        temp_csv_path = output_path / f"{base_name}_temperature.csv"
        temp_data = df['temperature_c'].copy()
        # 保存为无表头的单列数据
        temp_data.to_csv(temp_csv_path, index=False, header=False)
        logger.info(f"保存温度数据到: {temp_csv_path}")
        
        # 4. 保存带表头的完整时序文件（用于分析）
        irrad_analysis_path = output_path / f"{base_name}_irradiance_timeseries.csv"
        irrad_data_full = df[['hour', 'irradiance_pu']].copy()
        irrad_data_full.columns = ['Hour', 'Mult']
        irrad_data_full.to_csv(irrad_analysis_path, index=False)
        
        temp_analysis_path = output_path / f"{base_name}_temperature_timeseries.csv"
        temp_data_full = df[['hour', 'temperature_c']].copy()
        temp_data_full.columns = ['Hour', 'Temp']
        temp_data_full.to_csv(temp_analysis_path, index=False)
        
        # 4. 生成OpenDSS命令文件
        dss_cmd_path = output_path / f"{base_name}_opendss_commands.dss"
        with open(dss_cmd_path, 'w') as f:
            f.write("! OpenDSS PV System Commands\n")
            f.write(f"! Generated from {base_name} data\n\n")
            
            # 定义LoadShape - 使用无表头的CSV文件
            f.write(f"New LoadShape.MyIrrad npts={len(df)} sinterval=60 mult=(file={irrad_csv_path.name})\n")
            f.write(f"New TShape.MyTemp npts={len(df)} sinterval=60 temp=(file={temp_csv_path.name})\n\n")
            
            # 定义效率曲线
            f.write("New XYCurve.Myeff npts=4 xarray=[.1 .2 .4 1.0] yarray=[.86 .9 .93 .97]\n")
            f.write("New XYCurve.MyPvsT npts=4 xarray=[0 25 75 100] yarray=[1.2 1.0 0.8 0.6]\n\n")
            
            # PV系统示例
            f.write("! Example PV System Definition\n")
            f.write("New PVSystem.PV834 phases=3 bus1=trafo_pv kV=0.48 kVA=200 ")
            f.write(f"irrad={df['irradiance_pu'].mean():.3f} Pmpp=180 ")
            f.write(f"temperature={df['temperature_c'].mean():.1f} PF=1 ")
            f.write("%cutin=0.1 %cutout=0.1 effcurve=Myeff P-TCurve=MyPvsT ")
            f.write("Duty=MyIrrad TDuty=MyTemp\n")
            
        logger.info(f"保存OpenDSS命令到: {dss_cmd_path}")
        
        # 5. 生成数据统计报告
        stats_path = output_path / f"{base_name}_data_statistics.txt"
        with open(stats_path, 'w') as f:
            f.write(f"数据统计报告 - {base_name}\n")
            f.write("="*50 + "\n\n")
            f.write(f"数据时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}\n")
            f.write(f"数据点数: {len(df)}\n\n")
            
            f.write("辐照度统计 (W/m²):\n")
            f.write(f"  平均值: {df['irradiance_wm2'].mean():.2f}\n")
            f.write(f"  最大值: {df['irradiance_wm2'].max():.2f}\n")
            f.write(f"  最小值: {df['irradiance_wm2'].min():.2f}\n")
            f.write(f"  标准差: {df['irradiance_wm2'].std():.2f}\n\n")
            
            f.write("温度统计 (°C):\n")
            f.write(f"  平均值: {df['temperature_c'].mean():.2f}\n")
            f.write(f"  最大值: {df['temperature_c'].max():.2f}\n")
            f.write(f"  最小值: {df['temperature_c'].min():.2f}\n")
            f.write(f"  标准差: {df['temperature_c'].std():.2f}\n")
            
        logger.info(f"保存统计报告到: {stats_path}")
    
    def process_batch(self, input_dir, output_dir=None):
        """批量处理txt目录下的所有数据文件"""
        input_path = Path(input_dir)
        
        # 设置输出目录为同级的csv文件夹
        if output_dir is None:
            output_dir = input_path.parent / 'csv'
        else:
            output_dir = Path(output_dir)
        
        # 创建输出目录
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")
        
        # 查找所有txt文件
        txt_files = list(input_path.glob('*.txt'))
        if not txt_files:
            logger.error(f"在目录 {input_path} 中未找到任何.txt文件")
            return False
        
        logger.info(f"找到 {len(txt_files)} 个数据文件")
        
        success_count = 0
        for txt_file in sorted(txt_files):
            try:
                logger.info(f"\n处理文件: {txt_file.name}")
                
                # 生成输出文件的基础名称（去掉.txt扩展名）
                base_name = txt_file.stem
                
                # 读取数据
                df = self.read_midc_data(txt_file)
                
                # 处理数据
                processed_df = self.process_data(df)
                
                if len(processed_df) == 0:
                    logger.warning(f"文件 {txt_file.name} 处理后无有效数据")
                    continue
                
                # 保存结果
                self.save_opendss_files(processed_df, output_dir, base_name)
                
                success_count += 1
                logger.info(f"文件 {txt_file.name} 处理完成")
                
            except Exception as e:
                logger.error(f"处理文件 {txt_file.name} 时出错: {e}")
                continue
        
        logger.info(f"\n批量处理完成！成功处理 {success_count}/{len(txt_files)} 个文件")
        return success_count > 0

def main():
    parser = argparse.ArgumentParser(description='将MIDC气象数据转换为OpenDSS PV系统格式')
    parser.add_argument('input_path', help='输入文件路径或包含txt文件的目录路径')
    parser.add_argument('-o', '--output_dir', default=None, 
                        help='输出目录路径（默认: 输入目录同级的csv文件夹）')
    parser.add_argument('-n', '--name', default='pv_data',
                        help='单文件模式下输出文件的基础名称（默认: pv_data）')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理模式：处理目录下所有txt文件')
    
    args = parser.parse_args()
    
    converter = MIDCtoOpenDSSConverter()
    input_path = Path(args.input_path)
    
    try:
        if args.batch or input_path.is_dir():
            # 批量处理模式
            logger.info("批量处理模式")
            if not input_path.is_dir():
                logger.error(f"批量模式需要提供目录路径，但得到: {input_path}")
                return 1
            
            success = converter.process_batch(input_path, args.output_dir)
            if not success:
                return 1
                
        else:
            # 单文件处理模式
            logger.info("单文件处理模式")
            if not input_path.is_file():
                logger.error(f"文件不存在: {input_path}")
                return 1
            
            # 设置输出目录
            if args.output_dir is None:
                output_dir = input_path.parent / 'csv'
            else:
                output_dir = Path(args.output_dir)
            
            # 读取数据
            df = converter.read_midc_data(input_path)
            
            # 处理数据
            processed_df = converter.process_data(df)
            
            # 保存结果
            converter.save_opendss_files(processed_df, output_dir, args.name)
        
        logger.info("转换完成！")
        
    except Exception as e:
        logger.error(f"转换失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())