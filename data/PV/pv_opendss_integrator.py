#!/usr/bin/env python3
"""
PV数据与OpenDSS集成工具

功能：
1. 将PV日数据转换为OpenDSS兼容格式
2. 自动修改DSS配置文件
3. 支持任意日期的PV数据集成
4. 提供批量处理和验证功能

作者：Sheldon Zheng
日期：2025-07-27
"""

import os
import sys
import shutil
import argparse
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

class PVOpenDSSIntegrator:
    def __init__(self, pv_data_dir, dss_project_dir, max_workers=10):
        """
        初始化PV-OpenDSS集成器
        
        Args:
            pv_data_dir: PV数据目录路径
            dss_project_dir: OpenDSS项目目录路径
            max_workers: 最大worker数量，用于创建多worker数据
        """
        self.pv_data_dir = Path(pv_data_dir)
        self.dss_project_dir = Path(dss_project_dir)
        self.max_workers = max_workers
        
        # 创建必要的目录
        self.irradiation_dir = Path(dss_project_dir) / 'irradiation'
        self.temperature_dir = Path(dss_project_dir) / 'temperature'
        
        self.irradiation_dir.mkdir(parents=True, exist_ok=True)
        self.temperature_dir.mkdir(parents=True, exist_ok=True)
        
        # 为每个worker创建子目录
        for worker_idx in range(self.max_workers):
            worker_str = f"{worker_idx:03d}"
            (self.irradiation_dir / worker_str).mkdir(exist_ok=True)
            (self.temperature_dir / worker_str).mkdir(exist_ok=True)
    
    def get_available_dates(self):
        """
        获取可用的PV数据日期列表
        
        Returns:
            list: 可用日期列表 (YYYY-MM-DD格式)
        """
        dates = []
        daily_data_dir = self.pv_data_dir / "daily_data"
        
        if not daily_data_dir.exists():
            print(f"错误：PV日数据目录不存在: {daily_data_dir}")
            return dates
        
        for year_month in daily_data_dir.iterdir():
            if year_month.is_dir():
                for day in year_month.iterdir():
                    if day.is_dir():
                        date_str = f"{year_month.name}-{day.name.zfill(2)}"
                        dates.append(date_str)
        
        return sorted(dates)
    
    def convert_pv_data_format(self, date_str):
        """
        转换PV数据格式为OpenDSS兼容格式
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD)
            
        Returns:
            tuple: (辐照度文件路径, 温度文件路径, 数据点数)
        """
        # 解析日期
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            year_month = date_obj.strftime("%Y-%m")
            day = date_obj.strftime("%d")
            date_compact = date_obj.strftime("%Y%m%d")
        except ValueError:
            raise ValueError(f"无效的日期格式: {date_str}，应为YYYY-MM-DD")
        
        # 构建源文件路径
        source_dir = self.pv_data_dir / "daily_data" / year_month / day
        irrad_source = source_dir / f"{date_compact}_irradiance_timeseries.csv"
        temp_source = source_dir / f"{date_compact}_temperature_timeseries.csv"
        
        if not irrad_source.exists() or not temp_source.exists():
            raise FileNotFoundError(f"PV数据文件不存在: {date_str}")
        
        # 读取并验证数据
        irrad_df = pd.read_csv(irrad_source)
        temp_df = pd.read_csv(temp_source)
        
        npts = len(irrad_df)
        print(f"数据点数: {npts}")
        print(f"辐照度范围: {irrad_df['Mult'].min():.6f} - {irrad_df['Mult'].max():.6f}")
        print(f"温度范围: {temp_df['Temp'].min():.1f} - {temp_df['Temp'].max():.1f} °C")
        
        # 为每个worker复制文件到OpenDSS目录
        target_files = []
        for worker_idx in range(self.max_workers):
            worker_str = f"{worker_idx:03d}"
            
            irrad_dest = self.irradiation_dir / worker_str / f"{date_compact}_irradiance.csv"
            temp_dest = self.temperature_dir / worker_str / f"{date_compact}_temperature.csv"
            
            # 确保目标目录存在
            irrad_dest.parent.mkdir(parents=True, exist_ok=True)
            temp_dest.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(irrad_source, irrad_dest)
            shutil.copy2(temp_source, temp_dest)
            
            target_files.append((irrad_dest, temp_dest))
        
        print(f"文件已复制到 {self.max_workers} 个worker目录:")
        print(f"  辐照度: {self.irradiation_dir}/000-{self.max_workers-1:03d}/{date_compact}_irradiance.csv")
        print(f"  温度: {self.temperature_dir}/000-{self.max_workers-1:03d}/{date_compact}_temperature.csv")
        
        return target_files[0][0], target_files[0][1], npts
    
    def create_dss_config(self, date_str, irrad_file, temp_file, npts, 
                         sinterval=3600, pmpp=10, irradiance=1.0, temperature=25):
        """创建基础DSS配置文件"""
        
        date_compact = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
        
        dss_content = f"""! PV系统数据配置文件
! 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
! 注意: 路径中的000会在运行时被动态替换为worker_idx

! 定义PV温度系数曲线
New XYCurve.MyPvsT npts=4 xarray=[0 25 75 100] yarray=[1.2 1.0 0.8 0.6]

! 定义PV效率曲线
New XYCurve.MyEff npts=4 xarray=[.1 .2 1.0 1.2] yarray=[.86 .9 .98 .99]

! 定义辐照度LoadShape (路径中的000会被动态替换)
New Loadshape.MyIrrad npts={npts} sinterval={sinterval} mult=(file=./irradiation/000/{date_compact}_irradiance.csv)

! 定义温度TShape (路径中的000会被动态替换)
New TShape.MyTemp npts={npts} sinterval={sinterval} temp=(file=./temperature/000/{date_compact}_temperature.csv)
"""
        
        # 写入配置文件
        config_file = self.dss_project_dir / 'pv_data.dss'
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(dss_content)
        
        print(f"PV数据配置文件已创建: {config_file}")
        return config_file
    
    def _create_basic_dss_config(self, date_str, irrad_file, temp_file, npts, output_file):
        """
        创建基础的DSS配置文件
        """
        date_compact = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
        irrad_rel_path = os.path.relpath(irrad_file, self.dss_project_dir)
        temp_rel_path = os.path.relpath(temp_file, self.dss_project_dir)
        
        config_content = f"""! OpenDSS Configuration for PV Data - {date_str}
! Generated by PV-OpenDSS Integrator
! Data points: {npts}, Interval: 1 minute

! 设置仿真模式
Set Mode=duty
Set stepsize=60s
Set number={npts}

! 定义负载形状
New LoadShape.MyIrrad npts={npts} interval=1 mult=(file={irrad_rel_path})
New TShape.MyTemp npts={npts} interval=1 temp=(file={temp_rel_path})

! 定义效率曲线
New XYCurve.Myeff npts=4 xarray=[.1 .2 .4 1.0] yarray=[.86 .9 .93 .97]
New XYCurve.MyPvsT npts=4 xarray=[0 25 75 100] yarray=[1.2 1.0 0.8 0.6]

! 定义PV系统
New PVSystem.PV_{date_compact} phases=3 bus1=trafo_pv kV=0.48 kVA=200 irrad=0.5 Pmpp=180 temperature=25 PF=1 %cutin=0.1 %cutout=0.1 effcurve=Myeff P-TCurve=MyPvsT Duty=MyIrrad TDuty=MyTemp

! 求解
Solve
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
    
    def _update_simulation_params(self, content, npts):
        """
        更新仿真参数
        """
        import re
        
        # 更新仿真模式设置行 - 匹配完整的Set mode行
        mode_pattern = r'Set mode=duty number=\d+ hour=\d+ stepsize=\d+ sec=\d+'
        mode_replacement = f'Set mode=duty number={npts} hour=0 stepsize=60 sec=0'
        content = re.sub(mode_pattern, mode_replacement, content)
        
        # 备用：单独更新步长和点数（如果上面的模式没有匹配到）
        content = re.sub(r'Set stepsize=\d+s?', 'Set stepsize=60s', content)
        content = re.sub(r'Set number=\d+', f'Set number={npts}', content)
        
        return content
    
    def _update_loadshape_config(self, content, irrad_path, temp_path, npts):
        """
        更新LoadShape配置
        """
        import re
        
        # 更新辐照度LoadShape - 匹配原始文件中的sinterval格式
        irrad_pattern = r'New Loadshape\.MyIrrad\s+npts=\d+\s+sinterval=\d+\s+mult=\([^)]+\)'
        irrad_replacement = f'New Loadshape.MyIrrad npts={npts} interval=1 mult=(file={irrad_path})'
        content = re.sub(irrad_pattern, irrad_replacement, content, flags=re.IGNORECASE)
        
        # 更新温度TShape
        temp_pattern = r'New TShape\.MyTemp\s+npts=\d+\s+interval=\d+\s+temp=\([^)]+\)'
        temp_replacement = f'New TShape.MyTemp npts={npts} interval=1 temp=(file={temp_path})'
        content = re.sub(temp_pattern, temp_replacement, content, flags=re.IGNORECASE)
        
        return content
    
    def integrate_pv_data(self, date_str, output_dss_file=None):
        """
        完整的PV数据集成流程
        
        Args:
            date_str: 日期字符串 (YYYY-MM-DD)
            output_dss_file: 输出DSS文件路径
            
        Returns:
            str: 生成的DSS文件路径
        """
        print(f"\n=== 集成PV数据: {date_str} ===")
        
        # 1. 转换数据格式
        print("\n1. 转换PV数据格式...")
        irrad_file, temp_file, npts = self.convert_pv_data_format(date_str)
        
        # 2. 创建DSS配置
        print("\n2. 创建OpenDSS配置文件...")
        dss_file = self.create_dss_config(date_str, irrad_file, temp_file, npts, sinterval=60)
        
        print(f"\n=== 集成完成 ===")
        print(f"DSS文件: {dss_file}")
        print(f"使用方法: 在OpenDSS中运行 'compile {dss_file}'")
        
        return dss_file
    
    def batch_integrate(self, start_date=None, end_date=None, max_files=None):
        """
        批量集成PV数据
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            max_files: 最大处理文件数
        """
        available_dates = self.get_available_dates()
        
        if not available_dates:
            print("没有找到可用的PV数据")
            return
        
        # 过滤日期范围
        if start_date:
            available_dates = [d for d in available_dates if d >= start_date]
        if end_date:
            available_dates = [d for d in available_dates if d <= end_date]
        if max_files:
            available_dates = available_dates[:max_files]
        
        print(f"\n=== 批量集成 {len(available_dates)} 个日期的PV数据 ===")
        
        success_count = 0
        for i, date_str in enumerate(available_dates, 1):
            try:
                print(f"\n[{i}/{len(available_dates)}] 处理日期: {date_str}")
                self.integrate_pv_data(date_str)
                success_count += 1
            except Exception as e:
                print(f"错误：处理日期 {date_str} 失败: {e}")
        
        print(f"\n=== 批量集成完成 ===")
        print(f"成功处理: {success_count}/{len(available_dates)} 个文件")

def main():
    parser = argparse.ArgumentParser(description='PV数据与OpenDSS集成工具')
    parser.add_argument('--pv-data-dir', default='/home/zhengxiaodong/exps/DeepVVC-agent/data/PV',
                       help='PV数据目录路径')
    parser.add_argument('--dss-project-dir', default='/home/zhengxiaodong/exps/PowerZoo/node_systems/34BUS',
                       help='OpenDSS项目目录路径')
    parser.add_argument('--date', help='指定日期 (YYYY-MM-DD格式)')
    parser.add_argument('--max-workers', type=int, default=1, help='最大worker数量')
    parser.add_argument('--list-dates', action='store_true', help='列出可用的日期')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    parser.add_argument('--start-date', help='批量处理开始日期')
    parser.add_argument('--end-date', help='批量处理结束日期')
    parser.add_argument('--max-files', type=int, help='最大处理文件数')
    parser.add_argument('--output', help='输出DSS文件路径')
    
    args = parser.parse_args()
    
    # 创建集成器
    integrator = PVOpenDSSIntegrator(args.pv_data_dir, args.dss_project_dir, args.max_workers)
    
    if args.list_dates:
        # 列出可用日期
        dates = integrator.get_available_dates()
        print(f"\n可用的PV数据日期 ({len(dates)} 个):")
        for date in dates:
            print(f"  {date}")
        return
    
    if args.batch:
        # 批量处理
        integrator.batch_integrate(args.start_date, args.end_date, args.max_files)
    elif args.date:
        # 单个日期处理
        integrator.integrate_pv_data(args.date, args.output)
    else:
        # 交互模式
        dates = integrator.get_available_dates()
        if not dates:
            print("没有找到可用的PV数据")
            return
        
        print(f"\n可用的PV数据日期:")
        for i, date in enumerate(dates[:10], 1):  # 显示前10个
            print(f"  {i}. {date}")
        if len(dates) > 10:
            print(f"  ... 还有 {len(dates)-10} 个日期")
        
        try:
            choice = input(f"\n请选择日期 (1-{min(10, len(dates))}) 或输入日期 (YYYY-MM-DD): ").strip()
            
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < min(10, len(dates)):
                    selected_date = dates[idx]
                else:
                    print("无效的选择")
                    return
            else:
                selected_date = choice
            
            integrator.integrate_pv_data(selected_date, args.output)
            
        except KeyboardInterrupt:
            print("\n操作已取消")
        except Exception as e:
            print(f"错误: {e}")

if __name__ == '__main__':
    main()