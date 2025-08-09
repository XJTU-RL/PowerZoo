#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
光伏数据提取脚本（全数据扫描版）
从data/PV/daily_data/下所有月份的每日光伏数据中提取指定时间段的辐照度和温度数据
输出格式：按月份和时间段组织，无表头纯数据

文件夹命名格式： 
1. 年-月-时间间隔（比如8点-18点）/日期/irrad.csv 
2. 年-月-时间间隔/日期/temp.csv

功能特点:
- 自动扫描所有可用月份数据
- 支持自定义时间范围
- 按月份分别组织输出
- 详细的处理进度和统计信息
- 支持命令行参数配置


🚀 使用示例

# 按天分割所有月份数据
python data/PV/extract_pv_data_from_csv.py --split-by-day

# 按天分割并指定输出目录
python data/PV/extract_pv_data_from_csv.py --split-by-day --output-dir data/PV/daily_segments

# 按天分割指定时间段（8-16点）
python data/PV/extract_pv_data_from_csv.py --split-by-day --start-hour 8.0 --end-hour 17.0

作者: Xiaodong Zheng
创建时间: 2025-07-29
更新时间: 2025-07-30
"""

import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Optional, List
import sys
import time

# 配置日志
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class PVDataExtractor:
	"""光伏数据提取器 - 支持全数据扫描"""
	
	def __init__(self, 
	             base_dir: str = "data/PV/daily_data",
	             start_hour: float = 6.0, 
	             end_hour: float = 18.0,
	             output_base_dir: str = "data/PV/hourly_segments",
	             split_by_day: bool = False):
		"""
		初始化数据提取器
		
		Args:
			base_dir: 数据源根目录路径
			start_hour: 开始时间（小时）
			end_hour: 结束时间（小时，不包含）
			output_base_dir: 输出根目录
			split_by_day: 是否按天分割数据存储
		"""
		self.base_dir = Path(base_dir)
		self.output_base_dir = Path(output_base_dir)
		self.split_by_day = split_by_day
		
		# 时间过滤条件
		self.start_hour = start_hour
		self.end_hour = end_hour
		self.time_range_str = f"{int(start_hour):02d}-{int(end_hour-1):02d}"
		
		# 统计信息
		self.stats = {
			'processed_months': 0,
			'processed_days': 0,
			'failed_days': 0,
			'total_irrad_records': 0,
			'total_temp_records': 0,
			'failed_files': [],
			'month_stats': {}  # 每月统计
		}
		
		logger.info(f"📊 光伏数据提取器初始化（全数据扫描版）")
		logger.info(f"   数据源目录: {self.base_dir}")
		logger.info(f"   输出根目录: {self.output_base_dir}")
		logger.info(f"   时间范围: {self.start_hour}:00 - {self.end_hour-1}:59")
		logger.info(f"   时间标识: {self.time_range_str}")
		logger.info(f"   存储模式: {'按天分割' if self.split_by_day else '按月合并'}")
	
	def _validate_directory(self) -> bool:
		"""验证数据源目录是否存在"""
		if not self.base_dir.exists():
			logger.error(f"❌ 数据源目录不存在: {self.base_dir}")
			return False
		
		# 检查是否有月份文件夹 (格式: 2025-01, 2025-02等)
		month_folders = [d for d in self.base_dir.iterdir() 
		                if d.is_dir() and len(d.name) == 7 and d.name.count('-') == 1]
		if not month_folders:
			logger.error(f"❌ 在{self.base_dir}中未找到月份文件夹")
			return False
		
		logger.info(f"✅ 找到 {len(month_folders)} 个月份文件夹")
		return True
	
	def _get_month_folders(self) -> List[Path]:
		"""获取所有月份文件夹，按日期排序"""
		month_folders = []
		
		for folder in sorted(self.base_dir.iterdir()):
			if folder.is_dir() and len(folder.name) == 7 and folder.name.count('-') == 1:
				month_folders.append(folder)
		
		logger.info(f"📅 将处理 {len(month_folders)} 个月份的数据: {[f.name for f in month_folders]}")
		return month_folders
	
	def _get_date_folders_in_month(self, month_folder: Path) -> List[Path]:
		"""获取指定月份下的所有日期文件夹"""
		date_folders = []
		
		for day_folder in sorted(month_folder.iterdir()):
			if day_folder.is_dir() and day_folder.name.isdigit():
				date_folders.append(day_folder)
		
		return date_folders
	
	def _read_daily_data(self, day_folder: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
		"""
		读取单日的辐照度和温度数据
		
		Args:
			day_folder: 日期文件夹路径
			
		Returns:
			(irradiance_df, temperature_df) 或 (None, None) if failed
		"""
		day_name = day_folder.name
		
		# 构建文件路径
		irrad_file = None
		temp_file = None
		
		# 查找辐照度和温度文件
		for file_path in day_folder.glob("*.csv"):
			if "irradiance_timeseries" in file_path.name:
				irrad_file = file_path
			elif "temperature_timeseries" in file_path.name:
				temp_file = file_path
		
		if not irrad_file or not temp_file:
			logger.error(f"❌ {day_name}日: 缺失数据文件")
			self.stats['failed_files'].append(f"{day_name}: 文件缺失")
			return None, None
		
		try:
			# 读取辐照度数据
			irrad_df = pd.read_csv(irrad_file)
			if 'hour' not in irrad_df.columns or 'Mult' not in irrad_df.columns:
				logger.error(f"❌ {day_name}日: 辐照度文件格式错误，缺少必要列")
				self.stats['failed_files'].append(f"{day_name}: 辐照度格式错误")
				return None, None
			
			# 读取温度数据
			temp_df = pd.read_csv(temp_file)
			# 兼容不同的列名（Hour/hour）
			hour_col = 'Hour' if 'Hour' in temp_df.columns else 'hour'
			if hour_col not in temp_df.columns or 'Temp' not in temp_df.columns:
				logger.error(f"❌ {day_name}日: 温度文件格式错误，缺少必要列")
				self.stats['failed_files'].append(f"{day_name}: 温度格式错误")
				return None, None
			
			# 统一列名
			if hour_col == 'Hour':
				temp_df = temp_df.rename(columns={'Hour': 'hour'})
			
			logger.debug(f"✅ {day_name}日: 数据读取成功 - 辐照度: {len(irrad_df)}条, 温度: {len(temp_df)}条")
			return irrad_df, temp_df
			
		except Exception as e:
			logger.error(f"❌ {day_name}日: 读取数据失败 - {e}")
			self.stats['failed_files'].append(f"{day_name}: {str(e)}")
			return None, None
	
	def _filter_time_range(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		过滤时间范围（6:00-17:59）
		
		Args:
			df: 包含hour列的数据框
			
		Returns:
			过滤后的数据框
		"""
		return df[(df['hour'] >= self.start_hour) & (df['hour'] < self.end_hour)].copy()
	
	def _extract_all_data(self) -> dict:
		"""
		提取所有月份的6-17点数据
		
		Returns:
			按月份或按天组织的数据字典
			- 按月模式: {month_name: (irrad_values, temp_values)}
			- 按天模式: {month_name: {day_name: (irrad_values, temp_values)}}
		"""
		all_month_data = {}
		month_folders = self._get_month_folders()
		
		logger.info("🔄 开始批量数据提取...")
		total_start_time = time.time()
		
		for month_idx, month_folder in enumerate(month_folders, 1):
			month_name = month_folder.name
			logger.info(f"\n📅 处理月份: {month_name} ({month_idx}/{len(month_folders)})")
			
			# 获取该月的所有日期文件夹
			date_folders = self._get_date_folders_in_month(month_folder)
			if not date_folders:
				logger.warning(f"⚠️ {month_name}月无有效日期数据")
				continue
			
			if self.split_by_day:
				# 按天分割模式
				month_daily_data = {}
				month_processed_days = 0
				month_failed_days = 0
				
				# 处理该月的每一天
				month_start_time = time.time()
				for day_idx, day_folder in enumerate(date_folders, 1):
					day_name = day_folder.name
					
					# 读取当日数据
					irrad_df, temp_df = self._read_daily_data(day_folder)
					
					if irrad_df is None or temp_df is None:
						month_failed_days += 1
						self.stats['failed_days'] += 1
						continue
					
					# 过滤时间范围
					irrad_filtered = self._filter_time_range(irrad_df)
					temp_filtered = self._filter_time_range(temp_df)
					
					# 提取数值
					irrad_values = irrad_filtered['Mult'].tolist()
					temp_values = temp_filtered['Temp'].tolist()
					
					# 存储该天数据
					month_daily_data[day_name] = (irrad_values, temp_values)
					month_processed_days += 1
					
					# 进度提示（每5天或最后一天）
					if day_idx % 5 == 0 or day_idx == len(date_folders):
						month_elapsed = time.time() - month_start_time
						logger.info(f"   {month_name}: {day_idx}/{len(date_folders)}天 "
						           f"({day_idx/len(date_folders)*100:.1f}%) - "
						           f"已处理: {month_processed_days}天 - "
						           f"耗时: {month_elapsed:.1f}s")
				
				# 存储月份的每日数据
				if month_daily_data:
					all_month_data[month_name] = month_daily_data
			else:
				# 按月合并模式（原逻辑）
				month_irrad_values = []
				month_temp_values = []
				month_processed_days = 0
				month_failed_days = 0
				
				# 处理该月的每一天
				month_start_time = time.time()
				for day_idx, day_folder in enumerate(date_folders, 1):
					# 读取当日数据
					irrad_df, temp_df = self._read_daily_data(day_folder)
					
					if irrad_df is None or temp_df is None:
						month_failed_days += 1
						self.stats['failed_days'] += 1
						continue
					
					# 过滤时间范围
					irrad_filtered = self._filter_time_range(irrad_df)
					temp_filtered = self._filter_time_range(temp_df)
					
					# 提取数值
					irrad_values = irrad_filtered['Mult'].tolist()
					temp_values = temp_filtered['Temp'].tolist()
					
					# 添加到月份数据中
					month_irrad_values.extend(irrad_values)
					month_temp_values.extend(temp_values)
					
					month_processed_days += 1
					
					# 进度提示（每5天或最后一天）
					if day_idx % 5 == 0 or day_idx == len(date_folders):
						month_elapsed = time.time() - month_start_time
						logger.info(f"   {month_name}: {day_idx}/{len(date_folders)}天 "
						           f"({day_idx/len(date_folders)*100:.1f}%) - "
						           f"辐照度: {len(month_irrad_values)}, 温度: {len(month_temp_values)} - "
						           f"耗时: {month_elapsed:.1f}s")
				
				# 存储月份合并数据
				if month_irrad_values and month_temp_values:
					all_month_data[month_name] = (month_irrad_values, month_temp_values)
			
			# 更新统计
			if month_name in all_month_data:
				self.stats['processed_months'] += 1
				self.stats['processed_days'] += month_processed_days
				
				if self.split_by_day:
					# 按天模式统计
					total_irrad = sum(len(data[0]) for data in all_month_data[month_name].values())
					total_temp = sum(len(data[1]) for data in all_month_data[month_name].values())
					self.stats['total_irrad_records'] += total_irrad
					self.stats['total_temp_records'] += total_temp
					
					self.stats['month_stats'][month_name] = {
						'days': month_processed_days,
						'failed_days': month_failed_days,
						'irrad_records': total_irrad,
						'temp_records': total_temp
					}
					
					logger.info(f"✅ {month_name}月完成: {month_processed_days}天（按天分割）, "
					           f"辐照度 {total_irrad:,}条, 温度 {total_temp:,}条")
				else:
					# 按月模式统计（原逻辑）
					irrad_values, temp_values = all_month_data[month_name]
					self.stats['total_irrad_records'] += len(irrad_values)
					self.stats['total_temp_records'] += len(temp_values)
					
					self.stats['month_stats'][month_name] = {
						'days': month_processed_days,
						'failed_days': month_failed_days,
						'irrad_records': len(irrad_values),
						'temp_records': len(temp_values)
					}
					
					logger.info(f"✅ {month_name}月完成: {month_processed_days}天, "
					           f"辐照度 {len(irrad_values):,}条, 温度 {len(temp_values):,}条")
			else:
				logger.warning(f"⚠️ {month_name}月无有效数据")
		
		total_elapsed = time.time() - total_start_time
		logger.info(f"✅ 全部数据提取完成! 总耗时: {total_elapsed:.1f}s")
		return all_month_data
	
	def _save_month_data(self, month_data: dict) -> bool:
		"""
		保存按月份或按天组织的数据到CSV文件
		
		Args:
			month_data: 按月份或按天组织的数据字典
			
		Returns:
			保存是否成功
		"""
		try:
			saved_files = []
			
			if self.split_by_day:
				# 按天分割模式
				for month_name, daily_data in month_data.items():
					# 创建月份目录
					month_output_dir = self.output_base_dir / f"{month_name}-{self.time_range_str}"
					month_output_dir.mkdir(parents=True, exist_ok=True)
					
					for day_name, (irrad_values, temp_values) in daily_data.items():
						# 创建日期子目录
						day_output_dir = month_output_dir / day_name
						day_output_dir.mkdir(parents=True, exist_ok=True)
						
						# 保存辐照度数据
						irrad_file = day_output_dir / "irrad.csv"
						with open(irrad_file, 'w', newline='') as f:
							for value in irrad_values:
								f.write(f"{value}\n")
						
						# 保存温度数据
						temp_file = day_output_dir / "temperature.csv"
						with open(temp_file, 'w', newline='') as f:
							for value in temp_values:
								f.write(f"{value}\n")
						
						saved_files.extend([irrad_file, temp_file])
					
					logger.info(f"💾 {month_name}月数据已保存（按天分割）: {month_output_dir}")
					logger.info(f"   包含 {len(daily_data)} 天数据，每天 {len(list(daily_data.values())[0][0])} 条记录")
			else:
				# 按月合并模式（原逻辑）
				for month_name, (irrad_values, temp_values) in month_data.items():
					# 创建月份专用输出目录
					month_output_dir = self.output_base_dir / f"{month_name}-{self.time_range_str}"
					month_output_dir.mkdir(parents=True, exist_ok=True)
					
					# 保存辐照度数据
					irrad_file = month_output_dir / "irrad.csv"
					with open(irrad_file, 'w', newline='') as f:
						for value in irrad_values:
							f.write(f"{value}\n")
					
					# 保存温度数据
					temp_file = month_output_dir / "temperature.csv"
					with open(temp_file, 'w', newline='') as f:
						for value in temp_values:
							f.write(f"{value}\n")
					
					saved_files.extend([irrad_file, temp_file])
					logger.info(f"💾 {month_name}月数据已保存: {month_output_dir}")
					logger.info(f"   辐照度: {len(irrad_values):,}条, 温度: {len(temp_values):,}条")
			
			logger.info(f"✅ 所有数据保存完成，共 {len(saved_files)} 个文件")
			return True
			
		except Exception as e:
			logger.error(f"❌ 保存数据失败: {e}")
			return False
	
	def _print_summary(self):
		"""打印处理总结"""
		logger.info("\n" + "="*70)
		logger.info("📊 处理总结:")
		logger.info(f"   成功处理月份: {self.stats['processed_months']}")
		logger.info(f"   成功处理天数: {self.stats['processed_days']}")
		logger.info(f"   失败天数: {self.stats['failed_days']}")
		logger.info(f"   辐照度记录数: {self.stats['total_irrad_records']:,}")
		logger.info(f"   温度记录数: {self.stats['total_temp_records']:,}")
		
		# 按月份统计
		if self.stats['month_stats']:
			logger.info(f"\n📅 各月份统计:")
			for month, stats in self.stats['month_stats'].items():
				logger.info(f"   {month}: {stats['days']}天, "
				           f"辐照度 {stats['irrad_records']:,}条, "
				           f"温度 {stats['temp_records']:,}条")
		
		if self.stats['failed_files']:
			logger.warning(f"\n⚠️ 失败文件列表:")
			for failed in self.stats['failed_files']:
				logger.warning(f"   - {failed}")
		
		logger.info("="*70)
	
	def extract(self) -> bool:
		"""
		执行完整的数据提取流程
		
		Returns:
			提取是否成功
		"""
		logger.info("🚀 开始光伏数据提取（全数据扫描）...")
		
		# 验证目录
		if not self._validate_directory():
			return False
		
		# 提取数据
		month_data = self._extract_all_data()
		
		if not month_data:
			logger.error("❌ 未提取到任何有效数据")
			return False
		
		# 保存数据
		if not self._save_month_data(month_data):
			return False
		
		# 打印总结
		self._print_summary()
		
		logger.info("🎉 光伏数据提取完成!")
		return True


def main():
	"""主函数"""
	import argparse
	
	parser = argparse.ArgumentParser(description="PowerZoo 光伏数据提取工具（全数据扫描版）")
	parser.add_argument("--base-dir", type=str, default="data/PV/daily_data", 
	                   help="数据源根目录 (默认: data/PV/daily_data)")
	parser.add_argument("--output-dir", type=str, default="data/PV/hourly_segments",
	                   help="输出根目录 (默认: data/PV/hourly_segments)")
	parser.add_argument("--start-hour", type=float, default=6.0,
	                   help="开始时间（小时） (默认: 6.0)")
	parser.add_argument("--end-hour", type=float, default=18.0,
	                   help="结束时间（小时，不包含） (默认: 18.0)")
	parser.add_argument("--months", type=str, nargs="*",
	                   help="指定处理的月份 (如: 2025-01 2025-02)，不指定则处理所有月份")
	parser.add_argument("--split-by-day", action="store_true",
	                   help="按天分割数据存储（默认按月合并存储）")
	
	args = parser.parse_args()
	
	print("🌞 PowerZoo 光伏数据提取工具（全数据扫描版）")
	print("="*60)
	
	# 创建提取器
	extractor = PVDataExtractor(
		base_dir=args.base_dir,
		start_hour=args.start_hour,
		end_hour=args.end_hour,
		output_base_dir=args.output_dir,
		split_by_day=args.split_by_day
	)
	
	# 执行提取
	try:
		success = extractor.extract()
		
		if success:
			print("\n✅ 全数据扫描提取成功完成!")
			print("📁 输出目录结构:")
			
			# 显示生成的文件夹
			output_base = Path(args.output_dir)
			if output_base.exists():
				time_range = f"{int(args.start_hour):02d}-{int(args.end_hour-1):02d}"
				for folder in sorted(output_base.glob(f"*-{time_range}")):
					print(f"   - {folder}/")
					if args.split_by_day:
						# 按天模式：显示日期子文件夹
						day_folders = [d for d in folder.iterdir() if d.is_dir() and d.name.isdigit()]
						if day_folders:
							for i, day_folder in enumerate(sorted(day_folders)[:3]):  # 只显示前3天作为示例
								connector = "├──" if i < min(len(day_folders), 3) - 1 else "└──"
								print(f"     {connector} {day_folder.name}/")
								print(f"     │   ├── irrad.csv")
								print(f"     │   └── temperature.csv")
							if len(day_folders) > 3:
								print(f"     └── ... (共{len(day_folders)}天)")
					else:
						# 按月模式：直接显示CSV文件
						print(f"     ├── irrad.csv")
						print(f"     └── temperature.csv")
			
			sys.exit(0)
		else:
			print("\n❌ 数据提取失败!")
			sys.exit(1)
			
	except KeyboardInterrupt:
		print("\n⏹️ 用户中断操作")
		sys.exit(1)
	except Exception as e:
		logger.error(f"❌ 程序执行出错: {e}")
		import traceback
		traceback.print_exc()
		sys.exit(1)


if __name__ == "__main__":
	main()