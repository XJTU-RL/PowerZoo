#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PV数据Worker目录设置工具
确保所有worker目录都有正确的PV数据文件

@Author: Xiaodong Zheng
"""

import os
import shutil
from pathlib import Path
import argparse

def setup_pv_workers(base_dir, max_workers=32):
	"""设置PV数据的worker目录
	
	Args:
		base_dir: 基础目录路径
		max_workers: 最大worker数量
	"""
	base_path = Path(base_dir)
	irradiation_dir = base_path / "irradiation"
	temperature_dir = base_path / "temperature"
	
	# 确保基础目录存在
	irradiation_dir.mkdir(parents=True, exist_ok=True)
	temperature_dir.mkdir(parents=True, exist_ok=True)
	
	# 检查源数据（000目录）
	source_irrad_dir = irradiation_dir / "000"
	source_temp_dir = temperature_dir / "000"
	
	if not source_irrad_dir.exists() or not source_temp_dir.exists():
		print(f"❌ 源数据目录不存在: {source_irrad_dir} 或 {source_temp_dir}")
		return False
	
	# 获取所有数据文件
	irrad_files = list(source_irrad_dir.glob("*.csv"))
	temp_files = list(source_temp_dir.glob("*.csv"))
	
	if not irrad_files or not temp_files:
		print(f"❌ 源数据目录中没有CSV文件")
		return False
	
	print(f"📁 找到 {len(irrad_files)} 个辐照度文件和 {len(temp_files)} 个温度文件")
	
	# 为每个worker创建目录并复制文件
	for worker_idx in range(max_workers):
		worker_str = f"{worker_idx:03d}"
		
		# 创建worker目录
		worker_irrad_dir = irradiation_dir / worker_str
		worker_temp_dir = temperature_dir / worker_str
		
		worker_irrad_dir.mkdir(exist_ok=True)
		worker_temp_dir.mkdir(exist_ok=True)
		
		# 复制辐照度文件
		for irrad_file in irrad_files:
			dest_file = worker_irrad_dir / irrad_file.name
			if not dest_file.exists():
				shutil.copy2(irrad_file, dest_file)
		
		# 复制温度文件
		for temp_file in temp_files:
			dest_file = worker_temp_dir / temp_file.name
			if not dest_file.exists():
				shutil.copy2(temp_file, dest_file)
		
		print(f"✅ Worker {worker_str}: {len(irrad_files)} 辐照度文件, {len(temp_files)} 温度文件")
	
	print(f"\n🎯 成功设置 {max_workers} 个worker目录")
	return True

def cleanup_duplicate_dirs(base_dir):
	"""清理重复的嵌套目录结构"""
	base_path = Path(base_dir)
	irradiation_dir = base_path / "irradiation"
	temperature_dir = base_path / "temperature"
	
	cleaned = 0
	
	# 清理irradiation目录中的嵌套000目录
	for worker_dir in irradiation_dir.iterdir():
		if worker_dir.is_dir() and worker_dir.name.isdigit():
			nested_000 = worker_dir / "000"
			if nested_000.exists():
				print(f"🧹 清理嵌套目录: {nested_000}")
				# 将嵌套目录中的文件移动到父目录
				for file in nested_000.glob("*.csv"):
					dest_file = worker_dir / file.name
					if not dest_file.exists():
						shutil.move(file, dest_file)
				# 删除空的嵌套目录
				if not any(nested_000.iterdir()):
					nested_000.rmdir()
					cleaned += 1
	
	# 清理temperature目录中的嵌套000目录
	for worker_dir in temperature_dir.iterdir():
		if worker_dir.is_dir() and worker_dir.name.isdigit():
			nested_000 = worker_dir / "000"
			if nested_000.exists():
				print(f"🧹 清理嵌套目录: {nested_000}")
				# 将嵌套目录中的文件移动到父目录
				for file in nested_000.glob("*.csv"):
					dest_file = worker_dir / file.name
					if not dest_file.exists():
						shutil.move(file, dest_file)
				# 删除空的嵌套目录
				if not any(nested_000.iterdir()):
					nested_000.rmdir()
					cleaned += 1
	
	if cleaned > 0:
		print(f"✅ 清理了 {cleaned} 个嵌套目录")
	
	return cleaned

def verify_worker_data(base_dir, worker_count=32):
	"""验证worker数据完整性"""
	base_path = Path(base_dir)
	irradiation_dir = base_path / "irradiation"
	temperature_dir = base_path / "temperature"
	
	print(f"\n🔍 验证 {worker_count} 个worker的数据完整性...")
	
	# 获取期望的文件列表（从000目录）
	source_irrad_dir = irradiation_dir / "000"
	expected_files = [f.name for f in source_irrad_dir.glob("*.csv")]
	
	if not expected_files:
		print("❌ 源目录000中没有找到CSV文件")
		return False
	
	missing_workers = []
	incomplete_workers = []
	
	for worker_idx in range(worker_count):
		worker_str = f"{worker_idx:03d}"
		worker_irrad_dir = irradiation_dir / worker_str
		worker_temp_dir = temperature_dir / worker_str
		
		if not worker_irrad_dir.exists() or not worker_temp_dir.exists():
			missing_workers.append(worker_str)
			continue
		
		# 检查文件完整性
		irrad_files = set(f.name for f in worker_irrad_dir.glob("*.csv"))
		temp_files = set(f.name for f in worker_temp_dir.glob("*.csv"))
		expected_set = set(expected_files)
		
		if irrad_files != expected_set or temp_files != expected_set:
			incomplete_workers.append(worker_str)
	
	if missing_workers:
		print(f"❌ 缺失的worker目录: {missing_workers}")
	
	if incomplete_workers:
		print(f"⚠️  不完整的worker目录: {incomplete_workers}")
	
	if not missing_workers and not incomplete_workers:
		print(f"✅ 所有 {worker_count} 个worker数据完整")
		return True
	
	return False

def main():
	parser = argparse.ArgumentParser(description='PV数据Worker目录设置工具')
	parser.add_argument('--base-dir', 
	                   default='/home/zhengxiaodong/exps/PowerZoo/envs/power_envs/powerzoo_llm/node_systems_with_pv/34Bus',
	                   help='基础目录路径')
	parser.add_argument('--max-workers', type=int, default=32,
	                   help='最大worker数量')
	parser.add_argument('--cleanup', action='store_true',
	                   help='清理重复的嵌套目录')
	parser.add_argument('--verify', action='store_true',
	                   help='验证数据完整性')
	
	args = parser.parse_args()
	
	print("🔧 PV数据Worker目录设置工具")
	print("=" * 50)
	
	if args.cleanup:
		print("🧹 清理重复目录...")
		cleanup_duplicate_dirs(args.base_dir)
	
	if args.verify:
		print("🔍 验证数据完整性...")
		verify_worker_data(args.base_dir, args.max_workers)
	else:
		print(f"📁 设置 {args.max_workers} 个worker目录...")
		success = setup_pv_workers(args.base_dir, args.max_workers)
		
		if success:
			print("\n🔍 验证设置结果...")
			verify_worker_data(args.base_dir, args.max_workers)
		else:
			print("❌ 设置失败")
			return 1
	
	print("\n✅ 操作完成!")
	return 0

if __name__ == "__main__":
	exit(main())