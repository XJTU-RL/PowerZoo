#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Loadshape Analysis - Batch Processing
增强版负荷分析 - 批量处理版本

支持小时级和分钟级数据的批量分析，优化大数据量处理性能

Author: PowerZoo Team  
Date: 2025-07-30
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# 添加项目根路径
sys.path.append(str(Path(__file__).parent.parent))

from data.Loads.loadshape_analyzer import LoadshapeAnalyzer

# 忽略警告信息
warnings.filterwarnings('ignore')

class BatchLoadshapeAnalyzer:
	"""
	批量负荷分析器
	
	功能特点：
	- 批量处理多个负荷数据文件
	- 自动识别数据类型（小时级/分钟级）
	- 并行处理提升性能
	- 生成对比分析报告
	- 支持多种数据格式
	"""
	
	def __init__(self, input_dir: Union[str, Path], 
				 output_dir: Union[str, Path] = None,
				 max_workers: int = 4):
		"""
		初始化批量分析器
		
		Args:
			input_dir: 输入数据目录
			output_dir: 输出结果目录
			max_workers: 最大并行工作进程数
		"""
		self.input_dir = Path(input_dir)
		self.output_dir = Path(output_dir) if output_dir else self.input_dir / "analysis_results"
		self.max_workers = max_workers
		self.results = {}
		
		# 创建输出目录
		self.output_dir.mkdir(parents=True, exist_ok=True)
		
		# 支持的文件格式
		self.supported_extensions = {'.csv', '.txt', '.dat', '.npy'}
		
		print(f"批量分析器初始化完成")
		print(f"输入目录: {self.input_dir}")
		print(f"输出目录: {self.output_dir}")
		print(f"最大并行数: {self.max_workers}")
	
	def find_data_files(self) -> List[Path]:
		"""查找所有支持的数据文件"""
		data_files = []
		
		for ext in self.supported_extensions:
			files = self.input_dir.rglob(f"*{ext}")
			# 排除分析结果目录中的文件
			for file in files:
				if 'analysis_results' not in str(file):
					data_files.append(file)
		
		print(f"找到 {len(data_files)} 个数据文件")
		return sorted(data_files)
	
	def analyze_single_file(self, file_path: Path) -> Dict:
		"""
		Analyze single file
		
		Args:
			file_path: File path
			
		Returns:
			Analysis result dictionary
		"""
		try:
			print(f"Analyzing: {file_path.name}")
			
			# Create analyzer
			analyzer = LoadshapeAnalyzer(data_path=file_path)
			
			# Execute pattern analysis
			patterns = analyzer.analyze_patterns()
			
			# Generate visualization charts
			output_prefix = self.output_dir / f"analysis_{file_path.stem}"
			
			# Time series chart
			fig1 = analyzer.plot_timeseries(
				title=f"Load Time Series - {file_path.name}",
				save_path=f"{output_prefix}_timeseries.png"
			)
			plt.close(fig1)
			
			# Statistical analysis chart
			fig2 = analyzer.plot_statistics(
				save_path=f"{output_prefix}_statistics.png"
			)
			plt.close(fig2)
			
			# Export detailed report
			analyzer.export_summary(f"{output_prefix}_report.txt")
			
			# Build result
			result = {
				'file_name': file_path.name,
				'file_path': str(file_path),
				'analysis': patterns,
				'status': 'success',
				'output_prefix': str(output_prefix)
			}
			
			print(f"✅ Analysis completed: {file_path.name}")
			return result
			
		except Exception as e:
			print(f"❌ Analysis failed: {file_path.name} - {str(e)}")
			return {
				'file_name': file_path.name,
				'file_path': str(file_path),
				'status': 'failed',
				'error': str(e)
			}
	
	def run_batch_analysis(self, parallel: bool = True) -> Dict:
		"""
		Execute batch analysis
		
		Args:
			parallel: Whether to use parallel processing
			
		Returns:
			Batch analysis results
		"""
		data_files = self.find_data_files()
		
		if not data_files:
			print("No supported data files found")
			return {}
		
		print(f"\nStarting batch analysis of {len(data_files)} files...")
		print("=" * 60)
		
		results = {}
		
		if parallel and len(data_files) > 1:
			# Parallel processing
			with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
				# Submit all tasks
				future_to_file = {
					executor.submit(self.analyze_single_file, file_path): file_path 
					for file_path in data_files
				}
				
				# Collect results
				for future in as_completed(future_to_file):
					file_path = future_to_file[future]
					try:
						result = future.result()
						results[result['file_name']] = result
					except Exception as e:
						print(f"❌ Exception occurred while processing {file_path.name}: {e}")
						results[file_path.name] = {
							'file_name': file_path.name,
							'status': 'failed',
							'error': str(e)
						}
		else:
			# Serial processing
			for file_path in data_files:
				result = self.analyze_single_file(file_path)
				results[result['file_name']] = result
		
		self.results = results
		
		# Generate summary report
		self.generate_summary_report()
		
		print("\n" + "=" * 60)
		print("Batch analysis completed!")
		self._print_summary()
		
		return results
	
	def generate_summary_report(self) -> None:
		"""Generate batch analysis summary report"""
		summary_path = self.output_dir / "batch_analysis_summary.txt"
		
		successful_results = [r for r in self.results.values() if r['status'] == 'success']
		failed_results = [r for r in self.results.values() if r['status'] == 'failed']
		
		with open(summary_path, 'w', encoding='utf-8') as f:
			f.write("Batch Load Analysis Summary Report\n")
			f.write("=" * 50 + "\n\n")
			
			# Overall statistics
			f.write("1. Overall Statistics\n")
			f.write("-" * 20 + "\n")
			f.write(f"Total Files: {len(self.results)}\n")
			f.write(f"Successful Analysis: {len(successful_results)}\n")
			f.write(f"Failed Files: {len(failed_results)}\n")
			f.write(f"Success Rate: {len(successful_results)/len(self.results)*100:.1f}%\n\n")
			
			if failed_results:
				f.write("2. Failed Files List\n")
				f.write("-" * 20 + "\n")
				for result in failed_results:
					f.write(f"File: {result['file_name']}\n")
					f.write(f"Error: {result.get('error', 'Unknown error')}\n\n")
			
			if successful_results:
				f.write("3. Successful Files Overview\n")
				f.write("-" * 20 + "\n")
				
				# Data type statistics
				data_types = {}
				for result in successful_results:
					dtype = result['analysis']['basic_stats']['data_type']
					data_types[dtype] = data_types.get(dtype, 0) + 1
				
				f.write("Data Type Distribution:\n")
				for dtype, count in data_types.items():
					f.write(f"  {dtype}: {count} files\n")
				f.write("\n")
				
				# Load statistics comparison
				f.write("4. Load Statistics Comparison\n")
				f.write("-" * 20 + "\n")
				f.write(f"{'File Name':<30} {'Data Type':<12} {'Mean(MW)':<10} {'Peak(MW)':<10} {'Valley(MW)':<10} {'Load Factor':<12}\n")
				f.write("-" * 94 + "\n")
				
				for result in successful_results:
					name = result['file_name'][:28]
					stats = result['analysis']['basic_stats']
					pv = result['analysis']['peak_valley']
					
					f.write(f"{name:<30} {stats['data_type']:<12} "
							f"{stats['mean']:<10.2f} {pv['peak_load']:<10.2f} "
							f"{pv['valley_load']:<10.2f} {pv['load_factor']:<12.3f}\n")
		
		print(f"Summary report saved to: {summary_path}")
	
	def generate_comparison_charts(self, chart_types: List[str] = None) -> None:
		"""
		Generate comparison charts
		
		Args:
			chart_types: Chart type list ['load_comparison', 'stats_comparison', 'pattern_comparison']
		"""
		if chart_types is None:
			chart_types = ['load_comparison', 'stats_comparison']
		
		successful_results = [r for r in self.results.values() if r['status'] == 'success']
		
		if len(successful_results) < 2:
			print("Less than 2 successfully analyzed files, cannot generate comparison charts")
			return
		
		print("Generating comparison charts...")
		
		# 1. Load statistics comparison
		if 'stats_comparison' in chart_types:
			self._plot_stats_comparison(successful_results)
		
		# 2. Load distribution comparison
		if 'load_comparison' in chart_types:
			self._plot_load_comparison(successful_results)
		
		print("Comparison charts generation completed")
	
	def _plot_stats_comparison(self, results: List[Dict]) -> None:
		"""绘制统计指标对比图"""
		fig, axes = plt.subplots(2, 2, figsize=(15, 10))
		
		file_names = [r['file_name'][:15] for r in results]  # 截短文件名
		
		# 1. 均值对比
		means = [r['analysis']['basic_stats']['mean'] for r in results]
		axes[0, 0].bar(file_names, means, color='skyblue', alpha=0.7)
		axes[0, 0].set_title('Average Load Comparison')
		axes[0, 0].set_ylabel('Load (MW)')
		axes[0, 0].tick_params(axis='x', rotation=45)
		
		# 2. 峰谷对比
		peaks = [r['analysis']['peak_valley']['peak_load'] for r in results]
		valleys = [r['analysis']['peak_valley']['valley_load'] for r in results]
		
		x_pos = np.arange(len(file_names))
		width = 0.35
		
		axes[0, 1].bar(x_pos - width/2, peaks, width, label='Peak Load', color='red', alpha=0.7)
		axes[0, 1].bar(x_pos + width/2, valleys, width, label='Valley Load', color='blue', alpha=0.7)
		axes[0, 1].set_title('Peak-Valley Load Comparison')
		axes[0, 1].set_ylabel('Load (MW)')
		axes[0, 1].set_xticks(x_pos)
		axes[0, 1].set_xticklabels(file_names, rotation=45)
		axes[0, 1].legend()
		
		# 3. 负荷率对比
		load_factors = [r['analysis']['peak_valley']['load_factor'] for r in results]
		axes[1, 0].bar(file_names, load_factors, color='green', alpha=0.7)
		axes[1, 0].set_title('Load Factor Comparison')
		axes[1, 0].set_ylabel('Load Factor')
		axes[1, 0].tick_params(axis='x', rotation=45)
		axes[1, 0].set_ylim(0, 1)
		
		# 4. 变异系数对比
		cvs = [r['analysis']['variability']['coefficient_of_variation'] for r in results]
		axes[1, 1].bar(file_names, cvs, color='orange', alpha=0.7)
		axes[1, 1].set_title('Coefficient of Variation Comparison')
		axes[1, 1].set_ylabel('Coefficient of Variation')
		axes[1, 1].tick_params(axis='x', rotation=45)
		
		plt.tight_layout()
		save_path = self.output_dir / "stats_comparison.png"
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.close()
		
		print(f"统计对比图已保存至: {save_path}")
	
	def _plot_load_comparison(self, results: List[Dict]) -> None:
		"""绘制负荷分布对比图"""
		# 限制显示的文件数量，避免图表过于复杂
		max_files = 6
		if len(results) > max_files:
			results = results[:max_files]
			print(f"注意: 只显示前{max_files}个文件的负荷分布对比")
		
		fig, ax = plt.subplots(figsize=(12, 8))
		
		colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
		
		for i, result in enumerate(results):
			try:
				# 重新加载数据用于绘制分布
				file_path = Path(result['file_path'])
				analyzer = LoadshapeAnalyzer(data_path=file_path)
				
				# 智能采样以优化性能
				sampled_data, _ = analyzer._smart_sample(analyzer.raw_data, max_points=5000)
				
				# 绘制核密度估计
				ax.hist(sampled_data, bins=50, alpha=0.3, color=colors[i], 
						label=result['file_name'][:20], density=True)
				
			except Exception as e:
				print(f"绘制 {result['file_name']} 的分布图时出错: {e}")
				continue
		
		ax.set_title('Load Distribution Comparison')
		ax.set_xlabel('Load (MW)')
		ax.set_ylabel('Density')
		ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		ax.grid(True, alpha=0.3)
		
		plt.tight_layout()
		save_path = self.output_dir / "load_distribution_comparison.png"
		plt.savefig(save_path, dpi=300, bbox_inches='tight')
		plt.close()
		
		print(f"负荷分布对比图已保存至: {save_path}")
	
	def _print_summary(self) -> None:
		"""Print analysis summary"""
		successful = [r for r in self.results.values() if r['status'] == 'success']
		failed = [r for r in self.results.values() if r['status'] == 'failed']
		
		print(f"Analysis Results Summary:")
		print(f"  Total Files: {len(self.results)}")
		print(f"  Successful: {len(successful)}")
		print(f"  Failed: {len(failed)}")
		print(f"  Success Rate: {len(successful)/len(self.results)*100:.1f}%")
		
		if successful:
			print(f"\nData Type Statistics:")
			data_types = {}
			for result in successful:
				dtype = result['analysis']['basic_stats']['data_type']
				data_types[dtype] = data_types.get(dtype, 0) + 1
			
			for dtype, count in data_types.items():
				print(f"  {dtype}: {count} files")
		
		print(f"\nResults saved to: {self.output_dir}")


def main():
	"""Command line entry function"""
	parser = argparse.ArgumentParser(description='Batch load data analysis tool')
	parser.add_argument('input_dir', help='Input data directory path')
	parser.add_argument('-o', '--output', help='Output results directory path', default=None)
	parser.add_argument('-w', '--workers', type=int, default=4, help='Number of parallel worker processes')
	parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
	parser.add_argument('--charts', action='store_true', help='Generate comparison charts')
	parser.add_argument('--chart-types', nargs='+', 
						choices=['load_comparison', 'stats_comparison', 'pattern_comparison'],
						default=['stats_comparison', 'load_comparison'],
						help='Specify chart types to generate')
	
	args = parser.parse_args()
	
	# Check input directory
	if not Path(args.input_dir).exists():
		print(f"Error: Input directory does not exist: {args.input_dir}")
		return 1
	
	try:
		# Create batch analyzer
		analyzer = BatchLoadshapeAnalyzer(
			input_dir=args.input_dir,
			output_dir=args.output,
			max_workers=args.workers
		)
		
		# Execute analysis
		results = analyzer.run_batch_analysis(parallel=not args.no_parallel)
		
		# Generate comparison charts
		if args.charts:
			analyzer.generate_comparison_charts(args.chart_types)
		
		print(f"\n🎉 Batch analysis completed! Results saved to: {analyzer.output_dir}")
		return 0
		
	except Exception as e:
		print(f"❌ Error occurred during batch analysis: {e}")
		return 1


if __name__ == "__main__":
	main()