#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Loadshape Analyzer
Universal load data analyzer supporting intelligent processing of hourly and minute-level data

Author: PowerZoo Team
Date: 2025-07-30
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
from datetime import datetime, timedelta
import warnings

# 设置图表样式（使用系统默认字体）
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class LoadshapeAnalyzer:
	"""
	Universal load data analyzer
	
	Supported features:
	- Automatic detection of data time granularity (hourly/minute-level)
	- Intelligent data sampling and visualization optimization
	- Statistical analysis and trend detection
	- Performance-optimized big data processing
	"""
	
	# 数据类型常量
	HOURLY_POINTS = 8760    # 365*24
	MINUTELY_POINTS = 525600  # 365*24*60
	
	# Visualization parameters
	MAX_PLOT_POINTS = 10000  # Maximum plot points, sample if exceeded
	
	def __init__(self, data_path: Optional[Union[str, Path]] = None, 
				 data: Optional[np.ndarray] = None):
		"""
		Initialize analyzer
		
		Args:
			data_path: Data file path
			data: Directly passed data array
		"""
		self.data_path = Path(data_path) if data_path else None
		self.raw_data = None
		self.data_type = None
		self.time_index = None
		self.stats = {}
		
		if data is not None:
			self.raw_data = np.array(data)
		elif self.data_path and self.data_path.exists():
			self._load_data()
		
		if self.raw_data is not None:
			self._detect_data_type()
			self._generate_time_index()
			self._calculate_basic_stats()
	
	def _load_data(self) -> None:
		"""Load data from file"""
		try:
			if self.data_path.suffix.lower() == '.csv':
				df = pd.read_csv(self.data_path)
				# 假设第一列是负荷数据
				self.raw_data = df.iloc[:, 0].values
			elif self.data_path.suffix.lower() in ['.txt', '.dat']:
				self.raw_data = np.loadtxt(self.data_path)
			else:
				# 尝试直接读取为numpy数组
				self.raw_data = np.load(self.data_path)
		except Exception as e:
			raise ValueError(f"Unable to load data file {self.data_path}: {e}")
	
	def _detect_data_type(self) -> None:
		"""Automatically detect data type (hourly or minute-level)"""
		data_length = len(self.raw_data)
		
		if abs(data_length - self.HOURLY_POINTS) <= 24:  # 允许少量误差
			self.data_type = 'hourly'
		elif abs(data_length - self.MINUTELY_POINTS) <= 1440:  # 允许少量误差
			self.data_type = 'minutely'
		elif data_length < self.HOURLY_POINTS:
			self.data_type = 'custom_short'
		else:
			self.data_type = 'custom_long'
		
		print(f"Detected data type: {self.data_type}, data points: {data_length}")
	
	def _generate_time_index(self) -> None:
		"""Generate time index"""
		data_length = len(self.raw_data)
		
		if self.data_type == 'hourly':
			# Hourly data, starting from January 1, 2023
			start_time = datetime(2023, 1, 1)
			self.time_index = pd.date_range(
				start=start_time, 
				periods=data_length, 
				freq='h'
			)
		elif self.data_type == 'minutely':
			# Minute-level data
			start_time = datetime(2023, 1, 1)
			self.time_index = pd.date_range(
				start=start_time, 
				periods=data_length, 
				freq='min'
			)
		else:
			# Custom length, use sequence index
			self.time_index = pd.RangeIndex(data_length)
	
	def _calculate_basic_stats(self) -> None:
		"""Calculate basic statistical information"""
		self.stats = {
			'mean': np.mean(self.raw_data),
			'std': np.std(self.raw_data),
			'min': np.min(self.raw_data),
			'max': np.max(self.raw_data),
			'median': np.median(self.raw_data),
			'q25': np.percentile(self.raw_data, 25),
			'q75': np.percentile(self.raw_data, 75),
			'data_points': len(self.raw_data),
			'data_type': self.data_type
		}
	
	def _smart_sample(self, data: np.ndarray, 
					  time_idx: Optional[pd.Index] = None, 
					  max_points: int = None) -> Tuple[np.ndarray, pd.Index]:
		"""
		Intelligent sampling to optimize visualization performance for large datasets
		
		Args:
			data: Original data
			time_idx: Time index
			max_points: Maximum sampling points
			
		Returns:
			Sampled data and time index
		"""
		if max_points is None:
			max_points = self.MAX_PLOT_POINTS
		
		if len(data) <= max_points:
			return data, time_idx if time_idx is not None else np.arange(len(data))
		
		# Calculate sampling step
		step = len(data) // max_points
		
		# Use uniform sampling to maintain data distribution characteristics
		indices = np.arange(0, len(data), step)
		sampled_data = data[indices]
		
		if time_idx is not None:
			sampled_time = time_idx[indices]
		else:
			sampled_time = indices
		
		print(f"Data sampling: {len(data)} -> {len(sampled_data)} points")
		return sampled_data, sampled_time
	
	def get_time_range_data(self, start_time: Optional[str] = None, 
							end_time: Optional[str] = None) -> Tuple[np.ndarray, pd.Index]:
		"""
		Get data for specified time range
		
		Args:
			start_time: Start time string (e.g., '2023-01-01' or '2023-01-01 12:00')
			end_time: End time string
			
		Returns:
			Data and time index within the time range
		"""
		if isinstance(self.time_index, pd.RangeIndex):
			# Numeric index, use index range
			start_idx = 0 if start_time is None else int(start_time)
			end_idx = len(self.raw_data) if end_time is None else int(end_time)
			mask = (self.time_index >= start_idx) & (self.time_index <= end_idx)
		else:
			# Time index
			start_dt = pd.to_datetime(start_time) if start_time else self.time_index[0]
			end_dt = pd.to_datetime(end_time) if end_time else self.time_index[-1]
			mask = (self.time_index >= start_dt) & (self.time_index <= end_dt)
		
		return self.raw_data[mask], self.time_index[mask]
	
	def plot_timeseries(self, start_time: Optional[str] = None, 
						end_time: Optional[str] = None,
						title: str = "Load Time Series",
						figsize: Tuple[int, int] = (15, 6),
						save_path: Optional[str] = None) -> plt.Figure:
		"""
		Plot time series chart
		
		Args:
			start_time: Start time
			end_time: End time  
			title: Chart title
			figsize: Chart size
			save_path: Save path
			
		Returns:
			matplotlib chart object
		"""
		# Get data for specified time range
		data, time_idx = self.get_time_range_data(start_time, end_time)
		
		# Intelligent sampling for performance optimization
		sampled_data, sampled_time = self._smart_sample(data, time_idx)
		
		# Create chart
		fig, ax = plt.subplots(figsize=figsize)
		
		# Plot line chart
		if isinstance(sampled_time, pd.DatetimeIndex):
			ax.plot(sampled_time, sampled_data, linewidth=0.8, alpha=0.8)
			ax.set_xlabel('Time')
		else:
			ax.plot(sampled_time, sampled_data, linewidth=0.8, alpha=0.8)
			ax.set_xlabel('Data Points')
		
		ax.set_ylabel('Load (MW)')
		ax.set_title(f"{title}\nData Type: {self.data_type}, Original Points: {len(data)}")
		ax.grid(True, alpha=0.3)
		
		# Add statistical information text
		stats_text = f"Mean: {np.mean(sampled_data):.2f} MW\n"
		stats_text += f"Max: {np.max(sampled_data):.2f} MW\n"
		stats_text += f"Min: {np.min(sampled_data):.2f} MW"
		ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
				verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
		
		plt.tight_layout()
		
		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"Chart saved to: {save_path}")
		
		return fig
	
	def plot_statistics(self, figsize: Tuple[int, int] = (15, 10),
						save_path: Optional[str] = None) -> plt.Figure:
		"""
		Plot statistical analysis charts
		
		Args:
			figsize: Chart size
			save_path: Save path
			
		Returns:
			matplotlib chart object
		"""
		# Intelligent sampling for histogram and box plot
		sampled_data, _ = self._smart_sample(self.raw_data)
		
		fig, axes = plt.subplots(2, 2, figsize=figsize)
		
		# 1. Histogram
		axes[0, 0].hist(sampled_data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
		axes[0, 0].set_title('Load Distribution Histogram')
		axes[0, 0].set_xlabel('Load (MW)')
		axes[0, 0].set_ylabel('Frequency')
		axes[0, 0].grid(True, alpha=0.3)
		
		# 2. Box plot
		axes[0, 1].boxplot(sampled_data, patch_artist=True,
						   boxprops=dict(facecolor='lightgreen', alpha=0.7))
		axes[0, 1].set_title('Load Box Plot')
		axes[0, 1].set_ylabel('Load (MW)')
		axes[0, 1].grid(True, alpha=0.3)
		
		# 3. Daily average curve (if time data)
		if isinstance(self.time_index, pd.DatetimeIndex):
			df_temp = pd.DataFrame({'load': self.raw_data, 'time': self.time_index})
			daily_mean = df_temp.groupby(df_temp['time'].dt.date)['load'].mean()
			
			# Sample daily average data for performance optimization
			if len(daily_mean) > self.MAX_PLOT_POINTS // 10:
				step = len(daily_mean) // (self.MAX_PLOT_POINTS // 10)
				daily_mean = daily_mean.iloc[::step]
			
			axes[1, 0].plot(daily_mean.index, daily_mean.values, marker='o', markersize=2)
			axes[1, 0].set_title('Daily Average Load Trend')
			axes[1, 0].set_xlabel('Date')
			axes[1, 0].set_ylabel('Daily Average Load (MW)')
			axes[1, 0].tick_params(axis='x', rotation=45)
		else:
			# Non-time data, show moving average
			window = max(1, len(self.raw_data) // 100)
			moving_avg = pd.Series(self.raw_data).rolling(window=window).mean()
			sampled_ma, sampled_idx = self._smart_sample(moving_avg.values, np.arange(len(moving_avg)))
			
			axes[1, 0].plot(sampled_idx, sampled_ma)
			axes[1, 0].set_title(f'Moving Average Trend (Window={window})')
			axes[1, 0].set_xlabel('Data Points')
			axes[1, 0].set_ylabel('Moving Average Load (MW)')
		axes[1, 0].grid(True, alpha=0.3)
		
		# 4. Statistical information table
		axes[1, 1].axis('off')
		stats_text = []
		stats_text.append(['Statistic', 'Value'])
		stats_text.append(['Data Points', f"{self.stats['data_points']:,}"])
		stats_text.append(['Data Type', self.stats['data_type']])
		stats_text.append(['Mean', f"{self.stats['mean']:.2f} MW"])
		stats_text.append(['Std Dev', f"{self.stats['std']:.2f} MW"])
		stats_text.append(['Minimum', f"{self.stats['min']:.2f} MW"])
		stats_text.append(['Maximum', f"{self.stats['max']:.2f} MW"])
		stats_text.append(['Median', f"{self.stats['median']:.2f} MW"])
		stats_text.append(['25% Percentile', f"{self.stats['q25']:.2f} MW"])
		stats_text.append(['75% Percentile', f"{self.stats['q75']:.2f} MW"])
		
		table = axes[1, 1].table(cellText=stats_text[1:], colLabels=stats_text[0],
								cellLoc='center', loc='center')
		table.auto_set_font_size(False)
		table.set_fontsize(10)
		table.scale(1, 2)
		axes[1, 1].set_title('Statistical Information Table')
		
		plt.tight_layout()
		
		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"Statistical chart saved to: {save_path}")
		
		return fig
	
	def analyze_patterns(self) -> Dict:
		"""
		Analyze load patterns and characteristics
		
		Returns:
			Analysis results dictionary
		"""
		results = {}
		
		# Basic statistics
		results['basic_stats'] = self.stats.copy()
		
		# Peak-valley characteristics
		results['peak_valley'] = {
			'peak_load': np.max(self.raw_data),
			'valley_load': np.min(self.raw_data),
			'peak_valley_ratio': np.max(self.raw_data) / np.min(self.raw_data),
			'load_factor': np.mean(self.raw_data) / np.max(self.raw_data)
		}
		
		# If time data, analyze time patterns
		if isinstance(self.time_index, pd.DatetimeIndex):
			df_temp = pd.DataFrame({'load': self.raw_data, 'time': self.time_index})
			
			# Hourly pattern
			hourly_pattern = df_temp.groupby(df_temp['time'].dt.hour)['load'].mean()
			results['hourly_pattern'] = {
				'peak_hour': hourly_pattern.idxmax(),
				'valley_hour': hourly_pattern.idxmin(),
				'pattern': hourly_pattern.to_dict()
			}
			
			# Daily pattern (weekday/weekend)
			df_temp['weekday'] = df_temp['time'].dt.weekday
			weekday_load = df_temp[df_temp['weekday'] < 5]['load'].mean()  # Monday to Friday
			weekend_load = df_temp[df_temp['weekday'] >= 5]['load'].mean()  # Weekend
			results['weekly_pattern'] = {
				'weekday_avg': weekday_load,
				'weekend_avg': weekend_load,
				'weekday_weekend_ratio': weekday_load / weekend_load
			}
		
		# Coefficient of variation
		results['variability'] = {
			'coefficient_of_variation': self.stats['std'] / self.stats['mean'],
			'range_mean_ratio': (self.stats['max'] - self.stats['min']) / self.stats['mean']
		}
		
		return results
	
	def export_summary(self, output_path: str) -> None:
		"""
		Export analysis summary report
		
		Args:
			output_path: Output file path
		"""
		analysis = self.analyze_patterns()
		
		with open(output_path, 'w', encoding='utf-8') as f:
			f.write("Load Data Analysis Report\n")
			f.write("=" * 50 + "\n\n")
			
			# Basic information
			f.write("1. Basic Information\n")
			f.write("-" * 20 + "\n")
			f.write(f"Data File: {self.data_path}\n")
			f.write(f"Data Type: {analysis['basic_stats']['data_type']}\n")
			f.write(f"Data Points: {analysis['basic_stats']['data_points']:,}\n\n")
			
			# Statistical characteristics
			f.write("2. Statistical Characteristics\n")
			f.write("-" * 20 + "\n")
			f.write(f"Mean: {analysis['basic_stats']['mean']:.2f} MW\n")
			f.write(f"Std Dev: {analysis['basic_stats']['std']:.2f} MW\n")
			f.write(f"Maximum: {analysis['basic_stats']['max']:.2f} MW\n")
			f.write(f"Minimum: {analysis['basic_stats']['min']:.2f} MW\n")
			f.write(f"Median: {analysis['basic_stats']['median']:.2f} MW\n\n")
			
			# Peak-valley characteristics
			f.write("3. Peak-Valley Characteristics\n")
			f.write("-" * 20 + "\n")
			pv = analysis['peak_valley']
			f.write(f"Peak Load: {pv['peak_load']:.2f} MW\n")
			f.write(f"Valley Load: {pv['valley_load']:.2f} MW\n")
			f.write(f"Peak-Valley Ratio: {pv['peak_valley_ratio']:.2f}\n")
			f.write(f"Load Factor: {pv['load_factor']:.2f}\n\n")
			
			# Variability
			f.write("4. Variability Analysis\n")
			f.write("-" * 20 + "\n")
			var = analysis['variability']
			f.write(f"Coefficient of Variation: {var['coefficient_of_variation']:.3f}\n")
			f.write(f"Range-Mean Ratio: {var['range_mean_ratio']:.3f}\n\n")
			
			# Time patterns (if available)
			if 'hourly_pattern' in analysis:
				f.write("5. Time Patterns\n")
				f.write("-" * 20 + "\n")
				hp = analysis['hourly_pattern']
				f.write(f"Peak Hour: {hp['peak_hour']}:00\n")
				f.write(f"Valley Hour: {hp['valley_hour']}:00\n")
				
				wp = analysis['weekly_pattern']
				f.write(f"Weekday Average: {wp['weekday_avg']:.2f} MW\n")
				f.write(f"Weekend Average: {wp['weekend_avg']:.2f} MW\n")
				f.write(f"Weekday/Weekend Ratio: {wp['weekday_weekend_ratio']:.2f}\n")
	
		print(f"Analysis report saved to: {output_path}")


if __name__ == "__main__":
	# Usage example
	print("LoadshapeAnalyzer Universal Load Analyzer")
	print("Intelligent analysis supporting hourly and minute-level data")
	
	# Test data generation
	# Generate hourly test data
	hourly_data = np.random.normal(100, 20, 8760)  # 8760 hours
	analyzer_hourly = LoadshapeAnalyzer(data=hourly_data)
	
	print(f"\nHourly data test:")
	print(f"Data type: {analyzer_hourly.data_type}")
	print(f"Data points: {len(analyzer_hourly.raw_data)}")
	
	# Generate minute-level test data
	minutely_data = np.random.normal(100, 20, 525600)  # 525600 minutes
	analyzer_minutely = LoadshapeAnalyzer(data=minutely_data)
	
	print(f"\nMinute-level data test:")
	print(f"Data type: {analyzer_minutely.data_type}")
	print(f"Data points: {len(analyzer_minutely.raw_data)}")