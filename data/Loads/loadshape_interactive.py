#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Interactive Loadshape Analysis
增强版交互式负荷分析工具

支持小时级和分钟级数据的实时交互式分析和可视化

Author: PowerZoo Team
Date: 2025-07-30
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import threading
import queue
from datetime import datetime, timedelta

# 添加项目根路径
sys.path.append(str(Path(__file__).parent.parent))

from utils.loadshape_analyzer import LoadshapeAnalyzer

# 设置图表样式（使用系统默认字体）
plt.rcParams['axes.unicode_minus'] = False

class InteractiveLoadshapeAnalyzer:
	"""
	交互式负荷分析器GUI应用
	
	功能特点：
	- 图形化界面，易于操作
	- 实时数据可视化
	- 支持时间范围选择
	- 多种图表类型切换
	- 智能数据采样优化性能
	- 导出分析结果
	"""
	
	def __init__(self):
		"""初始化GUI应用"""
		self.root = tk.Tk()
		self.root.title("Enhanced Interactive Load Analysis Tool - PowerZoo")
		self.root.geometry("1400x900")
		
		# 数据相关属性
		self.analyzer = None
		self.current_file = None
		self.plot_queue = queue.Queue()
		
		# GUI组件
		self.setup_ui()
		
		# 绑定事件
		self.setup_events()
		
		print("交互式负荷分析器初始化完成")
	
	def setup_ui(self):
		"""设置用户界面"""
		# 创建主框架
		main_frame = ttk.Frame(self.root)
		main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
		
		# 左侧控制面板
		self.setup_control_panel(main_frame)
		
		# 右侧图表区域
		self.setup_plot_area(main_frame)
		
		# 底部状态栏
		self.setup_status_bar()
	
	def setup_control_panel(self, parent):
		"""设置控制面板"""
		# 控制面板框架
		control_frame = ttk.LabelFrame(parent, text="Control Panel", padding=10)
		control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
		control_frame.configure(width=350)
		
		# 文件操作区
		file_frame = ttk.LabelFrame(control_frame, text="File Operations", padding=5)
		file_frame.pack(fill=tk.X, pady=(0, 10))
		
		ttk.Button(file_frame, text="📁 Select Data File", 
				   command=self.load_file, width=25).pack(pady=2)
		
		self.file_label = ttk.Label(file_frame, text="No file selected", 
									foreground="gray", wraplength=300)
		self.file_label.pack(pady=2)
		
		# 数据信息区
		info_frame = ttk.LabelFrame(control_frame, text="Data Information", padding=5)
		info_frame.pack(fill=tk.X, pady=(0, 10))
		
		self.info_text = tk.Text(info_frame, height=8, width=40, 
								 font=("Consolas", 9), state=tk.DISABLED)
		info_scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, 
									   command=self.info_text.yview)
		self.info_text.configure(yscrollcommand=info_scrollbar.set)
		self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		
		# 时间范围选择区
		time_frame = ttk.LabelFrame(control_frame, text="Time Range Selection", padding=5)
		time_frame.pack(fill=tk.X, pady=(0, 10))
		
		# 快速选择按钮
		quick_frame = ttk.Frame(time_frame)
		quick_frame.pack(fill=tk.X, pady=(0, 5))
		
		ttk.Button(quick_frame, text="All", command=lambda: self.set_time_range("all"), 
				   width=8).pack(side=tk.LEFT, padx=2)
		ttk.Button(quick_frame, text="1 Day", command=lambda: self.set_time_range("1d"), 
				   width=8).pack(side=tk.LEFT, padx=2)
		ttk.Button(quick_frame, text="1 Week", command=lambda: self.set_time_range("1w"), 
				   width=8).pack(side=tk.LEFT, padx=2)
		ttk.Button(quick_frame, text="1 Month", command=lambda: self.set_time_range("1m"), 
				   width=8).pack(side=tk.LEFT, padx=2)
		
		# 自定义时间范围
		custom_frame = ttk.Frame(time_frame)
		custom_frame.pack(fill=tk.X, pady=(5, 0))
		
		ttk.Label(custom_frame, text="Start:").pack(anchor=tk.W)
		self.start_entry = ttk.Entry(custom_frame, width=25)
		self.start_entry.pack(fill=tk.X, pady=(0, 5))
		
		ttk.Label(custom_frame, text="End:").pack(anchor=tk.W)
		self.end_entry = ttk.Entry(custom_frame, width=25)
		self.end_entry.pack(fill=tk.X, pady=(0, 5))
		
		ttk.Button(custom_frame, text="Apply Time Range", 
				   command=self.apply_time_range, width=25).pack()
		
		# 图表选项区
		plot_frame = ttk.LabelFrame(control_frame, text="Chart Options", padding=5)
		plot_frame.pack(fill=tk.X, pady=(0, 10))
		
		# 图表类型选择
		ttk.Label(plot_frame, text="Chart Type:").pack(anchor=tk.W)
		self.plot_type_var = tk.StringVar(value="timeseries")
		plot_types = [
			("Time Series", "timeseries"),
			("Statistical Analysis", "statistics"),
			("Hourly Pattern", "hourly"),
			("Daily Pattern", "daily"),
			("Distribution", "distribution")
		]
		
		for text, value in plot_types:
			ttk.Radiobutton(plot_frame, text=text, variable=self.plot_type_var, 
							value=value, command=self.update_plot).pack(anchor=tk.W)
		
		# 显示选项
		options_frame = ttk.Frame(plot_frame)
		options_frame.pack(fill=tk.X, pady=(10, 0))
		
		self.show_stats_var = tk.BooleanVar(value=True)
		ttk.Checkbutton(options_frame, text="Show Statistics", 
						variable=self.show_stats_var, 
						command=self.update_plot).pack(anchor=tk.W)
		
		self.smart_sample_var = tk.BooleanVar(value=True)
		ttk.Checkbutton(options_frame, text="Smart Sampling", 
						variable=self.smart_sample_var, 
						command=self.update_plot).pack(anchor=tk.W)
		
		# 导出按钮区
		export_frame = ttk.LabelFrame(control_frame, text="Export Options", padding=5)
		export_frame.pack(fill=tk.X, pady=(0, 10))
		
		ttk.Button(export_frame, text="💾 Export Current Chart", 
				   command=self.export_current_plot, width=25).pack(pady=2)
		ttk.Button(export_frame, text="📊 Export Analysis Report", 
				   command=self.export_analysis_report, width=25).pack(pady=2)
		ttk.Button(export_frame, text="📋 Export All Charts", 
				   command=self.export_all_plots, width=25).pack(pady=2)
	
	def setup_plot_area(self, parent):
		"""设置图表区域"""
		# 图表框架
		plot_frame = ttk.LabelFrame(parent, text="Data Visualization", padding=5)
		plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
		
		# 创建matplotlib图表
		self.fig = Figure(figsize=(10, 8), dpi=100)
		self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
		self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
		
		# 添加工具栏
		toolbar_frame = ttk.Frame(plot_frame)
		toolbar_frame.pack(fill=tk.X, pady=(5, 0))
		
		self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
		self.toolbar.update()
		
		# 初始化空白图表
		self.show_welcome_plot()
	
	def setup_status_bar(self):
		"""设置状态栏"""
		self.status_var = tk.StringVar(value="Ready")
		status_bar = ttk.Label(self.root, textvariable=self.status_var, 
							   relief=tk.SUNKEN, anchor=tk.W)
		status_bar.pack(side=tk.BOTTOM, fill=tk.X)
		
		# 进度条
		self.progress_var = tk.DoubleVar()
		self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, 
											maximum=100)
		self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
		self.progress_bar.pack_forget()  # 初始隐藏
	
	def setup_events(self):
		"""设置事件绑定"""
		# 窗口关闭事件
		self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
		
		# 键盘快捷键
		self.root.bind('<Control-o>', lambda e: self.load_file())
		self.root.bind('<Control-s>', lambda e: self.export_current_plot())
		self.root.bind('<F5>', lambda e: self.update_plot())
	
	def show_welcome_plot(self):
		"""显示欢迎界面"""
		self.fig.clear()
		ax = self.fig.add_subplot(111)
		
		ax.text(0.5, 0.6, "Enhanced Interactive Load Analysis Tool", 
				horizontalalignment='center', verticalalignment='center',
				transform=ax.transAxes, fontsize=20, fontweight='bold')
		
		ax.text(0.5, 0.4, "Intelligent analysis for hourly and minute-level data", 
				horizontalalignment='center', verticalalignment='center',
				transform=ax.transAxes, fontsize=14, color='gray')
		
		ax.text(0.5, 0.2, "Please click 'Select Data File' to start analysis", 
				horizontalalignment='center', verticalalignment='center',
				transform=ax.transAxes, fontsize=12, color='blue')
		
		ax.set_xlim(0, 1)
		ax.set_ylim(0, 1)
		ax.axis('off')
		
		self.canvas.draw()
	
	def load_file(self):
		"""加载数据文件"""
		file_path = filedialog.askopenfilename(
			title="Select Load Data File",
			filetypes=[
				("All Supported Formats", "*.csv;*.txt;*.dat;*.npy"),
				("CSV Files", "*.csv"),
				("Text Files", "*.txt"),
				("Data Files", "*.dat"),
				("NumPy Files", "*.npy"),
				("All Files", "*.*")
			]
		)
		
		if not file_path:
			return
		
		try:
			self.set_status("Loading data file...")
			self.show_progress(True)
			
			# 在后台线程中加载数据
			threading.Thread(target=self._load_file_thread, 
							 args=(file_path,), daemon=True).start()
			
		except Exception as e:
			messagebox.showerror("Error", f"Failed to load file: {str(e)}")
			self.set_status("Ready")
			self.show_progress(False)
	
	def _load_file_thread(self, file_path):
		"""在后台线程中加载文件"""
		try:
			# 创建分析器
			analyzer = LoadshapeAnalyzer(data_path=file_path)
			
			# 更新UI（在主线程中执行）
			self.root.after(0, self._on_file_loaded, analyzer, file_path)
			
		except Exception as e:
			self.root.after(0, self._on_file_load_error, str(e))
	
	def _on_file_loaded(self, analyzer, file_path):
		"""文件加载完成回调"""
		self.analyzer = analyzer
		self.current_file = Path(file_path)
		
		# 更新文件标签
		self.file_label.config(text=f"📄 {self.current_file.name}", 
							   foreground="black")
		
		# 更新数据信息
		self.update_data_info()
		
		# 更新时间范围输入框
		self.update_time_range_entries()
		
		# 绘制默认图表
		self.update_plot()
		
		self.set_status(f"File loaded: {self.current_file.name}")
		self.show_progress(False)
	
	def _on_file_load_error(self, error_msg):
		"""文件加载错误回调"""
		messagebox.showerror("Load Error", f"Unable to load file: {error_msg}")
		self.set_status("File load failed")
		self.show_progress(False)
	
	def update_data_info(self):
		"""更新数据信息显示"""
		if not self.analyzer:
			return
		
		info_text = f"📊 Data Information\n"
		info_text += f"{'='*30}\n"
		info_text += f"File: {self.current_file.name}\n"
		info_text += f"Data Type: {self.analyzer.data_type}\n"
		info_text += f"Data Points: {self.analyzer.stats['data_points']:,}\n"
		info_text += f"\n📈 Statistical Information\n"
		info_text += f"{'='*30}\n"
		info_text += f"Mean: {self.analyzer.stats['mean']:.2f} MW\n"
		info_text += f"Std Dev: {self.analyzer.stats['std']:.2f} MW\n"
		info_text += f"Maximum: {self.analyzer.stats['max']:.2f} MW\n"
		info_text += f"Minimum: {self.analyzer.stats['min']:.2f} MW\n"
		info_text += f"Median: {self.analyzer.stats['median']:.2f} MW\n"
		
		# 如果有时间模式分析
		if hasattr(self.analyzer, 'time_index') and isinstance(self.analyzer.time_index, pd.DatetimeIndex):
			patterns = self.analyzer.analyze_patterns()
			if 'peak_valley' in patterns:
				pv = patterns['peak_valley']
				info_text += f"\n⚡ Peak-Valley Features\n"
				info_text += f"{'='*30}\n"
				info_text += f"Peak Load: {pv['peak_load']:.2f} MW\n"
				info_text += f"Valley Load: {pv['valley_load']:.2f} MW\n"
				info_text += f"Peak-Valley Ratio: {pv['peak_valley_ratio']:.2f}\n"
				info_text += f"Load Factor: {pv['load_factor']:.3f}\n"
		
		# 更新文本框
		self.info_text.config(state=tk.NORMAL)
		self.info_text.delete(1.0, tk.END)
		self.info_text.insert(1.0, info_text)
		self.info_text.config(state=tk.DISABLED)
	
	def update_time_range_entries(self):
		"""更新时间范围输入框"""
		if not self.analyzer or not isinstance(self.analyzer.time_index, pd.DatetimeIndex):
			# 非时间数据，使用索引
			self.start_entry.delete(0, tk.END)
			self.start_entry.insert(0, "0")
			self.end_entry.delete(0, tk.END)
			self.end_entry.insert(0, str(len(self.analyzer.raw_data)-1))
		else:
			# 时间数据
			start_time = self.analyzer.time_index[0].strftime("%Y-%m-%d %H:%M")
			end_time = self.analyzer.time_index[-1].strftime("%Y-%m-%d %H:%M")
			
			self.start_entry.delete(0, tk.END)
			self.start_entry.insert(0, start_time)
			self.end_entry.delete(0, tk.END)
			self.end_entry.insert(0, end_time)
	
	def set_time_range(self, range_type):
		"""设置预定义时间范围"""
		if not self.analyzer:
			return
		
		if not isinstance(self.analyzer.time_index, pd.DatetimeIndex):
			# 非时间数据，使用比例
			total_points = len(self.analyzer.raw_data)
			if range_type == "all":
				start_idx, end_idx = 0, total_points - 1
			elif range_type == "1d":
				end_idx = total_points - 1
				start_idx = max(0, end_idx - total_points // 365)  # 大约1天的数据
			elif range_type == "1w":
				end_idx = total_points - 1
				start_idx = max(0, end_idx - total_points // 52)   # 大约1周的数据
			elif range_type == "1m":
				end_idx = total_points - 1
				start_idx = max(0, end_idx - total_points // 12)   # 大约1月的数据
			
			self.start_entry.delete(0, tk.END)
			self.start_entry.insert(0, str(start_idx))
			self.end_entry.delete(0, tk.END)
			self.end_entry.insert(0, str(end_idx))
		else:
			# 时间数据
			end_time = self.analyzer.time_index[-1]
			
			if range_type == "all":
				start_time = self.analyzer.time_index[0]
			elif range_type == "1d":
				start_time = end_time - timedelta(days=1)
			elif range_type == "1w":
				start_time = end_time - timedelta(weeks=1)
			elif range_type == "1m":
				start_time = end_time - timedelta(days=30)
			
			self.start_entry.delete(0, tk.END)
			self.start_entry.insert(0, start_time.strftime("%Y-%m-%d %H:%M"))
			self.end_entry.delete(0, tk.END)
			self.end_entry.insert(0, end_time.strftime("%Y-%m-%d %H:%M"))
		
		# 自动应用时间范围
		self.apply_time_range()
	
	def apply_time_range(self):
		"""应用自定义时间范围"""
		if not self.analyzer:
			return
		
		try:
			start_str = self.start_entry.get().strip()
			end_str = self.end_entry.get().strip()
			
			if not start_str or not end_str:
				messagebox.showwarning("Warning", "Please enter complete time range")
				return
			
			# 更新图表
			self.update_plot()
			
		except Exception as e:
			messagebox.showerror("Error", f"Time range format error: {str(e)}")
	
	def update_plot(self):
		"""更新图表显示"""
		if not self.analyzer:
			return
		
		try:
			self.set_status("Updating chart...")
			
			# 在后台线程中生成图表
			threading.Thread(target=self._update_plot_thread, daemon=True).start()
			
		except Exception as e:
			messagebox.showerror("Error", f"Failed to update chart: {str(e)}")
			self.set_status("Chart update failed")
	
	def _update_plot_thread(self):
		"""在后台线程中更新图表"""
		try:
			plot_type = self.plot_type_var.get()
			
			# 获取时间范围
			start_str = self.start_entry.get().strip()
			end_str = self.end_entry.get().strip()
			
			start_time = start_str if start_str else None
			end_time = end_str if end_str else None
			
			# 生成图表数据
			if plot_type == "timeseries":
				plot_data = self._generate_timeseries_plot(start_time, end_time)
			elif plot_type == "statistics":
				plot_data = self._generate_statistics_plot()
			elif plot_type == "hourly":
				plot_data = self._generate_hourly_pattern_plot()
			elif plot_type == "daily":
				plot_data = self._generate_daily_pattern_plot()
			elif plot_type == "distribution":
				plot_data = self._generate_distribution_plot()
			else:
				plot_data = None
			
			# 在主线程中更新图表
			self.root.after(0, self._on_plot_ready, plot_data)
			
		except Exception as e:
			self.root.after(0, self._on_plot_error, str(e))
	
	def _generate_timeseries_plot(self, start_time, end_time):
		"""生成时间序列图数据"""
		data, time_idx = self.analyzer.get_time_range_data(start_time, end_time)
		
		# 智能采样
		if self.smart_sample_var.get():
			sampled_data, sampled_time = self.analyzer._smart_sample(data, time_idx)
		else:
			sampled_data, sampled_time = data, time_idx
		
		return {
			'type': 'timeseries',
			'data': sampled_data,
			'time': sampled_time,
			'original_length': len(data),
			'title': f"Load Time Series - {self.current_file.name}"
		}
	
	def _generate_statistics_plot(self):
		"""生成统计分析图数据"""
		# 智能采样用于统计分析
		if self.smart_sample_var.get():
			sampled_data, _ = self.analyzer._smart_sample(self.analyzer.raw_data)
		else:
			sampled_data = self.analyzer.raw_data
		
		return {
			'type': 'statistics',
			'data': sampled_data,
			'stats': self.analyzer.stats,
			'title': f"Statistical Analysis - {self.current_file.name}"
		}
	
	def _generate_hourly_pattern_plot(self):
		"""生成小时模式图数据"""
		if not isinstance(self.analyzer.time_index, pd.DatetimeIndex):
			raise ValueError("Hourly pattern analysis requires time series data")
		
		df_temp = pd.DataFrame({
			'load': self.analyzer.raw_data, 
			'time': self.analyzer.time_index
		})
		
		hourly_pattern = df_temp.groupby(df_temp['time'].dt.hour)['load'].agg(['mean', 'std', 'min', 'max'])
		
		return {
			'type': 'hourly',
			'pattern': hourly_pattern,
			'title': f"Hourly Load Pattern - {self.current_file.name}"
		}
	
	def _generate_daily_pattern_plot(self):
		"""生成日模式图数据"""
		if not isinstance(self.analyzer.time_index, pd.DatetimeIndex):
			raise ValueError("Daily pattern analysis requires time series data")
		
		df_temp = pd.DataFrame({
			'load': self.analyzer.raw_data, 
			'time': self.analyzer.time_index
		})
		
		daily_pattern = df_temp.groupby(df_temp['time'].dt.date)['load'].mean()
		
		# 智能采样日模式数据
		if self.smart_sample_var.get() and len(daily_pattern) > 1000:
			step = len(daily_pattern) // 1000
			daily_pattern = daily_pattern.iloc[::step]
		
		return {
			'type': 'daily',
			'pattern': daily_pattern,
			'title': f"Daily Load Pattern - {self.current_file.name}"
		}
	
	def _generate_distribution_plot(self):
		"""生成分布图数据"""
		if self.smart_sample_var.get():
			sampled_data, _ = self.analyzer._smart_sample(self.analyzer.raw_data)
		else:
			sampled_data = self.analyzer.raw_data
		
		return {
			'type': 'distribution',
			'data': sampled_data,
			'title': f"Load Distribution - {self.current_file.name}"
		}
	
	def _on_plot_ready(self, plot_data):
		"""图表准备完成回调"""
		if not plot_data:
			self.set_status("Chart generation failed")
			return
		
		# 清除之前的图表
		self.fig.clear()
		
		# 根据图表类型绘制
		if plot_data['type'] == 'timeseries':
			self._plot_timeseries(plot_data)
		elif plot_data['type'] == 'statistics':
			self._plot_statistics(plot_data)
		elif plot_data['type'] == 'hourly':
			self._plot_hourly_pattern(plot_data)
		elif plot_data['type'] == 'daily':
			self._plot_daily_pattern(plot_data)
		elif plot_data['type'] == 'distribution':
			self._plot_distribution(plot_data)
		
		# 更新画布
		self.canvas.draw()
		self.set_status("Chart update completed")
	
	def _plot_timeseries(self, plot_data):
		"""绘制时间序列图"""
		ax = self.fig.add_subplot(111)
		
		ax.plot(plot_data['time'], plot_data['data'], 
				linewidth=0.8, alpha=0.8, color='blue')
		
		if isinstance(plot_data['time'], pd.DatetimeIndex):
			ax.set_xlabel('Time')
		else:
			ax.set_xlabel('Data Points')
		
		ax.set_ylabel('Load (MW)')
		ax.set_title(plot_data['title'])
		ax.grid(True, alpha=0.3)
		
		# 添加统计信息
		if self.show_stats_var.get():
			stats_text = f"Mean: {np.mean(plot_data['data']):.2f} MW\n"
			stats_text += f"Max: {np.max(plot_data['data']):.2f} MW\n"
			stats_text += f"Min: {np.min(plot_data['data']):.2f} MW\n"
			stats_text += f"Original Points: {plot_data['original_length']:,}"
			
			ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
					verticalalignment='top', 
					bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
		
		self.fig.tight_layout()
	
	def _plot_statistics(self, plot_data):
		"""绘制统计分析图"""
		# 创建2x2子图
		((ax1, ax2), (ax3, ax4)) = self.fig.subplots(2, 2)
		
		data = plot_data['data']
		stats = plot_data['stats']
		
		# 1. 直方图
		ax1.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
		ax1.set_title('Load Distribution Histogram')
		ax1.set_xlabel('Load (MW)')
		ax1.set_ylabel('Frequency')
		ax1.grid(True, alpha=0.3)
		
		# 2. 箱线图
		ax2.boxplot(data, patch_artist=True,
					boxprops=dict(facecolor='lightgreen', alpha=0.7))
		ax2.set_title('Load Box Plot')
		ax2.set_ylabel('Load (MW)')
		ax2.grid(True, alpha=0.3)
		
		# 3. Q-Q图或概率图
		from scipy import stats as scipy_stats
		
		# 正态分布Q-Q图
		res = scipy_stats.probplot(data, dist="norm", plot=ax3)
		ax3.set_title('Normal Q-Q Plot')
		ax3.grid(True, alpha=0.3)
		
		# 4. 统计信息表
		ax4.axis('off')
		stats_data = [
			['Statistic', 'Value'],
			['Data Points', f"{stats['data_points']:,}"],
			['Mean', f"{stats['mean']:.2f} MW"],
			['Std Dev', f"{stats['std']:.2f} MW"],
			['Minimum', f"{stats['min']:.2f} MW"],
			['Maximum', f"{stats['max']:.2f} MW"],
			['Median', f"{stats['median']:.2f} MW"],
			['25% Percentile', f"{stats['q25']:.2f} MW"],
			['75% Percentile', f"{stats['q75']:.2f} MW"]
		]
		
		table = ax4.table(cellText=stats_data[1:], colLabels=stats_data[0],
						  cellLoc='center', loc='center')
		table.auto_set_font_size(False)
		table.set_fontsize(9)
		table.scale(1, 1.5)
		ax4.set_title('Statistical Information')
		
		self.fig.suptitle(plot_data['title'])
		self.fig.tight_layout()
	
	def _plot_hourly_pattern(self, plot_data):
		"""绘制小时模式图"""
		ax = self.fig.add_subplot(111)
		
		pattern = plot_data['pattern']
		hours = pattern.index
		
		# 绘制均值线
		ax.plot(hours, pattern['mean'], 'o-', linewidth=2, 
				markersize=6, label='Average', color='blue')
		
		# 绘制误差带
		ax.fill_between(hours, 
						pattern['mean'] - pattern['std'],
						pattern['mean'] + pattern['std'],
						alpha=0.3, color='blue', label='±1 Std Dev')
		
		# 绘制最大最小值
		ax.plot(hours, pattern['max'], '--', alpha=0.7, 
				color='red', label='Maximum')
		ax.plot(hours, pattern['min'], '--', alpha=0.7, 
				color='green', label='Minimum')
		
		ax.set_xlabel('Hour')
		ax.set_ylabel('Load (MW)')
		ax.set_title(plot_data['title'])
		ax.legend()
		ax.grid(True, alpha=0.3)
		ax.set_xlim(0, 23)
		
		self.fig.tight_layout()
	
	def _plot_daily_pattern(self, plot_data):
		"""绘制日模式图"""
		ax = self.fig.add_subplot(111)
		
		pattern = plot_data['pattern']
		
		ax.plot(pattern.index, pattern.values, 'o-', 
				linewidth=1, markersize=3, alpha=0.8)
		
		ax.set_xlabel('Date')
		ax.set_ylabel('Daily Average Load (MW)')
		ax.set_title(plot_data['title'])
		ax.grid(True, alpha=0.3)
		
		# 旋转x轴标签
		plt.setp(ax.get_xticklabels(), rotation=45)
		
		self.fig.tight_layout()
	
	def _plot_distribution(self, plot_data):
		"""绘制分布图"""
		ax = self.fig.add_subplot(111)
		
		data = plot_data['data']
		
		# 绘制直方图和核密度估计
		ax.hist(data, bins=50, density=True, alpha=0.7, 
				color='skyblue', edgecolor='black', label='Histogram')
		
		# 核密度估计
		from scipy.stats import gaussian_kde
		
		try:
			kde = gaussian_kde(data)
			x_range = np.linspace(data.min(), data.max(), 200)
			ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='Kernel Density Estimation')
		except:
			pass  # 如果KDE失败，只显示直方图
		
		ax.set_xlabel('Load (MW)')
		ax.set_ylabel('Density')
		ax.set_title(plot_data['title'])
		ax.legend()
		ax.grid(True, alpha=0.3)
		
		self.fig.tight_layout()
	
	def _on_plot_error(self, error_msg):
		"""图表生成错误回调"""
		messagebox.showerror("Chart Error", f"Error occurred while generating chart: {error_msg}")
		self.set_status("Chart generation failed")
	
	def export_current_plot(self):
		"""导出当前图表"""
		if not self.analyzer:
			messagebox.showwarning("Warning", "Please load data file first")
			return
		
		file_path = filedialog.asksaveasfilename(
			title="Save Chart",
			defaultextension=".png",
			filetypes=[
				("PNG Images", "*.png"),
				("PDF Files", "*.pdf"),
				("SVG Files", "*.svg"),
				("All Files", "*.*")
			]
		)
		
		if file_path:
			try:
				self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
				messagebox.showinfo("Success", f"Chart saved to: {file_path}")
			except Exception as e:
				messagebox.showerror("Error", f"Failed to save chart: {str(e)}")
	
	def export_analysis_report(self):
		"""导出分析报告"""
		if not self.analyzer:
			messagebox.showwarning("警告", "请先加载数据文件")
			return
		
		file_path = filedialog.asksaveasfilename(
			title="Save Analysis Report",
			defaultextension=".txt",
			filetypes=[
				("Text Files", "*.txt"),
				("All Files", "*.*")
			]
		)
		
		if file_path:
			try:
				self.analyzer.export_summary(file_path)
				messagebox.showinfo("Success", f"Analysis report saved to: {file_path}")
			except Exception as e:
				messagebox.showerror("Error", f"Failed to save report: {str(e)}")
	
	def export_all_plots(self):
		"""导出所有图表"""
		if not self.analyzer:
			messagebox.showwarning("警告", "请先加载数据文件")
			return
		
		output_dir = filedialog.askdirectory(title="Select Output Directory")
		if not output_dir:
			return
		
		try:
			self.set_status("Exporting all charts...")
			self.show_progress(True)
			
			# 在后台线程中导出
			threading.Thread(target=self._export_all_plots_thread, 
							 args=(output_dir,), daemon=True).start()
			
		except Exception as e:
			messagebox.showerror("Error", f"Export failed: {str(e)}")
			self.show_progress(False)
	
	def _export_all_plots_thread(self, output_dir):
		"""在后台线程中导出所有图表"""
		try:
			output_path = Path(output_dir)
			file_prefix = self.current_file.stem
			
			# 保存原始图表类型
			original_plot_type = self.plot_type_var.get()
			
			plot_types = ["timeseries", "statistics", "hourly", "daily", "distribution"]
			
			for i, plot_type in enumerate(plot_types):
				try:
					# 更新进度
					progress = (i + 1) / len(plot_types) * 100
					self.root.after(0, self.progress_var.set, progress)
					
					# 在主线程中切换图表类型
					self.root.after(0, self.plot_type_var.set, plot_type)
					
					# 等待图表更新
					import time
					time.sleep(1)
					
					# 保存图表
					save_path = output_path / f"{file_prefix}_{plot_type}.png"
					self.root.after(0, self._save_current_plot, str(save_path))
					
				except Exception as e:
					print(f"导出 {plot_type} 图表时出错: {e}")
					continue
			
			# 恢复原始图表类型
			self.root.after(0, self.plot_type_var.set, original_plot_type)
			
			# 导出分析报告
			report_path = output_path / f"{file_prefix}_report.txt"
			self.analyzer.export_summary(str(report_path))
			
			self.root.after(0, self._on_export_complete, output_dir)
			
		except Exception as e:
			self.root.after(0, self._on_export_error, str(e))
	
	def _save_current_plot(self, save_path):
		"""保存当前图表"""
		try:
			self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
		except Exception as e:
			print(f"保存图表失败: {e}")
	
	def _on_export_complete(self, output_dir):
		"""导出完成回调"""
		self.show_progress(False)
		self.set_status("All charts export completed")
		messagebox.showinfo("Success", f"All charts exported to: {output_dir}")
	
	def _on_export_error(self, error_msg):
		"""导出错误回调"""
		self.show_progress(False)
		self.set_status("Export failed")
		messagebox.showerror("Error", f"Export failed: {error_msg}")
	
	def set_status(self, message):
		"""设置状态栏信息"""
		self.status_var.set(message)
		self.root.update()
	
	def show_progress(self, show=True):
		"""显示或隐藏进度条"""
		if show:
			self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, before=self.status_var.master)
			self.progress_var.set(0)
		else:
			self.progress_bar.pack_forget()
	
	def on_closing(self):
		"""窗口关闭事件处理"""
		if messagebox.askokcancel("Exit", "Are you sure you want to exit the program?"):
			self.root.quit()
			self.root.destroy()
	
	def run(self):
		"""运行应用"""
		print("Starting Interactive Load Analyzer...")
		self.root.mainloop()


def main():
	"""主函数"""
	try:
		app = InteractiveLoadshapeAnalyzer()
		app.run()
	except Exception as e:
		print(f"Application startup failed: {e}")
		return 1
	
	return 0


if __name__ == "__main__":
	exit(main())