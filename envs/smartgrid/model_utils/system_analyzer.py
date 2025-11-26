# -*- coding: utf-8 -*-
"""
系统参数分析器 - PowerZoo LLM环境训练数据分析
专用于HAPPO训练过程中记录的系统参数的分析、可视化和报告生成

功能特点:
- 训练过程系统性能分析
- 电力系统运行状态统计
- 奖励函数分解分析
- 设备控制策略评估
- 可视化图表生成
- 训练报告自动化生成
"""

import h5py
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import yaml
from datetime import datetime
import logging
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

try:
	from envs.smartgrid.utils import get_logger
except ImportError:
	def get_logger(name):
		logging.basicConfig(level=logging.INFO)
		return logging.getLogger(name)

logger = get_logger(__name__)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class AnalysisConfig:
	"""分析配置"""
	# 数据处理配置
	smooth_window: int = 50  # 平滑窗口大小
	outlier_threshold: float = 3.0  # 异常值检测阈值（标准差倍数）
	min_episode_length: int = 10  # 最小有效回合长度
	
	# 可视化配置
	figure_size: Tuple[int, int] = (12, 8)
	dpi: int = 300
	style: str = 'seaborn-v0_8'
	color_palette: str = 'husl'
	
	# 统计分析配置
	confidence_level: float = 0.95  # 置信区间
	correlation_threshold: float = 0.3  # 相关性阈值
	
	# 报告生成配置
	include_detailed_plots: bool = True
	generate_summary_only: bool = False
	export_raw_data: bool = False


class SystemAnalyzer:
	"""
	系统参数分析器
	
	功能:
	1. 加载和预处理训练数据
	2. 电力系统性能分析
	3. 奖励函数分解分析
	4. 设备控制策略评估
	5. 可视化图表生成
	6. 训练报告生成
	"""
	
	def __init__(self, 
				log_dir: str,
				config: Optional[AnalysisConfig] = None,
				output_dir: str = "./analysis_results"):
		"""
		初始化分析器
		
		Args:
			log_dir: 日志数据目录
			config: 分析配置
			output_dir: 输出结果目录
		"""
		self.log_dir = Path(log_dir)
		self.config = config or AnalysisConfig()
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)
		
		# 数据存储
		self.raw_data = {}
		self.processed_data = {}
		self.statistics = {}
		self.metadata = {}
		
		# 分析结果
		self.analysis_results = {}
		self.visualizations = {}
		
		# 设置绘图样式
		plt.style.use(self.config.style)
		sns.set_palette(self.config.color_palette)
		
		logger.info(f"SystemAnalyzer initialized - Log dir: {log_dir}, Output dir: {output_dir}")
	
	def load_data(self, session_id: Optional[str] = None) -> bool:
		"""
		加载训练数据
		
		Args:
			session_id: 指定会话ID，如果为None则加载最新会话
			
		Returns:
			bool: 加载是否成功
		"""
		try:
			# 查找数据文件
			if session_id:
				hdf5_file = self.log_dir / f"system_data_{session_id}.h5"
				metadata_file = self.log_dir / f"metadata_{session_id}.yaml"
			else:
				# 找到最新的会话文件
				hdf5_files = list(self.log_dir.glob("system_data_*.h5"))
				if not hdf5_files:
					logger.error(f"No HDF5 files found in {self.log_dir}")
					return False
				
				hdf5_file = max(hdf5_files, key=lambda x: x.stat().st_mtime)
				session_id = hdf5_file.stem.replace("system_data_", "")
				metadata_file = self.log_dir / f"metadata_{session_id}.yaml"
			
			# 加载元数据
			if metadata_file.exists():
				with open(metadata_file, 'r') as f:
					self.metadata = yaml.safe_load(f)
				logger.info(f"Loaded metadata for session: {session_id}")
			
			# 加载HDF5数据
			with h5py.File(hdf5_file, 'r') as f:
				self._load_hdf5_data(f)
			
			# 加载最终统计（如果存在）
			stats_file = self.log_dir / f"final_stats_{session_id}.json"
			if stats_file.exists():
				with open(stats_file, 'r') as f:
					self.statistics = json.load(f)
			
			logger.info(f"Successfully loaded data from session: {session_id}")
			return True
			
		except Exception as e:
			logger.error(f"Failed to load data: {e}")
			return False
	
	def _load_hdf5_data(self, hdf5_file: h5py.File):
		"""从HDF5文件加载数据"""
		self.raw_data = {}
		
		# 加载时间序列数据
		if 'timeseries' in hdf5_file:
			ts_group = hdf5_file['timeseries']
			
			# 基础时间信息
			if 'timestamps' in ts_group:
				self.raw_data['timestamps'] = ts_group['timestamps'][:]
			if 'episodes' in ts_group:
				self.raw_data['episodes'] = ts_group['episodes'][:]
			if 'steps' in ts_group:
				self.raw_data['steps'] = ts_group['steps'][:]
			
			# 奖励数据
			if 'rewards' in ts_group:
				reward_group = ts_group['rewards']
				self.raw_data['rewards'] = {}
				for key in reward_group.keys():
					self.raw_data['rewards'][key] = reward_group[key][:]
			
			# 性能数据
			if 'performance' in ts_group:
				perf_group = ts_group['performance']
				self.raw_data['performance'] = {}
				for key in perf_group.keys():
					self.raw_data['performance'][key] = perf_group[key][:]
		
		logger.debug(f"Loaded {len(self.raw_data)} data categories")
	
	def preprocess_data(self):
		"""预处理数据"""
		if not self.raw_data:
			logger.error("No raw data to preprocess")
			return
		
		self.processed_data = {}
		
		try:
			# 创建主数据框
			main_df = self._create_main_dataframe()
			self.processed_data['main'] = main_df
			
			# 按回合聚合数据
			episode_df = self._aggregate_by_episode(main_df)
			self.processed_data['episodes'] = episode_df
			
			# 计算滑动统计
			smoothed_df = self._apply_smoothing(main_df)
			self.processed_data['smoothed'] = smoothed_df
			
			# 检测异常值
			anomalies = self._detect_anomalies(main_df)
			self.processed_data['anomalies'] = anomalies
			
			logger.info("Data preprocessing completed")
			
		except Exception as e:
			logger.error(f"Data preprocessing failed: {e}")
	
	def _create_main_dataframe(self) -> pd.DataFrame:
		"""创建主数据框"""
		data_dict = {}
		
		# 基础时间信息
		if 'timestamps' in self.raw_data:
			data_dict['timestamp'] = self.raw_data['timestamps']
		if 'episodes' in self.raw_data:
			data_dict['episode'] = self.raw_data['episodes']
		if 'steps' in self.raw_data:
			data_dict['step'] = self.raw_data['steps']
		
		# 奖励数据
		if 'rewards' in self.raw_data:
			for key, values in self.raw_data['rewards'].items():
				data_dict[f'reward_{key.replace("_rewards", "").replace("_penalties", "_penalty")}'] = values
		
		# 性能数据
		if 'performance' in self.raw_data:
			for key, values in self.raw_data['performance'].items():
				data_dict[f'perf_{key.replace("_indices", "_index")}'] = values
		
		df = pd.DataFrame(data_dict)
		
		# 添加时间相关列
		if 'timestamp' in df.columns:
			df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
			df['elapsed_time'] = (df['timestamp'] - df['timestamp'].min()) / 3600  # 小时
		
		return df
	
	def _aggregate_by_episode(self, df: pd.DataFrame) -> pd.DataFrame:
		"""按回合聚合数据"""
		if 'episode' not in df.columns:
			return pd.DataFrame()
		
		# 筛选有效回合（长度足够）
		episode_lengths = df.groupby('episode').size()
		valid_episodes = episode_lengths[episode_lengths >= self.config.min_episode_length].index
		df_valid = df[df['episode'].isin(valid_episodes)]
		
		# 聚合统计
		agg_funcs = {
			'step': 'count',  # 回合长度
			'elapsed_time': 'max',  # 回合持续时间
		}
		
		# 添加数值列的聚合函数
		numeric_cols = df_valid.select_dtypes(include=[np.number]).columns
		for col in numeric_cols:
			if col not in ['episode', 'step', 'timestamp']:
				agg_funcs[col] = ['mean', 'std', 'min', 'max', 'sum']
		
		episode_stats = df_valid.groupby('episode').agg(agg_funcs)
		
		# 展平多级列名
		episode_stats.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
								for col in episode_stats.columns]
		
		# 重命名step列
		episode_stats.rename(columns={'step_count': 'episode_length'}, inplace=True)
		
		return episode_stats.reset_index()
	
	def _apply_smoothing(self, df: pd.DataFrame) -> pd.DataFrame:
		"""应用平滑处理"""
		smoothed_df = df.copy()
		
		# 对数值列应用滑动平均
		numeric_cols = df.select_dtypes(include=[np.number]).columns
		window = min(self.config.smooth_window, len(df) // 10)  # 确保窗口不会太大
		
		for col in numeric_cols:
			if col not in ['episode', 'step', 'timestamp']:
				smoothed_df[f'{col}_smooth'] = df[col].rolling(
					window=window, min_periods=1, center=True
				).mean()
		
		return smoothed_df
	
	def _detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
		"""检测异常值"""
		anomalies = []
		
		numeric_cols = df.select_dtypes(include=[np.number]).columns
		
		for col in numeric_cols:
			if col not in ['episode', 'step', 'timestamp'] and not col.endswith('_smooth'):
				mean_val = df[col].mean()
				std_val = df[col].std()
				threshold = self.config.outlier_threshold * std_val
				
				outliers = df[
					(df[col] < mean_val - threshold) | 
					(df[col] > mean_val + threshold)
				].copy()
				
				if not outliers.empty:
					outliers['anomaly_type'] = col
					outliers['anomaly_value'] = outliers[col]
					outliers['anomaly_z_score'] = np.abs((outliers[col] - mean_val) / std_val)
					anomalies.append(outliers[['episode', 'step', 'anomaly_type', 
												'anomaly_value', 'anomaly_z_score']])
		
		return pd.concat(anomalies, ignore_index=True) if anomalies else pd.DataFrame()
	
	def analyze_system_performance(self) -> Dict:
		"""分析系统性能"""
		if 'main' not in self.processed_data:
			logger.error("No processed data available for analysis")
			return {}
		
		df = self.processed_data['main']
		analysis = {}
		
		try:
			# 1. 总体性能统计
			analysis['overall_stats'] = self._calculate_overall_stats(df)
			
			# 2. 训练收敛分析
			analysis['convergence'] = self._analyze_convergence(df)
			
			# 3. 奖励分解分析
			analysis['reward_analysis'] = self._analyze_rewards(df)
			
			# 4. 系统稳定性分析
			analysis['stability'] = self._analyze_stability(df)
			
			# 5. 性能指标趋势
			analysis['performance_trends'] = self._analyze_performance_trends(df)
			
			# 6. 异常情况分析
			if 'anomalies' in self.processed_data and not self.processed_data['anomalies'].empty:
				analysis['anomalies'] = self._analyze_anomalies()
			
			self.analysis_results['system_performance'] = analysis
			logger.info("System performance analysis completed")
			
		except Exception as e:
			logger.error(f"System performance analysis failed: {e}")
		
		return analysis
	
	def _calculate_overall_stats(self, df: pd.DataFrame) -> Dict:
		"""计算总体统计信息"""
		stats = {
			'total_steps': len(df),
			'total_episodes': df['episode'].nunique() if 'episode' in df.columns else 0,
			'training_duration_hours': df['elapsed_time'].max() if 'elapsed_time' in df.columns else 0,
		}
		
		# 计算各种奖励的统计
		reward_cols = [col for col in df.columns if col.startswith('reward_')]
		for col in reward_cols:
			reward_name = col.replace('reward_', '')
			stats[f'{reward_name}_mean'] = df[col].mean()
			stats[f'{reward_name}_std'] = df[col].std()
			stats[f'{reward_name}_min'] = df[col].min()
			stats[f'{reward_name}_max'] = df[col].max()
		
		# 计算性能指标统计
		perf_cols = [col for col in df.columns if col.startswith('perf_')]
		for col in perf_cols:
			perf_name = col.replace('perf_', '')
			stats[f'{perf_name}_mean'] = df[col].mean()
			stats[f'{perf_name}_std'] = df[col].std()
		
		return stats
	
	def _analyze_convergence(self, df: pd.DataFrame) -> Dict:
		"""分析训练收敛性"""
		convergence = {}
		
		if 'reward_total' in df.columns:
			# 使用滑动平均分析收敛
			if 'reward_total_smooth' in df.columns:
				smooth_rewards = df['reward_total_smooth'].dropna()
				
				# 计算收敛点（奖励不再显著增长的点）
				window = min(100, len(smooth_rewards) // 10)
				if window > 0:
					reward_diff = smooth_rewards.rolling(window).apply(
						lambda x: (x[-1] - x[0]) / window
					)
					
					convergence_threshold = smooth_rewards.std() * 0.01  # 1%标准差阈值
					converged_idx = np.where(np.abs(reward_diff) < convergence_threshold)[0]
					
					if len(converged_idx) > 0:
						convergence['convergence_step'] = converged_idx[0]
						convergence['convergence_episode'] = df.iloc[converged_idx[0]]['episode'] if 'episode' in df.columns else None
						convergence['final_performance'] = smooth_rewards.iloc[-window:].mean()
					
					convergence['reward_improvement'] = smooth_rewards.iloc[-1] - smooth_rewards.iloc[0]
					convergence['improvement_rate'] = convergence['reward_improvement'] / len(smooth_rewards)
		
		return convergence
	
	def _analyze_rewards(self, df: pd.DataFrame) -> Dict:
		"""分析奖励组成"""
		reward_analysis = {}
		
		reward_cols = [col for col in df.columns if col.startswith('reward_')]
		
		if reward_cols:
			# 奖励组成占比
			reward_data = df[reward_cols]
			reward_analysis['composition'] = {
				col.replace('reward_', ''): {
					'mean': reward_data[col].mean(),
					'contribution_pct': abs(reward_data[col].mean()) / abs(reward_data.mean(axis=1)).mean() * 100
				}
				for col in reward_cols
			}
			
			# 奖励相关性分析
			correlation_matrix = reward_data.corr()
			reward_analysis['correlations'] = correlation_matrix.to_dict()
			
			# 奖励稳定性分析
			reward_analysis['stability'] = {
				col.replace('reward_', ''): {
					'coefficient_of_variation': reward_data[col].std() / abs(reward_data[col].mean()) if reward_data[col].mean() != 0 else float('inf')
				}
				for col in reward_cols
			}
		
		return reward_analysis
	
	def _analyze_stability(self, df: pd.DataFrame) -> Dict:
		"""分析系统稳定性"""
		stability = {}
		
		# DSS收敛率
		if 'perf_dss_convergence' in df.columns:
			convergence_rate = df['perf_dss_convergence'].mean()
			stability['dss_convergence_rate'] = convergence_rate
			stability['dss_failures'] = len(df[df['perf_dss_convergence'] == 0])
		
		# 计算时间分析
		if 'perf_computation_times' in df.columns:
			comp_times = df['perf_computation_times']
			stability['avg_computation_time'] = comp_times.mean()
			stability['max_computation_time'] = comp_times.max()
			stability['computation_time_std'] = comp_times.std()
			stability['slow_steps_pct'] = (comp_times > 0.1).mean() * 100  # >100ms认为慢
		
		# 系统稳定性指标
		if 'perf_stability_index' in df.columns:
			stability_idx = df['perf_stability_index']
			stability['avg_stability_index'] = stability_idx.mean()
			stability['min_stability_index'] = stability_idx.min()
			stability['stability_degradation_pct'] = (stability_idx < 0.8).mean() * 100
		
		return stability
	
	def _analyze_performance_trends(self, df: pd.DataFrame) -> Dict:
		"""分析性能趋势"""
		trends = {}
		
		if 'episode' in df.columns:
			# 按回合分析趋势
			episode_stats = df.groupby('episode').agg({
				col: 'mean' for col in df.columns 
				if col.startswith(('reward_', 'perf_')) and col != 'episode'
			})
			
			# 计算线性趋势
			for col in episode_stats.columns:
				if len(episode_stats) > 5:  # 至少5个数据点
					x = np.arange(len(episode_stats))
					y = episode_stats[col].values
					
					# 去除NaN值
					valid_mask = ~np.isnan(y)
					if valid_mask.sum() > 3:
						x_valid = x[valid_mask]
						y_valid = y[valid_mask]
						
						# 线性拟合
						slope, intercept = np.polyfit(x_valid, y_valid, 1)
						r_squared = np.corrcoef(x_valid, y_valid)[0, 1] ** 2
						
						trends[col] = {
							'slope': slope,
							'r_squared': r_squared,
							'trend': 'improving' if slope > 0 else 'declining' if slope < 0 else 'stable'
						}
		
		return trends
	
	def _analyze_anomalies(self) -> Dict:
		"""分析异常情况"""
		anomalies_df = self.processed_data['anomalies']
		analysis = {}
		
		if not anomalies_df.empty:
			# 异常类型统计
			analysis['anomaly_counts'] = anomalies_df['anomaly_type'].value_counts().to_dict()
			
			# 异常严重程度分布
			analysis['severity_distribution'] = {
				'mild': len(anomalies_df[anomalies_df['anomaly_z_score'] < 4]),
				'moderate': len(anomalies_df[(anomalies_df['anomaly_z_score'] >= 4) & 
											(anomalies_df['anomaly_z_score'] < 6)]),
				'severe': len(anomalies_df[anomalies_df['anomaly_z_score'] >= 6])
			}
			
			# 异常回合分析
			if 'episode' in anomalies_df.columns:
				anomaly_episodes = anomalies_df['episode'].value_counts()
				analysis['most_problematic_episodes'] = anomaly_episodes.head(10).to_dict()
		
		return analysis
	
	def generate_visualizations(self) -> Dict[str, str]:
		"""生成可视化图表"""
		if 'main' not in self.processed_data:
			logger.error("No processed data available for visualization")
			return {}
		
		visualizations = {}
		
		try:
			# 1. 训练进度总览
			fig_path = self._plot_training_overview()
			if fig_path:
				visualizations['training_overview'] = fig_path
			
			# 2. 奖励分解分析
			fig_path = self._plot_reward_decomposition()
			if fig_path:
				visualizations['reward_decomposition'] = fig_path
			
			# 3. 系统性能监控
			fig_path = self._plot_system_performance()
			if fig_path:
				visualizations['system_performance'] = fig_path
			
			# 4. 收敛性分析
			fig_path = self._plot_convergence_analysis()
			if fig_path:
				visualizations['convergence_analysis'] = fig_path
			
			# 5. 异常检测结果
			if 'anomalies' in self.processed_data and not self.processed_data['anomalies'].empty:
				fig_path = self._plot_anomaly_detection()
				if fig_path:
					visualizations['anomaly_detection'] = fig_path
			
			# 6. 设备控制分析（如果有相关数据）
			fig_path = self._plot_device_control_analysis()
			if fig_path:
				visualizations['device_control'] = fig_path
			
			self.visualizations = visualizations
			logger.info(f"Generated {len(visualizations)} visualizations")
			
		except Exception as e:
			logger.error(f"Visualization generation failed: {e}")
		
		return visualizations
	
	def _plot_training_overview(self) -> Optional[str]:
		"""绘制训练总览图"""
		df = self.processed_data['main']
		
		fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
		fig.suptitle('PowerZoo训练总览', fontsize=16, fontweight='bold')
		
		# 1. 奖励趋势
		if 'reward_total' in df.columns:
			axes[0, 0].plot(df.index, df['reward_total'], alpha=0.3, color='blue', label='原始奖励')
			if 'reward_total_smooth' in df.columns:
				axes[0, 0].plot(df.index, df['reward_total_smooth'], color='red', linewidth=2, label='平滑奖励')
			axes[0, 0].set_title('奖励趋势')
			axes[0, 0].set_xlabel('训练步数')
			axes[0, 0].set_ylabel('奖励值')
			axes[0, 0].legend()
			axes[0, 0].grid(True, alpha=0.3)
		
		# 2. 计算时间分布
		if 'perf_computation_times' in df.columns:
			comp_times = df['perf_computation_times']
			axes[0, 1].hist(comp_times, bins=50, alpha=0.7, color='green', edgecolor='black')
			axes[0, 1].axvline(comp_times.mean(), color='red', linestyle='--', 
							  label=f'平均: {comp_times.mean():.4f}s')
			axes[0, 1].set_title('计算时间分布')
			axes[0, 1].set_xlabel('计算时间 (秒)')
			axes[0, 1].set_ylabel('频次')
			axes[0, 1].legend()
			axes[0, 1].grid(True, alpha=0.3)
		
		# 3. DSS收敛率
		if 'perf_dss_convergence' in df.columns:
			convergence_rolling = df['perf_dss_convergence'].rolling(window=100, min_periods=1).mean()
			axes[1, 0].plot(df.index, convergence_rolling, color='purple', linewidth=2)
			axes[1, 0].set_title('DSS收敛率 (滑动平均)')
			axes[1, 0].set_xlabel('训练步数')
			axes[1, 0].set_ylabel('收敛率')
			axes[1, 0].set_ylim([0, 1.1])
			axes[1, 0].grid(True, alpha=0.3)
		
		# 4. 系统稳定性指标
		if 'perf_stability_index' in df.columns:
			stability_rolling = df['perf_stability_index'].rolling(window=100, min_periods=1).mean()
			axes[1, 1].plot(df.index, stability_rolling, color='orange', linewidth=2)
			axes[1, 1].set_title('系统稳定性指标 (滑动平均)')
			axes[1, 1].set_xlabel('训练步数')
			axes[1, 1].set_ylabel('稳定性指标')
			axes[1, 1].set_ylim([0, 1.1])
			axes[1, 1].grid(True, alpha=0.3)
		
		plt.tight_layout()
		
		# 保存图表
		output_path = self.output_dir / 'training_overview.png'
		plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
		plt.close()
		
		logger.debug(f"Training overview plot saved to {output_path}")
		return str(output_path)
	
	def _plot_reward_decomposition(self) -> Optional[str]:
		"""绘制奖励分解图"""
		df = self.processed_data['main']
		reward_cols = [col for col in df.columns if col.startswith('reward_') and not col.endswith('_smooth')]
		
		if len(reward_cols) < 2:
			logger.warning("Insufficient reward data for decomposition plot")
			return None
		
		fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
		fig.suptitle('奖励函数分解分析', fontsize=16, fontweight='bold')
		
		# 1. 奖励组成时间序列
		ax1 = axes[0, 0]
		for col in reward_cols:
			label = col.replace('reward_', '').replace('_', ' ').title()
			if f'{col}_smooth' in df.columns:
				ax1.plot(df.index, df[f'{col}_smooth'], label=label, linewidth=2)
			else:
				ax1.plot(df.index, df[col], label=label, alpha=0.7)
		
		ax1.set_title('奖励组成时间序列')
		ax1.set_xlabel('训练步数')
		ax1.set_ylabel('奖励值')
		ax1.legend()
		ax1.grid(True, alpha=0.3)
		
		# 2. 奖励组成占比饼图
		reward_means = {col.replace('reward_', ''): abs(df[col].mean()) for col in reward_cols}
		if sum(reward_means.values()) > 0:
			axes[0, 1].pie(reward_means.values(), labels=reward_means.keys(), autopct='%1.1f%%')
			axes[0, 1].set_title('奖励组成占比')
		
		# 3. 奖励相关性热力图
		reward_data = df[reward_cols]
		correlation_matrix = reward_data.corr()
		im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
		axes[1, 0].set_xticks(range(len(reward_cols)))
		axes[1, 0].set_yticks(range(len(reward_cols)))
		axes[1, 0].set_xticklabels([col.replace('reward_', '') for col in reward_cols], rotation=45)
		axes[1, 0].set_yticklabels([col.replace('reward_', '') for col in reward_cols])
		axes[1, 0].set_title('奖励相关性矩阵')
		plt.colorbar(im, ax=axes[1, 0])
		
		# 4. 奖励分布箱线图
		reward_data_clean = reward_data.dropna()
		if not reward_data_clean.empty:
			box_data = [reward_data_clean[col] for col in reward_cols]
			box_labels = [col.replace('reward_', '') for col in reward_cols]
			axes[1, 1].boxplot(box_data, labels=box_labels)
			axes[1, 1].set_title('奖励分布')
			axes[1, 1].set_ylabel('奖励值')
			axes[1, 1].tick_params(axis='x', rotation=45)
		
		plt.tight_layout()
		
		# 保存图表
		output_path = self.output_dir / 'reward_decomposition.png'
		plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
		plt.close()
		
		logger.debug(f"Reward decomposition plot saved to {output_path}")
		return str(output_path)
	
	def _plot_system_performance(self) -> Optional[str]:
		"""绘制系统性能图"""
		df = self.processed_data['main']
		
		fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
		fig.suptitle('电力系统性能监控', fontsize=16, fontweight='bold')
		
		# 1. 计算时间趋势
		if 'perf_computation_times' in df.columns:
			comp_times = df['perf_computation_times']
			comp_times_smooth = comp_times.rolling(window=50, min_periods=1).mean()
			
			axes[0, 0].plot(df.index, comp_times, alpha=0.3, color='blue', label='原始')
			axes[0, 0].plot(df.index, comp_times_smooth, color='red', linewidth=2, label='平滑')
			axes[0, 0].axhline(0.1, color='orange', linestyle='--', label='100ms阈值')
			axes[0, 0].set_title('计算时间趋势')
			axes[0, 0].set_xlabel('训练步数')
			axes[0, 0].set_ylabel('计算时间 (秒)')
			axes[0, 0].legend()
			axes[0, 0].grid(True, alpha=0.3)
		
		# 2. DSS收敛性分析
		if 'perf_dss_convergence' in df.columns:
			convergence = df['perf_dss_convergence']
			convergence_rate = convergence.rolling(window=100, min_periods=1).mean()
			
			axes[0, 1].plot(df.index, convergence_rate, color='green', linewidth=2)
			axes[0, 1].fill_between(df.index, convergence_rate, alpha=0.3, color='green')
			axes[0, 1].set_title('DSS收敛率')
			axes[0, 1].set_xlabel('训练步数')
			axes[0, 1].set_ylabel('收敛率')
			axes[0, 1].set_ylim([0, 1.1])
			axes[0, 1].grid(True, alpha=0.3)
		
		# 3. 系统稳定性和电能质量
		if 'perf_stability_index' in df.columns and 'perf_power_quality_index' in df.columns:
			stability = df['perf_stability_index'].rolling(window=50, min_periods=1).mean()
			quality = df['perf_power_quality_index'].rolling(window=50, min_periods=1).mean()
			
			axes[1, 0].plot(df.index, stability, label='稳定性指标', linewidth=2)
			axes[1, 0].plot(df.index, quality, label='电能质量指标', linewidth=2)
			axes[1, 0].set_title('系统指标对比')
			axes[1, 0].set_xlabel('训练步数')
			axes[1, 0].set_ylabel('指标值')
			axes[1, 0].legend()
			axes[1, 0].grid(True, alpha=0.3)
		
		# 4. 性能指标分布
		perf_cols = [col for col in df.columns if col.startswith('perf_') and 'index' in col]
		if perf_cols:
			perf_data = df[perf_cols].dropna()
			if not perf_data.empty:
				perf_data.hist(ax=axes[1, 1], bins=20, alpha=0.7)
				axes[1, 1].set_title('性能指标分布')
				axes[1, 1].legend([col.replace('perf_', '').replace('_', ' ') for col in perf_cols])
		
		plt.tight_layout()
		
		# 保存图表
		output_path = self.output_dir / 'system_performance.png'
		plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
		plt.close()
		
		logger.debug(f"System performance plot saved to {output_path}")
		return str(output_path)
	
	def _plot_convergence_analysis(self) -> Optional[str]:
		"""绘制收敛性分析图"""
		if 'episodes' not in self.processed_data:
			logger.warning("No episode data available for convergence analysis")
			return None
		
		episode_df = self.processed_data['episodes']
		
		if episode_df.empty:
			logger.warning("Episode dataframe is empty")
			return None
		
		fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
		fig.suptitle('训练收敛性分析', fontsize=16, fontweight='bold')
		
		# 1. 回合平均奖励趋势
		reward_cols = [col for col in episode_df.columns if 'reward_total_mean' in col]
		if reward_cols:
			col = reward_cols[0]
			axes[0, 0].plot(episode_df.index, episode_df[col], 'b-', linewidth=2, label='回合平均奖励')
			
			# 添加趋势线
			if len(episode_df) > 5:
				z = np.polyfit(episode_df.index, episode_df[col], 1)
				p = np.poly1d(z)
				axes[0, 0].plot(episode_df.index, p(episode_df.index), 'r--', 
							   label=f'趋势线 (斜率: {z[0]:.4f})')
			
			axes[0, 0].set_title('回合平均奖励收敛')
			axes[0, 0].set_xlabel('回合数')
			axes[0, 0].set_ylabel('平均奖励')
			axes[0, 0].legend()
			axes[0, 0].grid(True, alpha=0.3)
		
		# 2. 回合长度趋势
		if 'episode_length' in episode_df.columns:
			axes[0, 1].plot(episode_df.index, episode_df['episode_length'], 'g-', linewidth=2)
			axes[0, 1].set_title('回合长度趋势')
			axes[0, 1].set_xlabel('回合数')
			axes[0, 1].set_ylabel('回合长度')
			axes[0, 1].grid(True, alpha=0.3)
		
		# 3. 奖励标准差趋势（反映训练稳定性）
		reward_std_cols = [col for col in episode_df.columns if 'reward_total_std' in col]
		if reward_std_cols:
			col = reward_std_cols[0]
			axes[1, 0].plot(episode_df.index, episode_df[col], 'purple', linewidth=2)
			axes[1, 0].set_title('奖励稳定性 (标准差)')
			axes[1, 0].set_xlabel('回合数')
			axes[1, 0].set_ylabel('奖励标准差')
			axes[1, 0].grid(True, alpha=0.3)
		
		# 4. 学习进展热力图
		if len(episode_df) > 10:
			# 选择几个关键指标创建热力图
			heatmap_cols = []
			for prefix in ['reward_total_mean', 'reward_voltage_mean', 'reward_control_mean']:
				matching_cols = [col for col in episode_df.columns if prefix in col]
				if matching_cols:
					heatmap_cols.append(matching_cols[0])
			
			if heatmap_cols:
				heatmap_data = episode_df[heatmap_cols].T
				im = axes[1, 1].imshow(heatmap_data, aspect='auto', cmap='viridis')
				axes[1, 1].set_title('学习进展热力图')
				axes[1, 1].set_xlabel('回合数')
				axes[1, 1].set_yticks(range(len(heatmap_cols)))
				axes[1, 1].set_yticklabels([col.replace('reward_', '').replace('_mean', '') for col in heatmap_cols])
				plt.colorbar(im, ax=axes[1, 1])
		
		plt.tight_layout()
		
		# 保存图表
		output_path = self.output_dir / 'convergence_analysis.png'
		plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
		plt.close()
		
		logger.debug(f"Convergence analysis plot saved to {output_path}")
		return str(output_path)
	
	def _plot_anomaly_detection(self) -> Optional[str]:
		"""绘制异常检测图"""
		anomalies_df = self.processed_data['anomalies']
		
		if anomalies_df.empty:
			logger.warning("No anomalies to plot")
			return None
		
		fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
		fig.suptitle('异常检测分析', fontsize=16, fontweight='bold')
		
		# 1. 异常类型分布
		anomaly_counts = anomalies_df['anomaly_type'].value_counts()
		axes[0, 0].bar(range(len(anomaly_counts)), anomaly_counts.values)
		axes[0, 0].set_xticks(range(len(anomaly_counts)))
		axes[0, 0].set_xticklabels(anomaly_counts.index, rotation=45)
		axes[0, 0].set_title('异常类型分布')
		axes[0, 0].set_ylabel('异常次数')
		
		# 2. 异常严重程度分布
		severity_bins = [0, 3, 4, 6, float('inf')]
		severity_labels = ['轻微', '中等', '严重', '极严重']
		severity_counts = pd.cut(anomalies_df['anomaly_z_score'], bins=severity_bins, labels=severity_labels).value_counts()
		
		axes[0, 1].pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%')
		axes[0, 1].set_title('异常严重程度分布')
		
		# 3. 异常时间分布
		if 'episode' in anomalies_df.columns:
			axes[1, 0].hist(anomalies_df['episode'], bins=20, alpha=0.7, edgecolor='black')
			axes[1, 0].set_title('异常回合分布')
			axes[1, 0].set_xlabel('回合数')
			axes[1, 0].set_ylabel('异常次数')
		
		# 4. Z-score分布
		axes[1, 1].hist(anomalies_df['anomaly_z_score'], bins=30, alpha=0.7, edgecolor='black')
		axes[1, 1].axvline(3, color='orange', linestyle='--', label='轻微阈值')
		axes[1, 1].axvline(6, color='red', linestyle='--', label='严重阈值')
		axes[1, 1].set_title('异常Z-score分布')
		axes[1, 1].set_xlabel('Z-score')
		axes[1, 1].set_ylabel('频次')
		axes[1, 1].legend()
		
		plt.tight_layout()
		
		# 保存图表
		output_path = self.output_dir / 'anomaly_detection.png'
		plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
		plt.close()
		
		logger.debug(f"Anomaly detection plot saved to {output_path}")
		return str(output_path)
	
	def _plot_device_control_analysis(self) -> Optional[str]:
		"""绘制设备控制分析图（占位实现）"""
		# 这里需要根据实际的设备控制数据来实现
		# 由于当前数据结构中没有详细的设备控制信息，这里提供占位实现
		
		logger.info("Device control analysis plot - placeholder implementation")
		return None
	
	def generate_report(self, 
					   report_name: str = "training_analysis_report",
					   include_raw_data: bool = False) -> str:
		"""生成分析报告"""
		
		report_path = self.output_dir / f"{report_name}.html"
		
		try:
			# HTML报告模板
			html_template = """
			<!DOCTYPE html>
			<html>
			<head>
				<title>PowerZoo训练分析报告</title>
				<meta charset="utf-8">
				<style>
					body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; line-height: 1.6; }}
					.header {{ text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
					.section {{ margin: 30px 0; padding: 20px; border-left: 4px solid #667eea; background: #f8f9fa; }}
					.metric {{ display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
					.chart {{ text-align: center; margin: 20px 0; }}
					.chart img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
					table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
					th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
					th {{ background-color: #667eea; color: white; }}
					.highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; border-left: 4px solid #ffc107; }}
					.success {{ color: #28a745; }}
					.warning {{ color: #ffc107; }}
					.danger {{ color: #dc3545; }}
				</style>
			</head>
			<body>
				<div class="header">
					<h1>PowerZoo LLM环境 - HAPPO训练分析报告</h1>
					<p>生成时间: {timestamp}</p>
					<p>训练会话: {session_id}</p>
				</div>
				
				{executive_summary}
				
				{system_performance}
				
				{visualizations}
				
				{detailed_analysis}
				
				{recommendations}
				
				{raw_data_section}
				
			</body>
			</html>
			"""
			
			# 生成各个部分的内容
			sections = {
				'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				'session_id': self.metadata.get('session_id', 'Unknown'),
				'executive_summary': self._generate_executive_summary(),
				'system_performance': self._generate_performance_section(),
				'visualizations': self._generate_visualization_section(),
				'detailed_analysis': self._generate_detailed_analysis_section(),
				'recommendations': self._generate_recommendations_section(),
				'raw_data_section': self._generate_raw_data_section() if include_raw_data else ""
			}
			
			# 填充模板
			html_content = html_template.format(**sections)
			
			# 写入文件
			with open(report_path, 'w', encoding='utf-8') as f:
				f.write(html_content)
			
			logger.info(f"Analysis report generated: {report_path}")
			return str(report_path)
			
		except Exception as e:
			logger.error(f"Failed to generate report: {e}")
			return ""
	
	def _generate_executive_summary(self) -> str:
		"""生成执行摘要"""
		if 'system_performance' not in self.analysis_results:
			return "<div class='section'><h2>执行摘要</h2><p>分析数据不足，无法生成摘要。</p></div>"
		
		analysis = self.analysis_results['system_performance']
		overall_stats = analysis.get('overall_stats', {})
		
		summary = f"""
		<div class="section">
			<h2>📊 执行摘要</h2>
			<div class="metric">
				<strong>训练总步数:</strong> {overall_stats.get('total_steps', 0):,}
			</div>
			<div class="metric">
				<strong>训练回合数:</strong> {overall_stats.get('total_episodes', 0):,}
			</div>
			<div class="metric">
				<strong>训练时长:</strong> {overall_stats.get('training_duration_hours', 0):.2f} 小时
			</div>
			<div class="metric">
				<strong>平均奖励:</strong> {overall_stats.get('total_mean', 0):.4f}
			</div>
			<div class="metric">
				<strong>平均计算时间:</strong> {overall_stats.get('computation_times_mean', 0):.4f} 秒
			</div>
		</div>
		"""
		
		return summary
	
	def _generate_performance_section(self) -> str:
		"""生成性能分析部分"""
		if 'system_performance' not in self.analysis_results:
			return "<div class='section'><h2>系统性能</h2><p>性能分析数据不可用。</p></div>"
		
		analysis = self.analysis_results['system_performance']
		stability = analysis.get('stability', {})
		convergence = analysis.get('convergence', {})
		
		performance_html = f"""
		<div class="section">
			<h2>⚡ 系统性能分析</h2>
			
			<h3>收敛性分析</h3>
			<div class="highlight">
				<p><strong>奖励改善:</strong> {convergence.get('reward_improvement', 0):.4f}</p>
				<p><strong>改善速率:</strong> {convergence.get('improvement_rate', 0):.6f}/步</p>
				<p><strong>最终性能:</strong> {convergence.get('final_performance', 0):.4f}</p>
			</div>
			
			<h3>系统稳定性</h3>
			<div class="highlight">
				<p><strong>DSS收敛率:</strong> <span class="{'success' if stability.get('dss_convergence_rate', 0) > 0.95 else 'warning'}">{stability.get('dss_convergence_rate', 0)*100:.2f}%</span></p>
				<p><strong>平均计算时间:</strong> {stability.get('avg_computation_time', 0):.4f} 秒</p>
				<p><strong>平均稳定性指标:</strong> {stability.get('avg_stability_index', 0):.4f}</p>
			</div>
		</div>
		"""
		
		return performance_html
	
	def _generate_visualization_section(self) -> str:
		"""生成可视化部分"""
		if not self.visualizations:
			return "<div class='section'><h2>可视化分析</h2><p>暂无可视化图表。</p></div>"
		
		viz_html = "<div class='section'><h2>📈 可视化分析</h2>"
		
		for name, path in self.visualizations.items():
			title = name.replace('_', ' ').title()
			# 使用相对路径
			relative_path = Path(path).name
			viz_html += f"""
			<div class="chart">
				<h3>{title}</h3>
				<img src="{relative_path}" alt="{title}">
			</div>
			"""
		
		viz_html += "</div>"
		return viz_html
	
	def _generate_detailed_analysis_section(self) -> str:
		"""生成详细分析部分"""
		if 'system_performance' not in self.analysis_results:
			return "<div class='section'><h2>详细分析</h2><p>详细分析数据不可用。</p></div>"
		
		analysis = self.analysis_results['system_performance']
		reward_analysis = analysis.get('reward_analysis', {})
		
		detailed_html = f"""
		<div class="section">
			<h2>🔍 详细分析</h2>
			
			<h3>奖励函数分析</h3>
			<table>
				<tr><th>奖励组件</th><th>平均值</th><th>贡献占比</th></tr>
		"""
		
		composition = reward_analysis.get('composition', {})
		for component, data in composition.items():
			detailed_html += f"""
				<tr>
					<td>{component.replace('_', ' ').title()}</td>
					<td>{data.get('mean', 0):.4f}</td>
					<td>{data.get('contribution_pct', 0):.1f}%</td>
				</tr>
			"""
		
		detailed_html += """
			</table>
		</div>
		"""
		
		return detailed_html
	
	def _generate_recommendations_section(self) -> str:
		"""生成建议部分"""
		recommendations = []
		
		if 'system_performance' in self.analysis_results:
			analysis = self.analysis_results['system_performance']
			stability = analysis.get('stability', {})
			
			# 基于DSS收敛率的建议
			dss_convergence_rate = stability.get('dss_convergence_rate', 1.0)
			if dss_convergence_rate < 0.95:
				recommendations.append("⚠️ DSS收敛率较低，建议检查系统参数配置或减小学习率")
			
			# 基于计算时间的建议
			avg_comp_time = stability.get('avg_computation_time', 0)
			if avg_comp_time > 0.1:
				recommendations.append("⚠️ 计算时间偏高，建议优化算法实现或简化网络模型")
			
			# 基于稳定性指标的建议
			stability_index = stability.get('avg_stability_index', 1.0)
			if stability_index < 0.8:
				recommendations.append("⚠️ 系统稳定性指标偏低，建议调整奖励函数权重或控制策略")
		
		if not recommendations:
			recommendations.append("✅ 系统运行正常，训练表现良好")
		
		rec_html = f"""
		<div class="section">
			<h2>💡 优化建议</h2>
			<ul>
				{''.join(f'<li>{rec}</li>' for rec in recommendations)}
			</ul>
		</div>
		"""
		
		return rec_html
	
	def _generate_raw_data_section(self) -> str:
		"""生成原始数据部分"""
		if not self.config.export_raw_data:
			return ""
		
		# 导出处理后的主要数据
		if 'main' in self.processed_data:
			csv_path = self.output_dir / "processed_data.csv"
			self.processed_data['main'].to_csv(csv_path, index=False)
			
			return f"""
			<div class="section">
				<h2>📄 原始数据</h2>
				<p>处理后的训练数据已导出至: <a href="{csv_path.name}">processed_data.csv</a></p>
			</div>
			"""
		
		return ""
	
	def run_full_analysis(self, session_id: Optional[str] = None) -> Dict:
		"""运行完整分析流程"""
		logger.info("Starting full analysis pipeline...")
		
		# 1. 加载数据
		if not self.load_data(session_id):
			logger.error("Failed to load data, aborting analysis")
			return {}
		
		# 2. 预处理数据
		self.preprocess_data()
		
		# 3. 系统性能分析
		performance_results = self.analyze_system_performance()
		
		# 4. 生成可视化
		visualizations = self.generate_visualizations()
		
		# 5. 生成报告
		report_path = self.generate_report()
		
		results = {
			'analysis_results': self.analysis_results,
			'visualizations': visualizations,
			'report_path': report_path,
			'output_directory': str(self.output_dir)
		}
		
		logger.info(f"Full analysis completed. Results saved to: {self.output_dir}")
		return results


# 便利函数
def analyze_training_session(log_dir: str, 
							session_id: Optional[str] = None,
							output_dir: str = "./analysis_results",
							config: Optional[AnalysisConfig] = None) -> Dict:
	"""
	便利函数：分析指定的训练会话
	
	Args:
		log_dir: 日志目录
		session_id: 会话ID（可选，默认使用最新）
		output_dir: 输出目录
		config: 分析配置
		
	Returns:
		分析结果字典
	"""
	analyzer = SystemAnalyzer(log_dir, config, output_dir)
	return analyzer.run_full_analysis(session_id)


if __name__ == "__main__":
	# 示例用法
	import argparse
	
	parser = argparse.ArgumentParser(description="PowerZoo系统参数分析器")
	parser.add_argument("--log_dir", type=str, required=True, help="日志目录路径")
	parser.add_argument("--session_id", type=str, help="指定会话ID")
	parser.add_argument("--output_dir", type=str, default="./analysis_results", help="输出目录")
	
	args = parser.parse_args()
	
	# 运行分析
	results = analyze_training_session(
		log_dir=args.log_dir,
		session_id=args.session_id,
		output_dir=args.output_dir
	)
	
	print(f"分析完成！结果保存至: {results.get('output_directory', 'Unknown')}")
	if results.get('report_path'):
		print(f"报告路径: {results['report_path']}")