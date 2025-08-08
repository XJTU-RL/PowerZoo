"""
奖励可视化模块

提供奖励组件、约束违反、拉格朗日乘子等的可视化功能
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# 设置matplotlib参数
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3


class RewardVisualizer:
	"""
	奖励系统可视化器
	
	提供训练过程中各种指标的可视化功能
	"""
	
	def __init__(
		self,
		save_dir: str = './visualization',
		figsize: Tuple[int, int] = (12, 8),
		style: str = 'seaborn-v0_8-darkgrid'
	):
		"""
		初始化可视化器
		
		Args:
			save_dir: 图片保存目录
			figsize: 图片尺寸
			style: matplotlib样式
		"""
		self.save_dir = Path(save_dir)
		self.save_dir.mkdir(parents=True, exist_ok=True)
		self.figsize = figsize
		
		# 设置样式
		try:
			plt.style.use(style)
		except:
			logger.warning(f"样式 {style} 不可用，使用默认样式")
		
		logger.info(f"可视化器初始化: 保存目录={save_dir}")
	
	def plot_training_curves(
		self,
		data: Dict[str, List[float]],
		title: str = "Training Progress",
		save_name: str = "training_curves.png",
		show_only_hours: bool = True
	):
		"""
		绘制训练曲线
		
		Args:
			data: 训练数据字典，键为指标名，值为数据列表
			title: 图表标题
			save_name: 保存文件名
			show_only_hours: 是否只在x轴显示整点
		"""
		fig, axes = plt.subplots(2, 2, figsize=self.figsize)
		fig.suptitle(title, fontsize=14, fontweight='bold')
		
		# 定义要绘制的指标
		metrics = [
			('total_reward_raw', 'Total Reward', axes[0, 0], 'tab:blue'),
			('cost_voltage', 'Voltage Cost', axes[0, 1], 'tab:red'),
			('lambda', 'Lagrangian Lambda', axes[1, 0], 'tab:green'),
			('voltage_violation_rate', 'Violation Rate (%)', axes[1, 1], 'tab:orange')
		]
		
		for metric_key, metric_name, ax, color in metrics:
			if metric_key in data and data[metric_key]:
				values = data[metric_key]
				steps = np.arange(len(values))
				
				# 绘制曲线
				ax.plot(steps, values, color=color, linewidth=1.5, alpha=0.8)
				
				# 添加移动平均
				if len(values) > 20:
					window = min(20, len(values) // 10)
					ma = np.convolve(values, np.ones(window) / window, mode='valid')
					ma_steps = steps[window-1:]
					ax.plot(ma_steps, ma, color=color, linewidth=2, alpha=0.9, 
						   label=f'MA({window})')
				
				# 设置标题和标签
				ax.set_title(metric_name, fontweight='bold')
				ax.set_xlabel('Step')
				ax.set_ylabel(metric_name)
				
				# 设置x轴刻度（只显示整点）
				if show_only_hours:
					self._set_hour_ticks(ax, len(values))
				
				# 添加统计信息
				mean_val = np.mean(values)
				std_val = np.std(values)
				ax.axhline(y=mean_val, color='gray', linestyle='--', alpha=0.5)
				ax.text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}',
					   transform=ax.transAxes, verticalalignment='top',
					   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
				
				if 'MA' in [l.get_label() for l in ax.lines]:
					ax.legend(loc='upper right')
		
		plt.tight_layout()
		
		# 保存图片
		save_path = self.save_dir / save_name
		plt.savefig(save_path, bbox_inches='tight')
		plt.close()
		
		logger.info(f"训练曲线已保存: {save_path}")
	
	def plot_reward_components(
		self,
		components: Dict[str, List[float]],
		title: str = "Reward Components",
		save_name: str = "reward_components.png"
	):
		"""
		绘制奖励组件分解图
		
		Args:
			components: 奖励组件数据
			title: 图表标题
			save_name: 保存文件名
		"""
		fig, ax = plt.subplots(figsize=self.figsize)
		
		# 准备数据
		labels = []
		values = []
		colors = []
		color_map = {
			'r_power': 'tab:blue',
			'r_ctrl': 'tab:orange',
			'r_pv': 'tab:green',
			'cost_voltage': 'tab:red'
		}
		
		for key, data in components.items():
			if data and key in color_map:
				labels.append(key)
				values.append(np.mean(np.abs(data)))
				colors.append(color_map[key])
		
		# 绘制条形图
		bars = ax.bar(labels, values, color=colors, alpha=0.8)
		
		# 添加数值标签
		for bar, value in zip(bars, values):
			height = bar.get_height()
			ax.text(bar.get_x() + bar.get_width()/2., height,
				   f'{value:.4f}',
				   ha='center', va='bottom')
		
		ax.set_title(title, fontsize=14, fontweight='bold')
		ax.set_ylabel('Average Magnitude')
		ax.set_xlabel('Component')
		
		# 保存图片
		save_path = self.save_dir / save_name
		plt.savefig(save_path, bbox_inches='tight')
		plt.close()
		
		logger.info(f"奖励组件图已保存: {save_path}")
	
	def plot_voltage_heatmap(
		self,
		voltage_violations: Dict[str, List[int]],
		title: str = "Voltage Violation Heatmap",
		save_name: str = "voltage_heatmap.png"
	):
		"""
		绘制电压违约热度图
		
		Args:
			voltage_violations: 各母线违约次数统计
			title: 图表标题
			save_name: 保存文件名
		"""
		if not voltage_violations:
			logger.warning("无电压违约数据可视化")
			return
		
		fig, ax = plt.subplots(figsize=(14, 6))
		
		# 准备数据
		bus_names = list(voltage_violations.keys())
		violations = list(voltage_violations.values())
		
		# 计算P95
		p95 = np.percentile(violations, 95)
		
		# 设置颜色
		colors = ['green' if v < p95 * 0.5 else 'yellow' if v < p95 else 'red' 
				 for v in violations]
		
		# 绘制条形图
		bars = ax.bar(bus_names, violations, color=colors, alpha=0.8)
		
		# 添加P95线
		ax.axhline(y=p95, color='red', linestyle='--', 
				  label=f'P95 = {p95:.0f}', linewidth=2)
		
		ax.set_title(title, fontsize=14, fontweight='bold')
		ax.set_xlabel('Bus Name')
		ax.set_ylabel('Violation Count')
		ax.set_xticklabels(bus_names, rotation=45, ha='right')
		ax.legend()
		
		# 保存图片
		save_path = self.save_dir / save_name
		plt.savefig(save_path, bbox_inches='tight')
		plt.close()
		
		logger.info(f"电压违约热度图已保存: {save_path}")
	
	def plot_lambda_evolution(
		self,
		lambda_history: List[float],
		cost_history: List[float],
		target_cost: float = 0.01,
		title: str = "Lambda Evolution",
		save_name: str = "lambda_evolution.png"
	):
		"""
		绘制拉格朗日乘子演化图
		
		Args:
			lambda_history: Lambda历史
			cost_history: 成本历史
			target_cost: 目标成本
			title: 图表标题
			save_name: 保存文件名
		"""
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
		
		steps = np.arange(len(lambda_history))
		
		# 绘制Lambda
		ax1.plot(steps, lambda_history, 'b-', linewidth=1.5, label='Lambda')
		ax1.set_ylabel('Lambda Value', color='b')
		ax1.tick_params(axis='y', labelcolor='b')
		ax1.set_title(title, fontsize=14, fontweight='bold')
		ax1.grid(True, alpha=0.3)
		
		# 绘制成本
		ax2.plot(steps, cost_history, 'r-', linewidth=1.5, label='Cost')
		ax2.axhline(y=target_cost, color='g', linestyle='--', 
				   linewidth=2, label=f'Target = {target_cost}')
		ax2.set_ylabel('Constraint Cost', color='r')
		ax2.tick_params(axis='y', labelcolor='r')
		ax2.set_xlabel('Update Step')
		ax2.legend(loc='upper right')
		ax2.grid(True, alpha=0.3)
		
		# 添加收敛区域
		if len(cost_history) > 20:
			recent_cost = np.mean(cost_history[-20:])
			if abs(recent_cost - target_cost) < target_cost * 0.3:
				ax2.axhspan(target_cost * 0.7, target_cost * 1.3, 
						   alpha=0.2, color='green', label='Target Zone')
		
		plt.tight_layout()
		
		# 保存图片
		save_path = self.save_dir / save_name
		plt.savefig(save_path, bbox_inches='tight')
		plt.close()
		
		logger.info(f"Lambda演化图已保存: {save_path}")
	
	def plot_p95_calibration(
		self,
		calibration_report: Dict[str, Any],
		title: str = "P95 Calibration Report",
		save_name: str = "p95_calibration.png"
	):
		"""
		绘制P95标定报告
		
		Args:
			calibration_report: 标定报告数据
			title: 图表标题
			save_name: 保存文件名
		"""
		fig, axes = plt.subplots(2, 2, figsize=self.figsize)
		fig.suptitle(title, fontsize=14, fontweight='bold')
		
		stats = calibration_report.get('statistics', {})
		weights = calibration_report.get('recommended_weights', {})
		
		# 1. P95值对比
		ax = axes[0, 0]
		if stats:
			components = ['r_power', 'r_ctrl', 'r_pv', 'cost_voltage']
			p95_values = []
			for comp in components:
				if comp in stats:
					p95_values.append(stats[comp].get('p95', 0))
				else:
					p95_values.append(0)
			
			bars = ax.bar(components, p95_values, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
			ax.set_title('P95 Values')
			ax.set_ylabel('P95 Magnitude')
			
			for bar, val in zip(bars, p95_values):
				ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
					   f'{val:.4f}', ha='center', va='bottom')
		
		# 2. 推荐权重
		ax = axes[0, 1]
		weight_keys = ['power_w', 'ctrl_w', 'pv_w']
		weight_values = [weights.get(k, 0) for k in weight_keys]
		
		bars = ax.bar(weight_keys, weight_values, color=['tab:blue', 'tab:orange', 'tab:green'])
		ax.set_title('Recommended Weights')
		ax.set_ylabel('Weight Value')
		
		for bar, val in zip(bars, weight_values):
			ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
				   f'{val:.4f}', ha='center', va='bottom')
		
		# 3. 分布箱线图
		ax = axes[1, 0]
		if stats:
			box_data = []
			box_labels = []
			for comp in ['r_power', 'r_ctrl', 'r_pv']:
				if comp in stats:
					# 模拟分布数据（实际应从原始数据获取）
					mean = stats[comp].get('mean', 0)
					std = stats[comp].get('std', 1)
					data = np.random.normal(mean, std, 100)
					box_data.append(np.abs(data))
					box_labels.append(comp)
			
			if box_data:
				bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
				colors = ['tab:blue', 'tab:orange', 'tab:green']
				for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
					patch.set_facecolor(color)
					patch.set_alpha(0.5)
				ax.set_title('Component Distributions')
				ax.set_ylabel('Magnitude')
		
		# 4. 权重比例
		ax = axes[1, 1]
		if weights:
			# 饼图显示权重比例
			sizes = [weights.get('power_w', 0.2), 
					weights.get('ctrl_w', 0.3),
					weights.get('pv_w', 0.5)]
			labels = ['Power', 'Control', 'PV']
			colors = ['tab:blue', 'tab:orange', 'tab:green']
			
			if sum(sizes) > 0:
				ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
					  startangle=90)
				ax.set_title('Weight Proportions')
		
		plt.tight_layout()
		
		# 保存图片
		save_path = self.save_dir / save_name
		plt.savefig(save_path, bbox_inches='tight')
		plt.close()
		
		logger.info(f"P95标定报告图已保存: {save_path}")
	
	def _set_hour_ticks(self, ax, data_length: int, hours_per_day: int = 24):
		"""
		设置x轴只显示整点
		
		Args:
			ax: matplotlib轴对象
			data_length: 数据长度
			hours_per_day: 每天小时数
		"""
		# 计算整点位置
		hour_indices = list(range(0, data_length, hours_per_day))
		if hour_indices[-1] != data_length - 1:
			hour_indices.append(data_length - 1)
		
		# 设置刻度
		ax.set_xticks(hour_indices)
		ax.set_xticklabels([f'{i//hours_per_day}d' if i % hours_per_day == 0 else f'{i}h' 
						   for i in hour_indices])
	
	def create_summary_dashboard(
		self,
		training_data: Dict[str, Any],
		save_name: str = "summary_dashboard.png"
	):
		"""
		创建综合仪表板
		
		Args:
			training_data: 包含所有训练数据的字典
			save_name: 保存文件名
		"""
		fig = plt.figure(figsize=(16, 10))
		gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
		
		# 主标题
		fig.suptitle('Training Summary Dashboard', fontsize=16, fontweight='bold')
		
		# 1. 总奖励趋势（大图）
		ax1 = fig.add_subplot(gs[0, :2])
		if 'total_reward' in training_data:
			self._plot_metric(ax1, training_data['total_reward'], 
							 'Total Reward Trend', 'tab:blue')
		
		# 2. 电压成本
		ax2 = fig.add_subplot(gs[0, 2])
		if 'cost_voltage' in training_data:
			self._plot_metric(ax2, training_data['cost_voltage'], 
							 'Voltage Cost', 'tab:red', compact=True)
		
		# 3. Lambda演化
		ax3 = fig.add_subplot(gs[1, 0])
		if 'lambda' in training_data:
			self._plot_metric(ax3, training_data['lambda'], 
							 'Lambda', 'tab:green', compact=True)
		
		# 4. 违约率
		ax4 = fig.add_subplot(gs[1, 1])
		if 'violation_rate' in training_data:
			self._plot_metric(ax4, training_data['violation_rate'], 
							 'Violation Rate', 'tab:orange', compact=True)
		
		# 5. 奖励组件饼图
		ax5 = fig.add_subplot(gs[1, 2])
		if 'component_weights' in training_data:
			self._plot_component_pie(ax5, training_data['component_weights'])
		
		# 6-8. 统计表格
		ax6 = fig.add_subplot(gs[2, :])
		if 'statistics' in training_data:
			self._plot_statistics_table(ax6, training_data['statistics'])
		
		# 保存
		save_path = self.save_dir / save_name
		plt.savefig(save_path, bbox_inches='tight', dpi=150)
		plt.close()
		
		logger.info(f"综合仪表板已保存: {save_path}")
	
	def _plot_metric(self, ax, data: List[float], title: str, color: str, compact: bool = False):
		"""辅助函数：绘制单个指标"""
		steps = np.arange(len(data))
		ax.plot(steps, data, color=color, linewidth=1.5 if not compact else 1.0)
		
		# 添加移动平均
		if len(data) > 20:
			window = min(20, len(data) // 10)
			ma = np.convolve(data, np.ones(window) / window, mode='valid')
			ax.plot(steps[window-1:], ma, color=color, linewidth=2.0 if not compact else 1.5, 
				   alpha=0.8, linestyle='--')
		
		ax.set_title(title, fontsize=10 if compact else 12)
		ax.grid(True, alpha=0.3)
		
		if compact:
			ax.tick_params(labelsize=8)
	
	def _plot_component_pie(self, ax, weights: Dict[str, float]):
		"""辅助函数：绘制组件权重饼图"""
		sizes = list(weights.values())
		labels = list(weights.keys())
		colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
		
		ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
			  startangle=90)
		ax.set_title('Component Weights', fontsize=10)
	
	def _plot_statistics_table(self, ax, stats: Dict[str, Any]):
		"""辅助函数：绘制统计表格"""
		ax.axis('tight')
		ax.axis('off')
		
		# 准备表格数据
		headers = ['Metric', 'Mean', 'Std', 'Min', 'Max', 'P95']
		rows = []
		
		for key, values in stats.items():
			if isinstance(values, dict):
				row = [
					key,
					f"{values.get('mean', 0):.4f}",
					f"{values.get('std', 0):.4f}",
					f"{values.get('min', 0):.4f}",
					f"{values.get('max', 0):.4f}",
					f"{values.get('p95', 0):.4f}"
				]
				rows.append(row)
		
		if rows:
			table = ax.table(cellText=rows, colLabels=headers, 
						   cellLoc='center', loc='center')
			table.auto_set_font_size(False)
			table.set_fontsize(9)
			table.scale(1.2, 1.5)