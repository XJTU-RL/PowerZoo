# -*- coding: utf-8 -*-
"""
可视化管理器 - 集成到日志系统中的奖励可视化功能

该模块负责在训练过程中收集数据并定期生成可视化图表，
自动保存到对应的日志目录。

@File      : visualization_manager.py  
@Time      : 2025-08-08
@Author    : Xiaodong Zheng (with Claude Code)
@Email     : zxd_xjtu@stu.xjtu.edu.cn
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import json
import time

from envs.smartgrid.rewards.visualize import RewardVisualizer
from envs.smartgrid.logging.base_logger import get_logger

logger = get_logger(__name__)


class VisualizationManager:
	"""
	可视化管理器
	
	负责在训练过程中收集CMDP相关数据并生成可视化
	"""
	
	def __init__(
		self,
		save_dir: str,
		plot_interval: int = 100,
		buffer_size: int = 10000,
		enable_plotting: bool = True
	):
		"""
		初始化可视化管理器
		
		Args:
			save_dir: 图表保存目录（通常是 log_dir/plots）
			plot_interval: 绘图间隔（步数）
			buffer_size: 数据缓冲区大小
			enable_plotting: 是否启用绘图
		"""
		self.save_dir = Path(save_dir)
		self.save_dir.mkdir(parents=True, exist_ok=True)
		
		self.plot_interval = plot_interval
		self.buffer_size = buffer_size
		self.enable_plotting = enable_plotting
		
		# 初始化可视化器
		self.visualizer = RewardVisualizer(
			save_dir=str(self.save_dir),
			figsize=(12, 8)
		)
		
		# 数据缓冲区
		self.data_buffers = {
			'total_reward': deque(maxlen=buffer_size),
			'reward_main': deque(maxlen=buffer_size),
			'cost_voltage': deque(maxlen=buffer_size),
			'lambda': deque(maxlen=buffer_size),
			'voltage_violation_rate': deque(maxlen=buffer_size),
			'powerloss_reward': deque(maxlen=buffer_size),
			'control_reward': deque(maxlen=buffer_size),
			'pv_reward': deque(maxlen=buffer_size),
			'voltage_compliance_rate': deque(maxlen=buffer_size)
		}
		
		# Lambda演化历史
		self.lambda_history = []
		self.cost_history = []
		
		# 电压违规统计
		self.voltage_violations_by_bus = {}
		
		# 计数器
		self.step_count = 0
		self.episode_count = 0
		self.last_plot_step = 0
		
		# 统计数据
		self.episode_stats = []
		
		logger.info(f"可视化管理器初始化完成: 保存目录={save_dir}, "
				   f"绘图间隔={plot_interval}步, 启用={enable_plotting}")
	
	def update(self, info: Dict[str, Any]) -> None:
		"""
		更新数据缓冲区
		
		Args:
			info: 包含奖励和CMDP信息的字典
		"""
		if not self.enable_plotting:
			return
		
		self.step_count += 1
		
		# 收集基本数据
		if 'reward_main' in info:
			self.data_buffers['reward_main'].append(float(info['reward_main']))
		
		if 'cost_voltage' in info:
			self.data_buffers['cost_voltage'].append(float(info['cost_voltage']))
		
		if 'lambda' in info:
			self.data_buffers['lambda'].append(float(info['lambda']))
		
		if 'voltage_violation_rate' in info:
			violation_rate = float(info['voltage_violation_rate']) * 100  # 转换为百分比
			self.data_buffers['voltage_violation_rate'].append(violation_rate)
		
		if 'voltage_compliance_rate' in info:
			self.data_buffers['voltage_compliance_rate'].append(float(info['voltage_compliance_rate']))
		
		# 收集奖励组件
		if 'powerloss_reward' in info:
			self.data_buffers['powerloss_reward'].append(float(info['powerloss_reward']))
		
		if 'control_reward' in info:
			self.data_buffers['control_reward'].append(float(info['control_reward']))
		
		if 'pv_reward' in info:
			self.data_buffers['pv_reward'].append(float(info['pv_reward']))
		
		# 计算总奖励（如果有Lagrangian惩罚）
		if 'reward_before_lagrangian' in info:
			total_reward = float(info.get('reward_before_lagrangian', 0))
		else:
			total_reward = float(info.get('reward_main', 0))
		self.data_buffers['total_reward'].append(total_reward)
		
		# 收集违规节点信息
		if 'violated_nodes' in info:
			for node in info['violated_nodes']:
				if node not in self.voltage_violations_by_bus:
					self.voltage_violations_by_bus[node] = 0
				self.voltage_violations_by_bus[node] += 1
		
		# 定期生成图表
		if self.step_count - self.last_plot_step >= self.plot_interval:
			self.generate_plots()
			self.last_plot_step = self.step_count
	
	def update_episode_end(
		self,
		episode_reward: float,
		episode_cost: float,
		lambda_value: float
	) -> None:
		"""
		Episode结束时的更新
		
		Args:
			episode_reward: Episode总奖励
			episode_cost: Episode平均成本
			lambda_value: 当前Lambda值
		"""
		self.episode_count += 1
		
		# 更新Lambda历史
		self.lambda_history.append(lambda_value)
		self.cost_history.append(episode_cost)
		
		# 记录episode统计
		self.episode_stats.append({
			'episode': self.episode_count,
			'reward': episode_reward,
			'cost': episode_cost,
			'lambda': lambda_value,
			'step': self.step_count
		})
		
		# 每10个episode生成一次综合报告
		if self.episode_count % 10 == 0:
			self.generate_episode_report()
	
	def generate_plots(self) -> None:
		"""生成所有可视化图表"""
		if not self.enable_plotting:
			return
		
		try:
			# 1. 训练曲线
			if len(self.data_buffers['total_reward']) > 0:
				training_data = {
					'total_reward_raw': list(self.data_buffers['total_reward']),
					'cost_voltage': list(self.data_buffers['cost_voltage']),
					'lambda': list(self.data_buffers['lambda']),
					'voltage_violation_rate': list(self.data_buffers['voltage_violation_rate'])
				}
				
				self.visualizer.plot_training_curves(
					data=training_data,
					title=f"Training Progress (Step {self.step_count})",
					save_name=f"training_curves_step{self.step_count}.png"
				)
			
			# 2. 奖励组件分解
			if len(self.data_buffers['powerloss_reward']) > 0:
				components = {
					'r_power': list(self.data_buffers['powerloss_reward']),
					'r_ctrl': list(self.data_buffers['control_reward']),
					'r_pv': list(self.data_buffers['pv_reward']),
					'cost_voltage': list(self.data_buffers['cost_voltage'])
				}
				
				self.visualizer.plot_reward_components(
					components=components,
					title=f"Reward Components (Step {self.step_count})",
					save_name=f"reward_components_step{self.step_count}.png"
				)
			
			# 3. Lambda演化图
			if len(self.lambda_history) > 0:
				self.visualizer.plot_lambda_evolution(
					lambda_history=self.lambda_history,
					cost_history=self.cost_history,
					target_cost=0.01,  # 可配置
					title=f"Lambda Evolution (Episode {self.episode_count})",
					save_name=f"lambda_evolution_ep{self.episode_count}.png"
				)
			
			# 4. 电压违规热度图
			if self.voltage_violations_by_bus:
				self.visualizer.plot_voltage_heatmap(
					voltage_violations=self.voltage_violations_by_bus,
					title=f"Voltage Violations Heatmap (Step {self.step_count})",
					save_name=f"voltage_heatmap_step{self.step_count}.png"
				)
			
			logger.debug(f"可视化图表已生成: Step {self.step_count}")
			
		except Exception as e:
			logger.error(f"生成可视化图表失败: {e}")
	
	def generate_episode_report(self) -> None:
		"""生成Episode综合报告"""
		if not self.enable_plotting or not self.episode_stats:
			return
		
		try:
			# 创建综合仪表板
			dashboard_data = {
				'total_reward': [stat['reward'] for stat in self.episode_stats],
				'cost_voltage': self.cost_history,
				'lambda': self.lambda_history,
				'violation_rate': list(self.data_buffers['voltage_violation_rate'])[-1000:],  # 最近1000步
				'component_weights': {
					'Power Loss': 1.0,
					'Control': 0.5,
					'PV': 0.8
				},
				'statistics': self._calculate_statistics()
			}
			
			self.visualizer.create_summary_dashboard(
				training_data=dashboard_data,
				save_name=f"summary_dashboard_ep{self.episode_count}.png"
			)
			
			# 保存统计数据到JSON
			self._save_statistics()
			
			logger.info(f"Episode综合报告已生成: Episode {self.episode_count}")
			
		except Exception as e:
			logger.error(f"生成Episode报告失败: {e}")
	
	def _calculate_statistics(self) -> Dict[str, Dict[str, float]]:
		"""计算统计数据"""
		stats = {}
		
		for key, buffer in self.data_buffers.items():
			if len(buffer) > 0:
				data = np.array(buffer)
				stats[key] = {
					'mean': float(np.mean(data)),
					'std': float(np.std(data)),
					'min': float(np.min(data)),
					'max': float(np.max(data)),
					'p95': float(np.percentile(data, 95))
				}
		
		return stats
	
	def _save_statistics(self) -> None:
		"""保存统计数据到JSON文件"""
		stats_file = self.save_dir / f"statistics_ep{self.episode_count}.json"
		
		try:
			stats_data = {
				'episode': self.episode_count,
				'step': self.step_count,
				'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
				'episode_stats': self.episode_stats[-10:],  # 最近10个episode
				'buffer_stats': self._calculate_statistics(),
				'lambda_current': self.lambda_history[-1] if self.lambda_history else 0,
				'cost_current': self.cost_history[-1] if self.cost_history else 0
			}
			
			with open(stats_file, 'w') as f:
				json.dump(stats_data, f, indent=2)
			
			logger.debug(f"统计数据已保存: {stats_file}")
			
		except Exception as e:
			logger.error(f"保存统计数据失败: {e}")
	
	def reset(self) -> None:
		"""重置管理器状态（用于新的训练会话）"""
		# 清空缓冲区
		for buffer in self.data_buffers.values():
			buffer.clear()
		
		# 重置历史
		self.lambda_history.clear()
		self.cost_history.clear()
		self.voltage_violations_by_bus.clear()
		self.episode_stats.clear()
		
		# 重置计数器
		self.step_count = 0
		self.episode_count = 0
		self.last_plot_step = 0
		
		logger.info("可视化管理器已重置")
	
	def finalize(self) -> None:
		"""训练结束时的最终处理"""
		if not self.enable_plotting:
			return
		
		# 生成最终的综合报告
		self.generate_episode_report()
		
		# 生成最终的所有图表
		self.generate_plots()
		
		# 保存完整的训练历史
		history_file = self.save_dir / "training_history.json"
		try:
			history_data = {
				'total_episodes': self.episode_count,
				'total_steps': self.step_count,
				'episode_stats': self.episode_stats,
				'final_lambda': self.lambda_history[-1] if self.lambda_history else 0,
				'final_cost': self.cost_history[-1] if self.cost_history else 0,
				'final_statistics': self._calculate_statistics()
			}
			
			with open(history_file, 'w') as f:
				json.dump(history_data, f, indent=2)
			
			logger.info(f"训练历史已保存: {history_file}")
			
		except Exception as e:
			logger.error(f"保存训练历史失败: {e}")
		
		logger.info("可视化管理器已完成最终处理")