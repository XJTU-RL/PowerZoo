# -*- coding: utf-8 -*-
"""
系统参数记录器 - PowerZoo LLM环境系统监控
专用于HAPPO训练过程中的电力系统关键参数记录与监控

功能特点:
- 实时记录电力系统核心参数
- 奖励函数详细分解记录
- 控制设备状态全面监控
- 系统性能指标跟踪
- 高效存储与访问机制
"""

import json
import numpy as np
import os
import pandas as pd
import time
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import h5py
import logging
from functools import wraps
from dataclasses import dataclass, asdict
import threading
from queue import Queue, Empty
import yaml

# 使用统一日志系统
from envs.smartgrid.logging.base_logger import get_logger

logger = get_logger(__name__)


@dataclass
class SystemState:
	"""系统状态数据结构"""
	timestamp: float
	episode: int
	step: int
	
	# 电力系统核心参数
	bus_voltages: Dict[str, np.ndarray]  # 节点电压幅值
	voltage_angles: Dict[str, np.ndarray]  # 节点电压相角
	active_powers: Dict[str, float]  # 有功功率流
	reactive_powers: Dict[str, float]  # 无功功率流
	power_losses: Dict[str, float]  # 功率损耗
	voltage_violations: Dict[str, int]  # 电压违规统计
	load_distribution: Dict[str, float]  # 负荷分布
	
	# 控制设备状态
	capacitor_states: Dict[str, int]  # 电容器开关状态
	regulator_taps: Dict[str, int]  # 调压器抽头位置
	battery_powers: Dict[str, float]  # 电池充放电功率
	battery_socs: Dict[str, float]  # 电池SOC状态
	pv_outputs: Dict[str, float]  # PV输出功率
	pv_control_params: Dict[str, Dict]  # PV控制参数
	
	# 奖励函数组成
	total_reward: float
	voltage_reward: float
	control_reward: float
	power_loss_penalty: float
	voltage_violation_penalty: float
	control_cost: float
	pv_utilization_reward: float
	
	# 系统性能指标
	dss_convergence: bool
	computation_time: float
	system_stability_index: float
	power_quality_index: float


class PerformanceBuffer:
	"""高性能循环缓冲区，用于实时数据存储"""
	
	def __init__(self, max_size: int = 10000):
		self.max_size = max_size
		self.buffer = deque(maxlen=max_size)
		self.lock = threading.Lock()
	
	def append(self, item: SystemState):
		"""线程安全的添加操作"""
		with self.lock:
			self.buffer.append(item)
	
	def get_recent(self, n: int = 100) -> List[SystemState]:
		"""获取最近的n个状态"""
		with self.lock:
			return list(self.buffer)[-n:]
	
	def clear(self):
		"""清空缓冲区"""
		with self.lock:
			self.buffer.clear()


class SystemLogger:
	"""
	电力系统参数记录器
	
	核心功能:
	1. 实时记录电力系统状态
	2. 奖励函数分解记录
	3. 设备控制状态监控
	4. 性能指标统计
	5. 高效数据存储与访问
	"""
	
	def __init__(self, 
			   log_dir: str = "./logs/system_params",
			   buffer_size: int = 10000,
			   save_interval: int = 100,
			   enable_realtime_log: bool = True,
			   compression_level: int = 6):
		"""
		初始化系统记录器
		
		Args:
			log_dir: 日志存储目录
			buffer_size: 内存缓冲区大小
			save_interval: 数据保存间隔(steps)
			enable_realtime_log: 是否启用实时日志
			compression_level: HDF5压缩级别(0-9)
		"""
		self.log_dir = log_dir
		self.buffer_size = buffer_size
		self.save_interval = save_interval
		self.enable_realtime_log = enable_realtime_log
		self.compression_level = compression_level
		
		# 创建日志目录
		os.makedirs(log_dir, exist_ok=True)
		
		# 初始化缓冲区和统计
		self.buffer = PerformanceBuffer(buffer_size)
		self.step_counter = 0
		self.episode_counter = 0
		self.total_logged_steps = 0
		
		# 性能统计
		self.performance_stats = defaultdict(list)
		self.reward_history = defaultdict(list)
		self.device_usage_stats = defaultdict(int)
		
		# 文件句柄
		self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.hdf5_file = None
		self.metadata_file = os.path.join(log_dir, f"metadata_{self.current_session}.yaml")
		
		# 异步写入队列
		self.write_queue = Queue(maxsize=1000)
		self.writer_thread = None
		self.stop_writing = threading.Event()
		
		# 初始化记录器
		self._initialize_logger()
		logger.info(f"SystemLogger initialized - Session: {self.current_session}")
	
	def _initialize_logger(self):
		"""初始化记录器组件"""
		# 创建HDF5文件
		hdf5_path = os.path.join(self.log_dir, f"system_data_{self.current_session}.h5")
		self.hdf5_file = h5py.File(hdf5_path, 'w')
		
		# 创建数据组
		self._create_hdf5_structure()
		
		# 启动异步写入线程
		if self.enable_realtime_log:
			self.writer_thread = threading.Thread(target=self._async_writer, daemon=True)
			self.writer_thread.start()
		
		# 保存元数据
		self._save_metadata()
	
	def _create_hdf5_structure(self):
		"""创建HDF5文件结构"""
		# 时间序列数据组
		ts_group = self.hdf5_file.create_group("timeseries")
		
		# 系统状态数据集
		ts_group.create_group("voltages")
		ts_group.create_group("powers")
		ts_group.create_group("devices")
		ts_group.create_group("rewards")
		ts_group.create_group("performance")
		
		# 统计数据组
		stats_group = self.hdf5_file.create_group("statistics")
		stats_group.create_group("episode_summary")
		stats_group.create_group("device_usage")
		stats_group.create_group("reward_analysis")
		
		logger.debug("HDF5 file structure created")
	
	def _save_metadata(self):
		"""保存会话元数据"""
		metadata = {
			'session_id': self.current_session,
			'created_at': datetime.now().isoformat(),
			'buffer_size': self.buffer_size,
			'save_interval': self.save_interval,
			'compression_level': self.compression_level,
			'log_structure': {
				'timeseries': ['voltages', 'powers', 'devices', 'rewards', 'performance'],
				'statistics': ['episode_summary', 'device_usage', 'reward_analysis']
			}
		}
		
		with open(self.metadata_file, 'w') as f:
			yaml.dump(metadata, f, default_flow_style=False)
	
	def log_system_state(self, 
						env,
						actions: np.ndarray,
						reward: float,
						info: Dict,
						computation_time: float = 0.0):
		"""
		记录系统状态
		
		Args:
			env: 环境实例
			actions: 执行的动作
			reward: 获得的奖励
			info: 额外信息字典
			computation_time: 计算耗时
		"""
		start_time = time.time()
		
		try:
			# 提取系统状态
			state = self._extract_system_state(env, actions, reward, info, computation_time)
			
			# 添加到缓冲区
			self.buffer.append(state)
			
			# 更新计数器
			self.step_counter += 1
			self.total_logged_steps += 1
			
			# 实时日志记录
			if self.enable_realtime_log:
				self._log_realtime_info(state)
			
			# 定期保存数据
			if self.step_counter % self.save_interval == 0:
				self._flush_buffer_to_file()
			
			# 记录性能
			log_time = time.time() - start_time
			self.performance_stats['log_time'].append(log_time)
			
			if log_time > 0.01:  # 记录耗时超过10ms的操作
				logger.debug(f"System logging took {log_time:.4f}s")
				
		except Exception as e:
			logger.error(f"Failed to log system state: {e}")
	
	def _extract_system_state(self, 
							env, 
							actions: np.ndarray,
							reward: float, 
							info: Dict,
							computation_time: float) -> SystemState:
		"""从环境中提取系统状态"""
		
		# 获取基本信息
		current_step = getattr(env, 't', 0)
		episode = getattr(env, '_episode_count', 0)
		
		# 提取电压信息
		bus_voltages = {}
		voltage_angles = {}
		if hasattr(env, 'obs') and 'bus_voltages' in env.obs:
			bus_voltages = env.obs['bus_voltages'].copy()
			voltage_angles = env.obs.get('voltage_angles', {})
		
		# 提取功率信息
		active_powers = info.get('active_powers', {})
		reactive_powers = info.get('reactive_powers', {})
		power_losses = {
			'total_loss': info.get('power_loss_ratio', 0.0),
			'line_losses': info.get('line_losses', {}),
			'transformer_losses': info.get('transformer_losses', {})
		}
		
		# 电压违规统计
		voltage_violations = self._calculate_voltage_violations(bus_voltages)
		
		# 负荷分布
		load_distribution = info.get('load_distribution', {})
		
		# 设备状态提取
		device_states = self._extract_device_states(env, actions)
		
		# 奖励分解
		reward_components = self._extract_reward_components(reward, info)
		
		# 系统性能指标
		performance_metrics = self._calculate_performance_metrics(env, info, computation_time)
		
		return SystemState(
			timestamp=time.time(),
			episode=episode,
			step=current_step,
			
			# 电力系统参数
			bus_voltages=bus_voltages,
			voltage_angles=voltage_angles,
			active_powers=active_powers,
			reactive_powers=reactive_powers,
			power_losses=power_losses,
			voltage_violations=voltage_violations,
			load_distribution=load_distribution,
			
			# 设备状态
			capacitor_states=device_states['capacitors'],
			regulator_taps=device_states['regulators'],
			battery_powers=device_states['battery_powers'],
			battery_socs=device_states['battery_socs'],
			pv_outputs=device_states['pv_outputs'],
			pv_control_params=device_states['pv_control_params'],
			
			# 奖励组成
			total_reward=reward_components['total'],
			voltage_reward=reward_components['voltage'],
			control_reward=reward_components['control'],
			power_loss_penalty=reward_components['power_loss_penalty'],
			voltage_violation_penalty=reward_components['voltage_violation_penalty'],
			control_cost=reward_components['control_cost'],
			pv_utilization_reward=reward_components['pv_utilization'],
			
			# 性能指标
			dss_convergence=performance_metrics['convergence'],
			computation_time=performance_metrics['computation_time'],
			system_stability_index=performance_metrics['stability_index'],
			power_quality_index=performance_metrics['power_quality_index']
		)
	
	def _calculate_voltage_violations(self, bus_voltages: Dict) -> Dict[str, int]:
		"""计算电压违规统计"""
		violations = {
			'under_voltage': 0,
			'over_voltage': 0,
			'total_violations': 0,
			'violation_buses': []
		}
		
		voltage_limits = {'min': 0.95, 'max': 1.05}  # 标准电压限制
		
		for bus_name, voltages in bus_voltages.items():
			if isinstance(voltages, (list, np.ndarray)):
				for i, v in enumerate(voltages):
					if v < voltage_limits['min']:
						violations['under_voltage'] += 1
						violations['violation_buses'].append(f"{bus_name}_{i}")
					elif v > voltage_limits['max']:
						violations['over_voltage'] += 1
						violations['violation_buses'].append(f"{bus_name}_{i}")
			else:
				if voltages < voltage_limits['min']:
					violations['under_voltage'] += 1
					violations['violation_buses'].append(bus_name)
				elif voltages > voltage_limits['max']:
					violations['over_voltage'] += 1
					violations['violation_buses'].append(bus_name)
		
		violations['total_violations'] = violations['under_voltage'] + violations['over_voltage']
		return violations
	
	def _extract_device_states(self, env, actions: np.ndarray) -> Dict:
		"""提取控制设备状态"""
		device_states = {
			'capacitors': {},
			'regulators': {},
			'battery_powers': {},
			'battery_socs': {},
			'pv_outputs': {},
			'pv_control_params': {}
		}
		
		try:
			# 电容器状态
			if hasattr(env, 'cap_names'):
				for i, cap_name in enumerate(env.cap_names):
					if i < len(actions):
						device_states['capacitors'][cap_name] = int(actions[i])
			
			# 调压器状态
			if hasattr(env, 'reg_names'):
				reg_start_idx = getattr(env, 'cap_num', 0)
				for i, reg_name in enumerate(env.reg_names):
					action_idx = reg_start_idx + i
					if action_idx < len(actions):
						device_states['regulators'][reg_name] = int(actions[action_idx])
			
			# 电池状态
			if hasattr(env, 'bat_names'):
				bat_start_idx = getattr(env, 'cap_num', 0) + getattr(env, 'reg_num', 0)
				for i, bat_name in enumerate(env.bat_names):
					action_idx = bat_start_idx + i
					if action_idx < len(actions):
						device_states['battery_powers'][bat_name] = float(actions[action_idx])
				
				# 获取SOC状态（如果可用）
				if hasattr(env, 'circuit') and hasattr(env.circuit, 'batteries'):
					for bat_name in env.bat_names:
						if bat_name in env.circuit.batteries:
							soc = getattr(env.circuit.batteries[bat_name], 'soc', 0.5)
							device_states['battery_socs'][bat_name] = float(soc)
			
			# PV系统状态
			if hasattr(env, 'pv_names') and getattr(env, 'pv_control_enabled', False):
				pv_start_idx = (getattr(env, 'cap_num', 0) + 
							   getattr(env, 'reg_num', 0) + 
							   getattr(env, 'bat_num', 0))
				
				for i, pv_name in enumerate(env.pv_names):
					action_idx = pv_start_idx + i
					if action_idx < len(actions):
						device_states['pv_outputs'][pv_name] = float(actions[action_idx])
				
				# 获取PV控制参数（如果可用）
				if hasattr(env, 'circuit') and hasattr(env.circuit, 'pvs'):
					for pv_name in env.pv_names:
						if pv_name in env.circuit.pvs:
							pv_system = env.circuit.pvs[pv_name]
							device_states['pv_control_params'][pv_name] = {
								'irradiance': getattr(pv_system, 'irradiance', 1000),
								'temperature': getattr(pv_system, 'temperature', 25),
								'power_factor': getattr(pv_system, 'power_factor', 1.0)
							}
		
		except Exception as e:
			logger.debug(f"Error extracting device states: {e}")
		
		return device_states
	
	def _extract_reward_components(self, total_reward: float, info: Dict) -> Dict:
		"""提取奖励函数组成"""
		return {
			'total': total_reward,
			'voltage': info.get('vol_reward', 0.0),
			'control': info.get('ctrl_reward', 0.0),
			'power_loss_penalty': info.get('power_loss_penalty', 0.0),
			'voltage_violation_penalty': info.get('voltage_violation_penalty', 0.0),
			'control_cost': info.get('control_cost', 0.0),
			'pv_utilization': info.get('pv_utilization_reward', 0.0)
		}
	
	def _calculate_performance_metrics(self, env, info: Dict, computation_time: float) -> Dict:
		"""计算系统性能指标"""
		metrics = {
			'convergence': info.get('dss_convergence', True),
			'computation_time': computation_time,
			'stability_index': 1.0,  # 默认值，需要根据实际系统计算
			'power_quality_index': 1.0  # 默认值，需要根据实际系统计算
		}
		
		# 计算系统稳定性指标（基于电压偏差）
		if hasattr(env, 'obs') and 'bus_voltages' in env.obs:
			voltage_deviations = []
			for voltages in env.obs['bus_voltages'].values():
				if isinstance(voltages, (list, np.ndarray)):
					for v in voltages:
						voltage_deviations.append(abs(v - 1.0))  # 偏离标称值
				else:
					voltage_deviations.append(abs(voltages - 1.0))
			
			if voltage_deviations:
				avg_deviation = np.mean(voltage_deviations)
				metrics['stability_index'] = max(0.0, 1.0 - avg_deviation * 10)  # 归一化
				
		# 计算电能质量指标（基于电压违规和损耗）
		power_loss_ratio = info.get('power_loss_ratio', 0.0)
		voltage_violations_count = len(info.get('voltage_violations', []))
		
		quality_penalty = power_loss_ratio * 0.5 + voltage_violations_count * 0.1
		metrics['power_quality_index'] = max(0.0, 1.0 - quality_penalty)
		
		return metrics
	
	def _log_realtime_info(self, state: SystemState):
		"""实时日志信息记录"""
		# 电压违规告警
		if state.voltage_violations['total_violations'] > 0:
			logger.warning(f"Episode {state.episode} Step {state.step}: "
						  f"{state.voltage_violations['total_violations']} voltage violations detected")
		
		# DSS收敛性检查
		if not state.dss_convergence:
			logger.error(f"Episode {state.episode} Step {state.step}: DSS failed to converge")
		
		# 性能警告
		if state.computation_time > 0.1:
			logger.warning(f"Episode {state.episode} Step {state.step}: "
						  f"High computation time: {state.computation_time:.4f}s")
	
	def _flush_buffer_to_file(self):
		"""将缓冲区数据刷写到文件"""
		if not self.buffer.buffer:
			return
		
		try:
			# 获取缓冲区数据
			states = list(self.buffer.buffer)
			
			# 异步写入队列
			if self.enable_realtime_log and not self.write_queue.full():
				self.write_queue.put(states)
			else:
				# 同步写入
				self._write_states_to_hdf5(states)
			
			# 更新统计
			self._update_statistics(states)
			
			logger.debug(f"Flushed {len(states)} states to storage")
			
		except Exception as e:
			logger.error(f"Failed to flush buffer: {e}")
	
	def _write_states_to_hdf5(self, states: List[SystemState]):
		"""将状态数据写入HDF5文件"""
		if not states or not self.hdf5_file:
			return
		
		try:
			# 准备数据数组
			timestamps = [s.timestamp for s in states]
			episodes = [s.episode for s in states]
			steps = [s.step for s in states]
			
			# 写入基础时间序列
			ts_group = self.hdf5_file["timeseries"]
			
			# 创建或扩展数据集
			self._write_dataset(ts_group, "timestamps", timestamps, compression='gzip')
			self._write_dataset(ts_group, "episodes", episodes, compression='gzip')
			self._write_dataset(ts_group, "steps", steps, compression='gzip')
			
			# 写入奖励数据
			reward_group = ts_group["rewards"]
			reward_data = {
				'total_rewards': [s.total_reward for s in states],
				'voltage_rewards': [s.voltage_reward for s in states],
				'control_rewards': [s.control_reward for s in states],
				'power_loss_penalties': [s.power_loss_penalty for s in states]
			}
			
			for name, data in reward_data.items():
				self._write_dataset(reward_group, name, data, compression='gzip')
			
			# 写入性能数据
			perf_group = ts_group["performance"]
			perf_data = {
				'computation_times': [s.computation_time for s in states],
				'dss_convergence': [s.dss_convergence for s in states],
				'stability_indices': [s.system_stability_index for s in states],
				'power_quality_indices': [s.power_quality_index for s in states]
			}
			
			for name, data in perf_data.items():
				self._write_dataset(perf_group, name, data, compression='gzip')
			
			# 确保数据写入磁盘
			self.hdf5_file.flush()
			
		except Exception as e:
			logger.error(f"Failed to write to HDF5: {e}")
	
	def _write_dataset(self, group, name: str, data: List, compression: str = 'gzip'):
		"""写入或扩展HDF5数据集"""
		if name in group:
			# 扩展现有数据集
			dataset = group[name]
			old_size = dataset.shape[0]
			new_size = old_size + len(data)
			dataset.resize((new_size,))
			dataset[old_size:new_size] = data
		else:
			# 创建新数据集
			group.create_dataset(name, data=data, 
							   maxshape=(None,), 
							   compression=compression,
							   compression_opts=self.compression_level)
	
	def _async_writer(self):
		"""异步写入线程"""
		while not self.stop_writing.is_set():
			try:
				states = self.write_queue.get(timeout=1.0)
				self._write_states_to_hdf5(states)
				self.write_queue.task_done()
			except Empty:
				continue
			except Exception as e:
				logger.error(f"Async writer error: {e}")
	
	def _update_statistics(self, states: List[SystemState]):
		"""更新统计信息"""
		for state in states:
			# 奖励历史
			self.reward_history['total'].append(state.total_reward)
			self.reward_history['voltage'].append(state.voltage_reward)
			self.reward_history['control'].append(state.control_reward)
			
			# 设备使用统计
			for cap_name, cap_state in state.capacitor_states.items():
				self.device_usage_stats[f"cap_{cap_name}"] += cap_state
			
			for bat_name, bat_power in state.battery_powers.items():
				if abs(bat_power) > 0.01:  # 电池有输出
					self.device_usage_stats[f"bat_{bat_name}"] += 1
	
	def get_episode_summary(self, episode: int) -> Dict:
		"""获取指定回合的汇总统计"""
		recent_states = self.buffer.get_recent(1000)
		episode_states = [s for s in recent_states if s.episode == episode]
		
		if not episode_states:
			return {}
		
		return {
			'episode': episode,
			'total_steps': len(episode_states),
			'avg_reward': np.mean([s.total_reward for s in episode_states]),
			'total_voltage_violations': sum(s.voltage_violations['total_violations'] for s in episode_states),
			'avg_computation_time': np.mean([s.computation_time for s in episode_states]),
			'convergence_rate': np.mean([s.dss_convergence for s in episode_states]),
			'avg_stability_index': np.mean([s.system_stability_index for s in episode_states]),
			'avg_power_quality': np.mean([s.power_quality_index for s in episode_states])
		}
	
	def get_realtime_metrics(self) -> Dict:
		"""获取实时系统指标"""
		recent_states = self.buffer.get_recent(100)
		
		if not recent_states:
			return {}
		
		return {
			'current_episode': recent_states[-1].episode if recent_states else 0,
			'current_step': recent_states[-1].step if recent_states else 0,
			'recent_avg_reward': np.mean([s.total_reward for s in recent_states]),
			'recent_voltage_violations': sum(s.voltage_violations['total_violations'] for s in recent_states),
			'recent_avg_computation_time': np.mean([s.computation_time for s in recent_states]),
			'total_logged_steps': self.total_logged_steps,
			'buffer_utilization': len(self.buffer.buffer) / self.buffer_size,
			'logging_performance': {
				'avg_log_time': np.mean(self.performance_stats['log_time'][-100:]) if self.performance_stats['log_time'] else 0,
				'max_log_time': max(self.performance_stats['log_time'][-100:]) if self.performance_stats['log_time'] else 0
			}
		}
	
	def export_data(self, output_path: str, format: str = 'csv'):
		"""导出数据到指定格式"""
		try:
			recent_states = self.buffer.get_recent()
			
			if format.lower() == 'csv':
				self._export_to_csv(recent_states, output_path)
			elif format.lower() == 'json':
				self._export_to_json(recent_states, output_path)
			else:
				raise ValueError(f"Unsupported export format: {format}")
			
			logger.info(f"Data exported to {output_path}")
			
		except Exception as e:
			logger.error(f"Failed to export data: {e}")
	
	def _export_to_csv(self, states: List[SystemState], output_path: str):
		"""导出为CSV格式"""
		# 准备数据
		data = []
		for state in states:
			row = {
				'timestamp': state.timestamp,
				'episode': state.episode,
				'step': state.step,
				'total_reward': state.total_reward,
				'voltage_reward': state.voltage_reward,
				'control_reward': state.control_reward,
				'voltage_violations': state.voltage_violations['total_violations'],
				'computation_time': state.computation_time,
				'dss_convergence': state.dss_convergence,
				'stability_index': state.system_stability_index,
				'power_quality_index': state.power_quality_index
			}
			data.append(row)
		
		# 保存为CSV
		df = pd.DataFrame(data)
		df.to_csv(output_path, index=False)
	
	def _export_to_json(self, states: List[SystemState], output_path: str):
		"""导出为JSON格式"""
		data = [asdict(state) for state in states]
		
		# 处理numpy数组
		def convert_numpy(obj):
			if isinstance(obj, np.ndarray):
				return obj.tolist()
			elif isinstance(obj, np.integer):
				return int(obj)
			elif isinstance(obj, np.floating):
				return float(obj)
			return obj
		
		def recursive_convert(data):
			if isinstance(data, dict):
				return {k: recursive_convert(v) for k, v in data.items()}
			elif isinstance(data, list):
				return [recursive_convert(item) for item in data]
			else:
				return convert_numpy(data)
		
		converted_data = recursive_convert(data)
		
		with open(output_path, 'w') as f:
			json.dump(converted_data, f, indent=2)
	
	def close(self):
		"""关闭记录器并清理资源"""
		try:
			# 停止异步写入
			if self.writer_thread and self.writer_thread.is_alive():
				self.stop_writing.set()
				self.writer_thread.join(timeout=5.0)
			
			# 刷写剩余数据
			self._flush_buffer_to_file()
			
			# 关闭HDF5文件
			if self.hdf5_file:
				self.hdf5_file.close()
			
			# 保存最终统计
			self._save_final_statistics()
			
			logger.info(f"SystemLogger closed - Total steps logged: {self.total_logged_steps}")
			
		except Exception as e:
			logger.error(f"Error closing SystemLogger: {e}")
	
	def _save_final_statistics(self):
		"""保存最终统计信息"""
		stats_path = os.path.join(self.log_dir, f"final_stats_{self.current_session}.json")
		
		final_stats = {
			'session_id': self.current_session,
			'total_logged_steps': self.total_logged_steps,
			'total_episodes': self.episode_counter,
			'device_usage': dict(self.device_usage_stats),
			'performance_summary': {
				'avg_log_time': np.mean(self.performance_stats['log_time']) if self.performance_stats['log_time'] else 0,
				'max_log_time': max(self.performance_stats['log_time']) if self.performance_stats['log_time'] else 0,
				'total_log_operations': len(self.performance_stats['log_time'])
			},
			'reward_summary': {
				'avg_total_reward': np.mean(self.reward_history['total']) if self.reward_history['total'] else 0,
				'avg_voltage_reward': np.mean(self.reward_history['voltage']) if self.reward_history['voltage'] else 0,
				'avg_control_reward': np.mean(self.reward_history['control']) if self.reward_history['control'] else 0
			}
		}
		
		with open(stats_path, 'w') as f:
			json.dump(final_stats, f, indent=2)


# 全局记录器实例
_global_system_logger = None

def get_system_logger(**kwargs) -> SystemLogger:
	"""获取全局系统记录器实例"""
	global _global_system_logger
	if _global_system_logger is None:
		_global_system_logger = SystemLogger(**kwargs)
	return _global_system_logger

def close_system_logger():
	"""关闭全局系统记录器"""
	global _global_system_logger
	if _global_system_logger is not None:
		_global_system_logger.close()
		_global_system_logger = None