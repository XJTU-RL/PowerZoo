"""
P95权重标定模块

用于自动标定奖励函数权重，使各组件在同一量级
通过统计分析确定合适的权重参数
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)


class RewardCalibrator:
	"""
	奖励权重标定器
	
	通过收集环境运行数据，自动计算各奖励组件的P95分位数
	并生成建议权重，使各组件贡献均衡
	"""
	
	def __init__(
		self,
		env,
		reward_obj,
		episodes: int = 20,
		steps_per_episode: Optional[int] = None,
		use_random_policy: bool = True,
		target_scale: float = 1.0
	):
		"""
		初始化标定器
		
		Args:
			env: 环境实例
			reward_obj: 奖励函数对象
			episodes: 标定使用的episode数量
			steps_per_episode: 每个episode的步数（None表示使用环境默认）
			use_random_policy: 是否使用随机策略（True）或现有策略
			target_scale: 目标奖励尺度
		"""
		self.env = env
		self.reward_obj = reward_obj
		self.episodes = episodes
		self.steps_per_episode = steps_per_episode or getattr(env, 'horizon', 96)
		self.use_random_policy = use_random_policy
		self.target_scale = target_scale
		
		# 统计数据容器
		self.stats = {
			'r_power': [],
			'r_ctrl': [],
			'r_pv': [],
			'cost_voltage': [],
			'total_reward': []
		}
		
		# 动作历史（用于计算平滑项）
		self.prev_actions = {}
		
		logger.info(
			f"奖励标定器初始化: {episodes} episodes, "
			f"{self.steps_per_episode} steps/episode, "
			f"随机策略={use_random_policy}"
		)
	
	def collect_statistics(self) -> Dict[str, List[float]]:
		"""
		收集奖励组件统计数据
		
		Returns:
			Dict[str, List[float]]: 各组件的数据列表
		"""
		logger.info("开始收集奖励统计数据...")
		
		for ep in range(self.episodes):
			obs = self.env.reset()
			self.prev_actions = {}  # 重置动作历史
			
			for step in range(self.steps_per_episode):
				# 生成动作
				if self.use_random_policy:
					action = self._random_action()
				else:
					action = self._get_policy_action(obs)
				
				# 执行动作
				obs, _, done, info = self.env.step(action)
				
				# 收集奖励组件
				self._collect_reward_components(action)
				
				# 更新动作历史
				self._update_action_history(action)
				
				if done:
					break
			
			# 记录进度
			if (ep + 1) % 5 == 0:
				logger.debug(f"完成 {ep + 1}/{self.episodes} episodes")
		
		logger.info(f"统计数据收集完成: 共{len(self.stats['total_reward'])}个样本")
		return self.stats
	
	def _collect_reward_components(self, action: Any):
		"""收集单步的奖励组件"""
		# 计算各组件（直接调用奖励函数的子方法）
		
		# 功率损失奖励
		r_power = abs(self.reward_obj.powerloss_reward())
		self.stats['r_power'].append(r_power)
		
		# 控制成本（需要计算动作差异）
		capdiff, regdiff, soc_err, dis_err, pv_diff = self._calculate_control_diffs(action)
		r_ctrl = abs(self.reward_obj.control_reward(capdiff, regdiff, soc_err, dis_err, pv_diff))
		self.stats['r_ctrl'].append(r_ctrl)
		
		# PV奖励
		r_pv = abs(self.reward_obj.pv_reward())
		self.stats['r_pv'].append(r_pv)
		
		# 电压成本
		cost_v = self.reward_obj.voltage_cost()
		self.stats['cost_voltage'].append(cost_v)
		
		# 总奖励
		total = r_power + r_ctrl + r_pv
		self.stats['total_reward'].append(total)
	
	def _calculate_control_diffs(self, action: Any) -> Tuple[List[float], ...]:
		"""计算控制动作差异（简化版）"""
		# 这里返回简化的差异值，实际应从环境获取
		capdiff = [0.1] * getattr(self.env, 'cap_num', 1)
		regdiff = [0.05] * getattr(self.env, 'reg_num', 1)
		soc_err = [0.0] * getattr(self.env, 'bat_num', 1)
		dis_err = [0.0] * getattr(self.env, 'bat_num', 1)
		pv_diff = [0.1] * getattr(self.env, 'pv_num', 1) if hasattr(self.env, 'pv_num') else []
		
		return capdiff, regdiff, soc_err, dis_err, pv_diff
	
	def _random_action(self) -> Any:
		"""生成随机动作"""
		if hasattr(self.env, 'action_space'):
			return self.env.action_space.sample()
		else:
			# 备用随机动作生成
			return np.random.uniform(-1, 1, size=self.env.action_dim)
	
	def _get_policy_action(self, obs: Any) -> Any:
		"""从现有策略获取动作（需要实现）"""
		# 这里应该调用训练好的策略
		# 暂时返回随机动作
		return self._random_action()
	
	def _update_action_history(self, action: Any):
		"""更新动作历史"""
		self.prev_actions['last'] = action
	
	def calculate_p95_weights(self) -> Dict[str, float]:
		"""
		计算P95分位数并生成建议权重
		
		Returns:
			Dict[str, float]: 建议的权重字典
		"""
		p95_stats = {}
		weights = {}
		
		# 计算各组件的P95分位数
		for key in ['r_power', 'r_ctrl', 'r_pv', 'cost_voltage']:
			if key in self.stats and self.stats[key]:
				data = np.array(self.stats[key])
				# 过滤掉异常值
				data = data[np.isfinite(data)]
				if len(data) > 0:
					p95 = np.percentile(np.abs(data), 95)
					p95_stats[key] = p95
				else:
					p95_stats[key] = 1.0
			else:
				p95_stats[key] = 1.0
		
		# 计算权重，使各组件在P95处贡献相等
		target_contribution = self.target_scale
		
		# 主奖励权重
		weights['power_w'] = target_contribution / max(p95_stats['r_power'], 1e-6)
		weights['ctrl_w'] = target_contribution / max(p95_stats['r_ctrl'], 1e-6)
		weights['pv_w'] = target_contribution / max(p95_stats['r_pv'], 1e-6)
		
		# 约束权重（不参与主奖励，但记录供参考）
		weights['voltage_cost_scale'] = target_contribution / max(p95_stats['cost_voltage'], 1e-6)
		
		# 细分控制权重（按比例分配）
		ctrl_base = weights['ctrl_w']
		weights['cap_w'] = ctrl_base * 0.2
		weights['reg_w'] = ctrl_base * 0.2
		weights['soc_w'] = ctrl_base * 0.4
		weights['dis_w'] = ctrl_base * 0.2
		
		logger.info("P95统计:")
		for key, value in p95_stats.items():
			logger.info(f"  {key}: {value:.6f}")
		
		logger.info("建议权重:")
		for key, value in weights.items():
			logger.info(f"  {key}: {value:.6f}")
		
		return weights
	
	def generate_report(self) -> Dict[str, Any]:
		"""
		生成标定报告
		
		Returns:
			Dict[str, Any]: 包含统计信息和建议权重的报告
		"""
		weights = self.calculate_p95_weights()
		
		# 计算统计信息
		statistics = {}
		for key, data in self.stats.items():
			if data:
				arr = np.array(data)
				arr = arr[np.isfinite(arr)]  # 过滤无效值
				if len(arr) > 0:
					statistics[key] = {
						'mean': float(np.mean(arr)),
						'std': float(np.std(arr)),
						'min': float(np.min(arr)),
						'max': float(np.max(arr)),
						'p25': float(np.percentile(arr, 25)),
						'p50': float(np.percentile(arr, 50)),
						'p75': float(np.percentile(arr, 75)),
						'p95': float(np.percentile(arr, 95)),
						'p99': float(np.percentile(arr, 99))
					}
		
		report = {
			'calibration_config': {
				'episodes': self.episodes,
				'steps_per_episode': self.steps_per_episode,
				'total_samples': len(self.stats.get('total_reward', [])),
				'use_random_policy': self.use_random_policy,
				'target_scale': self.target_scale
			},
			'statistics': statistics,
			'recommended_weights': weights,
			'weight_ratios': self._calculate_weight_ratios(weights)
		}
		
		return report
	
	def _calculate_weight_ratios(self, weights: Dict[str, float]) -> Dict[str, float]:
		"""计算权重比例关系"""
		base_weight = weights.get('power_w', 1.0)
		ratios = {}
		
		for key, value in weights.items():
			if key.endswith('_w'):
				ratios[f"{key}_ratio"] = value / base_weight if base_weight > 0 else 0.0
		
		return ratios
	
	def save_report(self, filepath: str):
		"""
		保存标定报告到文件
		
		Args:
			filepath: 保存路径
		"""
		report = self.generate_report()
		
		path = Path(filepath)
		path.parent.mkdir(parents=True, exist_ok=True)
		
		with open(path, 'w', encoding='utf-8') as f:
			json.dump(report, f, indent=2, ensure_ascii=False)
		
		logger.info(f"标定报告已保存到: {filepath}")
	
	def apply_weights(self, weights: Optional[Dict[str, float]] = None):
		"""
		应用标定的权重到奖励对象
		
		Args:
			weights: 权重字典（None则使用自动计算的权重）
		"""
		if weights is None:
			weights = self.calculate_p95_weights()
		
		# 应用权重到奖励对象
		for key, value in weights.items():
			if hasattr(self.reward_obj, key):
				setattr(self.reward_obj, key, value)
				logger.debug(f"应用权重: {key} = {value:.6f}")
		
		logger.info("权重已应用到奖励函数")


def quick_calibrate(env, reward_obj, episodes: int = 10) -> Dict[str, float]:
	"""
	快速标定函数
	
	Args:
		env: 环境实例
		reward_obj: 奖励对象
		episodes: 标定episode数
		
	Returns:
		Dict[str, float]: 建议权重
	"""
	calibrator = RewardCalibrator(env, reward_obj, episodes=episodes)
	calibrator.collect_statistics()
	weights = calibrator.calculate_p95_weights()
	
	# 自动应用权重
	calibrator.apply_weights(weights)
	
	return weights


def load_calibration(filepath: str) -> Dict[str, Any]:
	"""
	加载标定报告
	
	Args:
		filepath: 报告文件路径
		
	Returns:
		Dict[str, Any]: 标定报告内容
	"""
	with open(filepath, 'r', encoding='utf-8') as f:
		return json.load(f)