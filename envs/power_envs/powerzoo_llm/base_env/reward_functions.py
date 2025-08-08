"""
PowerZoo环境奖励函数模块（向后兼容封装）

该模块为新的CMDP奖励系统提供向后兼容接口
"""

import logging
from typing import Dict, List, Any, Tuple, Optional

# 导入新的CMDP奖励系统
from envs.power_envs.powerzoo_llm.rewards import PowerZooReward as CMDPReward

logger = logging.getLogger(__name__)


class PowerZooReward(CMDPReward):
	"""
	PowerZoo环境的奖励函数类（向后兼容版本）
	
	该类继承自新的CMDP奖励系统，提供向后兼容接口
	以确保现有代码无需修改即可使用新系统
	"""
	
	def __init__(self, env, info: Dict[str, Any]):
		"""
		初始化奖励函数（向后兼容）
		
		Args:
			env: 环境实例
			info: 配置信息字典
		"""
		# 映射旧参数到新参数
		if 'power_w' in info:
			info['powerloss_weight'] = info.get('power_w', 0.1)
		if 'cap_w' in info or 'reg_w' in info:
			# 将旧的控制权重合并
			cap_w = info.get('cap_w', 0.1)
			reg_w = info.get('reg_w', 0.1)
			soc_w = info.get('soc_w', 0.5)
			dis_w = info.get('dis_w', 0.1)
			info['control_weight'] = cap_w + reg_w + soc_w + dis_w
		if 'pv_w' in info:
			info['pv_weight'] = info.get('pv_w', 0.5)
		
		# 设置电压范围
		if 'voltage_target_range' not in info:
			info['voltage_range'] = info.get('voltage_target_range', (0.95, 1.05))
		
		# 调用父类初始化
		super().__init__(env, info)
		
		# 保存旧的权重（向后兼容）
		self.power_w = info.get('power_w', 0.1)
		self.cap_w = info.get('cap_w', 0.1)
		self.reg_w = info.get('reg_w', 0.1)
		self.soc_w = info.get('soc_w', 0.5)
		self.dis_w = info.get('dis_w', 0.1)
		self.pv_w = info.get('pv_w', 0.5)
		
		# 兼容性标志
		self.constraint_aware = info.get('constraint_aware', True)
		self.use_reward_normalization = info.get('use_reward_normalization', False)
		self.reward_clip_range = info.get('reward_clip_range', (-10, 10))
		self.voltage_penalty_scale = info.get('voltage_penalty_scale', 1.0)
		self.progressive_penalty = info.get('progressive_penalty', True)
		self.reward_scale = info.get('reward_scale', 1.0)
		
		# 组件权重（兼容性）
		self.component_weights = {
			'power_loss': info.get('power_loss_weight', 0.1),
			'voltage': info.get('voltage_weight', 0.4),
			'control': info.get('control_weight', 0.2),
			'power_balance': info.get('power_balance_weight', 0.0),  # 默认禁用
			'pv_optimization': info.get('pv_optimization_weight', 0.1)
		}
		
		# 统计信息（兼容性）
		self.reward_stats = {
			'mean': 0.0,
			'std': 1.0,
			'count': 0,
			'running_mean': 0.0,
			'running_std': 1.0
		}
		
		logger.info("PowerZoo奖励函数初始化完成（向后兼容模式）")
	
	def powerloss_reward(self) -> float:
		"""功率损失奖励（向后兼容）"""
		# 调用父类方法并应用旧权重
		base_reward = super().powerloss_reward()
		return base_reward * self.power_w / self.weights.get('powerloss', 1.0)
	
	def ctrl_reward(self, capdiff: List[float], regdiff: List[float], 
				   soc_err: List[float], discharge_err: List[float],
				   pv_diff: Optional[List[float]] = None) -> float:
		"""控制动作成本奖励（向后兼容）"""
		# 使用旧的权重计算
		pv_diff = pv_diff or []
		
		cap_cost = self.cap_w * sum(capdiff) if capdiff else 0.0
		reg_cost = self.reg_w * sum(regdiff) if regdiff else 0.0
		
		soc_cost = 0.0
		if hasattr(self.env, 't') and hasattr(self.env, 'horizon'):
			if self.env.t == self.env.horizon and soc_err:
				soc_cost = self.soc_w * sum(soc_err)
		
		dis_cost = self.dis_w * sum(discharge_err) if discharge_err else 0.0
		pv_cost = self.pv_w * sum(pv_diff) if pv_diff else 0.0
		
		total_cost = cap_cost + reg_cost + soc_cost + dis_cost + pv_cost
		
		return -total_cost
	
	def voltage_reward(self, record_node: bool = False) -> Tuple[float, List[str]]:
		"""
		电压违规奖励（向后兼容）
		
		将新的电压成本转换为旧的奖励形式
		"""
		violated_nodes = []
		voltage_cost = self.voltage_cost()
		
		# 转换成本为奖励（负值）
		reward = -voltage_cost * self.voltage_penalty_scale * 10.0
		
		# 记录违规节点
		if record_node:
			bus_voltages = self.env.obs.get('bus_voltages', {})
			v_min, v_max = self.voltage_range
			
			for bus_name, voltages in bus_voltages.items():
				if not voltages:
					continue
				for v in voltages:
					if not (v_min <= v <= v_max):
						if bus_name not in violated_nodes:
							violated_nodes.append(bus_name)
		
		return reward, violated_nodes
	
	def enhanced_voltage_reward(self, bus_voltages: Dict) -> float:
		"""增强电压约束奖励（向后兼容）"""
		# 使用新的电压成本计算
		voltage_cost = self.voltage_cost()
		
		# 如果启用渐进式惩罚，增加一些正奖励
		if self.progressive_penalty and voltage_cost < 0.01:
			bonus = 0.1 * (1.0 - voltage_cost / 0.01)
			return -voltage_cost * self.voltage_penalty_scale + bonus
		else:
			return -voltage_cost * self.voltage_penalty_scale
	
	def power_balance_reward(self) -> float:
		"""功率平衡奖励（向后兼容，默认禁用）"""
		if self.component_weights['power_balance'] == 0:
			return 0.0
		
		power_loss = self.env.obs.get('power_loss', 0)
		
		if hasattr(power_loss, '__len__'):
			power_loss = float(power_loss[0]) if len(power_loss) > 0 else 0.0
		else:
			power_loss = float(power_loss)
		
		# 简单的平衡奖励
		if power_loss < 0.02:
			return 1.0
		elif power_loss < 0.05:
			return 0.5
		elif power_loss < 0.10:
			return 0.0
		else:
			return -power_loss * 5.0
	
	def pv_optimization_reward(self) -> float:
		"""光伏优化奖励（向后兼容）"""
		# 调用父类的pv_reward方法
		base_reward = self.pv_reward()
		# 应用旧的权重比例
		return base_reward * self.pv_w / self.weights.get('pv', 1.0)
	
	def _normalize_reward(self, reward: float) -> float:
		"""归一化奖励值（向后兼容）"""
		if not self.use_reward_normalization:
			return reward
		
		# 更新运行统计
		self.reward_stats['count'] += 1
		delta = reward - self.reward_stats['running_mean']
		self.reward_stats['running_mean'] += delta / self.reward_stats['count']
		delta2 = reward - self.reward_stats['running_mean']
		
		import numpy as np
		self.reward_stats['running_std'] = np.sqrt(
			(self.reward_stats['running_std'] ** 2 * (self.reward_stats['count'] - 1) + 
			 delta * delta2) / self.reward_stats['count']
		)
		
		# 归一化
		if self.reward_stats['running_std'] > 0.01:
			normalized = (reward - self.reward_stats['running_mean']) / self.reward_stats['running_std']
		else:
			normalized = reward
		
		# 裁剪到合理范围
		clipped = np.clip(normalized, self.reward_clip_range[0], self.reward_clip_range[1])
		
		return clipped
	
	def _calculate_voltage_compliance(self) -> float:
		"""计算电压合格率（继承自父类）"""
		return super()._calculate_voltage_compliance()
	
	def composite_reward(self, cd: List[float], rd: List[float], 
					   soc: List[float], dis: List[float], 
					   pv_diff: Optional[List[float]] = None,
					   full: bool = True, record_node: bool = False) -> Tuple[float, Dict[str, Any]]:
		"""
		综合奖励计算 - 主入口函数（向后兼容）
		
		根据constraint_aware标志选择不同的奖励模式
		"""
		info = {}
		
		if self.constraint_aware:
			# CMDP模式：使用新系统
			main_reward, base_info = super().composite_reward(
				cd, rd, soc, dis, pv_diff, full, record_node
			)
			
			# 添加兼容性字段
			info.update(base_info)
			
			# 添加旧字段名称的映射
			info['power_loss_reward'] = base_info.get('powerloss_reward', 0)
			info['voltage_reward'] = -base_info.get('cost_voltage', 0) * 10.0
			info['control_reward'] = base_info.get('control_reward', 0)
			info['power_balance'] = self.power_balance_reward()
			info['pv_optimization'] = base_info.get('pv_reward', 0)
			
			# 组合奖励（包含电压项，向后兼容）
			if self.component_weights['voltage'] > 0:
				# 旧模式：电压参与主奖励
				total_reward = (
					main_reward + 
					info['voltage_reward'] * self.component_weights['voltage']
				)
			else:
				# 新模式：电压仅作为约束
				total_reward = main_reward
			
			# 记录原始奖励
			info['total_reward_raw'] = total_reward
			info['total_reward'] = total_reward
			
		else:
			# 基础奖励模式（完全向后兼容）
			p = self.powerloss_reward()
			v, vio_nodes = self.voltage_reward(record_node)
			t = self.ctrl_reward(cd, rd, soc, dis, pv_diff)
			
			total_reward = p + v + t
			
			if record_node:
				info['violated_nodes'] = vio_nodes
			
			if full:
				info.update({
					'power_loss_reward': p,
					'voltage_reward': v,
					'control_reward': t,
					'total_reward': total_reward,
					'total_reward_raw': total_reward
				})
		
		# 应用全局缩放
		total_reward *= self.reward_scale
		
		# 归一化奖励（如果启用）
		normalized_reward = self._normalize_reward(total_reward)
		
		# 记录归一化信息
		if full:
			info['raw_reward'] = total_reward
			info['normalized_reward'] = normalized_reward
			
			# 确保包含电压成本（CMDP关键字段）
			if 'cost_voltage' not in info:
				info['cost_voltage'] = self.voltage_cost()
			
			# 添加违约率
			info['voltage_violation_rate'] = 1.0 - self._calculate_voltage_compliance()
		
		# 记录到日志（调试模式）
		if logger.isEnabledFor(logging.DEBUG):
			self._log_reward_components(info)
		
		return normalized_reward, info
	
	def _log_reward_components(self, components: Dict[str, Any]):
		"""记录奖励组件到日志"""
		log_msg = "奖励组件: "
		for key, value in components.items():
			if isinstance(value, (int, float)):
				log_msg += f"{key}={value:.3f}, "
		logger.debug(log_msg.rstrip(', '))