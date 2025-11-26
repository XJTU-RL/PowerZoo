"""
拉格朗日更新器模块

用于CMDP（约束马尔可夫决策过程）的拉格朗日乘子自适应更新
支持标准拉格朗日和增广拉格朗日方法
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class LagrangianUpdater:
	"""
	拉格朗日乘子更新器
	
	用于处理约束优化问题中的拉格朗日乘子自适应调整
	支持多种更新策略和约束类型
	"""
	
	def __init__(
		self,
		init_lambda: float = 1.0,
		lr: float = 1e-3,
		target_cost: float = 0.01,
		lambda_max: float = 1e3,
		lambda_min: float = 0.0,
		update_strategy: str = 'standard',
		momentum: float = 0.0,
		adaptive_lr: bool = False
	):
		"""
		初始化拉格朗日更新器
		
		Args:
			init_lambda: 初始拉格朗日乘子
			lr: 学习率
			target_cost: 目标约束值
			lambda_max: 最大拉格朗日乘子
			lambda_min: 最小拉格朗日乘子
			update_strategy: 更新策略 ('standard', 'augmented', 'adaptive')
			momentum: 动量系数（用于平滑更新）
			adaptive_lr: 是否使用自适应学习率
		"""
		self.lmbda = init_lambda
		self.lr = lr
		self.target = target_cost
		self.lambda_max = lambda_max
		self.lambda_min = lambda_min
		self.update_strategy = update_strategy
		self.momentum = momentum
		self.adaptive_lr = adaptive_lr
		
		# 内部状态
		self.velocity = 0.0  # 动量项
		self.update_count = 0
		self.cost_history: List[float] = []
		self.lambda_history: List[float] = [init_lambda]
		
		# 自适应学习率参数
		self.lr_decay = 0.999
		self.lr_min = 1e-5
		
		logger.info(
			f"拉格朗日更新器初始化: λ={init_lambda:.3f}, "
			f"lr={lr:.4f}, target={target_cost:.3f}, "
			f"strategy={update_strategy}"
		)
	
	def update(self, batch_cost_mean: float) -> float:
		"""
		更新拉格朗日乘子
		
		Args:
			batch_cost_mean: 批次平均约束违反成本
			
		Returns:
			float: 更新后的拉格朗日乘子
		"""
		self.update_count += 1
		self.cost_history.append(batch_cost_mean)
		
		# 计算约束违反程度
		constraint_violation = batch_cost_mean - self.target
		
		# 选择更新策略
		if self.update_strategy == 'standard':
			delta = self._standard_update(constraint_violation)
		elif self.update_strategy == 'augmented':
			delta = self._augmented_update(constraint_violation)
		elif self.update_strategy == 'adaptive':
			delta = self._adaptive_update(constraint_violation)
		else:
			raise ValueError(f"未知的更新策略: {self.update_strategy}")
		
		# 应用动量
		if self.momentum > 0:
			self.velocity = self.momentum * self.velocity + (1 - self.momentum) * delta
			delta = self.velocity
		
		# 更新拉格朗日乘子
		self.lmbda = np.clip(
			self.lmbda + delta,
			self.lambda_min,
			self.lambda_max
		)
		
		# 记录历史
		self.lambda_history.append(self.lmbda)
		
		# 自适应学习率衰减
		if self.adaptive_lr:
			self.lr = max(self.lr * self.lr_decay, self.lr_min)
		
		# 日志记录
		if self.update_count % 10 == 0:
			logger.debug(
				f"Lambda更新 #{self.update_count}: "
				f"cost={batch_cost_mean:.4f}, target={self.target:.4f}, "
				f"λ={self.lmbda:.3f}, lr={self.lr:.5f}"
			)
		
		return self.lmbda
	
	def _standard_update(self, violation: float) -> float:
		"""标准拉格朗日更新"""
		return self.lr * violation
	
	def _augmented_update(self, violation: float) -> float:
		"""增广拉格朗日更新（带二次惩罚项）"""
		# 增广项提供额外的梯度信息
		augmented_grad = violation + 0.5 * violation * abs(violation)
		return self.lr * augmented_grad
	
	def _adaptive_update(self, violation: float) -> float:
		"""自适应更新（基于历史违反程度）"""
		if len(self.cost_history) < 2:
			return self._standard_update(violation)
		
		# 计算违反趋势
		recent_costs = self.cost_history[-10:]
		cost_trend = np.mean(recent_costs) - self.target
		
		# 根据趋势调整学习率
		if abs(cost_trend) < 0.001:  # 接近目标
			adaptive_lr = self.lr * 0.5
		elif cost_trend * violation > 0:  # 同向，加速收敛
			adaptive_lr = self.lr * 1.2
		else:  # 震荡，减小步长
			adaptive_lr = self.lr * 0.8
		
		return adaptive_lr * violation
	
	def reset(self, init_lambda: Optional[float] = None):
		"""重置更新器状态"""
		if init_lambda is not None:
			self.lmbda = init_lambda
		else:
			self.lmbda = self.lambda_history[0] if self.lambda_history else 1.0
		
		self.velocity = 0.0
		self.update_count = 0
		self.cost_history.clear()
		self.lambda_history = [self.lmbda]
		
		logger.info(f"拉格朗日更新器重置: λ={self.lmbda:.3f}")
	
	def get_stats(self) -> Dict[str, Any]:
		"""获取统计信息"""
		stats = {
			'current_lambda': self.lmbda,
			'update_count': self.update_count,
			'learning_rate': self.lr,
			'target_cost': self.target
		}
		
		if self.cost_history:
			stats.update({
				'avg_cost': np.mean(self.cost_history),
				'std_cost': np.std(self.cost_history),
				'recent_cost': np.mean(self.cost_history[-10:]) if len(self.cost_history) >= 10 else np.mean(self.cost_history),
				'cost_trend': self._calculate_trend(self.cost_history)
			})
		
		if self.lambda_history:
			stats.update({
				'lambda_mean': np.mean(self.lambda_history),
				'lambda_std': np.std(self.lambda_history),
				'lambda_trend': self._calculate_trend(self.lambda_history)
			})
		
		return stats
	
	def _calculate_trend(self, data: List[float]) -> float:
		"""计算数据趋势（线性回归斜率）"""
		if len(data) < 2:
			return 0.0
		
		x = np.arange(len(data))
		y = np.array(data)
		
		# 简单线性回归
		x_mean = x.mean()
		y_mean = y.mean()
		
		numerator = ((x - x_mean) * (y - y_mean)).sum()
		denominator = ((x - x_mean) ** 2).sum()
		
		if denominator == 0:
			return 0.0
		
		return numerator / denominator


class MultiConstraintLagrangian:
	"""
	多约束拉格朗日更新器
	
	处理多个约束的CMDP问题
	"""
	
	def __init__(
		self,
		constraint_names: List[str],
		init_lambdas: Optional[Dict[str, float]] = None,
		lr: float = 1e-3,
		target_costs: Optional[Dict[str, float]] = None,
		lambda_max: float = 1e3,
		lambda_min: float = 0.0
	):
		"""
		初始化多约束更新器
		
		Args:
			constraint_names: 约束名称列表
			init_lambdas: 初始拉格朗日乘子字典
			lr: 学习率
			target_costs: 目标约束值字典
			lambda_max: 最大拉格朗日乘子
			lambda_min: 最小拉格朗日乘子
		"""
		self.constraint_names = constraint_names
		self.updaters = {}
		
		# 为每个约束创建独立的更新器
		for name in constraint_names:
			init_lambda = init_lambdas.get(name, 1.0) if init_lambdas else 1.0
			target_cost = target_costs.get(name, 0.01) if target_costs else 0.01
			
			self.updaters[name] = LagrangianUpdater(
				init_lambda=init_lambda,
				lr=lr,
				target_cost=target_cost,
				lambda_max=lambda_max,
				lambda_min=lambda_min
			)
		
		logger.info(f"多约束拉格朗日更新器初始化: {len(constraint_names)}个约束")
	
	def update(self, costs: Dict[str, float]) -> Dict[str, float]:
		"""
		更新所有约束的拉格朗日乘子
		
		Args:
			costs: 各约束的成本字典
			
		Returns:
			Dict[str, float]: 更新后的拉格朗日乘子字典
		"""
		lambdas = {}
		for name in self.constraint_names:
			if name in costs:
				lambdas[name] = self.updaters[name].update(costs[name])
			else:
				lambdas[name] = self.updaters[name].lmbda
		
		return lambdas
	
	def get_lambdas(self) -> Dict[str, float]:
		"""获取当前所有拉格朗日乘子"""
		return {name: updater.lmbda for name, updater in self.updaters.items()}
	
	def get_stats(self) -> Dict[str, Dict[str, Any]]:
		"""获取所有约束的统计信息"""
		return {name: updater.get_stats() for name, updater in self.updaters.items()}
	
	def reset(self, init_lambdas: Optional[Dict[str, float]] = None):
		"""重置所有更新器"""
		for name, updater in self.updaters.items():
			init_lambda = init_lambdas.get(name) if init_lambdas else None
			updater.reset(init_lambda)