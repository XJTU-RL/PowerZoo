#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HAPPO算法诊断工具模块

用于诊断HAPPO训练过程中的数值稳定性问题，包括：
- GAE计算过程监控
- Advantage归一化分析
- Policy Loss组件分解
- Value Loss误差分析
- Explained Variance详细分解
- 数值稳定性检查（NaN/Inf检测）
"""

import torch
import numpy as np
from typing import Optional, Union, Dict, Any
from torch.utils.tensorboard import SummaryWriter


class HAPPODiagnostics:
	"""HAPPO算法诊断工具类"""
	
	def __init__(self, writer: Optional[SummaryWriter] = None):
		"""
		初始化诊断工具
		
		Args:
			writer: TensorBoard writer实例
		"""
		self.writer = writer
		self.global_step = 0
		self.enabled = True  # 可以通过此标志关闭诊断
		
	def set_writer(self, writer: SummaryWriter):
		"""设置TensorBoard writer"""
		self.writer = writer
		
	def set_global_step(self, step: int):
		"""设置全局步数"""
		self.global_step = step
		
	def enable(self):
		"""启用诊断"""
		self.enabled = True
		
	def disable(self):
		"""禁用诊断"""
		self.enabled = False
		
	def log_gae_computation(self, step: int, delta: float, gae: float, 
							reward: float, value_curr: float, value_next: float,
							mask: float, bad_mask: float):
		"""
		记录GAE计算过程的中间值
		
		Args:
			step: 当前步数
			delta: TD误差
			gae: GAE值
			reward: 当前奖励
			value_curr: 当前价值估计
			value_next: 下一步价值估计
			mask: 普通mask
			bad_mask: bad mask
		"""
		if not self.enabled or self.writer is None:
			return
			
		prefix = f'gae_debug/step_{step}'
		self.writer.add_scalar(f'{prefix}/delta', delta, self.global_step)
		self.writer.add_scalar(f'{prefix}/gae', gae, self.global_step)
		self.writer.add_scalar(f'{prefix}/reward', reward, self.global_step)
		self.writer.add_scalar(f'{prefix}/value_curr', value_curr, self.global_step)
		self.writer.add_scalar(f'{prefix}/value_next', value_next, self.global_step)
		self.writer.add_scalar(f'{prefix}/mask', mask, self.global_step)
		self.writer.add_scalar(f'{prefix}/bad_mask', bad_mask, self.global_step)
		
		# 每20步打印一次到控制台
		if step % 20 == 0:
			print(f"[GAE Debug Step {step}] Delta={delta:.4f}, GAE={gae:.4f}, "
				  f"Reward={reward:.4f}, V_curr={value_curr:.4f}, V_next={value_next:.4f}")
	
	def log_advantages_normalization(self, advantages_raw: np.ndarray, 
									 advantages_norm: np.ndarray):
		"""
		记录优势函数归一化前后的统计信息
		
		Args:
			advantages_raw: 原始优势函数
			advantages_norm: 归一化后的优势函数
		"""
		if not self.enabled or self.writer is None:
			return
			
		# 原始优势统计
		self.writer.add_scalar('advantages_debug/mean_before', 
							  np.nanmean(advantages_raw), self.global_step)
		self.writer.add_scalar('advantages_debug/std_before', 
							  np.nanstd(advantages_raw), self.global_step)
		self.writer.add_scalar('advantages_debug/max_before', 
							  np.nanmax(advantages_raw), self.global_step)
		self.writer.add_scalar('advantages_debug/min_before', 
							  np.nanmin(advantages_raw), self.global_step)
		
		# 归一化后优势统计
		self.writer.add_scalar('advantages_debug/mean_after', 
							  np.nanmean(advantages_norm), self.global_step)
		self.writer.add_scalar('advantages_debug/std_after', 
							  np.nanstd(advantages_norm), self.global_step)
		
		# 数值稳定性检查
		nan_count = np.isnan(advantages_raw).sum()
		inf_count = np.isinf(advantages_raw).sum()
		self.writer.add_scalar('advantages_debug/nan_count', nan_count, self.global_step)
		self.writer.add_scalar('advantages_debug/inf_count', inf_count, self.global_step)
		
		# 警告打印
		if nan_count > 0 or inf_count > 0:
			print(f"[WARNING] Advantages contain {nan_count} NaN and {inf_count} Inf values!")
			
	def log_policy_loss_components(self, policy_loss: torch.Tensor,
								   imp_weights: torch.Tensor,
								   advantages: torch.Tensor,
								   surr1: torch.Tensor,
								   surr2: torch.Tensor,
								   clip_coef: float):
		"""
		记录策略损失的各个组件
		
		Args:
			policy_loss: 策略损失
			imp_weights: 重要性权重 (ratio)
			advantages: 优势函数
			surr1: 未裁剪的surrogate损失
			surr2: 裁剪后的surrogate损失
			clip_coef: 裁剪系数
		"""
		if not self.enabled or self.writer is None:
			return
			
		with torch.no_grad():
			# 策略损失
			self.writer.add_scalar('policy_debug/policy_loss', 
								  policy_loss.mean().item(), self.global_step)
			
			# 重要性权重统计
			self.writer.add_scalar('policy_debug/imp_weights_mean', 
								  imp_weights.mean().item(), self.global_step)
			self.writer.add_scalar('policy_debug/imp_weights_std', 
								  imp_weights.std().item(), self.global_step)
			self.writer.add_scalar('policy_debug/imp_weights_max', 
								  imp_weights.max().item(), self.global_step)
			self.writer.add_scalar('policy_debug/imp_weights_min', 
								  imp_weights.min().item(), self.global_step)
			
			# 优势函数统计
			self.writer.add_scalar('policy_debug/advantages_mean', 
								  advantages.mean().item(), self.global_step)
			self.writer.add_scalar('policy_debug/advantages_std', 
								  advantages.std().item(), self.global_step)
			
			# Surrogate损失
			self.writer.add_scalar('policy_debug/surr1_mean', 
								  surr1.mean().item(), self.global_step)
			self.writer.add_scalar('policy_debug/surr2_mean', 
								  surr2.mean().item(), self.global_step)
			
			# 计算裁剪比例
			clipped_mask = (imp_weights > 1 + clip_coef) | (imp_weights < 1 - clip_coef)
			clip_fraction = clipped_mask.float().mean().item()
			self.writer.add_scalar('policy_debug/clip_fraction', 
								  clip_fraction, self.global_step)
			
			# 每10步打印一次
			if self.global_step % 10 == 0:
				print(f"[Policy Debug] Loss={policy_loss.mean().item():.4f}, "
					  f"Ratio Mean={imp_weights.mean().item():.4f}, "
					  f"Clip Fraction={clip_fraction:.4f}")
				
	def log_value_loss_components(self, value_loss: torch.Tensor,
								  values: torch.Tensor,
								  returns: torch.Tensor,
								  value_pred_clipped: Optional[torch.Tensor] = None,
								  old_values: Optional[torch.Tensor] = None):
		"""
		记录价值损失的各个组件
		
		Args:
			value_loss: 价值损失
			values: 当前价值预测
			returns: 目标回报
			value_pred_clipped: 裁剪后的价值预测（如果使用）
			old_values: 旧的价值预测（如果使用裁剪）
		"""
		if not self.enabled or self.writer is None:
			return
			
		with torch.no_grad():
			# 价值损失
			self.writer.add_scalar('value_debug/value_loss', 
								  value_loss.mean().item(), self.global_step)
			
			# 价值预测统计
			self.writer.add_scalar('value_debug/values_mean', 
								  values.mean().item(), self.global_step)
			self.writer.add_scalar('value_debug/values_std', 
								  values.std().item(), self.global_step)
			
			# 目标回报统计
			self.writer.add_scalar('value_debug/returns_mean', 
								  returns.mean().item(), self.global_step)
			self.writer.add_scalar('value_debug/returns_std', 
								  returns.std().item(), self.global_step)
			
			# 误差分析
			error_original = returns - values
			self.writer.add_scalar('value_debug/error_mean', 
								  error_original.mean().item(), self.global_step)
			self.writer.add_scalar('value_debug/error_std', 
								  error_original.std().item(), self.global_step)
			
			# 如果使用价值裁剪
			if value_pred_clipped is not None and old_values is not None:
				error_clipped = returns - value_pred_clipped
				self.writer.add_scalar('value_debug/error_clipped_mean', 
									  error_clipped.mean().item(), self.global_step)
				
			# 每10步打印一次
			if self.global_step % 10 == 0:
				print(f"[Value Debug] Loss={value_loss.mean().item():.4f}, "
					  f"Values Mean={values.mean().item():.4f}, "
					  f"Returns Mean={returns.mean().item():.4f}")
				
	def log_explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray):
		"""
		记录解释方差的详细分解
		
		Args:
			y_true: 真实值（returns）
			y_pred: 预测值（values）
			
		Returns:
			explained_variance: 计算得到的解释方差
		"""
		if not self.enabled:
			return None
			
		# 计算解释方差
		var_y = np.var(y_true)
		explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8)
		
		if self.writer is not None:
			# 记录解释方差
			self.writer.add_scalar('explained_var_debug/explained_variance', 
								  explained_var, self.global_step)
			
			# 记录方差分解
			self.writer.add_scalar('explained_var_debug/returns_variance', 
								  var_y, self.global_step)
			self.writer.add_scalar('explained_var_debug/residual_variance', 
								  np.var(y_true - y_pred), self.global_step)
			
			# 记录相关性
			if len(y_true) > 1:
				correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
				self.writer.add_scalar('explained_var_debug/correlation', 
									  correlation, self.global_step)
			
			# 警告：如果解释方差过低
			if explained_var < 0.1:
				print(f"[WARNING] Explained variance is very low: {explained_var:.4f}")
				
		return explained_var
		
	def check_tensor_stability(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
		"""
		检查张量的数值稳定性
		
		Args:
			tensor: 要检查的张量
			name: 张量名称（用于日志）
			
		Returns:
			包含稳定性信息的字典
		"""
		with torch.no_grad():
			nan_count = torch.isnan(tensor).sum().item()
			inf_count = torch.isinf(tensor).sum().item()
			total_elements = tensor.numel()
			
			stability_info = {
				'nan_count': nan_count,
				'inf_count': inf_count,
				'total_elements': total_elements,
				'nan_ratio': nan_count / total_elements if total_elements > 0 else 0,
				'inf_ratio': inf_count / total_elements if total_elements > 0 else 0,
				'is_stable': nan_count == 0 and inf_count == 0
			}
			
			if self.writer is not None and self.enabled:
				prefix = f'stability/{name}'
				self.writer.add_scalar(f'{prefix}/nan_count', nan_count, self.global_step)
				self.writer.add_scalar(f'{prefix}/inf_count', inf_count, self.global_step)
				self.writer.add_scalar(f'{prefix}/nonfinite_ratio', 
									  (nan_count + inf_count) / total_elements, 
									  self.global_step)
				
				# 如果张量稳定，记录其统计信息
				if stability_info['is_stable']:
					self.writer.add_scalar(f'{prefix}/mean', 
										  tensor.mean().item(), self.global_step)
					self.writer.add_scalar(f'{prefix}/std', 
										  tensor.std().item(), self.global_step)
					
			# 打印警告
			if not stability_info['is_stable']:
				print(f"[STABILITY WARNING] {name} contains {nan_count} NaN and "
					  f"{inf_count} Inf values out of {total_elements} total!")
					  
			return stability_info


# 全局诊断实例
_global_diagnostics = HAPPODiagnostics()


def get_diagnostics() -> HAPPODiagnostics:
	"""获取全局诊断实例"""
	return _global_diagnostics


def set_writer(writer: SummaryWriter):
	"""设置全局诊断的TensorBoard writer"""
	_global_diagnostics.set_writer(writer)


def set_global_step(step: int):
	"""设置全局步数"""
	_global_diagnostics.set_global_step(step)


def enable_diagnostics():
	"""启用诊断"""
	_global_diagnostics.enable()


def disable_diagnostics():
	"""禁用诊断"""
	_global_diagnostics.disable()


# 便捷函数
def log_gae_computation(step: int, delta: float, gae: float, 
						reward: float, value_curr: float, value_next: float,
						mask: float, bad_mask: float):
	"""记录GAE计算（便捷函数）"""
	_global_diagnostics.log_gae_computation(
		step, delta, gae, reward, value_curr, value_next, mask, bad_mask
	)


def log_advantages_normalization(advantages_raw: np.ndarray, 
								 advantages_norm: np.ndarray):
	"""记录优势归一化（便捷函数）"""
	_global_diagnostics.log_advantages_normalization(advantages_raw, advantages_norm)


def log_policy_loss_components(policy_loss: torch.Tensor,
							   imp_weights: torch.Tensor,
							   advantages: torch.Tensor,
							   surr1: torch.Tensor,
							   surr2: torch.Tensor,
							   clip_coef: float):
	"""记录策略损失组件（便捷函数）"""
	_global_diagnostics.log_policy_loss_components(
		policy_loss, imp_weights, advantages, surr1, surr2, clip_coef
	)


def log_value_loss_components(value_loss: torch.Tensor,
							  values: torch.Tensor,
							  returns: torch.Tensor,
							  value_pred_clipped: Optional[torch.Tensor] = None,
							  old_values: Optional[torch.Tensor] = None):
	"""记录价值损失组件（便捷函数）"""
	_global_diagnostics.log_value_loss_components(
		value_loss, values, returns, value_pred_clipped, old_values
	)


def log_explained_variance(y_true: np.ndarray, y_pred: np.ndarray):
	"""记录解释方差（便捷函数）"""
	return _global_diagnostics.log_explained_variance(y_true, y_pred)


def check_tensor_stability(tensor: torch.Tensor, name: str) -> Dict[str, Any]:
	"""检查张量稳定性（便捷函数）"""
	return _global_diagnostics.check_tensor_stability(tensor, name)