"""
SmartGrid Snapshot Assembler
快照组装器

将各 data extractor 的输出组装为统一的快照字典格式，
供 Plotly / Matplotlib 可视化使用。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from envs.smartgrid.render.data.bus_data_extractor import (
	extract_bus_data,
	compute_voltage_stats,
)
from envs.smartgrid.render.data.line_data_extractor import (
	extract_line_data,
	extract_transformer_data,
)
from envs.smartgrid.render.data.component_data_extractor import (
	extract_all_components,
	get_component_summary,
)
from envs.smartgrid.render.data.circuit_data_extractor import (
	extract_circuit_summary,
)
from envs.smartgrid.render.data.lagrangian_data_extractor import (
	extract_lagrangian_state,
)

logger = logging.getLogger(__name__)


def assemble_snapshot(
	env,
	step: int,
	actions: Optional[np.ndarray] = None,
	rewards: Optional[np.ndarray] = None,
	infos: Optional[Any] = None,
) -> Dict[str, Any]:
	"""组装完整快照

	Args:
		env: SmartGrid Env 实例
		step: 当前步编号
		actions: 动作数组
		rewards: 奖励数组
		infos: step() 返回的 info 字典

	Returns:
		统一格式的快照字典
	"""
	snapshot: Dict[str, Any] = {
		"step": step,
	}

	# 母线数据
	snapshot["buses"] = extract_bus_data(env)

	# 线路数据
	lines = extract_line_data(env)
	xfmrs = extract_transformer_data(env)
	snapshot["lines"] = {**lines, **xfmrs}

	# 设备数据（OOP 组件）
	snapshot["devices"] = extract_all_components(env)

	# 电路汇总
	snapshot["circuit"] = extract_circuit_summary(env)

	# Lagrangian 状态
	lagrangian_state = extract_lagrangian_state(env)
	snapshot["lagrangian"] = lagrangian_state

	# 动作
	if actions is not None:
		if isinstance(actions, np.ndarray):
			snapshot["actions"] = actions.tolist()
		else:
			snapshot["actions"] = actions

	# 奖励
	if rewards is not None:
		if isinstance(rewards, np.ndarray):
			snapshot["rewards"] = rewards.tolist()
		else:
			snapshot["rewards"] = rewards

	# Info (来自 env.step())
	if infos is not None:
		if isinstance(infos, dict):
			snapshot["info"] = infos
			# 提取奖励分解
			snapshot["reward_components"] = _extract_reward_components(infos)

	return snapshot


def _extract_reward_components(info: Dict[str, Any]) -> Dict[str, Any]:
	"""从 info 中提取奖励分解组件"""
	components: Dict[str, Any] = {}

	reward_keys = [
		"vol_reward", "ctrl_reward", "power_loss_reward",
		"cap_penalty", "reg_penalty", "soc_penalty", "dis_penalty",
		"pv_penalty", "cost_voltage", "lagrangian_penalty",
		"reward_before_lagrangian", "lambda",
		"power_loss_kw", "power_loss_ratio", "power_loss_percentage",
		"voltage_violation_count", "voltage_violation_rate",
		"voltage_violation_rate_buses", "voltage_violation_rate_phases",
		"pv_utilization", "battery_avg_soc",
	]

	for key in reward_keys:
		if key in info:
			val = info[key]
			if isinstance(val, (int, float)):
				components[key] = float(val)

	return components
