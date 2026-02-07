"""
SmartGrid Lagrangian Data Extractor (SmartGrid 独有)
拉格朗日数据提取器

从 SmartGrid 环境的 CMDP 框架中提取 Lagrangian 乘子 lambda、
约束违反量、目标成本等数据，用于 Lagrangian 轨迹图的渲染。
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_lagrangian_state(env) -> Dict[str, Any]:
	"""从 SmartGrid 环境提取当前 Lagrangian 状态

	Args:
		env: SmartGrid Env 实例（base_env.env.Env）

	Returns:
		{lambda_value, target_cost, update_count, lambda_history, cost_history, ...}
	"""
	state: Dict[str, Any] = {
		"use_cmdp": False,
		"lambda_value": 0.0,
		"target_cost": 0.0,
		"update_count": 0,
		"lambda_history": [],
		"cost_history": [],
		"episode_costs": [],
		"update_strategy": "standard",
	}

	try:
		use_cmdp = getattr(env, "use_cmdp", False)
		state["use_cmdp"] = use_cmdp

		if not use_cmdp:
			return state

		updater = getattr(env, "lagrangian_updater", None)
		if updater is not None:
			state["lambda_value"] = float(getattr(updater, "lmbda", 0.0))
			state["target_cost"] = float(getattr(updater, "target", 0.0))
			state["update_count"] = int(getattr(updater, "update_count", 0))
			state["update_strategy"] = str(getattr(updater, "update_strategy", "standard"))

			# 历史记录（来自 updater 自身）
			updater_lambda_hist = getattr(updater, "lambda_history", [])
			state["lambda_history"] = [float(v) for v in updater_lambda_hist]

			updater_cost_hist = getattr(updater, "cost_history", [])
			state["cost_history"] = [float(v) for v in updater_cost_hist]

		# 来自 env 的跨 episode 历史
		env_lambda_hist = getattr(env, "lambda_history", [])
		if env_lambda_hist:
			state["lambda_history"] = [float(v) for v in env_lambda_hist]

		env_cost_hist = getattr(env, "cost_history", [])
		if env_cost_hist:
			state["cost_history"] = [float(v) for v in env_cost_hist]

		episode_costs = getattr(env, "episode_costs", [])
		state["episode_costs"] = [float(v) for v in episode_costs]

	except Exception as exc:
		logger.error(f"Lagrangian state extraction failed: {exc}")

	return state


def extract_lagrangian_from_infos(
	snapshots: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
	"""从快照 info 中提取 Lagrangian 时间序列

	env.step() 返回的 info 中包含:
	- 'lambda': 当前 lambda 值
	- 'cost_voltage': 电压约束违反成本
	- 'lagrangian_penalty': 拉格朗日惩罚项

	Args:
		snapshots: 快照列表

	Returns:
		{lambda_values, cost_voltage, lagrangian_penalty, constraint_violations}
	"""
	lambda_values: List[float] = []
	cost_voltage: List[float] = []
	lagrangian_penalty: List[float] = []
	constraint_violations: List[float] = []

	for snap in snapshots:
		info = snap.get("info", {})
		if not info:
			info = snap.get("reward_components", {})

		lmbda = info.get("lambda", None)
		if lmbda is not None:
			lambda_values.append(float(lmbda))

		cost = info.get("cost_voltage", None)
		if cost is not None:
			cost_voltage.append(float(cost))

		penalty = info.get("lagrangian_penalty", None)
		if penalty is not None:
			lagrangian_penalty.append(float(penalty))

		violation = info.get("voltage_violation_rate", info.get("voltage_violation_rate_buses", None))
		if violation is not None:
			constraint_violations.append(float(violation))

	return {
		"lambda_values": lambda_values,
		"cost_voltage": cost_voltage,
		"lagrangian_penalty": lagrangian_penalty,
		"constraint_violations": constraint_violations,
	}


def compute_lagrangian_summary(
	snapshots: List[Dict[str, Any]],
) -> Dict[str, Any]:
	"""计算 Lagrangian 汇总统计

	Args:
		snapshots: 快照列表

	Returns:
		汇总字典
	"""
	series = extract_lagrangian_from_infos(snapshots)

	summary: Dict[str, Any] = {
		"has_lagrangian_data": False,
	}

	lambda_vals = series["lambda_values"]
	if lambda_vals:
		summary["has_lagrangian_data"] = True
		summary["lambda_init"] = lambda_vals[0]
		summary["lambda_final"] = lambda_vals[-1]
		summary["lambda_min"] = float(np.min(lambda_vals))
		summary["lambda_max"] = float(np.max(lambda_vals))
		summary["lambda_mean"] = float(np.mean(lambda_vals))
		summary["n_lambda_updates"] = len(lambda_vals)

	cost_vals = series["cost_voltage"]
	if cost_vals:
		summary["avg_cost_voltage"] = float(np.mean(cost_vals))
		summary["max_cost_voltage"] = float(np.max(cost_vals))
		summary["final_cost_voltage"] = cost_vals[-1]

	violations = series["constraint_violations"]
	if violations:
		summary["avg_violation_rate"] = float(np.mean(violations))
		summary["max_violation_rate"] = float(np.max(violations))
		# 约束满足率：违反率 < 目标的步数占比
		summary["constraint_satisfaction_rate"] = float(
			np.mean([1.0 if v < 0.05 else 0.0 for v in violations])
		)

	return summary
