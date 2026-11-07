"""
JSON Exporter
JSON 数据导出器 -- 将 episode 数据导出为可读 JSON

提供完整 episode 导出和汇总统计导出两种模式。
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def export_episode_json(
	episode_data: Any,
	output_path: Optional[str] = None,
) -> str:
	"""将 EpisodeData 导出为可读 JSON 字符串

	包含元数据、汇总统计和每步精简数据（不含原始电气数据，
	只保留关键指标）。

	Args:
		episode_data: EpisodeData 对象 (来自 episode_reader)
		output_path: 可选输出文件路径

	Returns:
		JSON 字符串
	"""
	snapshots = episode_data.snapshots

	# Episode 元数据
	result: Dict[str, Any] = {
		"metadata": {
			"algorithm": episode_data.metadata.get("algorithm", "unknown"),
			"seed": episode_data.metadata.get("seed", -1),
			"environment": episode_data.config_summary.get("env_name", "unknown"),
			"total_steps": episode_data.episode_length,
			"total_reward": episode_data.total_reward,
			"export_timestamp": datetime.now().isoformat(),
		},
		"config_summary": episode_data.config_summary,
	}

	# 汇总统计
	result["summary"] = _compute_summary(snapshots, episode_data.total_reward)

	# 每步精简数据
	result["steps"] = []
	for snap in snapshots:
		step_data = _extract_step_summary(snap)
		result["steps"].append(step_data)

	json_str = json.dumps(result, indent=2, ensure_ascii=False, default=_json_default)

	if output_path is not None:
		with open(output_path, "w", encoding="utf-8") as f:
			f.write(json_str)

	return json_str


def export_summary_json(
	snapshots: List[Dict[str, Any]],
	output_path: Optional[str] = None,
) -> str:
	"""导出快照序列的汇总统计 JSON

	仅包含汇总统计，不含每步数据，适合快速概览。

	Args:
		snapshots: 快照列表
		output_path: 可选输出文件路径

	Returns:
		JSON 字符串
	"""
	total_reward = 0.0
	for snap in snapshots:
		rewards = snap.get("rewards")
		if isinstance(rewards, list) and rewards:
			total_reward += sum(
				r for r in rewards if isinstance(r, (int, float))
			)

	result = {
		"export_timestamp": datetime.now().isoformat(),
		"n_steps": len(snapshots),
		"summary": _compute_summary(snapshots, total_reward),
	}

	json_str = json.dumps(result, indent=2, ensure_ascii=False, default=_json_default)

	if output_path is not None:
		with open(output_path, "w", encoding="utf-8") as f:
			f.write(json_str)

	return json_str


# ======================================================================
# 内部函数
# ======================================================================

def _compute_summary(
	snapshots: List[Dict[str, Any]],
	total_reward: float,
) -> Dict[str, Any]:
	"""从快照序列中计算汇总统计

	Args:
		snapshots: 快照列表
		total_reward: 累计总奖励

	Returns:
		汇总统计字典
	"""
	if not snapshots:
		return {"total_reward": total_reward}

	# 电压统计
	all_v_mean: List[float] = []
	all_v_min: List[float] = []
	all_v_max: List[float] = []

	# 功率统计
	total_loss_sum = 0.0
	total_load_sum = 0.0
	total_pv_sum = 0.0

	for snap in snapshots:
		circuit = snap.get("circuit", {})

		v_mean = circuit.get("v_mean_pu")
		if isinstance(v_mean, (int, float)):
			all_v_mean.append(v_mean)

		v_min = circuit.get("v_min_pu")
		if isinstance(v_min, (int, float)):
			all_v_min.append(v_min)

		v_max = circuit.get("v_max_pu")
		if isinstance(v_max, (int, float)):
			all_v_max.append(v_max)

		loss = circuit.get("total_loss_kw")
		if isinstance(loss, (int, float)):
			total_loss_sum += loss

		load = circuit.get("total_load_kw")
		if isinstance(load, (int, float)):
			total_load_sum += load

		pv = circuit.get("total_pv_kw")
		if isinstance(pv, (int, float)):
			total_pv_sum += pv

	n = len(snapshots)
	summary: Dict[str, Any] = {
		"total_reward": total_reward,
		"n_steps": n,
	}

	# 电压
	if all_v_mean:
		summary["avg_voltage_pu"] = sum(all_v_mean) / len(all_v_mean)
	if all_v_min:
		summary["min_voltage_pu"] = min(all_v_min)
	if all_v_max:
		summary["max_voltage_pu"] = max(all_v_max)

	# 功率
	summary["total_loss_kwh"] = total_loss_sum * 0.25  # 15min 分辨率
	summary["avg_loss_kw"] = total_loss_sum / n if n > 0 else 0.0
	summary["avg_load_kw"] = total_load_sum / n if n > 0 else 0.0
	summary["total_pv_kwh"] = total_pv_sum * 0.25

	# PV 利用率 (PV / Load)
	if total_load_sum > 0:
		summary["pv_utilization_ratio"] = total_pv_sum / total_load_sum
	else:
		summary["pv_utilization_ratio"] = 0.0

	return summary


def _extract_step_summary(snap: Dict[str, Any]) -> Dict[str, Any]:
	"""从单个快照中提取精简指标（不含原始电气数据）

	Args:
		snap: 快照字典

	Returns:
		精简指标字典
	"""
	circuit = snap.get("circuit", {})

	step_data: Dict[str, Any] = {
		"step": snap.get("step", 0),
		"timestamp_h": snap.get("timestamp_h", 0.0),
	}

	# 系统级指标
	for key in [
		"total_loss_kw", "total_load_kw", "total_gen_kw",
		"total_pv_kw", "total_storage_kw",
		"v_mean_pu", "v_min_pu", "v_max_pu",
		"converged",
	]:
		val = circuit.get(key)
		if val is not None:
			step_data[key] = val

	# 动作摘要
	actions = snap.get("actions")
	if actions is not None:
		step_data["n_agents"] = len(actions) if isinstance(actions, list) else 0

	# 奖励
	rewards = snap.get("rewards")
	if isinstance(rewards, list):
		step_data["rewards"] = rewards
		step_data["reward_sum"] = sum(
			r for r in rewards if isinstance(r, (int, float))
		)

	# 奖励分量
	rc = snap.get("reward_components")
	if rc:
		step_data["reward_components"] = rc

	return step_data


def _json_default(obj: Any) -> Any:
	"""JSON 序列化回调 -- 处理非标准类型

	Args:
		obj: 待序列化对象

	Returns:
		可序列化的 Python 原生类型
	"""
	# numpy 类型兼容
	try:
		import numpy as np
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		if isinstance(obj, (np.integer,)):
			return int(obj)
		if isinstance(obj, (np.floating,)):
			return float(obj)
		if isinstance(obj, (np.bool_,)):
			return bool(obj)
	except ImportError:
		pass

	return str(obj)
