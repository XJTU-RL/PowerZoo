# -*- coding: utf-8 -*-
"""
压力测试运行器

参数扫描 + 批量推理，支持二维网格搜索和单参数灵敏度分析。
通过修改环境配置参数并调用 EpisodeRunner 运行 episode，
收集 total_reward, voltage_violation_pct, total_loss_kw, pv_utilization 等指标。
"""

import copy
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs.district_dispatch.render.engine.episode_runner import (
	EpisodeData,
	EpisodeRunner,
)
from envs.district_dispatch.render.engine.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)

# 支持的压力测试参数及其默认范围
SWEEP_PARAM_DEFAULTS: Dict[str, Dict[str, Any]] = {
	"load_multiplier": {"min": 0.5, "max": 2.0, "default": 1.0},
	"pv_output_ratio": {"min": 0.0, "max": 1.0, "default": 0.8},
	"initial_soc": {"min": 0.1, "max": 0.9, "default": 0.5},
	"ev_demand_mult": {"min": 0.0, "max": 3.0, "default": 1.0},
	"carbon_intensity": {"min": 0.1, "max": 1.5, "default": 0.5},
	"price_multiplier": {"min": 0.5, "max": 3.0, "default": 1.0},
}


def _extract_metrics(episode_data: EpisodeData) -> Dict[str, float]:
	"""从 episode 数据中提取关键指标。

	Args:
		episode_data: 一个 episode 的完整数据

	Returns:
		Dict[str, float]: {指标名: 值}
	"""
	snapshots = episode_data.snapshots
	total_reward = episode_data.total_reward
	n_steps = max(len(snapshots), 1)

	# 电压越限率
	violation_count = 0
	bus_count = 0
	for snap in snapshots:
		buses = snap.get("buses", snap.get("bus_data", {}))
		for bus_name, bus_data in buses.items():
			v = bus_data.get("v_mean", 1.0)
			bus_count += 1
			if v < 0.95 or v > 1.05:
				violation_count += 1

	voltage_violation_pct = (
		100.0 * violation_count / bus_count if bus_count > 0 else 0.0
	)

	# 总损耗 (kWh)
	dt_h = 0.25
	total_loss_kwh = 0.0
	for snap in snapshots:
		circuit = snap.get("circuit", {})
		total_loss_kwh += circuit.get("total_loss_kw", 0.0) * dt_h

	# PV 利用率
	total_pv_capacity = 0.0
	total_pv_output = 0.0
	for snap in snapshots:
		devices = snap.get("devices", {})
		for pv_data in devices.get("pv", {}).values():
			rated = pv_data.get("kw_rated", pv_data.get("pmpp", 0.0))
			output = pv_data.get("kw_output", 0.0)
			total_pv_capacity += abs(rated) * dt_h
			total_pv_output += abs(output) * dt_h

	pv_utilization = (
		total_pv_output / total_pv_capacity * 100.0
		if total_pv_capacity > 0 else 0.0
	)

	return {
		"total_reward": total_reward,
		"voltage_violation_pct": voltage_violation_pct,
		"total_loss_kwh": total_loss_kwh,
		"pv_utilization": pv_utilization,
		"episode_length": episode_data.episode_length,
	}


def _apply_params_to_config(
	config: Any, params: Dict[str, float]
) -> Any:
	"""将压力测试参数应用到环境配置上。

	通过 deepcopy 创建配置副本，避免修改原始配置。
	参数映射到 config 对象的属性或 dict 的键。

	Args:
		config: 环境配置对象
		params: {参数名: 值}

	Returns:
		修改后的配置副本
	"""
	cfg = copy.deepcopy(config)

	for key, value in params.items():
		if hasattr(cfg, key):
			setattr(cfg, key, value)
		elif isinstance(cfg, dict) and key in cfg:
			cfg[key] = value
		else:
			# 尝试嵌套属性: 检查 stress_test_params
			if hasattr(cfg, "stress_test_params"):
				if isinstance(cfg.stress_test_params, dict):
					cfg.stress_test_params[key] = value
			else:
				logger.warning(
					f"参数 '{key}' 无法映射到配置对象，跳过"
				)

	return cfg


class StressTestRunner:
	"""压力测试运行器

	支持三种测试模式：
	1. 单场景运行：修改指定参数运行一个 episode
	2. 参数扫描：二维网格搜索
	3. 灵敏度分析：one-at-a-time 参数变化

	Args:
		config: 环境配置对象
		inference_engine: 推理引擎 (None 则使用随机动作)
	"""

	def __init__(
		self,
		config: Any,
		inference_engine: Optional[InferenceEngine] = None,
	):
		self.base_config = config
		self.inference_engine = inference_engine
		self._run_count = 0

	def run_single(
		self,
		params: Dict[str, float],
		seed: Optional[int] = None,
		collect_snapshots: bool = True,
	) -> Dict[str, Any]:
		"""运行单个场景。

		修改环境参数并执行一个完整 episode，返回指标和元数据。

		Args:
			params: 参数字典，如 {"load_multiplier": 1.5, "pv_output_ratio": 0.3}
			seed: 随机种子
			collect_snapshots: 是否收集快照 (关闭可加速)

		Returns:
			{
				"params": Dict,
				"metrics": Dict[str, float],
				"episode_data": EpisodeData (if collect_snapshots),
				"elapsed_seconds": float,
			}
		"""
		t_start = time.time()
		self._run_count += 1

		cfg = _apply_params_to_config(self.base_config, params)
		runner = EpisodeRunner(
			config=cfg,
			inference_engine=self.inference_engine,
			collect_snapshots=collect_snapshots,
		)

		try:
			episode_data = runner.run_episode(seed=seed)
			metrics = _extract_metrics(episode_data)
		finally:
			runner.close()

		elapsed = round(time.time() - t_start, 3)

		logger.info(
			f"Stress test #{self._run_count}: params={params}, "
			f"reward={metrics['total_reward']:.2f}, "
			f"violations={metrics['voltage_violation_pct']:.1f}%, "
			f"elapsed={elapsed}s"
		)

		result: Dict[str, Any] = {
			"params": params,
			"metrics": metrics,
			"elapsed_seconds": elapsed,
		}
		if collect_snapshots:
			result["episode_data"] = episode_data

		return result

	def run_sweep(
		self,
		param1_name: str,
		param1_range: np.ndarray,
		param2_name: str,
		param2_range: np.ndarray,
		seed: int = 42,
		base_params: Optional[Dict[str, float]] = None,
	) -> Dict[str, Any]:
		"""二维参数扫描。

		对两个参数的网格组合运行 episode，收集指标热力图。

		Args:
			param1_name: 第一个参数名 (行)
			param1_range: 第一个参数的取值范围 (np.ndarray)
			param2_name: 第二个参数名 (列)
			param2_range: 第二个参数的取值范围 (np.ndarray)
			seed: 随机种子
			base_params: 基准参数 (其余参数使用此值)

		Returns:
			{
				"param1_name": str,
				"param1_range": list,
				"param2_name": str,
				"param2_range": list,
				"results": 2D list of metrics dicts,
				"heatmaps": {metric_name: 2D np.ndarray},
				"total_runs": int,
				"elapsed_seconds": float,
			}
		"""
		t_start = time.time()
		base = dict(base_params or {})

		n1 = len(param1_range)
		n2 = len(param2_range)
		total_runs = n1 * n2

		logger.info(
			f"Starting sweep: {param1_name}({n1}) x {param2_name}({n2}) = {total_runs} runs"
		)

		results_grid: List[List[Dict[str, float]]] = []
		metric_names: List[str] = []

		for i, v1 in enumerate(param1_range):
			row: List[Dict[str, float]] = []
			for j, v2 in enumerate(param2_range):
				params = dict(base)
				params[param1_name] = float(v1)
				params[param2_name] = float(v2)

				result = self.run_single(
					params, seed=seed, collect_snapshots=False,
				)
				metrics = result["metrics"]
				row.append(metrics)

				if not metric_names:
					metric_names = list(metrics.keys())

				run_idx = i * n2 + j + 1
				logger.debug(
					f"Sweep [{run_idx}/{total_runs}]: "
					f"{param1_name}={v1:.3f}, {param2_name}={v2:.3f} -> "
					f"reward={metrics['total_reward']:.2f}"
				)

			results_grid.append(row)

		# 构建热力图矩阵
		heatmaps: Dict[str, np.ndarray] = {}
		for metric_name in metric_names:
			matrix = np.zeros((n1, n2))
			for i in range(n1):
				for j in range(n2):
					matrix[i, j] = results_grid[i][j].get(metric_name, 0.0)
			heatmaps[metric_name] = matrix

		elapsed = round(time.time() - t_start, 3)
		logger.info(f"Sweep complete: {total_runs} runs in {elapsed}s")

		return {
			"param1_name": param1_name,
			"param1_range": param1_range.tolist(),
			"param2_name": param2_name,
			"param2_range": param2_range.tolist(),
			"results": results_grid,
			"heatmaps": heatmaps,
			"total_runs": total_runs,
			"elapsed_seconds": elapsed,
		}

	def run_sensitivity(
		self,
		base_params: Dict[str, float],
		param_names: List[str],
		n_points: int = 5,
		seed: int = 42,
	) -> Dict[str, Any]:
		"""One-at-a-time 灵敏度分析。

		固定基准参数，每次只变化一个参数在其默认范围内均匀取 n_points 个值，
		收集所有指标用于灵敏度评估。

		Args:
			base_params: 基准参数字典
			param_names: 待分析的参数名列表
			n_points: 每个参数的采样点数
			seed: 随机种子

		Returns:
			{
				"base_params": Dict,
				"param_analyses": {
					param_name: {
						"values": list,
						"metrics": list of dicts,
					}
				},
				"sensitivity_scores": {param_name: {metric: float}},
				"total_runs": int,
				"elapsed_seconds": float,
			}
		"""
		t_start = time.time()
		total_runs = 0

		param_analyses: Dict[str, Dict[str, Any]] = {}
		sensitivity_scores: Dict[str, Dict[str, float]] = {}

		for param_name in param_names:
			info = SWEEP_PARAM_DEFAULTS.get(param_name, {})
			p_min = info.get("min", 0.0)
			p_max = info.get("max", 2.0)

			values = np.linspace(p_min, p_max, n_points)
			metrics_list: List[Dict[str, float]] = []

			logger.info(
				f"Sensitivity: {param_name} [{p_min:.2f} -> {p_max:.2f}], "
				f"{n_points} points"
			)

			for val in values:
				params = dict(base_params)
				params[param_name] = float(val)

				result = self.run_single(
					params, seed=seed, collect_snapshots=False,
				)
				metrics_list.append(result["metrics"])
				total_runs += 1

			param_analyses[param_name] = {
				"values": values.tolist(),
				"metrics": metrics_list,
			}

			# 计算灵敏度分数 (各指标的变化范围 / 参数变化范围)
			scores: Dict[str, float] = {}
			if metrics_list:
				for metric_key in metrics_list[0]:
					metric_vals = [m[metric_key] for m in metrics_list]
					metric_range = max(metric_vals) - min(metric_vals)
					param_range = p_max - p_min
					scores[metric_key] = (
						metric_range / param_range if param_range > 0 else 0.0
					)
			sensitivity_scores[param_name] = scores

		elapsed = round(time.time() - t_start, 3)
		logger.info(
			f"Sensitivity analysis complete: {total_runs} runs, "
			f"{len(param_names)} params, {elapsed}s"
		)

		return {
			"base_params": base_params,
			"param_analyses": param_analyses,
			"sensitivity_scores": sensitivity_scores,
			"total_runs": total_runs,
			"elapsed_seconds": elapsed,
		}
