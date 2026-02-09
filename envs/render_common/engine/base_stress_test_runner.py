"""
Base Stress Test Runner (Abstract)
压力测试运行器抽象基类

提供参数扫描 + 批量推理的标准流程，
子类需定义 SWEEP_DEFAULTS 和 _extract_metrics()。
"""

import copy
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from envs.render_common.engine.base_episode_runner import BaseEpisodeRunner
from envs.render_common.engine.episode_reader import EpisodeData
from envs.render_common.engine.inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


class BaseStressTestRunner(ABC):
	"""压力测试运行器抽象基类

	子类需:
	1. 定义 SWEEP_DEFAULTS: Dict[str, Dict[str, Any]] — 参数范围
	2. 实现 _extract_metrics(episode_data) -> Dict[str, float]
	3. 实现 _create_episode_runner(config) -> BaseEpisodeRunner
	4. 实现 _apply_params_to_config(config, params) -> config

	Args:
		config: 环境配置对象
		inference_engine: 推理引擎
	"""

	# 子类覆写: 支持的压力测试参数及其默认范围
	SWEEP_DEFAULTS: Dict[str, Dict[str, Any]] = {}

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
		"""运行单个场景

		Args:
			params: 参数字典
			seed: 随机种子
			collect_snapshots: 是否收集快照

		Returns:
			结果字典 {params, metrics, elapsed_seconds, ...}
		"""
		t_start = time.time()
		self._run_count += 1

		cfg = self._apply_params_to_config(
			copy.deepcopy(self.base_config), params
		)
		runner = self._create_episode_runner(cfg)

		try:
			episode_data = runner.run_episode(seed=seed)
			metrics = self._extract_metrics(episode_data)
		finally:
			runner.close()

		elapsed = round(time.time() - t_start, 3)

		logger.info(
			f"Stress test #{self._run_count}: params={params}, "
			f"reward={metrics.get('total_reward', 0):.2f}, "
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
		"""二维参数扫描

		Args:
			param1_name: 第一个参数名 (行)
			param1_range: 第一个参数取值范围
			param2_name: 第二个参数名 (列)
			param2_range: 第二个参数取值范围
			seed: 随机种子
			base_params: 基准参数

		Returns:
			扫描结果字典
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

			results_grid.append(row)

		heatmaps: Dict[str, np.ndarray] = {}
		for metric_name in metric_names:
			matrix = np.zeros((n1, n2))
			for i in range(n1):
				for j in range(n2):
					matrix[i, j] = results_grid[i][j].get(metric_name, 0.0)
			heatmaps[metric_name] = matrix

		elapsed = round(time.time() - t_start, 3)

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
		"""One-at-a-time 灵敏度分析

		Args:
			base_params: 基准参数字典
			param_names: 待分析的参数名列表
			n_points: 每个参数的采样点数
			seed: 随机种子

		Returns:
			灵敏度分析结果字典
		"""
		t_start = time.time()
		total_runs = 0

		param_analyses: Dict[str, Dict[str, Any]] = {}
		sensitivity_scores: Dict[str, Dict[str, float]] = {}

		for param_name in param_names:
			info = self.SWEEP_DEFAULTS.get(param_name, {})
			p_min = info.get("min", 0.0)
			p_max = info.get("max", 2.0)

			values = np.linspace(p_min, p_max, n_points)
			metrics_list: List[Dict[str, float]] = []

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

		return {
			"base_params": base_params,
			"param_analyses": param_analyses,
			"sensitivity_scores": sensitivity_scores,
			"total_runs": total_runs,
			"elapsed_seconds": elapsed,
		}

	# ------------------------------------------------------------------
	# 抽象方法 (子类必须实现)
	# ------------------------------------------------------------------

	@abstractmethod
	def _extract_metrics(self, episode_data: EpisodeData) -> Dict[str, float]:
		"""从 episode 数据中提取关键指标

		Args:
			episode_data: 完整 episode 数据

		Returns:
			{指标名: 值}
		"""
		...

	@abstractmethod
	def _create_episode_runner(self, config: Any) -> BaseEpisodeRunner:
		"""创建 episode runner 实例

		Args:
			config: 修改后的环境配置

		Returns:
			EpisodeRunner 实例
		"""
		...

	@abstractmethod
	def _apply_params_to_config(
		self, config: Any, params: Dict[str, float]
	) -> Any:
		"""将压力测试参数应用到环境配置

		Args:
			config: 环境配置副本
			params: {参数名: 值}

		Returns:
			修改后的配置
		"""
		...
