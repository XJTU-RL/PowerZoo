# -*- coding: utf-8 -*-
"""
VVC 压力测试运行器

继承 BaseStressTestRunner，支持 VVC 环境的参数扫描:
- load_mult: 负荷倍率
- pv_ratio: 光伏出力比例
"""

import copy
import logging
from typing import Any, Dict, Optional

import numpy as np

from envs.render_common.engine.base_episode_runner import BaseEpisodeRunner
from envs.render_common.engine.base_stress_test_runner import BaseStressTestRunner
from envs.render_common.engine.episode_reader import EpisodeData
from envs.render_common.engine.inference_engine import InferenceEngine
from envs.vvc.render.engine.episode_runner import VVCEpisodeRunner

logger = logging.getLogger(__name__)


class VVCStressTestRunner(BaseStressTestRunner):
	"""VVC 压力测试运行器

	对负荷倍率和光伏比例进行二维扫描，
	评估 VVC 控制策略在不同工况下的性能。

	Args:
		config: VVC 环境配置字典
		inference_engine: 推理引擎
	"""

	SWEEP_DEFAULTS: Dict[str, Dict[str, Any]] = {
		"load_mult": {
			"min": 0.5,
			"max": 2.0,
			"default": 1.0,
			"label": "Load Multiplier",
		},
		"pv_ratio": {
			"min": 0.0,
			"max": 2.0,
			"default": 1.0,
			"label": "PV Output Ratio",
		},
	}

	def __init__(
		self,
		config: Dict[str, Any],
		inference_engine: Optional[InferenceEngine] = None,
	):
		super().__init__(config, inference_engine)

	def _extract_metrics(self, episode_data: EpisodeData) -> Dict[str, float]:
		"""从 episode 数据中提取关键性能指标。

		Args:
			episode_data: 完整 episode 数据

		Returns:
			{total_reward, avg_voltage_pu, min_voltage_pu, max_voltage_pu,
			 avg_loss_kw, n_violations}
		"""
		metrics: Dict[str, float] = {
			"total_reward": episode_data.total_reward,
			"avg_voltage_pu": 1.0,
			"min_voltage_pu": 1.0,
			"max_voltage_pu": 1.0,
			"avg_loss_kw": 0.0,
			"n_violations": 0.0,
		}

		if not episode_data.snapshots:
			return metrics

		all_v_mean = []
		all_v_min = []
		all_v_max = []
		all_loss = []
		total_violations = 0

		for snap in episode_data.snapshots:
			circuit = snap.get("circuit", {})

			v_mean = circuit.get("v_mean_pu")
			if isinstance(v_mean, (int, float)):
				all_v_mean.append(v_mean)

			v_min = circuit.get("v_min_pu")
			if isinstance(v_min, (int, float)):
				all_v_min.append(v_min)
				if v_min < 0.95:
					total_violations += 1

			v_max = circuit.get("v_max_pu")
			if isinstance(v_max, (int, float)):
				all_v_max.append(v_max)
				if v_max > 1.05:
					total_violations += 1

			loss = circuit.get("total_loss_kw")
			if isinstance(loss, (int, float)):
				all_loss.append(loss)

		if all_v_mean:
			metrics["avg_voltage_pu"] = sum(all_v_mean) / len(all_v_mean)
		if all_v_min:
			metrics["min_voltage_pu"] = min(all_v_min)
		if all_v_max:
			metrics["max_voltage_pu"] = max(all_v_max)
		if all_loss:
			metrics["avg_loss_kw"] = sum(all_loss) / len(all_loss)
		metrics["n_violations"] = float(total_violations)

		return metrics

	def _create_episode_runner(self, config: Any) -> BaseEpisodeRunner:
		"""创建 VVC episode runner 实例。

		Args:
			config: 修改后的环境配置

		Returns:
			VVCEpisodeRunner 实例
		"""
		return VVCEpisodeRunner(
			config=config,
			inference_engine=self.inference_engine,
			collect_snapshots=True,
		)

	def _apply_params_to_config(
		self, config: Any, params: Dict[str, float]
	) -> Any:
		"""将压力测试参数应用到 VVC 配置。

		Args:
			config: VVC 配置副本
			params: {参数名: 值}

		Returns:
			修改后的配置
		"""
		cfg = copy.deepcopy(config) if not isinstance(config, dict) else dict(config)

		if "load_mult" in params:
			cfg["load_mult"] = params["load_mult"]

		if "pv_ratio" in params:
			cfg["pv_ratio"] = params["pv_ratio"]

		return cfg
