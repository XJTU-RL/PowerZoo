# -*- coding: utf-8 -*-
"""
Stackelberg 压力测试运行器

支持负荷倍率、TOU 峰值电价、DR 强度等参数的扫描和灵敏度分析。
"""

import copy
import logging
from typing import Any, Dict, Optional

import numpy as np

from envs.render_common.engine.base_episode_runner import BaseEpisodeRunner
from envs.render_common.engine.base_stress_test_runner import BaseStressTestRunner
from envs.render_common.engine.episode_reader import EpisodeData
from envs.render_common.engine.inference_engine import InferenceEngine
from envs.stackelberg.render.engine.episode_runner import StackelbergEpisodeRunner

logger = logging.getLogger(__name__)


class StackelbergStressTestRunner(BaseStressTestRunner):
	"""Stackelberg 压力测试运行器

	支持 Stackelberg 博弈特有的参数扫描：
	- load_mult: 负荷倍率
	- tou_peak_price: TOU 峰值电价
	- dr_intensity: DR 信号强度
	- ess_capacity_mult: 储能容量倍率
	- pv_output_mult: PV 输出倍率
	- n_consumers: Consumer 数量

	Args:
		config: Stackelberg 环境配置字典
		inference_engine: 推理引擎
	"""

	SWEEP_DEFAULTS: Dict[str, Dict[str, Any]] = {
		"load_mult": {
			"min": 0.5,
			"max": 2.0,
			"default": 1.0,
			"label": "Load Multiplier",
			"step": 0.1,
		},
		"tou_peak_price": {
			"min": 0.05,
			"max": 0.50,
			"default": 0.15,
			"label": "TOU Peak Price ($/kWh)",
			"step": 0.05,
		},
		"dr_intensity": {
			"min": 0.0,
			"max": 1.0,
			"default": 0.5,
			"label": "DR Signal Intensity",
			"step": 0.1,
		},
		"ess_capacity_mult": {
			"min": 0.0,
			"max": 3.0,
			"default": 1.0,
			"label": "ESS Capacity Multiplier",
			"step": 0.25,
		},
		"pv_output_mult": {
			"min": 0.0,
			"max": 2.0,
			"default": 1.0,
			"label": "PV Output Multiplier",
			"step": 0.2,
		},
	}

	def _extract_metrics(self, episode_data: EpisodeData) -> Dict[str, float]:
		"""从 episode 数据中提取 Stackelberg 特有指标

		Args:
			episode_data: 完整 episode 数据

		Returns:
			指标字典
		"""
		metrics: Dict[str, float] = {
			"total_reward": episode_data.total_reward,
			"episode_length": float(episode_data.episode_length),
		}

		snapshots = episode_data.snapshots
		if not snapshots:
			return metrics

		# 电压指标
		v_mins = []
		v_maxs = []
		v_means = []
		for snap in snapshots:
			vs = snap.get("voltage_summary", {})
			if vs:
				v_mins.append(vs.get("v_min", 1.0))
				v_maxs.append(vs.get("v_max", 1.0))
				v_means.append(vs.get("v_mean", 1.0))

		if v_mins:
			metrics["v_min"] = float(np.min(v_mins))
			metrics["v_max"] = float(np.max(v_maxs))
			metrics["v_mean"] = float(np.mean(v_means))

		# UC 和 Consumer 奖励
		uc_rewards = []
		consumer_rewards = []
		for snap in snapshots:
			if "uc_reward" in snap:
				uc_rewards.append(snap["uc_reward"])
			if "avg_consumer_reward" in snap:
				consumer_rewards.append(snap["avg_consumer_reward"])

		if uc_rewards:
			metrics["uc_total_reward"] = float(np.sum(uc_rewards))
			metrics["uc_avg_reward"] = float(np.mean(uc_rewards))
		if consumer_rewards:
			metrics["consumer_avg_reward"] = float(np.mean(consumer_rewards))

		# 损耗
		losses = []
		for snap in snapshots:
			circuit = snap.get("circuit", {})
			loss = circuit.get("total_loss_kw", 0.0)
			if loss > 0:
				losses.append(loss)

		if losses:
			metrics["avg_loss_kw"] = float(np.mean(losses))
			metrics["max_loss_kw"] = float(np.max(losses))

		# 市场数据
		effective_prices = []
		for snap in snapshots:
			market = snap.get("market_data", {})
			uc_actions = market.get("uc_actions", {})
			ep = uc_actions.get("effective_price")
			if ep is not None:
				effective_prices.append(ep)

		if effective_prices:
			metrics["avg_effective_price"] = float(np.mean(effective_prices))
			metrics["price_volatility"] = float(np.std(effective_prices))

		return metrics

	def _create_episode_runner(self, config: Any) -> BaseEpisodeRunner:
		"""创建 Stackelberg episode runner

		Args:
			config: 修改后的环境配置

		Returns:
			StackelbergEpisodeRunner 实例
		"""
		return StackelbergEpisodeRunner(
			config=config,
			inference_engine=self.inference_engine,
			collect_snapshots=True,
		)

	def _apply_params_to_config(
		self, config: Any, params: Dict[str, float]
	) -> Any:
		"""将压力测试参数应用到 Stackelberg 配置

		Args:
			config: 环境配置副本
			params: 参数字典

		Returns:
			修改后的配置
		"""
		cfg = config if isinstance(config, dict) else {}
		env_args = cfg.get("env_args", cfg)

		if "load_mult" in params:
			env_args["load_multiplier"] = params["load_mult"]

		if "tou_peak_price" in params:
			tou_cfg = env_args.setdefault("tou_config", {})
			tou_cfg["on_peak_price"] = params["tou_peak_price"]

		if "dr_intensity" in params:
			env_args["dr_intensity"] = params["dr_intensity"]

		if "ess_capacity_mult" in params:
			env_args["ess_capacity_multiplier"] = params["ess_capacity_mult"]

		if "pv_output_mult" in params:
			env_args["pv_output_multiplier"] = params["pv_output_mult"]

		return cfg
