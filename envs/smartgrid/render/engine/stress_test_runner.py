"""
SmartGrid Stress Test Runner
SmartGrid 压力测试运行器

继承 BaseStressTestRunner，定义 SmartGrid 特定的
参数扫描范围和指标提取逻辑。
"""

import copy
import logging
from typing import Any, Dict, Optional

import numpy as np

from envs.render_common.engine.base_episode_runner import BaseEpisodeRunner
from envs.render_common.engine.base_stress_test_runner import BaseStressTestRunner
from envs.render_common.engine.episode_reader import EpisodeData
from envs.render_common.engine.inference_engine import InferenceEngine
from envs.smartgrid.render.engine.episode_runner import SmartGridEpisodeRunner

logger = logging.getLogger(__name__)


class SmartGridStressTestRunner(BaseStressTestRunner):
	"""SmartGrid 压力测试运行器

	Args:
		config: SmartGridConfig 或等效字典
		inference_engine: 推理引擎
		project_root: 项目根目录
	"""

	SWEEP_DEFAULTS: Dict[str, Dict[str, Any]] = {
		"load_mult": {
			"min": 0.5,
			"max": 2.0,
			"default": 1.0,
			"description": "Load multiplier (uniform scaling)",
		},
		"pv_ratio": {
			"min": 0.0,
			"max": 2.0,
			"default": 1.0,
			"description": "PV penetration ratio",
		},
		"lambda_init": {
			"min": 0.01,
			"max": 10.0,
			"default": 1.0,
			"description": "Initial Lagrangian lambda value",
		},
	}

	def __init__(
		self,
		config: Any,
		inference_engine: Optional[InferenceEngine] = None,
		project_root: str = "",
	):
		super().__init__(config, inference_engine)
		self.project_root = project_root

	def _extract_metrics(self, episode_data: EpisodeData) -> Dict[str, float]:
		"""从 episode 数据中提取关键指标

		Args:
			episode_data: 完整 episode 数据

		Returns:
			{指标名: 值}
		"""
		metrics: Dict[str, float] = {
			"total_reward": episode_data.total_reward,
			"episode_length": float(episode_data.episode_length),
		}

		snapshots = episode_data.snapshots
		if not snapshots:
			return metrics

		# 电压统计
		v_min_vals = []
		v_max_vals = []
		v_mean_vals = []
		loss_vals = []
		violation_vals = []

		for snap in snapshots:
			circuit = snap.get("circuit", {})
			v_min = circuit.get("v_min_pu")
			v_max = circuit.get("v_max_pu")
			v_mean = circuit.get("v_mean_pu")
			loss = circuit.get("total_loss_kw")

			if isinstance(v_min, (int, float)):
				v_min_vals.append(v_min)
			if isinstance(v_max, (int, float)):
				v_max_vals.append(v_max)
			if isinstance(v_mean, (int, float)):
				v_mean_vals.append(v_mean)
			if isinstance(loss, (int, float)):
				loss_vals.append(loss)

			info = snap.get("info", {})
			viol = info.get("voltage_violation_rate_buses")
			if isinstance(viol, (int, float)):
				violation_vals.append(viol)

		if v_min_vals:
			metrics["min_voltage_pu"] = float(np.min(v_min_vals))
		if v_max_vals:
			metrics["max_voltage_pu"] = float(np.max(v_max_vals))
		if v_mean_vals:
			metrics["avg_voltage_pu"] = float(np.mean(v_mean_vals))
		if loss_vals:
			metrics["avg_loss_kw"] = float(np.mean(loss_vals))
			metrics["total_loss_kwh"] = float(np.sum(loss_vals))
		if violation_vals:
			metrics["avg_violation_rate"] = float(np.mean(violation_vals))

		return metrics

	def _create_episode_runner(self, config: Any) -> BaseEpisodeRunner:
		"""创建 episode runner 实例

		Args:
			config: 修改后的环境配置

		Returns:
			SmartGridEpisodeRunner 实例
		"""
		return SmartGridEpisodeRunner(
			config=config,
			inference_engine=self.inference_engine,
			collect_snapshots=True,
			project_root=self.project_root,
		)

	def _apply_params_to_config(
		self, config: Any, params: Dict[str, float],
	) -> Any:
		"""将压力测试参数应用到环境配置

		Args:
			config: 环境配置副本
			params: {参数名: 值}

		Returns:
			修改后的配置
		"""
		from envs.smartgrid.base_env.env_config import SmartGridConfig

		if isinstance(config, SmartGridConfig):
			cfg = copy.deepcopy(config)
		elif isinstance(config, dict):
			cfg = dict(config)
		else:
			cfg = copy.deepcopy(config)

		for param_name, value in params.items():
			if param_name == "load_mult":
				if isinstance(cfg, SmartGridConfig):
					cfg.scale = float(value)
				elif isinstance(cfg, dict):
					cfg["scale"] = float(value)

			elif param_name == "lambda_init":
				if isinstance(cfg, SmartGridConfig):
					pass  # lambda_init 不在 SmartGridConfig 中
				elif isinstance(cfg, dict):
					cfg["lambda_init"] = float(value)

			elif param_name == "pv_ratio":
				# PV 比例通过 scale 间接影响
				pass

		return cfg
