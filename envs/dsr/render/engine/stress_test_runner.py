"""
DSR Stress Test Runner
DSR 压力测试运行器

继承 BaseStressTestRunner，定义 DSR 特有的扫描参数:
- n_faults: 故障数量
- fault_severity: 故障严重程度
- load_priority_ratio: 优先级负荷比例
"""

import copy
import logging
from typing import Any, Dict, Optional

import numpy as np

from envs.render_common.engine.base_stress_test_runner import BaseStressTestRunner
from envs.render_common.engine.base_episode_runner import BaseEpisodeRunner
from envs.render_common.engine.episode_reader import EpisodeData
from envs.render_common.engine.inference_engine import InferenceEngine
from envs.dsr.render.engine.episode_runner import DSREpisodeRunner

logger = logging.getLogger(__name__)


class DSRStressTestRunner(BaseStressTestRunner):
	"""DSR 压力测试运行器

	通过扫描故障数量、严重程度、负荷优先级等参数，
	评估恢复策略在不同场景下的鲁棒性。

	Args:
		config: DSR 环境配置字典
		inference_engine: 推理引擎
	"""

	SWEEP_DEFAULTS = {
		"n_faults": {
			"min": 1.0,
			"max": 5.0,
			"default": 2.0,
			"step": 1.0,
			"label": "Number of Faults",
			"description": "Number of simultaneous line faults",
		},
		"fault_severity": {
			"min": 0.1,
			"max": 1.0,
			"default": 0.5,
			"step": 0.1,
			"label": "Fault Severity",
			"description": "Severity of faults (0=minor, 1=critical)",
		},
		"load_priority_ratio": {
			"min": 0.0,
			"max": 1.0,
			"default": 0.5,
			"step": 0.1,
			"label": "High-Priority Load Ratio",
			"description": "Ratio of high-priority loads in the system",
		},
		"max_episode_steps": {
			"min": 5.0,
			"max": 30.0,
			"default": 20.0,
			"step": 5.0,
			"label": "Max Steps",
			"description": "Maximum episode length",
		},
		"load_noise": {
			"min": 0.0,
			"max": 0.5,
			"default": 0.1,
			"step": 0.05,
			"label": "Load Noise",
			"description": "Random noise added to load values",
		},
	}

	def __init__(
		self,
		config: Dict[str, Any],
		inference_engine: Optional[InferenceEngine] = None,
	):
		super().__init__(config, inference_engine)

	def _extract_metrics(self, episode_data: EpisodeData) -> Dict[str, float]:
		"""从 episode 数据中提取 DSR 关键指标

		Args:
			episode_data: 完整 episode 数据

		Returns:
			{指标名: 值}
		"""
		metrics: Dict[str, float] = {
			"total_reward": episode_data.total_reward,
			"episode_length": float(episode_data.episode_length),
			"final_restoration_pct": 0.0,
			"avg_restoration_pct": 0.0,
			"max_restoration_pct": 0.0,
			"n_voltage_violations": 0.0,
			"n_overloads": 0.0,
			"restoration_speed": 0.0,
		}

		if not episode_data.snapshots:
			return metrics

		restoration_pcts = []
		total_violations = 0
		total_overloads = 0

		for snap in episode_data.snapshots:
			restoration = snap.get("restoration_data", {})
			pct = restoration.get("restoration_pct", 0.0)
			restoration_pcts.append(pct)

			circuit = snap.get("circuit", {})
			total_overloads += circuit.get("n_overloaded_lines", 0)

			# 电压违规
			buses = snap.get("buses", {})
			for bus_data in buses.values():
				if bus_data.get("is_energized", False):
					for v in bus_data.get("v_mag_pu", []):
						if v < 0.95 or v > 1.05:
							total_violations += 1

		if restoration_pcts:
			metrics["final_restoration_pct"] = restoration_pcts[-1]
			metrics["avg_restoration_pct"] = float(np.mean(restoration_pcts))
			metrics["max_restoration_pct"] = float(np.max(restoration_pcts))

			# 恢复速度 = 最终恢复率 / 步数
			if episode_data.episode_length > 0:
				metrics["restoration_speed"] = (
					metrics["final_restoration_pct"] / episode_data.episode_length
				)

		metrics["n_voltage_violations"] = float(total_violations)
		metrics["n_overloads"] = float(total_overloads)

		return metrics

	def _create_episode_runner(self, config: Dict[str, Any]) -> BaseEpisodeRunner:
		"""创建 episode runner 实例

		Args:
			config: 修改后的环境配置

		Returns:
			DSREpisodeRunner 实例
		"""
		return DSREpisodeRunner(
			config=config,
			inference_engine=self.inference_engine,
			collect_snapshots=True,
		)

	def _apply_params_to_config(
		self, config: Dict[str, Any], params: Dict[str, float]
	) -> Dict[str, Any]:
		"""将压力测试参数应用到环境配置

		Args:
			config: 环境配置副本
			params: {参数名: 值}

		Returns:
			修改后的配置
		"""
		env_args = config.setdefault("env_args", {})

		for param_name, value in params.items():
			if param_name == "n_faults":
				n = int(value)
				config["min_faults"] = n
				config["max_faults"] = n
				env_args["min_faults"] = n
				env_args["max_faults"] = n

			elif param_name == "fault_severity":
				# 严重程度映射到最大可故障线路数
				max_faultable = int(value * 10)
				config["max_faultable_lines"] = max(max_faultable, 1)
				env_args["max_faultable_lines"] = max(max_faultable, 1)

			elif param_name == "load_priority_ratio":
				# 调整优先级权重
				config["priority_weights"] = {
					"critical": value * 0.4,
					"high": value * 0.3,
					"medium": (1 - value) * 0.5,
					"low": (1 - value) * 0.5,
				}

			elif param_name == "max_episode_steps":
				config["max_episode_steps"] = int(value)
				env_args["max_episode_steps"] = int(value)

			elif param_name == "load_noise":
				config["load_noise"] = float(value)
				env_args["load_noise"] = float(value)

			else:
				config[param_name] = value
				env_args[param_name] = value

		return config
