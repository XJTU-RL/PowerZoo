"""
DSR Circuit Data Extractor
电路级数据提取器

从 DSR 环境中提取系统级汇总数据：总损耗、总负荷、发电、
潮流收敛状态等。与 VVC/SmartGrid 的 circuit extractor 类似，
但增加了 DSR 特有的恢复率和故障状态指标。
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CircuitDataExtractor:
	"""DSR 电路级数据提取器

	提取系统级的汇总指标，包括损耗、负荷、发电、
	以及 DSR 特有的恢复状态指标。

	Args:
		env: DSREnv 实例
	"""

	def __init__(self, env: Any):
		self.env = env
		self._core = getattr(env, "core_env", None) or getattr(env, "dsr_core", None)

	def extract(self) -> Dict[str, Any]:
		"""提取电路级汇总数据

		Returns:
			{total_loss_kw, total_load_kw, total_gen_kw, total_pv_kw,
			 v_mean_pu, v_min_pu, v_max_pu, converged,
			 restoration_pct, n_faults, n_energized_buses, ...} 汇总字典
		"""
		circuit_data: Dict[str, Any] = {
			"total_loss_kw": 0.0,
			"total_loss_kvar": 0.0,
			"total_load_kw": 0.0,
			"total_load_kvar": 0.0,
			"total_gen_kw": 0.0,
			"total_gen_kvar": 0.0,
			"total_pv_kw": 0.0,
			"total_storage_kw": 0.0,
			"converged": True,
			"v_mean_pu": 1.0,
			"v_min_pu": 1.0,
			"v_max_pu": 1.0,
		}

		# 从 OpenDSS 获取电路数据
		dss_data = self._extract_from_opendss()
		if dss_data:
			circuit_data.update(dss_data)

		# 添加 DSR 特有指标
		circuit_data.update(self._extract_dsr_metrics())

		return circuit_data

	def _extract_from_opendss(self) -> Dict[str, Any]:
		"""从 OpenDSS 提取电路数据"""
		data: Dict[str, Any] = {}

		try:
			circuit = getattr(self._core, "circuit", None)
			if circuit is None:
				return data
			dss = getattr(circuit, "dss", None)
			if dss is None:
				return data

			# 总损耗
			losses = dss.ActiveCircuit.Losses
			if losses is not None and len(losses) >= 2:
				data["total_loss_kw"] = float(losses[0]) / 1000.0
				data["total_loss_kvar"] = float(losses[1]) / 1000.0

			# 总负荷
			data["total_load_kw"] = float(dss.ActiveCircuit.TotalPower[0]) * -1
			data["total_load_kvar"] = float(dss.ActiveCircuit.TotalPower[1]) * -1

			# 收敛状态
			data["converged"] = bool(dss.ActiveCircuit.Solution.Converged)

			# 电压统计
			voltages = []
			for bus_name in dss.ActiveCircuit.AllBusNames:
				dss.ActiveCircuit.SetActiveBus(bus_name)
				v_pu = dss.ActiveCircuit.Buses.puVmagAngle
				if v_pu is not None and len(v_pu) >= 2:
					# 取幅值（偶数索引）
					for i in range(0, len(v_pu), 2):
						v = float(v_pu[i])
						if 0.1 < v < 2.0:  # 过滤异常值
							voltages.append(v)

			if voltages:
				data["v_mean_pu"] = float(np.mean(voltages))
				data["v_min_pu"] = float(np.min(voltages))
				data["v_max_pu"] = float(np.max(voltages))

			# PV 发电
			total_pv = 0.0
			try:
				pv_names = dss.ActiveCircuit.PVSystems.AllNames
				if pv_names:
					for pv_name in pv_names:
						dss.ActiveCircuit.SetActiveElement(f"PVSystem.{pv_name}")
						powers = dss.ActiveCircuit.ActiveElement.Powers
						if powers is not None and len(powers) >= 2:
							total_pv += abs(float(powers[0]))
			except Exception:
				pass
			data["total_pv_kw"] = total_pv

			# 发电机
			total_gen = 0.0
			total_gen_kvar = 0.0
			try:
				gen_names = dss.ActiveCircuit.Generators.AllNames
				if gen_names:
					for gen_name in gen_names:
						dss.ActiveCircuit.SetActiveElement(f"Generator.{gen_name}")
						powers = dss.ActiveCircuit.ActiveElement.Powers
						if powers is not None and len(powers) >= 2:
							total_gen += abs(float(powers[0]))
							total_gen_kvar += abs(float(powers[1]))
			except Exception:
				pass
			data["total_gen_kw"] = total_gen
			data["total_gen_kvar"] = total_gen_kvar

		except Exception as exc:
			logger.debug(f"OpenDSS extraction failed: {exc}")

		return data

	def _extract_dsr_metrics(self) -> Dict[str, Any]:
		"""提取 DSR 特有的恢复和故障指标"""
		metrics: Dict[str, Any] = {
			"restoration_pct": 0.0,
			"n_faults": 0,
			"n_energized_buses": 0,
			"n_total_buses": 0,
			"n_overloaded_lines": 0,
			"current_step": 0,
			"max_steps": 20,
		}

		if self._core is None:
			return metrics

		# 恢复率
		try:
			ratio = self._core._get_restored_load_ratio()
			metrics["restoration_pct"] = float(ratio) * 100.0
		except Exception:
			pass

		# 故障数
		fault_lines = getattr(self._core, "fault_lines", [])
		metrics["n_faults"] = len(fault_lines)

		# 带电母线数
		try:
			obs = self._core._get_observations()
			energized = obs.get("energized_buses", [])
			metrics["n_energized_buses"] = len(energized)
		except Exception:
			pass

		metrics["n_total_buses"] = getattr(self._core, "n_bus", 0)

		# 过载线路数
		try:
			overloads = self._core._get_line_overloads()
			metrics["n_overloaded_lines"] = int(overloads)
		except Exception:
			pass

		# 步数信息
		metrics["current_step"] = getattr(self._core, "current_step", 0)
		config = getattr(self._core, "config", None)
		if config is not None:
			metrics["max_steps"] = getattr(config, "max_episode_steps", 20)

		return metrics
