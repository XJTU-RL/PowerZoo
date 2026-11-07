# -*- coding: utf-8 -*-
"""
电路级数据提取器

从 OpenDSS 引擎提取系统级汇总数据：总损耗、总负荷、
总发电、收敛状态等。
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class CircuitDataExtractor:
	"""电路级数据提取器

	提取电路整体运行状态和汇总指标。

	Args:
		dss: dss-python DSS 引擎实例
	"""

	def __init__(self, dss: Any):
		self._dss = dss

	def extract(self) -> Dict[str, Any]:
		"""提取电路级汇总数据。

		Returns:
			{total_loss_kw, total_loss_kvar, total_load_kw, total_load_kvar,
			 total_gen_kw, total_gen_kvar, total_pv_kw, total_storage_kw,
			 converged, v_mean_pu, v_min_pu, v_max_pu, n_buses, n_elements}
		"""
		result: Dict[str, Any] = {
			"total_loss_kw": 0.0,
			"total_loss_kvar": 0.0,
			"total_load_kw": 0.0,
			"total_load_kvar": 0.0,
			"total_gen_kw": 0.0,
			"total_gen_kvar": 0.0,
			"total_pv_kw": 0.0,
			"total_storage_kw": 0.0,
			"converged": False,
			"v_mean_pu": 1.0,
			"v_min_pu": 1.0,
			"v_max_pu": 1.0,
			"n_buses": 0,
			"n_elements": 0,
		}

		try:
			# 损耗
			losses = self._dss.Circuit.Losses()
			result["total_loss_kw"] = losses[0] / 1000.0
			result["total_loss_kvar"] = losses[1] / 1000.0

			# 负荷
			result["total_load_kw"] = self._get_total_load_kw()
			result["total_load_kvar"] = self._get_total_load_kvar()

			# PV
			result["total_pv_kw"] = self._get_total_pv_kw()

			# 储能
			result["total_storage_kw"] = self._get_total_storage_kw()

			# 收敛
			result["converged"] = bool(self._dss.Solution.Converged())

			# 电压统计
			v_stats = self._get_voltage_stats()
			result.update(v_stats)

			# 元素和母线计数
			result["n_buses"] = self._dss.Circuit.NumBuses()
			result["n_elements"] = self._dss.Circuit.NumCktElements()

		except Exception as exc:
			logger.warning(f"Circuit data extraction failed: {exc}")

		return result

	def _get_total_load_kw(self) -> float:
		"""获取系统总有功负荷。"""
		total = 0.0
		try:
			self._dss.Loads.First()
			while True:
				total += self._dss.Loads.kW()
				if not self._dss.Loads.Next():
					break
		except Exception:
			pass
		return total

	def _get_total_load_kvar(self) -> float:
		"""获取系统总无功负荷。"""
		total = 0.0
		try:
			self._dss.Loads.First()
			while True:
				total += self._dss.Loads.kvar()
				if not self._dss.Loads.Next():
					break
		except Exception:
			pass
		return total

	def _get_total_pv_kw(self) -> float:
		"""获取 PV 总输出。"""
		total = 0.0
		try:
			self._dss.PVsystems.First()
			while True:
				self._dss.Circuit.SetActiveElement(f"PVsystem.{self._dss.PVsystems.Name()}")
				powers = self._dss.CktElement.Powers()
				if powers:
					total += abs(powers[0])
				if not self._dss.PVsystems.Next():
					break
		except Exception:
			pass
		return total

	def _get_total_storage_kw(self) -> float:
		"""获取储能总功率。"""
		total = 0.0
		try:
			self._dss.Storages.First()
			while True:
				self._dss.Circuit.SetActiveElement(f"Storage.{self._dss.Storages.Name()}")
				powers = self._dss.CktElement.Powers()
				if powers:
					total += powers[0]
				if not self._dss.Storages.Next():
					break
		except Exception:
			pass
		return total

	def _get_voltage_stats(self) -> Dict[str, float]:
		"""获取系统电压统计。"""
		all_v = []
		try:
			bus_names = self._dss.Circuit.AllBusNames()
			for bus in bus_names:
				self._dss.Circuit.SetActiveBus(bus)
				v_pu = list(self._dss.Bus.puVmagAngle()[::2])
				n_ph = self._dss.Bus.NumNodes()
				all_v.extend(v_pu[:n_ph])
		except Exception:
			pass

		if not all_v:
			return {"v_mean_pu": 1.0, "v_min_pu": 1.0, "v_max_pu": 1.0}

		return {
			"v_mean_pu": sum(all_v) / len(all_v),
			"v_min_pu": min(all_v),
			"v_max_pu": max(all_v),
		}
