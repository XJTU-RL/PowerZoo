# -*- coding: utf-8 -*-
"""
系统级数据提取器 -- 从 OpenDSS 提取全局电路统计信息。
聚合损耗、负荷、发电、收敛状态、电压越限等系统级指标。
"""

import logging
import math
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class CircuitDataExtractor:
	"""系统级数据提取器

	提取整个电路的聚合信息，不涉及逐元件明细（那些由专用提取器负责）。

	参数:
		dss_engine: dss-python 的 DSS 全局单例
	"""

	def __init__(self, dss_engine) -> None:
		self.dss = dss_engine

	# ------------------------------------------------------------------
	# 公开接口
	# ------------------------------------------------------------------

	def extract_all(self) -> Dict[str, Any]:
		"""提取系统级数据，包含损耗/负荷/发电/网络规模/求解状态/电压统计/线路汇总"""
		ckt = self.dss.ActiveCircuit
		if ckt is None:
			logger.warning("ActiveCircuit 为 None，无法提取系统数据")
			return self._empty_result()

		result: Dict[str, Any] = {}

		# -- 损耗 --
		loss_kw, loss_kvar = self._get_total_losses(ckt)
		result["total_loss_kw"] = loss_kw
		result["total_loss_kvar"] = loss_kvar

		# -- 负荷 --
		load_kw, load_kvar = self._get_total_load(ckt)
		result["total_load_kw"] = load_kw
		result["total_load_kvar"] = load_kvar

		# -- 发电/电源 --
		gen_kw, gen_kvar = self._get_total_generation(ckt)
		result["total_gen_kw"] = gen_kw
		result["total_gen_kvar"] = gen_kvar

		# -- 分布式设备汇总 --
		result["total_pv_kw"] = self._get_total_pv_kw(ckt)
		result["total_storage_kw"] = self._get_total_storage_kw(ckt)

		# -- 网络规模 --
		result["n_buses"] = int(ckt.NumBuses)
		result["n_nodes"] = int(ckt.NumNodes)
		result.update(self._count_elements(ckt))

		# -- 求解状态 --
		sol = ckt.Solution
		result["converged"] = bool(sol.Converged)
		result["iterations"] = int(sol.Iterations)
		result["frequency"] = float(sol.Frequency)
		result["base_frequency"] = self._safe_float(
			sol, "DefaultBaseFreq", 50.0
		)

		# -- 电压统计 --
		result.update(self._compute_voltage_stats(ckt))

		# -- 线路汇总 --
		result.update(self._compute_line_summary(ckt))

		return result

	@staticmethod
	def _get_total_losses(ckt) -> tuple:
		"""获取系统总损耗 -> (loss_kw, loss_kvar)"""
		losses = ckt.Losses
		try:
			if hasattr(losses, "__len__") and len(losses) > 0:
				real_w = float(losses[0]) if len(losses) > 0 else 0.0
				imag_w = float(losses[1]) if len(losses) > 1 else 0.0
			else:
				real_w = float(losses.real)
				imag_w = float(losses.imag)
		except Exception:
			real_w, imag_w = 0.0, 0.0
		return abs(real_w) / 1000.0, abs(imag_w) / 1000.0

	@staticmethod
	def _get_total_load(ckt) -> tuple:
		"""遍历所有 Load 元素求和 -> (total_kw, total_kvar)"""
		total_kw = 0.0
		total_kvar = 0.0
		loads = ckt.Loads
		if loads.First != 0:
			while True:
				total_kw += abs(float(loads.kW))
				total_kvar += abs(float(loads.kvar))
				if loads.Next == 0:
					break
		return total_kw, total_kvar

	@staticmethod
	def _get_total_generation(ckt) -> tuple:
		"""从 TotalPower 获取电源注入功率 -> (gen_kw, gen_kvar)"""
		try:
			power = ckt.TotalPower
			if hasattr(power, "__len__") and len(power) > 0:
				real_kw = float(power[0]) if len(power) > 0 else 0.0
				imag_kvar = float(power[1]) if len(power) > 1 else 0.0
			else:
				real_kw = float(power.real)
				imag_kvar = float(power.imag)
		except Exception:
			real_kw, imag_kvar = 0.0, 0.0
		return abs(real_kw), abs(imag_kvar)

	def _get_total_pv_kw(self, ckt) -> float:
		"""汇总所有 PV 有功出力 (kW)"""
		total = 0.0
		pv = ckt.PVSystems
		if pv.First == 0:
			return total

		while True:
			try:
				name = pv.Name
				self.dss.ActiveCircuit.SetActiveElement(
					f"PVSystem.{name}"
				)
				powers = self._safe_list(
					self.dss.ActiveCircuit.ActiveElement.Powers
				)
				# PV 发电为负值，取绝对值
				kw = abs(sum(powers[i] for i in range(0, len(powers), 2)))
				total += kw
			except Exception:
				pass
			if pv.Next == 0:
				break

		return total

	def _get_total_storage_kw(self, ckt) -> float:
		"""汇总所有储能功率 (kW)，充电为负、放电为正"""
		total = 0.0
		try:
			ckt.SetActiveClass("Storage")
			ac = ckt.ActiveClass
			if ac.First == 0:
				return total

			while True:
				try:
					name = ac.Name
					self.dss.ActiveCircuit.SetActiveElement(
						f"Storage.{name}"
					)
					powers = self._safe_list(
						self.dss.ActiveCircuit.ActiveElement.Powers
					)
					kw = sum(powers[i] for i in range(0, len(powers), 2))
					total += kw
				except Exception:
					pass
				if ac.Next == 0:
					break
		except Exception:
			pass

		return total

	def _count_elements(self, ckt) -> Dict[str, int]:
		"""统计各类元件数量"""
		return {
			"n_lines": self._count_class(ckt, "Line"),
			"n_transformers": self._count_class(ckt, "Transformer"),
			"n_loads": self._count_class(ckt, "Load"),
			"n_pv": self._count_class(ckt, "PVSystem"),
			"n_storage": self._count_class(ckt, "Storage"),
		}

	@staticmethod
	def _count_class(ckt, class_name: str) -> int:
		"""统计指定 DSS 类别的元素数量"""
		try:
			ckt.SetActiveClass(class_name)
			count = 0
			if ckt.ActiveClass.First != 0:
				count = 1
				while ckt.ActiveClass.Next != 0:
					count += 1
			return count
		except Exception:
			return 0

	@staticmethod
	def _compute_voltage_stats(ckt) -> Dict[str, Any]:
		"""计算全网电压统计 (ANSI C84.1: 0.95~1.05 pu)"""
		_empty = {
			"v_min_pu": 0.0, "v_max_pu": 0.0, "v_mean_pu": 0.0,
			"voltage_violation_count": 0, "voltage_violation_pct": 0.0,
		}
		try:
			all_vmag = list(ckt.AllBusVmagPu)
		except Exception:
			return _empty
		if not all_vmag:
			return _empty
		# 过滤零值（未连接节点）
		valid = [v for v in all_vmag if v > 0.01]
		if not valid:
			return _empty

		arr = np.array(valid, dtype=np.float64)
		violations = int(np.sum((arr < 0.95) | (arr > 1.05)))

		return {
			"v_min_pu": float(np.min(arr)),
			"v_max_pu": float(np.max(arr)),
			"v_mean_pu": float(np.mean(arr)),
			"voltage_violation_count": violations,
			"voltage_violation_pct": violations / len(arr) * 100.0,
		}

	def _compute_line_summary(self, ckt) -> Dict[str, float]:
		"""计算线路损耗和负载率汇总"""
		total_loss = 0.0
		max_loading = 0.0

		lines = ckt.Lines
		if lines.First == 0:
			return {
				"total_line_loss_kw": 0.0,
				"max_line_loading_pct": 0.0,
			}

		while True:
			try:
				name = lines.Name
				normal_amps = float(lines.NormalAmps)

				self.dss.ActiveCircuit.SetActiveElement(f"Line.{name}")
				elem = self.dss.ActiveCircuit.ActiveElement

				# 损耗
				losses = self._safe_list(elem.Losses)
				if losses and len(losses) >= 2:
					total_loss += abs(float(losses[0])) / 1000.0

				# 负载率
				if normal_amps > 1e-6:
					currents = self._safe_list(elem.Currents)
					i_max = self._max_current_magnitude(currents)
					loading = i_max / normal_amps * 100.0
					if loading > max_loading:
						max_loading = loading
			except Exception:
				pass

			if lines.Next == 0:
				break

		return {
			"total_line_loss_kw": total_loss,
			"max_line_loading_pct": max_loading,
		}

	@staticmethod
	def _max_current_magnitude(currents_raw: List[float]) -> float:
		"""从复数电流数组中求最大幅值 (A)"""
		if not currents_raw or len(currents_raw) < 2:
			return 0.0
		mags = [
			math.sqrt(currents_raw[2 * i] ** 2 + currents_raw[2 * i + 1] ** 2)
			for i in range(len(currents_raw) // 2)
		]
		return max(mags) if mags else 0.0

	@staticmethod
	def _safe_float(obj, attr: str, default: float) -> float:
		"""安全获取对象的浮点属性"""
		try:
			return float(getattr(obj, attr, default))
		except (TypeError, ValueError):
			return default

	@staticmethod
	def _safe_list(obj) -> List:
		"""安全地将 DSS 返回值转为 list"""
		if obj is None:
			return []
		try:
			return list(obj)
		except (TypeError, ValueError):
			return []

	@staticmethod
	def _empty_result() -> Dict[str, Any]:
		"""返回空的系统数据（全零）"""
		return {
			"total_loss_kw": 0.0,
			"total_loss_kvar": 0.0,
			"total_load_kw": 0.0,
			"total_load_kvar": 0.0,
			"total_gen_kw": 0.0,
			"total_gen_kvar": 0.0,
			"total_pv_kw": 0.0,
			"total_storage_kw": 0.0,
			"n_buses": 0,
			"n_nodes": 0,
			"n_lines": 0,
			"n_transformers": 0,
			"n_loads": 0,
			"n_pv": 0,
			"n_storage": 0,
			"converged": False,
			"iterations": 0,
			"frequency": 0.0,
			"base_frequency": 50.0,
			"v_min_pu": 0.0,
			"v_max_pu": 0.0,
			"v_mean_pu": 0.0,
			"voltage_violation_count": 0,
			"voltage_violation_pct": 0.0,
			"total_line_loss_kw": 0.0,
			"max_line_loading_pct": 0.0,
		}
