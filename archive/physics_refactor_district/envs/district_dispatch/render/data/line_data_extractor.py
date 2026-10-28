# -*- coding: utf-8 -*-
"""
线路数据提取器 -- 从 OpenDSS 提取所有 Line 元素属性。

包含线路功率、电流、损耗、负载率、拓扑连接等完整信息。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LineDataExtractor:
	"""线路数据提取器

	通过 ActiveCircuit.Lines 迭代器 + SetActiveElement 获取线路运行数据。

	参数:
		dss_engine: dss-python 的 DSS 全局单例
	"""

	def __init__(self, dss_engine) -> None:
		self.dss = dss_engine

	# ------------------------------------------------------------------
	# 公开接口
	# ------------------------------------------------------------------

	def extract_all(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有线路数据

		返回:
			{line_name: {
				from_bus:     str,
				to_bus:       str,
				n_phases:     int,
				length_km:    float,    -- 线路长度 (km)
				p_from_kw:    float,    -- 送端有功 (kW)
				q_from_kvar:  float,    -- 送端无功 (kvar)
				p_to_kw:      float,    -- 受端有功 (kW)
				q_to_kvar:    float,    -- 受端无功 (kvar)
				loss_kw:      float,    -- 有功损耗 (kW)
				loss_kvar:    float,    -- 无功损耗 (kvar)
				i_from_a:     List[float],  -- 送端各相电流幅值 (A)
				i_to_a:       List[float],  -- 受端各相电流幅值 (A)
				i_max_a:      float,    -- 最大相电流 (A)
				normal_amps:  float,    -- 正常载流量 (A)
				emerg_amps:   float,    -- 紧急载流量 (A)
				loading_pct:  float,    -- 负载率 (%)
				enabled:      bool,
			}}
		"""
		ckt = self.dss.ActiveCircuit
		if ckt is None:
			logger.warning("ActiveCircuit 为 None，无法提取线路数据")
			return {}

		lines = ckt.Lines
		result: Dict[str, Dict[str, Any]] = {}

		if lines.First == 0:
			return result

		while True:
			try:
				data = self._extract_current_line(lines)
				if data is not None:
					result[data.pop("_name")] = data
			except Exception as exc:
				logger.warning(f"提取线路数据异常: {exc}")

			if lines.Next == 0:
				break

		return result

	def get_loss_summary(self) -> Dict[str, float]:
		"""获取全网线路损耗摘要

		返回:
			{total_loss_kw, total_loss_kvar, max_loading_pct, avg_loading_pct}
		"""
		all_data = self.extract_all()
		if not all_data:
			return {
				"total_loss_kw": 0.0,
				"total_loss_kvar": 0.0,
				"max_loading_pct": 0.0,
				"avg_loading_pct": 0.0,
			}

		loss_kw = sum(d["loss_kw"] for d in all_data.values())
		loss_kvar = sum(d["loss_kvar"] for d in all_data.values())
		loadings = [d["loading_pct"] for d in all_data.values()]

		return {
			"total_loss_kw": loss_kw,
			"total_loss_kvar": loss_kvar,
			"max_loading_pct": max(loadings) if loadings else 0.0,
			"avg_loading_pct": float(np.mean(loadings)) if loadings else 0.0,
		}

	# ------------------------------------------------------------------
	# 内部实现
	# ------------------------------------------------------------------

	def _extract_current_line(
		self, lines
	) -> Optional[Dict[str, Any]]:
		"""提取当前 Lines 迭代器指向的线路数据

		参数:
			lines: ActiveCircuit.Lines 迭代器对象

		返回:
			包含 _name 键的数据字典，失败时返回 None
		"""
		name = lines.Name
		bus1 = lines.Bus1
		bus2 = lines.Bus2
		n_phases = int(lines.Phases)
		length = float(lines.Length)
		normal_amps = float(lines.NormalAmps)
		emerg_amps = float(lines.EmergAmps)
		enabled = bool(lines.IsSwitch) or True  # Lines 没有直接 enabled 属性

		# 通过 ActiveElement 获取运行数据
		self.dss.ActiveCircuit.SetActiveElement(f"Line.{name}")
		elem = self.dss.ActiveCircuit.ActiveElement

		# -- 功率 --
		powers = self._safe_list(elem.Powers)
		p_from, q_from, p_to, q_to = self._parse_powers(
			powers, n_phases
		)

		# -- 损耗 --
		losses = self._safe_list(elem.Losses)
		loss_kw, loss_kvar = self._parse_losses(losses)

		# -- 电流 --
		currents = self._safe_list(elem.Currents)
		i_from, i_to = self._parse_currents(currents, n_phases)
		i_max = max(i_from) if i_from else 0.0

		# -- 负载率 --
		loading_pct = (
			(i_max / normal_amps * 100.0)
			if normal_amps > 1e-6
			else 0.0
		)

		# -- enabled: 通过 ActiveElement 属性获取 --
		try:
			enabled = bool(elem.Enabled)
		except Exception:
			enabled = True

		# 清理母线名称（去除相标识 .1.2.3 等）
		from_bus = self._strip_bus_phases(bus1)
		to_bus = self._strip_bus_phases(bus2)

		return {
			"_name": name,
			"from_bus": from_bus,
			"to_bus": to_bus,
			"n_phases": n_phases,
			"length_km": length,
			"p_from_kw": p_from,
			"q_from_kvar": q_from,
			"p_to_kw": p_to,
			"q_to_kvar": q_to,
			"loss_kw": loss_kw,
			"loss_kvar": loss_kvar,
			"i_from_a": i_from,
			"i_to_a": i_to,
			"i_max_a": i_max,
			"normal_amps": normal_amps,
			"emerg_amps": emerg_amps,
			"loading_pct": loading_pct,
			"enabled": enabled,
		}

	@staticmethod
	def _parse_powers(
		powers: List[float], n_phases: int
	) -> Tuple[float, float, float, float]:
		"""解析 ActiveElement.Powers 数组

		Powers 格式: [P1, Q1, P2, Q2, ...] 前半为送端，后半为受端。
		注意: OpenDSS 中功率符号约定 -- 流入元件为正。

		参数:
			powers: 功率数组
			n_phases: 相数

		返回:
			(p_from_kw, q_from_kvar, p_to_kw, q_to_kvar)
		"""
		if not powers or len(powers) < 4:
			return 0.0, 0.0, 0.0, 0.0

		# 每相 2 个值 (P, Q)，两端各 n_phases 相
		# 前 2*n_phases 个值是送端，后 2*n_phases 个值是受端
		half = n_phases * 2
		if len(powers) < 2 * half:
			half = len(powers) // 2

		from_powers = powers[:half]
		to_powers = powers[half: 2 * half]

		p_from = sum(from_powers[i] for i in range(0, len(from_powers), 2))
		q_from = sum(from_powers[i] for i in range(1, len(from_powers), 2))
		p_to = sum(to_powers[i] for i in range(0, len(to_powers), 2))
		q_to = sum(to_powers[i] for i in range(1, len(to_powers), 2))

		return p_from, q_from, p_to, q_to

	@staticmethod
	def _parse_losses(losses: List[float]) -> Tuple[float, float]:
		"""解析 ActiveElement.Losses 数组

		Losses 返回 [real_W, imag_W]（单位是瓦特）。

		参数:
			losses: 损耗数组

		返回:
			(loss_kw, loss_kvar)
		"""
		if not losses or len(losses) < 2:
			return 0.0, 0.0
		return abs(float(losses[0])) / 1000.0, abs(float(losses[1])) / 1000.0

	@staticmethod
	def _parse_currents(
		currents: List[float], n_phases: int
	) -> Tuple[List[float], List[float]]:
		"""解析 ActiveElement.Currents 数组

		Currents 格式: [I1_re, I1_im, I2_re, I2_im, ...]
		前半为送端各相复数电流，后半为受端。

		参数:
			currents: 电流复数数组 (实部虚部交替)
			n_phases: 相数

		返回:
			(i_from_magnitudes, i_to_magnitudes) 各相电流幅值 (A)
		"""
		if not currents or len(currents) < 4:
			return [], []

		# 每相 2 个值 (re, im)，两端各 n_phases 相
		half = n_phases * 2  # 每端的浮点数数量
		if len(currents) < 2 * half:
			half = len(currents) // 2

		from_raw = currents[:half]
		to_raw = currents[half: 2 * half]

		i_from = [
			math.sqrt(from_raw[2 * i] ** 2 + from_raw[2 * i + 1] ** 2)
			for i in range(len(from_raw) // 2)
		]
		i_to = [
			math.sqrt(to_raw[2 * i] ** 2 + to_raw[2 * i + 1] ** 2)
			for i in range(len(to_raw) // 2)
		]
		return i_from, i_to

	@staticmethod
	def _strip_bus_phases(bus_str: str) -> str:
		"""去除母线名称中的相标识

		例如 '650.1.2.3' -> '650'

		参数:
			bus_str: 原始母线字符串

		返回:
			不含相标识的母线名称
		"""
		return bus_str.split(".")[0]

	@staticmethod
	def _safe_list(obj) -> List:
		"""安全地将 DSS 返回值转为 list

		参数:
			obj: DSS 属性返回值

		返回:
			Python list
		"""
		if obj is None:
			return []
		try:
			return list(obj)
		except (TypeError, ValueError):
			return []
