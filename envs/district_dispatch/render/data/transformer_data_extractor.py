# -*- coding: utf-8 -*-
"""
变压器数据提取器 -- 从 OpenDSS 提取所有 Transformer 元素属性。

包含各绕组参数、分接头位置、损耗、负载率等完整运行信息。
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TransformerDataExtractor:
	"""变压器数据提取器

	参数:
		dss_engine: dss-python 的 DSS 全局单例
	"""

	def __init__(self, dss_engine) -> None:
		self.dss = dss_engine

	# ------------------------------------------------------------------
	# 公开接口
	# ------------------------------------------------------------------

	def extract_all(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有变压器数据

		返回:
			{xfm_name: {
				buses:          List[str],  -- 各绕组连接母线
				kva_rating:     float,      -- 额定容量 (kVA)
				n_windings:     int,        -- 绕组数
				n_phases:       int,        -- 相数
				tap_pu:         float,      -- 当前分接头 (pu)
				min_tap:        float,      -- 最小分接头
				max_tap:        float,      -- 最大分接头
				loss_kw:        float,      -- 有功损耗 (kW)
				loss_kvar:      float,      -- 无功损耗 (kvar)
				currents_a:     List[float],-- 各端口电流幅值 (A)
				i_max_a:        float,      -- 最大电流 (A)
				loading_pct:    float,      -- 负载率 (%)
				winding_kvs:    List[float],-- 各绕组额定电压 (kV)
				winding_kvas:   List[float],-- 各绕组额定容量 (kVA)
				r_pct:          float,      -- 电阻百分比 (%)
				xhl:            float,      -- 漏抗百分比 (%)
				enabled:        bool,
			}}
		"""
		ckt = self.dss.ActiveCircuit
		if ckt is None:
			logger.warning("ActiveCircuit 为 None，无法提取变压器数据")
			return {}

		xfm = ckt.Transformers
		result: Dict[str, Dict[str, Any]] = {}

		if xfm.First == 0:
			return result

		while True:
			try:
				data = self._extract_current_transformer(xfm)
				if data is not None:
					result[data.pop("_name")] = data
			except Exception as exc:
				logger.warning(f"提取变压器数据异常: {exc}")

			if xfm.Next == 0:
				break

		return result

	def get_loading_summary(self) -> Dict[str, float]:
		"""获取变压器负载率摘要

		返回:
			{max_loading_pct, avg_loading_pct, total_loss_kw, total_loss_kvar, count}
		"""
		all_data = self.extract_all()
		if not all_data:
			return {
				"max_loading_pct": 0.0,
				"avg_loading_pct": 0.0,
				"total_loss_kw": 0.0,
				"total_loss_kvar": 0.0,
				"count": 0,
			}

		loadings = [d["loading_pct"] for d in all_data.values()]
		total_loss_kw = sum(d["loss_kw"] for d in all_data.values())
		total_loss_kvar = sum(d["loss_kvar"] for d in all_data.values())

		return {
			"max_loading_pct": max(loadings) if loadings else 0.0,
			"avg_loading_pct": sum(loadings) / len(loadings) if loadings else 0.0,
			"total_loss_kw": total_loss_kw,
			"total_loss_kvar": total_loss_kvar,
			"count": len(all_data),
		}

	# ------------------------------------------------------------------
	# 内部实现
	# ------------------------------------------------------------------

	def _extract_current_transformer(
		self, xfm
	) -> Optional[Dict[str, Any]]:
		"""提取当前 Transformers 迭代器指向的变压器数据

		参数:
			xfm: ActiveCircuit.Transformers 迭代器

		返回:
			数据字典（含 _name 键），失败时返回 None
		"""
		name = xfm.Name
		kva = float(xfm.kVA)
		n_windings = int(xfm.NumWindings)
		tap = float(xfm.Tap)
		min_tap = float(xfm.MinTap)
		max_tap = float(xfm.MaxTap)

		# 通过 ActiveElement 获取运行数据
		self.dss.ActiveCircuit.SetActiveElement(f"Transformer.{name}")
		elem = self.dss.ActiveCircuit.ActiveElement
		n_phases = int(elem.NumPhases)

		# -- 损耗 --
		losses = self._safe_list(elem.Losses)
		loss_kw, loss_kvar = self._parse_losses(losses)

		# -- 电流 --
		currents_raw = self._safe_list(elem.Currents)
		currents_mag = self._compute_current_magnitudes(currents_raw)
		i_max = max(currents_mag) if currents_mag else 0.0

		# -- 母线 --
		bus_names_raw = self._safe_list(elem.BusNames)
		buses = [self._strip_bus_phases(b) for b in bus_names_raw]

		# -- enabled --
		try:
			enabled = bool(elem.Enabled)
		except Exception:
			enabled = True

		# -- 各绕组参数 --
		winding_kvs, winding_kvas = self._extract_winding_params(
			xfm, n_windings
		)

		# -- 负载率: 基于电流和额定容量 --
		loading_pct = self._compute_loading(
			i_max, kva, winding_kvs, n_phases
		)

		# -- 阻抗参数 (通过 Text 命令) --
		r_pct = self._query_float(
			f"? Transformer.{name}.%R", default=0.0
		)
		xhl = self._query_float(
			f"? Transformer.{name}.XHL", default=0.0
		)

		return {
			"_name": name,
			"buses": buses,
			"kva_rating": kva,
			"n_windings": n_windings,
			"n_phases": n_phases,
			"tap_pu": tap,
			"min_tap": min_tap,
			"max_tap": max_tap,
			"loss_kw": loss_kw,
			"loss_kvar": loss_kvar,
			"currents_a": currents_mag,
			"i_max_a": i_max,
			"loading_pct": loading_pct,
			"winding_kvs": winding_kvs,
			"winding_kvas": winding_kvas,
			"r_pct": r_pct,
			"xhl": xhl,
			"enabled": enabled,
		}

	def _extract_winding_params(
		self, xfm, n_windings: int
	) -> Tuple[List[float], List[float]]:
		"""提取各绕组的额定电压和容量

		参数:
			xfm: Transformers 迭代器（指向当前变压器）
			n_windings: 绕组数

		返回:
			(winding_kvs, winding_kvas)
		"""
		winding_kvs: List[float] = []
		winding_kvas: List[float] = []

		for wdg_idx in range(1, n_windings + 1):
			try:
				xfm.Wdg = wdg_idx
				winding_kvs.append(float(xfm.kV))
				winding_kvas.append(float(xfm.kVA))
			except Exception:
				winding_kvs.append(0.0)
				winding_kvas.append(0.0)

		return winding_kvs, winding_kvas

	@staticmethod
	def _compute_loading(
		i_max: float,
		kva: float,
		winding_kvs: List[float],
		n_phases: int,
	) -> float:
		"""计算变压器负载率

		基于一次侧额定电流: I_rated = kVA / (sqrt(3) * kV) (三相)
		                 或 I_rated = kVA / kV (单相)

		参数:
			i_max: 最大测量电流 (A)
			kva: 额定容量 (kVA)
			winding_kvs: 各绕组额定电压列表
			n_phases: 相数

		返回:
			负载率 (%)
		"""
		if not winding_kvs or kva < 1e-6:
			return 0.0

		kv_primary = winding_kvs[0]
		if kv_primary < 1e-6:
			return 0.0

		if n_phases >= 3:
			i_rated = kva / (math.sqrt(3) * kv_primary)
		else:
			i_rated = kva / kv_primary

		if i_rated < 1e-6:
			return 0.0

		return i_max / i_rated * 100.0

	@staticmethod
	def _compute_current_magnitudes(
		currents_raw: List[float],
	) -> List[float]:
		"""将复数电流数组转为幅值列表

		参数:
			currents_raw: [I1_re, I1_im, I2_re, I2_im, ...] 实部虚部交替

		返回:
			各端口电流幅值 (A)
		"""
		if not currents_raw or len(currents_raw) < 2:
			return []
		return [
			math.sqrt(currents_raw[2 * i] ** 2 + currents_raw[2 * i + 1] ** 2)
			for i in range(len(currents_raw) // 2)
		]

	@staticmethod
	def _parse_losses(losses: List[float]) -> Tuple[float, float]:
		"""解析 ActiveElement.Losses 数组

		参数:
			losses: [real_W, imag_W]

		返回:
			(loss_kw, loss_kvar)
		"""
		if not losses or len(losses) < 2:
			return 0.0, 0.0
		return abs(float(losses[0])) / 1000.0, abs(float(losses[1])) / 1000.0

	@staticmethod
	def _strip_bus_phases(bus_str: str) -> str:
		"""去除母线名称中的相标识

		参数:
			bus_str: 原始母线字符串

		返回:
			不含相标识的母线名称
		"""
		return bus_str.split(".")[0]

	def _query_float(self, cmd: str, default: float = 0.0) -> float:
		"""通过 Text 命令查询浮点属性

		参数:
			cmd: DSS Text 命令
			default: 查询失败时的默认值

		返回:
			浮点数结果
		"""
		try:
			self.dss.Text.Command = cmd
			return float(self.dss.Text.Result)
		except (ValueError, TypeError, Exception):
			return default

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
