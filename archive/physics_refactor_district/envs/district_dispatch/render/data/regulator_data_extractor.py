# -*- coding: utf-8 -*-
"""
调压器数据提取器 -- 从 OpenDSS 提取所有 RegControl 元素属性。

RegControl 是 OpenDSS 中的调压器控制元件，控制变压器分接头。
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RegulatorDataExtractor:
	"""调压器数据提取器

	参数:
		dss_engine: dss-python 的 DSS 全局单例
	"""

	def __init__(self, dss_engine) -> None:
		self.dss = dss_engine

	# ------------------------------------------------------------------
	# 公开接口
	# ------------------------------------------------------------------

	def extract_all(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有调压器 (RegControl) 数据

		返回:
			{reg_name: {
				transformer:    str,    -- 受控变压器名称
				winding:        int,    -- 受控绕组编号
				tap_number:     int,    -- 当前分接头档位
				v_reg:          float,  -- 正向调压目标 (V)
				bandwidth:      float,  -- 正向调压带宽 (V)
				pt_ratio:       float,  -- PT 变比
				ct_primary:     float,  -- CT 一次侧额定值 (A)
				delay:          float,  -- 动作延时 (s)
				max_tap_change: int,    -- 每步最大分接头变化
				forward_band:   float,  -- 正向带宽 (V)
				forward_vreg:   float,  -- 正向调压值 (V)
				reverse_band:   float,  -- 反向带宽 (V)
				reverse_vreg:   float,  -- 反向调压值 (V)
				is_reversible:  bool,   -- 是否可反向
				tap_winding:    int,    -- 分接头绕组号
				enabled:        bool,
			}}
		"""
		ckt = self.dss.ActiveCircuit
		if ckt is None:
			logger.warning("ActiveCircuit 为 None，无法提取调压器数据")
			return {}

		reg = ckt.RegControls
		result: Dict[str, Dict[str, Any]] = {}

		if reg.First == 0:
			return result

		while True:
			try:
				data = self._extract_current_regcontrol(reg)
				if data is not None:
					result[data.pop("_name")] = data
			except Exception as exc:
				logger.warning(f"提取调压器数据异常: {exc}")

			if reg.Next == 0:
				break

		return result

	def get_tap_summary(self) -> Dict[str, Any]:
		"""获取分接头位置摘要

		返回:
			{count, tap_positions: {reg_name: tap_number},
			 min_tap, max_tap, avg_tap}
		"""
		all_data = self.extract_all()
		if not all_data:
			return {
				"count": 0,
				"tap_positions": {},
				"min_tap": 0,
				"max_tap": 0,
				"avg_tap": 0.0,
			}

		taps = {
			name: d["tap_number"] for name, d in all_data.items()
		}
		tap_values = list(taps.values())

		return {
			"count": len(taps),
			"tap_positions": taps,
			"min_tap": min(tap_values),
			"max_tap": max(tap_values),
			"avg_tap": sum(tap_values) / len(tap_values),
		}

	# ------------------------------------------------------------------
	# 内部实现
	# ------------------------------------------------------------------

	def _extract_current_regcontrol(
		self, reg
	) -> Optional[Dict[str, Any]]:
		"""提取当前 RegControls 迭代器指向的调压器数据

		参数:
			reg: ActiveCircuit.RegControls 迭代器

		返回:
			数据字典（含 _name 键），失败时返回 None
		"""
		name = reg.Name

		# 基本属性
		transformer = reg.Transformer
		winding = int(reg.Winding)
		tap_number = int(reg.TapNumber)
		delay = float(reg.Delay)
		pt_ratio = float(reg.PTratio)
		ct_primary = float(reg.CTPrimary)
		max_tap_change = int(reg.MaxTapChange)

		# 正向/反向调压参数
		forward_vreg = float(reg.ForwardVreg)
		forward_band = float(reg.ForwardBand)
		reverse_vreg = float(reg.ReverseVreg)
		reverse_band = float(reg.ReverseBand)

		# 可反向性
		is_reversible = bool(reg.IsReversible)

		# 分接头绕组
		tap_winding = int(reg.TapWinding)

		# enabled: 通过 Text 命令查询
		enabled = self._query_enabled(name)

		# v_reg 和 bandwidth 使用正向值作为主要指标
		v_reg = forward_vreg
		bandwidth = forward_band

		return {
			"_name": name,
			"transformer": transformer,
			"winding": winding,
			"tap_number": tap_number,
			"v_reg": v_reg,
			"bandwidth": bandwidth,
			"pt_ratio": pt_ratio,
			"ct_primary": ct_primary,
			"delay": delay,
			"max_tap_change": max_tap_change,
			"forward_band": forward_band,
			"forward_vreg": forward_vreg,
			"reverse_band": reverse_band,
			"reverse_vreg": reverse_vreg,
			"is_reversible": is_reversible,
			"tap_winding": tap_winding,
			"enabled": enabled,
		}

	def _query_enabled(self, name: str) -> bool:
		"""查询 RegControl 的启用状态

		参数:
			name: RegControl 名称

		返回:
			是否启用
		"""
		try:
			self.dss.Text.Command = f"? RegControl.{name}.enabled"
			result = self.dss.Text.Result.strip().lower()
			return result in ("true", "yes", "1")
		except Exception:
			return True
