# -*- coding: utf-8 -*-
"""
线路数据提取器

通过 dss-python API 提取馈线线路的电气参数：
端点母线、相数、电流幅值、负载率和损耗。
"""

import logging
import math
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class LineDataExtractor:
	"""线路数据提取器

	遍历 OpenDSS 线路元素，提取线路两端母线和电气指标。

	Args:
		dss: dss-python DSS 引擎实例
	"""

	def __init__(self, dss: Any) -> None:
		self._dss = dss

	def extract_all(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有线路的电气数据。

		Returns:
			{line_name: {bus1, bus2, phases, current_mag, loading_pct, losses_kw}}
		"""
		result: Dict[str, Dict[str, Any]] = {}
		circuit = self._dss.ActiveCircuit

		try:
			lines = circuit.Lines
			idx = lines.First
		except Exception as exc:
			logger.warning(f"Failed to iterate lines: {exc}")
			return result

		while idx > 0:
			try:
				name = lines.Name
				bus1_raw = lines.Bus1
				bus2_raw = lines.Bus2
				# 去除相标识 (e.g., "632.1.2.3" -> "632")
				bus1 = bus1_raw.split(".")[0] if bus1_raw else ""
				bus2 = bus2_raw.split(".")[0] if bus2_raw else ""
				phases = lines.Phases

				# 获取电流和损耗
				current_mag = 0.0
				loading_pct = 0.0
				losses_kw = 0.0

				try:
					# 设置活动元素以获取电流
					circuit.SetActiveElement(f"Line.{name}")
					elem = circuit.ActiveCktElement

					# 电流 (复数交替: [re1, im1, re2, im2, ...])
					currents = list(elem.Currents)
					max_current = 0.0
					for i in range(0, min(len(currents), phases * 2), 2):
						re_val = currents[i]
						im_val = currents[i + 1]
						mag = math.sqrt(re_val ** 2 + im_val ** 2)
						max_current = max(max_current, mag)
					current_mag = max_current

					# 额定电流 (用于计算负载率)
					normal_amps = lines.NormAmps
					if normal_amps > 0:
						loading_pct = (max_current / normal_amps) * 100.0

					# 损耗
					losses = list(elem.Losses)
					if len(losses) >= 2:
						# Losses 返回 [kW, kVar] (总计，单位 W -> kW 需除 1000)
						losses_kw = losses[0] / 1000.0

				except Exception as exc:
					logger.debug(f"Line '{name}' current/loss extraction failed: {exc}")

				result[name] = {
					"bus1": bus1,
					"bus2": bus2,
					"phases": phases,
					"current_mag": current_mag,
					"loading_pct": loading_pct,
					"losses_kw": losses_kw,
				}
			except Exception as exc:
				logger.debug(f"Failed to extract line data: {exc}")

			idx = lines.Next

		return result

	def get_loading_summary(self) -> Dict[str, Any]:
		"""获取全系统线路负载率摘要。

		Returns:
			{avg_loading_pct, max_loading_pct, max_loading_line,
			 n_lines, n_overloaded, total_losses_kw}
		"""
		all_data = self.extract_all()

		if not all_data:
			return {
				"avg_loading_pct": 0.0,
				"max_loading_pct": 0.0,
				"max_loading_line": "",
				"n_lines": 0,
				"n_overloaded": 0,
				"total_losses_kw": 0.0,
			}

		loadings: List[float] = []
		max_loading = 0.0
		max_loading_line = ""
		total_losses = 0.0
		n_overloaded = 0

		for line_name, line_data in all_data.items():
			loading = line_data.get("loading_pct", 0.0)
			loadings.append(loading)

			if loading > max_loading:
				max_loading = loading
				max_loading_line = line_name

			if loading > 100.0:
				n_overloaded += 1

			total_losses += line_data.get("losses_kw", 0.0)

		return {
			"avg_loading_pct": sum(loadings) / len(loadings) if loadings else 0.0,
			"max_loading_pct": max_loading,
			"max_loading_line": max_loading_line,
			"n_lines": len(all_data),
			"n_overloaded": n_overloaded,
			"total_losses_kw": total_losses,
		}
