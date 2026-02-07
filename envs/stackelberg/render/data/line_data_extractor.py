# -*- coding: utf-8 -*-
"""
线路数据提取器

从 OpenDSS 引擎提取线路电流、功率、损耗和负载率数据。
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class LineDataExtractor:
	"""线路数据提取器

	提取所有线路的电流幅值、功率流、损耗和负载率信息。

	Args:
		dss: dss-python DSS 引擎实例
	"""

	def __init__(self, dss: Any):
		self._dss = dss

	def extract_all(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有线路数据。

		Returns:
			{线路名: {current_mag, power, losses, loading_pct, from_bus, to_bus, length}}
		"""
		result: Dict[str, Dict[str, Any]] = {}

		try:
			self._dss.Lines.First()
		except Exception:
			return result

		while True:
			try:
				name = self._dss.Lines.Name()
				from_bus = self._dss.Lines.Bus1().split(".")[0]
				to_bus = self._dss.Lines.Bus2().split(".")[0]
				length = self._dss.Lines.Length()

				# 获取电流和功率
				self._dss.Circuit.SetActiveElement(f"Line.{name}")
				currents = self._dss.CktElement.CurrentsMagAng()
				powers = self._dss.CktElement.Powers()
				losses = self._dss.CktElement.Losses()

				n_phases = self._dss.CktElement.NumPhases()

				# 提取电流幅值 (仅 from 端)
				current_mag = [currents[2 * i] for i in range(n_phases)]

				# 提取功率 (kW, kvar per phase, from 端)
				power_kw = [powers[2 * i] for i in range(n_phases)]
				power_kvar = [powers[2 * i + 1] for i in range(n_phases)]

				# 损耗 (W -> kW)
				loss_kw = losses[0] / 1000.0
				loss_kvar = losses[1] / 1000.0

				# 负载率估算
				normal_amps = self._dss.Lines.NormAmps()
				if normal_amps > 0 and current_mag:
					loading_pct = max(current_mag) / normal_amps * 100.0
				else:
					loading_pct = 0.0

				result[f"{from_bus}_{to_bus}"] = {
					"name": name,
					"from_bus": from_bus,
					"to_bus": to_bus,
					"current_mag": current_mag,
					"power_kw": power_kw,
					"power_kvar": power_kvar,
					"power": power_kw,
					"losses": {"kw": loss_kw, "kvar": loss_kvar},
					"loading_pct": loading_pct,
					"length": length,
					"n_phases": n_phases,
				}
			except Exception as exc:
				logger.debug(f"Line extraction failed: {exc}")

			if not self._dss.Lines.Next():
				break

		return result

	def get_total_losses(self) -> Dict[str, float]:
		"""获取系统总损耗。

		Returns:
			{total_loss_kw, total_loss_kvar}
		"""
		try:
			losses = self._dss.Circuit.Losses()
			return {
				"total_loss_kw": losses[0] / 1000.0,
				"total_loss_kvar": losses[1] / 1000.0,
			}
		except Exception:
			return {"total_loss_kw": 0.0, "total_loss_kvar": 0.0}
