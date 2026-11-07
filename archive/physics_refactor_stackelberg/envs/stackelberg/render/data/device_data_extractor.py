# -*- coding: utf-8 -*-
"""
设备数据提取器

从 OpenDSS 提取 PV、储能 (Storage)、负荷 (Load) 等设备数据。
Stackelberg 环境使用 CircuitAdapter，设备包括 PV、ESS 和变压器。
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DeviceDataExtractor:
	"""设备数据提取器

	提取 PV、储能、负荷和变压器设备的运行状态。

	Args:
		dss: dss-python DSS 引擎实例
	"""

	def __init__(self, dss: Any):
		self._dss = dss

	def extract_all(self) -> Dict[str, Any]:
		"""提取所有设备数据。

		Returns:
			{pv: [...], storage: [...], loads: [...], transformers: [...]}
		"""
		return {
			"pv": self._extract_pv(),
			"storage": self._extract_storage(),
			"loads": self._extract_loads(),
			"transformers": self._extract_transformers(),
		}

	def _extract_pv(self) -> List[Dict[str, Any]]:
		"""提取 PV 系统数据。"""
		pv_list: List[Dict[str, Any]] = []

		try:
			self._dss.PVsystems.First()
		except Exception:
			return pv_list

		while True:
			try:
				name = self._dss.PVsystems.Name()
				bus = self._dss.Properties.Value("bus1").split(".")[0]

				self._dss.Circuit.SetActiveElement(f"PVsystem.{name}")
				powers = self._dss.CktElement.Powers()
				output_kw = -powers[0] if powers else 0.0
				output_kvar = -powers[1] if len(powers) > 1 else 0.0

				pmpp = self._dss.PVsystems.Pmpp()
				irradiance = self._dss.PVsystems.Irradiance()

				pv_list.append({
					"name": name,
					"bus": bus,
					"output_kw": output_kw,
					"output_kvar": output_kvar,
					"pmpp": pmpp,
					"irradiance": irradiance,
					"available_kw": pmpp * irradiance,
				})
			except Exception as exc:
				logger.debug(f"PV extraction failed: {exc}")

			if not self._dss.PVsystems.Next():
				break

		return pv_list

	def _extract_storage(self) -> List[Dict[str, Any]]:
		"""提取储能系统数据。"""
		storage_list: List[Dict[str, Any]] = []

		try:
			self._dss.Storages.First()
		except Exception:
			return storage_list

		while True:
			try:
				name = self._dss.Storages.Name()
				bus = self._dss.Properties.Value("bus1").split(".")[0]

				self._dss.Circuit.SetActiveElement(f"Storage.{name}")
				powers = self._dss.CktElement.Powers()
				power_kw = -powers[0] if powers else 0.0

				soc = self._dss.Storages.puSOC()
				kwhrated = self._dss.Storages.kWhRated()
				kwrated = self._dss.Storages.kWRated()

				storage_list.append({
					"name": name,
					"bus": bus,
					"power_kw": power_kw,
					"soc": soc,
					"kwh_rated": kwhrated,
					"kw_rated": kwrated,
					"state": "charging" if power_kw > 0 else ("discharging" if power_kw < 0 else "idle"),
				})
			except Exception as exc:
				logger.debug(f"Storage extraction failed: {exc}")

			if not self._dss.Storages.Next():
				break

		return storage_list

	def _extract_loads(self) -> List[Dict[str, Any]]:
		"""提取负荷数据。"""
		load_list: List[Dict[str, Any]] = []

		try:
			self._dss.Loads.First()
		except Exception:
			return load_list

		while True:
			try:
				name = self._dss.Loads.Name()
				bus = self._dss.Properties.Value("bus1").split(".")[0]
				kw = self._dss.Loads.kW()
				kvar = self._dss.Loads.kvar()

				load_list.append({
					"name": name,
					"bus": bus,
					"kw": kw,
					"kvar": kvar,
				})
			except Exception as exc:
				logger.debug(f"Load extraction failed: {exc}")

			if not self._dss.Loads.Next():
				break

		return load_list

	def _extract_transformers(self) -> List[Dict[str, Any]]:
		"""提取变压器数据。"""
		xfm_list: List[Dict[str, Any]] = []

		try:
			self._dss.Transformers.First()
		except Exception:
			return xfm_list

		while True:
			try:
				name = self._dss.Transformers.Name()
				n_windings = self._dss.Transformers.NumWindings()

				self._dss.Circuit.SetActiveElement(f"Transformer.{name}")
				powers = self._dss.CktElement.Powers()
				losses = self._dss.CktElement.Losses()

				xfm_list.append({
					"name": name,
					"n_windings": n_windings,
					"loss_kw": losses[0] / 1000.0 if losses else 0.0,
					"power_kw": powers[0] if powers else 0.0,
				})
			except Exception as exc:
				logger.debug(f"Transformer extraction failed: {exc}")

			if not self._dss.Transformers.Next():
				break

		return xfm_list
