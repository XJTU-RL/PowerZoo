# -*- coding: utf-8 -*-
"""
设备数据提取器

提取 VVC 环境中四类可控设备的运行状态：
- 电容器 (Capacitor): 开关状态、容量
- 调压器 (Regulator): 分接头位置、电压设定
- 电池储能 (Storage): SOC、功率、容量
- 光伏系统 (PVSystem): 输出功率、削减率
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class DeviceDataExtractor:
	"""VVC 设备数据提取器

	遍历 OpenDSS 电路中的电容器、调压器、储能和光伏元素。

	Args:
		dss: dss-python DSS 引擎实例
	"""

	def __init__(self, dss: Any) -> None:
		self._dss = dss

	def extract_capacitors(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有电容器状态。

		Returns:
			{cap_name: {bus, phases, kvar, n_steps, state, is_on}}
		"""
		result: Dict[str, Dict[str, Any]] = {}
		circuit = self._dss.ActiveCircuit

		try:
			caps = circuit.Capacitors
			idx = caps.First
		except Exception as exc:
			logger.debug(f"No capacitors or iteration failed: {exc}")
			return result

		while idx > 0:
			try:
				name = caps.Name
				kvar = caps.kvar
				n_steps = caps.NumSteps
				states = list(caps.States)
				is_on = any(s > 0 for s in states) if states else False

				# 获取连接母线
				bus = ""
				try:
					circuit.SetActiveElement(f"Capacitor.{name}")
					bus_names = list(circuit.ActiveCktElement.BusNames)
					if bus_names:
						bus = bus_names[0].split(".")[0]
				except Exception:
					pass

				result[name] = {
					"bus": bus,
					"kvar": kvar,
					"n_steps": n_steps,
					"states": states,
					"is_on": is_on,
				}
			except Exception as exc:
				logger.debug(f"Failed to extract capacitor: {exc}")

			idx = caps.Next

		return result

	def extract_regulators(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有调压器状态。

		Returns:
			{reg_name: {bus, tap, min_tap, max_tap, tap_number, forward_vreg, winding}}
		"""
		result: Dict[str, Dict[str, Any]] = {}
		circuit = self._dss.ActiveCircuit

		try:
			regs = circuit.RegControls
			idx = regs.First
		except Exception as exc:
			logger.debug(f"No regulators or iteration failed: {exc}")
			return result

		while idx > 0:
			try:
				name = regs.Name
				tap = regs.TapNumber
				forward_vreg = regs.ForwardVreg
				winding = regs.Winding

				# 获取分接头范围
				min_tap = -16
				max_tap = 16
				try:
					min_tap = int(regs.MinTap) if hasattr(regs, "MinTap") else -16
					max_tap = int(regs.MaxTap) if hasattr(regs, "MaxTap") else 16
				except Exception:
					pass

				# 获取连接母线
				bus = ""
				try:
					transformer_name = regs.Transformer
					circuit.SetActiveElement(f"Transformer.{transformer_name}")
					bus_names = list(circuit.ActiveCktElement.BusNames)
					if bus_names:
						bus = bus_names[0].split(".")[0]
				except Exception:
					pass

				result[name] = {
					"bus": bus,
					"tap": tap,
					"min_tap": min_tap,
					"max_tap": max_tap,
					"tap_number": tap,
					"forward_vreg": forward_vreg,
					"winding": winding,
				}
			except Exception as exc:
				logger.debug(f"Failed to extract regulator: {exc}")

			idx = regs.Next

		return result

	def extract_batteries(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有电池储能系统状态。

		Returns:
			{bat_name: {bus, kw, kwrated, kwhrated, kwhstored, soc, state}}
		"""
		result: Dict[str, Dict[str, Any]] = {}
		circuit = self._dss.ActiveCircuit

		try:
			storages = circuit.Storages
			idx = storages.First
		except Exception as exc:
			logger.debug(f"No storage elements or iteration failed: {exc}")
			return result

		while idx > 0:
			try:
				name = storages.Name

				# 通过 CktElement 获取详细数据
				circuit.SetActiveElement(f"Storage.{name}")
				elem = circuit.ActiveCktElement

				bus = ""
				bus_names = list(elem.BusNames)
				if bus_names:
					bus = bus_names[0].split(".")[0]

				# 获取功率 (正值=放电, 负值=充电)
				powers = list(elem.Powers)
				kw = powers[0] if powers else 0.0

				# 获取额定值和 SOC
				kw_rated = 0.0
				kwh_rated = 0.0
				kwh_stored = 0.0
				soc = 0.0
				state_str = "IDLING"

				try:
					props = circuit.ActiveDSSElement
					kw_rated = float(props.Properties("kWRated").Val)
					kwh_rated = float(props.Properties("kWhRated").Val)
					kwh_stored = float(props.Properties("kWhStored").Val)
					soc = float(props.Properties("%stored").Val) / 100.0
					state_str = str(props.Properties("State").Val)
				except Exception:
					# 备用: 从 SOC 计算
					if kwh_rated > 0:
						soc = kwh_stored / kwh_rated

				result[name] = {
					"bus": bus,
					"kw": kw,
					"kw_rated": kw_rated,
					"kwh_rated": kwh_rated,
					"kwh_stored": kwh_stored,
					"soc": soc,
					"state": state_str,
				}
			except Exception as exc:
				logger.debug(f"Failed to extract storage: {exc}")

			idx = storages.Next

		return result

	def extract_pvsystems(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有光伏系统状态。

		Returns:
			{pv_name: {bus, kw, kva, pmpp, irradiance, pf, curtail_pct}}
		"""
		result: Dict[str, Dict[str, Any]] = {}
		circuit = self._dss.ActiveCircuit

		try:
			pvs = circuit.PVSystems
			idx = pvs.First
		except Exception as exc:
			logger.debug(f"No PV systems or iteration failed: {exc}")
			return result

		while idx > 0:
			try:
				name = pvs.Name

				circuit.SetActiveElement(f"PVSystem.{name}")
				elem = circuit.ActiveCktElement

				bus = ""
				bus_names = list(elem.BusNames)
				if bus_names:
					bus = bus_names[0].split(".")[0]

				# 当前输出功率
				powers = list(elem.Powers)
				kw = abs(powers[0]) if powers else 0.0
				kvar = abs(powers[1]) if len(powers) > 1 else 0.0
				kva = (kw ** 2 + kvar ** 2) ** 0.5

				# 额定参数
				pmpp = 0.0
				irradiance = 1.0
				pf = 1.0
				curtail_pct = 0.0

				try:
					props = circuit.ActiveDSSElement
					pmpp = float(props.Properties("Pmpp").Val)
					irradiance = float(props.Properties("irradiance").Val)
					pf = float(props.Properties("pf").Val)

					# 削减率: (理论最大 - 实际) / 理论最大
					theoretical_max = pmpp * irradiance
					if theoretical_max > 0:
						curtail_pct = max(0.0, (theoretical_max - kw) / theoretical_max * 100.0)
				except Exception:
					pass

				result[name] = {
					"bus": bus,
					"kw": kw,
					"kva": kva,
					"pmpp": pmpp,
					"irradiance": irradiance,
					"pf": pf,
					"curtail_pct": curtail_pct,
				}
			except Exception as exc:
				logger.debug(f"Failed to extract PV system: {exc}")

			idx = pvs.Next

		return result

	def extract_all(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有设备数据。

		Returns:
			{capacitors: {...}, regulators: {...}, batteries: {...}, pvsystems: {...}}
		"""
		return {
			"capacitors": self.extract_capacitors(),
			"regulators": self.extract_regulators(),
			"batteries": self.extract_batteries(),
			"pvsystems": self.extract_pvsystems(),
		}
