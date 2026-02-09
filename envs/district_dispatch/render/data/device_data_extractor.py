# -*- coding: utf-8 -*-
"""
设备数据提取器 -- PV、储能、EV 充电桩。

三类分布式设备的运行数据提取，统一封装在 DeviceDataExtractor 中。
EV 充电桩在 OpenDSS 中建模为带 'ev_' 前缀的 Load 元素。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DeviceDataExtractor:
	"""设备数据提取器 -- PV、储能、充电桩

	参数:
		dss_engine: dss-python 的 DSS 全局单例
	"""

	def __init__(self, dss_engine) -> None:
		self.dss = dss_engine

	# ------------------------------------------------------------------
	# 公开接口
	# ------------------------------------------------------------------

	def extract_all(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有设备数据

		返回:
			{
				"pv":      {pv_name: {...}, ...},
				"storage": {storage_name: {...}, ...},
				"ev":      {ev_name: {...}, ...},
			}
		"""
		return {
			"pv": self.extract_pv_data(),
			"storage": self.extract_storage_data(),
			"ev": self.extract_ev_data(),
		}

	def extract_pv_data(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有 PV 系统数据

		返回:
			{pv_name: {
				bus:          str,
				kw_rated:     float,   -- 额定容量 (kW)
				kw_output:    float,   -- 实际有功出力 (kW)
				kvar_output:  float,   -- 实际无功出力 (kvar)
				pmpp:         float,   -- 最大功率点 (kW)
				pct_pmpp:     float,   -- 出力百分比 (%)
				irradiance:   float,   -- 辐照度 (kW/m2)
				temperature:  float,   -- 温度 (C)
				pf:           float,   -- 功率因数
				n_phases:     int,
				enabled:      bool,
			}}
		"""
		ckt = self.dss.ActiveCircuit
		if ckt is None:
			return {}

		result: Dict[str, Dict[str, Any]] = {}
		pv = ckt.PVSystems
		if pv.First == 0:
			return result

		while True:
			try:
				data = self._extract_current_pv(pv)
				if data is not None:
					result[data.pop("_name")] = data
			except Exception as exc:
				logger.warning(f"提取 PV 数据异常: {exc}")

			if pv.Next == 0:
				break

		return result

	def extract_storage_data(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有储能数据

		返回:
			{storage_name: {
				bus:            str,
				kwh_rated:      float,  -- 额定容量 (kWh)
				kwh_stored:     float,  -- 当前存储能量 (kWh)
				soc_pct:        float,  -- SOC (%)
				kw_output:      float,  -- 实际功率 (kW)
				kvar_output:    float,  -- 实际无功 (kvar)
				kw_rated:       float,  -- 额定功率 (kW)
				state:          str,    -- CHARGING / DISCHARGING / IDLING
				pct_reserve:    float,  -- 储备百分比 (%)
				charge_eff:     float,  -- 充电效率 (%)
				discharge_eff:  float,  -- 放电效率 (%)
				n_phases:       int,
				enabled:        bool,
			}}
		"""
		ckt = self.dss.ActiveCircuit
		if ckt is None:
			return {}

		result: Dict[str, Dict[str, Any]] = {}

		# 通过 ActiveClass 迭代 Storage 元素
		# NOTE: dss-python 中 Storages 接口可能不完整，
		#       使用 ActiveClass + Text 命令更可靠
		try:
			ckt.SetActiveClass("Storage")
			ac = ckt.ActiveClass
			if ac.First == 0:
				return result

			while True:
				try:
					data = self._extract_current_storage(ac)
					if data is not None:
						result[data.pop("_name")] = data
				except Exception as exc:
					logger.warning(f"提取储能数据异常: {exc}")

				if ac.Next == 0:
					break
		except Exception as exc:
			logger.warning(f"迭代 Storage ActiveClass 失败: {exc}")

		return result

	def extract_ev_data(self) -> Dict[str, Dict[str, Any]]:
		"""提取 EV 充电桩数据

		EV 充电桩在 OpenDSS 中建模为带 'ev_' 前缀的 Load 元素。

		返回:
			{ev_name: {
				bus:        str,
				kw:         float,   -- 当前有功负荷 (kW)
				kvar:       float,   -- 当前无功负荷 (kvar)
				kw_rated:   float,   -- 额定功率 (kW) (使用 kW 作为近似)
				connected:  bool,    -- 是否正在充电 (kW > 0)
				n_phases:   int,
				enabled:    bool,
			}}
		"""
		ckt = self.dss.ActiveCircuit
		if ckt is None:
			return {}

		result: Dict[str, Dict[str, Any]] = {}
		loads = ckt.Loads
		if loads.First == 0:
			return result

		while True:
			try:
				name = loads.Name
				if name.lower().startswith("ev_"):
					data = self._extract_ev_load(loads, name)
					if data is not None:
						result[name] = data
			except Exception as exc:
				logger.warning(f"提取 EV 数据异常: {exc}")

			if loads.Next == 0:
				break

		return result

	def get_pv_summary(self) -> Dict[str, float]:
		"""获取 PV 快速摘要

		返回:
			{total_kw, total_kvar, count, avg_utilization_pct}
		"""
		pv_data = self.extract_pv_data()
		if not pv_data:
			return {
				"total_kw": 0.0,
				"total_kvar": 0.0,
				"count": 0,
				"avg_utilization_pct": 0.0,
			}

		total_kw = sum(d["kw_output"] for d in pv_data.values())
		total_kvar = sum(d["kvar_output"] for d in pv_data.values())
		total_rated = sum(d["kw_rated"] for d in pv_data.values())
		utilization = (
			(total_kw / total_rated * 100.0)
			if total_rated > 1e-6
			else 0.0
		)

		return {
			"total_kw": total_kw,
			"total_kvar": total_kvar,
			"count": len(pv_data),
			"avg_utilization_pct": utilization,
		}

	def get_storage_summary(self) -> Dict[str, float]:
		"""获取储能快速摘要

		返回:
			{total_kw, total_kvar, count, avg_soc_pct}
		"""
		st_data = self.extract_storage_data()
		if not st_data:
			return {
				"total_kw": 0.0,
				"total_kvar": 0.0,
				"count": 0,
				"avg_soc_pct": 0.0,
			}

		total_kw = sum(d["kw_output"] for d in st_data.values())
		total_kvar = sum(d["kvar_output"] for d in st_data.values())
		soc_values = [d["soc_pct"] for d in st_data.values()]
		avg_soc = sum(soc_values) / len(soc_values) if soc_values else 0.0

		return {
			"total_kw": total_kw,
			"total_kvar": total_kvar,
			"count": len(st_data),
			"avg_soc_pct": avg_soc,
		}

	# ------------------------------------------------------------------
	# 内部实现: PV
	# ------------------------------------------------------------------

	def _extract_current_pv(self, pv) -> Optional[Dict[str, Any]]:
		"""提取当前 PVSystems 迭代器指向的 PV 数据

		参数:
			pv: ActiveCircuit.PVSystems 迭代器

		返回:
			数据字典（含 _name 键），失败时返回 None
		"""
		name = pv.Name

		# 额定参数
		pmpp = float(pv.Pmpp)
		irradiance = float(pv.Irradiance)
		pf = float(pv.pf)
		kva_rated = float(pv.kVARated)

		# 通过 ActiveElement 获取实际出力
		self.dss.ActiveCircuit.SetActiveElement(f"PVSystem.{name}")
		elem = self.dss.ActiveCircuit.ActiveElement
		powers = self._safe_list(elem.Powers)
		bus_names = self._safe_list(elem.BusNames)
		n_phases = int(elem.NumPhases)

		# 解析功率
		kw_output, kvar_output = self._sum_power_pairs(powers)
		# PV 发电为负值（流出元件），取绝对值
		kw_output = abs(kw_output)
		kvar_output = abs(kvar_output)

		# 出力百分比
		pct_pmpp = (
			(kw_output / pmpp * 100.0) if pmpp > 1e-6 else 0.0
		)

		# 温度: 通过 Text 命令查询
		temperature = self._query_float(
			f"? PVSystem.{name}.Temperature", default=25.0
		)

		# enabled
		try:
			enabled = bool(elem.Enabled)
		except Exception:
			enabled = True

		bus = self._strip_bus_phases(bus_names[0]) if bus_names else ""

		return {
			"_name": name,
			"bus": bus,
			"kw_rated": kva_rated,
			"kw_output": kw_output,
			"kvar_output": kvar_output,
			"pmpp": pmpp,
			"pct_pmpp": pct_pmpp,
			"irradiance": irradiance,
			"temperature": temperature,
			"pf": pf,
			"n_phases": n_phases,
			"enabled": enabled,
		}

	# ------------------------------------------------------------------
	# 内部实现: Storage
	# ------------------------------------------------------------------

	def _extract_current_storage(
		self, ac
	) -> Optional[Dict[str, Any]]:
		"""提取当前 ActiveClass 迭代器指向的储能数据

		参数:
			ac: ActiveCircuit.ActiveClass 迭代器

		返回:
			数据字典（含 _name 键），失败时返回 None
		"""
		name = ac.Name

		# 通过 Text 命令查询储能特有属性
		soc_pct = self._query_float(
			f"? Storage.{name}.%stored", default=50.0
		)
		state = self._query_string(
			f"? Storage.{name}.State", default="IDLING"
		)
		kwh_rated = self._query_float(
			f"? Storage.{name}.kWhrated", default=0.0
		)
		kw_rated = self._query_float(
			f"? Storage.{name}.kWrated", default=0.0
		)
		pct_reserve = self._query_float(
			f"? Storage.{name}.%reserve", default=20.0
		)
		charge_eff = self._query_float(
			f"? Storage.{name}.%EffCharge", default=90.0
		)
		discharge_eff = self._query_float(
			f"? Storage.{name}.%EffDischarge", default=90.0
		)

		# 计算当前存储能量
		kwh_stored = kwh_rated * soc_pct / 100.0

		# 通过 ActiveElement 获取实际功率
		self.dss.ActiveCircuit.SetActiveElement(f"Storage.{name}")
		elem = self.dss.ActiveCircuit.ActiveElement
		powers = self._safe_list(elem.Powers)
		bus_names = self._safe_list(elem.BusNames)
		n_phases = int(elem.NumPhases)

		kw_output, kvar_output = self._sum_power_pairs(powers)

		try:
			enabled = bool(elem.Enabled)
		except Exception:
			enabled = True

		bus = self._strip_bus_phases(bus_names[0]) if bus_names else ""

		return {
			"_name": name,
			"bus": bus,
			"kwh_rated": kwh_rated,
			"kwh_stored": kwh_stored,
			"soc_pct": soc_pct,
			"kw_output": kw_output,
			"kvar_output": kvar_output,
			"kw_rated": kw_rated,
			"state": state.strip().upper(),
			"pct_reserve": pct_reserve,
			"charge_eff": charge_eff,
			"discharge_eff": discharge_eff,
			"n_phases": n_phases,
			"enabled": enabled,
		}

	# ------------------------------------------------------------------
	# 内部实现: EV
	# ------------------------------------------------------------------

	def _extract_ev_load(
		self, loads, name: str
	) -> Optional[Dict[str, Any]]:
		"""提取 EV 充电桩数据（建模为 Load）

		参数:
			loads: ActiveCircuit.Loads 迭代器
			name: Load 名称（已确认 ev_ 前缀）

		返回:
			数据字典，失败时返回 None
		"""
		kw = float(loads.kW)
		kvar = float(loads.kvar)

		# 通过 ActiveElement 获取更多属性
		self.dss.ActiveCircuit.SetActiveElement(f"Load.{name}")
		elem = self.dss.ActiveCircuit.ActiveElement
		bus_names = self._safe_list(elem.BusNames)
		n_phases = int(elem.NumPhases)

		try:
			enabled = bool(elem.Enabled)
		except Exception:
			enabled = True

		bus = self._strip_bus_phases(bus_names[0]) if bus_names else ""

		return {
			"bus": bus,
			"kw": kw,
			"kvar": kvar,
			"kw_rated": kw,  # 使用当前设定值作为额定值近似
			"connected": kw > 0.01,
			"n_phases": n_phases,
			"enabled": enabled,
		}

	# ------------------------------------------------------------------
	# 工具方法
	# ------------------------------------------------------------------

	def _query_float(self, cmd: str, default: float = 0.0) -> float:
		"""通过 Text 命令查询浮点属性

		参数:
			cmd: DSS Text 命令（如 '? Storage.bat1.%stored'）
			default: 查询失败时的默认值

		返回:
			浮点数结果
		"""
		try:
			self.dss.Text.Command = cmd
			return float(self.dss.Text.Result)
		except (ValueError, TypeError, Exception):
			return default

	def _query_string(self, cmd: str, default: str = "") -> str:
		"""通过 Text 命令查询字符串属性

		参数:
			cmd: DSS Text 命令
			default: 查询失败时的默认值

		返回:
			字符串结果
		"""
		try:
			self.dss.Text.Command = cmd
			result = self.dss.Text.Result
			return result if result else default
		except Exception:
			return default

	@staticmethod
	def _sum_power_pairs(powers: List[float]) -> Tuple[float, float]:
		"""将 Powers 数组中的 P, Q 分别求和

		Powers 格式: [P1, Q1, P2, Q2, ...]

		参数:
			powers: 功率数组

		返回:
			(total_p, total_q)
		"""
		if not powers or len(powers) < 2:
			return 0.0, 0.0
		total_p = sum(powers[i] for i in range(0, len(powers), 2))
		total_q = sum(powers[i] for i in range(1, len(powers), 2))
		return total_p, total_q

	@staticmethod
	def _strip_bus_phases(bus_str: str) -> str:
		"""去除母线名称中的相标识

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
