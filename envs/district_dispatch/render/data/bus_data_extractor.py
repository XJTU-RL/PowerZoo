# -*- coding: utf-8 -*-
"""
母线数据提取器 -- 从 OpenDSS ActiveCircuit 提取所有母线属性。

每条母线返回电压幅值/角度（逐相 + 统计汇总）、电流、功率、
基准电压、拓扑距离、坐标等完整信息。
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BusDataExtractor:
	"""母线数据提取器 -- 从 OpenDSS 提取所有母线属性

	DSS 引擎作为外部参数注入，本类不创建也不释放引擎实例。

	参数:
		dss_engine: dss-python 的 DSS 全局单例
	"""

	def __init__(self, dss_engine) -> None:
		self.dss = dss_engine

	# ------------------------------------------------------------------
	# 公开接口
	# ------------------------------------------------------------------

	def extract_all(self) -> Dict[str, Dict[str, Any]]:
		"""提取所有母线数据

		返回:
			{bus_name: {
				v_mag_pu:      List[float],  -- 各相 pu 电压幅值
				v_angle_deg:   List[float],  -- 各相电压角度 (度)
				v_mag_kv:      List[float],  -- 各相电压幅值 (kV)
				kv_base:       float,        -- 基准电压 (kV)
				n_phases:      int,          -- 相数 (仅计相导体)
				nodes:         List[int],    -- 相节点编号
				distance:      float,        -- 到源母线的拓扑距离
				coords_x:      float | None, -- X 坐标
				coords_y:      float | None, -- Y 坐标
				seq_voltages:  List[float],  -- 序电压 [V0, V1, V2] (V)
				isc:           List[float],  -- 短路电流 (A, 实部+虚部交替)
				v_mean:        float,        -- 各相 pu 电压均值
				v_min:         float,        -- 各相 pu 电压最小值
				v_max:         float,        -- 各相 pu 电压最大值
				v_unbalance:   float,        -- 电压不平衡度
			}}
		"""
		ckt = self.dss.ActiveCircuit
		if ckt is None:
			logger.warning("ActiveCircuit 为 None，无法提取母线数据")
			return {}

		all_names = list(ckt.AllBusNames)
		result: Dict[str, Dict[str, Any]] = {}

		for bus_name in all_names:
			try:
				data = self._extract_single_bus(bus_name)
				if data is not None:
					result[bus_name] = data
			except Exception as exc:
				logger.warning(
					f"提取母线 {bus_name} 数据异常: {exc}"
				)
		return result

	def extract_by_names(
		self, bus_names: List[str]
	) -> Dict[str, Dict[str, Any]]:
		"""按名称列表提取指定母线数据

		参数:
			bus_names: 母线名称列表

		返回:
			与 extract_all 相同结构，仅含指定母线
		"""
		result: Dict[str, Dict[str, Any]] = {}
		for bus_name in bus_names:
			try:
				data = self._extract_single_bus(bus_name)
				if data is not None:
					result[bus_name] = data
			except Exception as exc:
				logger.warning(
					f"提取母线 {bus_name} 数据异常: {exc}"
				)
		return result

	def get_voltage_summary(self) -> Dict[str, float]:
		"""获取全网电压快速摘要（不含逐母线明细）

		返回:
			{v_min_pu, v_max_pu, v_mean_pu, violation_count, violation_pct}
		"""
		ckt = self.dss.ActiveCircuit
		if ckt is None:
			return self._empty_voltage_summary()

		all_vmag = self._safe_list(ckt.AllBusVmagPu)
		if not all_vmag:
			return self._empty_voltage_summary()

		# 过滤掉零值母线（通常是未连接节点）
		valid = [v for v in all_vmag if v > 0.01]
		if not valid:
			return self._empty_voltage_summary()

		arr = np.array(valid, dtype=np.float64)
		violations = int(np.sum((arr < 0.95) | (arr > 1.05)))
		return {
			"v_min_pu": float(np.min(arr)),
			"v_max_pu": float(np.max(arr)),
			"v_mean_pu": float(np.mean(arr)),
			"violation_count": violations,
			"violation_pct": violations / len(arr) * 100.0,
		}

	# ------------------------------------------------------------------
	# 内部实现
	# ------------------------------------------------------------------

	def _extract_single_bus(
		self, bus_name: str
	) -> Optional[Dict[str, Any]]:
		"""提取单条母线的完整数据

		参数:
			bus_name: 母线名称

		返回:
			母线数据字典，提取失败时返回 None
		"""
		ckt = self.dss.ActiveCircuit
		bus = ckt.Buses(bus_name)

		# -- 基本属性 --
		nodes = self._safe_list(bus.Nodes)
		kv_base = float(bus.kVBase)
		distance = float(bus.Distance)

		# -- 逐相电压 --
		pu_v_angle = self._safe_list(bus.puVmagAngle)
		v_mag_pu, v_angle_deg, phase_count = self._parse_pu_v_angle(
			pu_v_angle, nodes
		)

		# 物理电压 (kV) = pu * kVBase
		v_mag_kv = [v * kv_base for v in v_mag_pu]

		# -- 序电压 --
		seq_voltages = self._safe_list(bus.SeqVoltages)

		# -- 短路电流 --
		isc = self._safe_list(bus.Isc)

		# -- 坐标 --
		coords_x, coords_y = self._get_coords(bus)

		# -- 统计汇总 --
		v_mean, v_min, v_max, v_unbalance = self._compute_voltage_stats(
			v_mag_pu
		)

		return {
			"v_mag_pu": v_mag_pu,
			"v_angle_deg": v_angle_deg,
			"v_mag_kv": v_mag_kv,
			"kv_base": kv_base,
			"n_phases": phase_count,
			"nodes": [int(n) for n in nodes],
			"distance": distance,
			"coords_x": coords_x,
			"coords_y": coords_y,
			"seq_voltages": [float(s) for s in seq_voltages],
			"isc": [float(c) for c in isc],
			"v_mean": v_mean,
			"v_min": v_min,
			"v_max": v_max,
			"v_unbalance": v_unbalance,
		}

	def _parse_pu_v_angle(
		self, pu_v_angle: List[float], nodes: List
	) -> Tuple[List[float], List[float], int]:
		"""解析 puVmagAngle 数组，仅保留相导体 (1, 2, 3)

		参数:
			pu_v_angle: [V1, theta1, V2, theta2, ...] 原始数组
			nodes: 节点编号列表

		返回:
			(v_mag_pu_list, v_angle_deg_list, phase_count)
		"""
		if not pu_v_angle or len(pu_v_angle) < 2:
			return [1.0], [0.0], 1

		n_entries = len(pu_v_angle) // 2
		v_mag_all = [float(pu_v_angle[2 * i]) for i in range(n_entries)]
		v_ang_all = [float(pu_v_angle[2 * i + 1]) for i in range(n_entries)]

		# 仅保留相导体 (节点号 1, 2, 3)
		phase_nodes = {1, 2, 3}
		v_mag_pu: List[float] = []
		v_angle_deg: List[float] = []
		for idx, node_num in enumerate(nodes):
			if idx < n_entries and int(node_num) in phase_nodes:
				v_mag_pu.append(v_mag_all[idx])
				v_angle_deg.append(v_ang_all[idx])

		if not v_mag_pu:
			# 退化: 没有标准相导体，取全部
			v_mag_pu = v_mag_all
			v_angle_deg = v_ang_all

		return v_mag_pu, v_angle_deg, len(v_mag_pu)

	@staticmethod
	def _compute_voltage_stats(
		v_mag_pu: List[float],
	) -> Tuple[float, float, float, float]:
		"""计算电压统计量

		参数:
			v_mag_pu: 各相 pu 电压列表

		返回:
			(v_mean, v_min, v_max, v_unbalance)
		"""
		if not v_mag_pu:
			return 1.0, 1.0, 1.0, 0.0

		arr = np.array(v_mag_pu, dtype=np.float64)
		v_mean = float(np.mean(arr))
		v_min = float(np.min(arr))
		v_max = float(np.max(arr))
		v_unbalance = (
			(v_max - v_min) / v_mean if v_mean > 1e-6 else 0.0
		)
		return v_mean, v_min, v_max, v_unbalance

	@staticmethod
	def _get_coords(bus) -> Tuple[Optional[float], Optional[float]]:
		"""获取母线坐标，未定义时返回 None

		参数:
			bus: OpenDSS Bus 对象

		返回:
			(x, y) 或 (None, None)
		"""
		try:
			if bus.Coorddefined:
				return float(bus.x), float(bus.y)
		except Exception:
			pass
		return None, None

	@staticmethod
	def _safe_list(obj) -> List:
		"""安全地将 DSS 返回值转为 list

		参数:
			obj: DSS 属性返回值（可能是 ndarray / tuple / None）

		返回:
			Python list，失败时返回空列表
		"""
		if obj is None:
			return []
		try:
			return list(obj)
		except (TypeError, ValueError):
			return []

	@staticmethod
	def _empty_voltage_summary() -> Dict[str, float]:
		"""返回空的电压摘要

		返回:
			全零的摘要字典
		"""
		return {
			"v_min_pu": 0.0,
			"v_max_pu": 0.0,
			"v_mean_pu": 0.0,
			"violation_count": 0,
			"violation_pct": 0.0,
		}
