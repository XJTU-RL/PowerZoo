"""
SmartGrid Bus Coordinates
SmartGrid 环境各系统母线坐标定义

为 13Bus, 34Bus_PV, 123Bus, 8500-Node 系统提供预定义的
母线 (x, y) 坐标，用于拓扑图绘制。坐标来源于 OpenDSS 模型
中的 BusCoords 文件。
"""

from typing import Dict, Tuple

# 类型别名
BusCoords = Dict[str, Tuple[float, float]]


def get_bus_coordinates(system_name: str) -> BusCoords:
	"""获取指定系统的母线坐标

	Args:
		system_name: 系统名称 ('13Bus', '34Bus_PV', '123Bus', '8500-Node')

	Returns:
		{bus_name: (x, y)} 坐标字典
	"""
	coords_map = {
		"13Bus": _get_13bus_coords,
		"34Bus_PV": _get_34bus_pv_coords,
		"34Bus": _get_34bus_pv_coords,
		"34Bus_PV_Aggressive": _get_34bus_pv_coords,
		"34Bus_PV_Conservative": _get_34bus_pv_coords,
		"34Bus_PV_Optimized": _get_34bus_pv_coords,
		"123Bus": _get_123bus_coords,
		"8500-Node": _get_8500node_coords,
	}
	factory = coords_map.get(system_name)
	if factory is None:
		return {}
	return factory()


def get_available_systems() -> list[str]:
	"""获取所有支持的系统名称列表"""
	return ["13Bus", "34Bus_PV", "123Bus", "8500-Node"]


def _get_13bus_coords() -> BusCoords:
	"""IEEE 13 Bus 系统坐标"""
	return {
		"sourcebus": (200.0, 400.0),
		"650": (200.0, 350.0),
		"rg60": (200.0, 300.0),
		"632": (200.0, 250.0),
		"633": (350.0, 250.0),
		"634": (450.0, 250.0),
		"645": (200.0, 150.0),
		"646": (200.0, 75.0),
		"671": (200.0, 450.0),
		"680": (200.0, 525.0),
		"684": (100.0, 450.0),
		"611": (100.0, 525.0),
		"652": (25.0, 450.0),
		"692": (300.0, 450.0),
		"675": (400.0, 450.0),
	}


def _get_34bus_pv_coords() -> BusCoords:
	"""IEEE 34 Bus PV 系统坐标"""
	return {
		"sourcebus": (0.0, 500.0),
		"800": (50.0, 500.0),
		"802": (150.0, 500.0),
		"806": (250.0, 500.0),
		"808": (350.0, 500.0),
		"810": (350.0, 400.0),
		"812": (450.0, 500.0),
		"814": (550.0, 500.0),
		"850": (600.0, 500.0),
		"816": (650.0, 500.0),
		"818": (650.0, 400.0),
		"820": (650.0, 300.0),
		"822": (650.0, 200.0),
		"824": (750.0, 500.0),
		"826": (750.0, 400.0),
		"828": (850.0, 500.0),
		"830": (950.0, 500.0),
		"854": (1050.0, 500.0),
		"852": (1150.0, 500.0),
		"832": (1250.0, 500.0),
		"858": (1350.0, 500.0),
		"834": (1450.0, 500.0),
		"842": (1550.0, 500.0),
		"844": (1650.0, 500.0),
		"846": (1750.0, 500.0),
		"848": (1850.0, 500.0),
		"860": (1350.0, 400.0),
		"836": (1450.0, 400.0),
		"840": (1450.0, 300.0),
		"862": (1350.0, 300.0),
		"838": (1450.0, 200.0),
		"864": (1350.0, 200.0),
		"888": (1250.0, 400.0),
		"890": (1250.0, 300.0),
	}


def _get_123bus_coords() -> BusCoords:
	"""IEEE 123 Bus 系统坐标（简化版关键节点）"""
	coords: BusCoords = {}
	# 主线路
	main_buses = [
		"149", "1", "2", "3", "4", "5", "6", "7", "8",
		"13", "18", "21", "23", "25", "28", "29", "30",
		"47", "48", "49", "50", "51", "52", "53", "54",
		"57", "60", "61", "62", "63", "64", "65", "66",
		"67", "72", "76", "77", "78", "79", "80", "81",
		"82", "83", "84", "85", "86", "87", "88", "89",
		"90", "91", "92", "93", "94", "95", "96", "97",
		"98", "99", "100", "150", "135", "197",
		"250", "300", "350", "450", "610",
	]
	for i, bus in enumerate(main_buses):
		x = (i % 15) * 80.0 + 50.0
		y = (i // 15) * 100.0 + 50.0
		coords[bus] = (x, y)
	# 补充 sourcebus
	coords["sourcebus"] = (0.0, 50.0)
	coords["150"] = (25.0, 50.0)
	return coords


def _get_8500node_coords() -> BusCoords:
	"""8500 Node 系统坐标（仅关键母线，其余从 OpenDSS 获取）"""
	# 8500-Node 系统过大，预定义几个关键节点
	coords: BusCoords = {
		"sourcebus": (0.0, 0.0),
		"regxfmr_hsb": (100.0, 0.0),
		"regxfmr_lsb": (200.0, 0.0),
		"_hvmv_sub_lsb": (300.0, 0.0),
	}
	return coords


def try_load_from_opendss(env) -> BusCoords:
	"""尝试从 OpenDSS 环境实例动态获取母线坐标

	Args:
		env: SmartGrid Env 实例

	Returns:
		{bus_name: (x, y)} 坐标字典
	"""
	coords: BusCoords = {}
	try:
		circuit = env.circuit
		dss = circuit.dss
		for bus_name in dss.ActiveCircuit.AllBusNames:
			dss.ActiveCircuit.SetActiveBus(bus_name)
			if dss.ActiveCircuit.Buses.Coorddefined:
				x = float(dss.ActiveCircuit.Buses.x)
				y = float(dss.ActiveCircuit.Buses.y)
				coords[bus_name] = (x, y)
	except Exception:
		pass
	return coords
