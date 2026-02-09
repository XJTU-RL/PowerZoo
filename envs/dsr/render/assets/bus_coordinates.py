"""
DSR Bus Coordinates
DSR 环境各系统母线坐标定义

为 13Bus, 123Bus, 8500-Node 系统提供预定义的母线 (x, y) 坐标，
用于拓扑图绘制。坐标来源于 OpenDSS 模型中的 BusCoords 文件。
DSR 场景下故障线路和恢复区域的可视化依赖这些坐标。
"""

from typing import Dict, List, Tuple

# 类型别名
BusCoords = Dict[str, Tuple[float, float]]


def get_bus_coordinates(system_name: str) -> BusCoords:
	"""获取指定系统的母线坐标

	Args:
		system_name: 系统名称 ('13Bus', '123Bus', '8500-Node')

	Returns:
		{bus_name: (x, y)} 坐标字典
	"""
	coords_map = {
		"13Bus": _get_13bus_coords,
		"123Bus": _get_123bus_coords,
		"8500-Node": _get_8500node_coords,
	}
	factory = coords_map.get(system_name)
	if factory is None:
		return {}
	return factory()


def get_available_systems() -> List[str]:
	"""获取 DSR 环境支持的系统名称列表"""
	return ["13Bus", "123Bus", "8500-Node"]


def _get_13bus_coords() -> BusCoords:
	"""IEEE 13 Bus 系统坐标

	适用于 DSR 故障恢复场景，母线数量少但拓扑清晰，
	适合演示开关操作和逐步恢复过程。
	"""
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


def _get_123bus_coords() -> BusCoords:
	"""IEEE 123 Bus 系统坐标（简化版关键节点）

	123 母线系统拓扑较为复杂，DSR 场景中故障点多样，
	开关操作路径丰富，适合测试恢复策略。
	"""
	coords: BusCoords = {}
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
	coords["sourcebus"] = (0.0, 50.0)
	coords["150"] = (25.0, 50.0)
	return coords


def _get_8500node_coords() -> BusCoords:
	"""8500 Node 系统坐标（仅关键母线）

	超大规模系统，母线数量过多无法全部预定义，
	仅提供关键节点坐标，其余从 OpenDSS 动态获取。
	"""
	return {
		"sourcebus": (0.0, 0.0),
		"regxfmr_hsb": (100.0, 0.0),
		"regxfmr_lsb": (200.0, 0.0),
		"_hvmv_sub_lsb": (300.0, 0.0),
	}


def try_load_from_opendss(env) -> BusCoords:
	"""尝试从 DSR 环境实例动态获取母线坐标

	Args:
		env: DSREnv 实例

	Returns:
		{bus_name: (x, y)} 坐标字典
	"""
	coords: BusCoords = {}
	try:
		core = getattr(env, "core_env", None) or getattr(env, "dsr_core", None)
		if core is None:
			return coords
		circuit = getattr(core, "circuit", None)
		if circuit is None:
			return coords
		dss = getattr(circuit, "dss", None)
		if dss is None:
			return coords
		for bus_name in dss.ActiveCircuit.AllBusNames:
			dss.ActiveCircuit.SetActiveBus(bus_name)
			if dss.ActiveCircuit.Buses.Coorddefined:
				x = float(dss.ActiveCircuit.Buses.x)
				y = float(dss.ActiveCircuit.Buses.y)
				coords[bus_name] = (x, y)
	except Exception:
		pass
	return coords


def generate_fallback_coords(bus_names: List[str]) -> BusCoords:
	"""为未知系统生成网格布局坐标

	当预定义坐标和 OpenDSS 坐标均不可用时，
	按网格排列生成应急坐标，保证拓扑图可绘制。

	Args:
		bus_names: 母线名称列表

	Returns:
		{bus_name: (x, y)} 坐标字典
	"""
	coords: BusCoords = {}
	cols = max(int(len(bus_names) ** 0.5), 1)
	for i, name in enumerate(bus_names):
		x = (i % cols) * 80.0 + 50.0
		y = (i // cols) * 80.0 + 50.0
		coords[name] = (x, y)
	return coords
