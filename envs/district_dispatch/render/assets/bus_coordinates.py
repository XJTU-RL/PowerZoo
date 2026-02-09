# -*- coding: utf-8 -*-
"""
@File      : bus_coordinates.py
@Description: IEEE 34-bus 系统母线坐标加载与处理。
			  从 CSV 文件读取母线 (x, y) 坐标，支持归一化和分区着色。
"""

import csv
import os
from typing import Dict, Optional, Tuple

from utils.path_utils import get_project_root


# -- Zone 定义：34-bus 系统三分区母线归属 --

ZONE_BUSES: Dict[int, list[str]] = {
	0: ["800", "802", "806", "808", "810", "812", "814", "814r", "850"],
	1: [
		"816", "818", "820", "822", "824", "826", "828", "830",
		"832", "852", "852r", "854", "856", "858", "864", "888", "890",
	],
	2: ["834", "836", "838", "840", "842", "844", "846", "848", "860", "862"],
}

BUS_TO_ZONE: Dict[str, int] = {
	bus: zone for zone, buses in ZONE_BUSES.items() for bus in buses
}


def get_default_csv_path() -> str:
	"""获取默认的 IEEE34 母线坐标 CSV 文件绝对路径。

	Returns:
		str: CSV 文件的绝对路径
	"""
	root = get_project_root()
	return str(root / "node_systems" / "District_34Bus_3Zone" / "IEEE34_BusXY.csv")


def load_bus_coordinates(
	csv_path: Optional[str] = None,
) -> Dict[str, Tuple[float, float]]:
	"""从 CSV 文件加载母线坐标。

	CSV 格式为无表头的三列: bus_name, x, y

	Args:
		csv_path: CSV 文件路径。为 None 时使用默认路径。

	Returns:
		Dict[str, Tuple[float, float]]: {母线名称: (x, y)} 坐标字典

	Raises:
		FileNotFoundError: CSV 文件不存在
		ValueError: CSV 数据格式异常
	"""
	if csv_path is None:
		csv_path = get_default_csv_path()

	if not os.path.isfile(csv_path):
		raise FileNotFoundError(f"母线坐标文件不存在: {csv_path}")

	coords: Dict[str, Tuple[float, float]] = {}

	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.reader(f)
		for line_num, row in enumerate(reader, start=1):
			# 跳过空行
			if not row or all(cell.strip() == "" for cell in row):
				continue

			if len(row) < 3:
				raise ValueError(
					f"CSV 第 {line_num} 行格式错误，期望 3 列，实际 {len(row)} 列: {row}"
				)

			bus_name = row[0].strip()
			try:
				x = float(row[1].strip())
				y = float(row[2].strip())
			except ValueError as e:
				raise ValueError(
					f"CSV 第 {line_num} 行坐标无法解析为浮点数: {row}"
				) from e

			coords[bus_name] = (x, y)

	return coords


def normalize_coordinates(
	coords: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
	"""将母线坐标归一化到 [0, 1] 范围，用于绘图。

	使用 min-max 归一化。若所有坐标相同（退化情况），全部映射到 0.5。

	Args:
		coords: 原始坐标字典 {母线名称: (x, y)}

	Returns:
		Dict[str, Tuple[float, float]]: 归一化后的坐标字典
	"""
	if not coords:
		return {}

	xs = [c[0] for c in coords.values()]
	ys = [c[1] for c in coords.values()]

	x_min, x_max = min(xs), max(xs)
	y_min, y_max = min(ys), max(ys)

	x_range = x_max - x_min
	y_range = y_max - y_min

	normalized: Dict[str, Tuple[float, float]] = {}
	for bus_name, (x, y) in coords.items():
		nx = (x - x_min) / x_range if x_range > 0 else 0.5
		ny = (y - y_min) / y_range if y_range > 0 else 0.5
		normalized[bus_name] = (nx, ny)

	return normalized
