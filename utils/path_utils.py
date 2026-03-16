# -*- coding: utf-8 -*-
"""
@File      : path_utils.py
@Description: 项目路径工具函数，为所有环境提供统一的路径定位方式。
			  避免各环境各自实现 dirname/parent 层级计算。
"""

import os
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=1)
def get_project_root() -> Path:
	"""获取 PowerZoo 项目根目录。

	通过向上查找包含 'node_systems' 目录的祖先目录来定位项目根。
	结果会被缓存，后续调用零开销。

	Returns:
		Path: 项目根目录的绝对路径

	Raises:
		FileNotFoundError: 无法定位项目根目录
	"""
	# 从当前文件 (utils/path_utils.py) 向上一层即为项目根
	candidate = Path(__file__).resolve().parent.parent

	# 验证：项目根目录应包含 node_systems 目录
	if (candidate / 'node_systems').is_dir():
		return candidate

	# 备选方案：从当前文件向上逐级搜索
	current = Path(__file__).resolve()
	for parent in current.parents:
		if (parent / 'node_systems').is_dir() and (parent / 'envs').is_dir():
			return parent

	raise FileNotFoundError(
		"无法定位 PowerZoo 项目根目录。"
		"请确保 node_systems/ 和 envs/ 目录存在于项目根下。"
	)


def get_node_systems_dir() -> Path:
	"""获取 node_systems 目录路径。

	Returns:
		Path: node_systems 目录的绝对路径
	"""
	return get_project_root() / 'node_systems'


def get_system_folder(system_name: str) -> Path:
	"""获取指定电力系统的 DSS 文件目录。

	Args:
		system_name: 系统名称，如 '13Bus', '34Bus_PV_Aggressive', '123Bus'

	Returns:
		Path: 系统 DSS 文件所在目录

	Raises:
		FileNotFoundError: 指定的系统目录不存在
	"""
	folder = get_node_systems_dir() / system_name
	if not folder.is_dir():
		raise FileNotFoundError(
			f"系统目录不存在: {folder}\n"
			f"可用系统: {list_available_systems()}"
		)
	return folder


def get_dss_file_path(system_name: str, dss_file: str) -> Path:
	"""获取指定系统的 DSS 文件完整路径。

	Args:
		system_name: 系统名称
		dss_file: DSS 文件名

	Returns:
		Path: DSS 文件的绝对路径

	Raises:
		FileNotFoundError: DSS 文件不存在
	"""
	path = get_system_folder(system_name) / dss_file
	if not path.is_file():
		available = [f.name for f in get_system_folder(system_name).glob('*.dss')]
		raise FileNotFoundError(
			f"DSS 文件不存在: {path}\n"
			f"可用 DSS 文件: {available}"
		)
	return path


def list_available_systems() -> list[str]:
	"""列出所有可用的电力系统。

	Returns:
		list[str]: 系统名称列表
	"""
	ns_dir = get_node_systems_dir()
	return sorted([
		d.name for d in ns_dir.iterdir()
		if d.is_dir() and not d.name.startswith('.')
	])


def resolve_system_path(system_name: str) -> Path:
	"""将 system_name 解析为 node_systems 下的绝对路径

	Args:
		system_name: 系统名称（如 '13Bus', '34Bus_PV'）
		             或已含 'node_systems/' 前缀的路径

	Returns:
		node_systems/{system_name} 的绝对路径

	Raises:
		FileNotFoundError: 系统目录不存在
	"""
	project_root = get_project_root()

	# 已含路径前缀
	if 'node_systems' in str(system_name):
		candidate = project_root / system_name
		if candidate.exists():
			return candidate

	# 标准解析
	system_dir = project_root / 'node_systems' / system_name
	if system_dir.exists():
		return system_dir

	raise FileNotFoundError(f"System '{system_name}' not found at {system_dir}")
