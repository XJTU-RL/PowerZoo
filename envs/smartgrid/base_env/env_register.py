# -*- coding: utf-8 -*-
"""
SmartGrid 环境注册和工厂模块

简化版本 - 删除冗余配置加载逻辑，统一使用 config_loader
"""
import os
import re
import glob
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from envs.smartgrid.base_env.env import Env
from envs.smartgrid.base_env.config_loader import load_config, get_env_config
from envs.smartgrid.base_env.env_config import SmartGridConfig


def get_data_root() -> Path:
	"""获取数据根目录（项目根目录）"""
	return Path(__file__).resolve().parent.parent.parent.parent


def get_node_systems_path() -> Path:
	"""获取 node_systems 目录路径"""
	return get_data_root() / 'node_systems'


def make_env(
	env_name: str,
	dss_act: bool = False,
	worker_idx: Optional[int] = None,
	config_dict: Optional[Dict[str, Any]] = None
) -> Env:
	"""创建 SmartGrid 环境 - 主入口

	这是创建环境的推荐方式。

	Args:
		env_name: 环境名称（如 '34Bus_pv', '13Bus'）
		dss_act: 是否使用 OpenDSS 自动控制
		worker_idx: 工作进程索引（用于并行训练）
		config_dict: 额外配置字典（可选）

	Returns:
		Env: 环境实例

	示例:
		env = make_env('34Bus_pv')
		env = make_env('34Bus_pv', worker_idx=0)
	"""
	# 构建覆盖参数
	overrides = _extract_overrides(config_dict)
	overrides['dss_act'] = dss_act

	# 加载配置
	config = load_config(env_name, overrides)

	# 获取数据路径
	folder_path = str(get_data_root())

	# 处理 worker_idx
	if worker_idx is not None:
		config = config.with_worker_idx(worker_idx)
		_setup_worker_files(folder_path, config.system_name, config.dss_file, worker_idx)

	# 创建环境
	return Env(folder_path, config)


def make_base_env(
	env_name: str,
	dss_act: bool = False,
	worker_idx: Optional[int] = None,
	config_dict: Optional[Dict[str, Any]] = None
) -> Env:
	"""创建环境实例 - 向后兼容接口

	保留此函数以兼容旧代码，新代码应使用 make_env()。
	"""
	return make_env(env_name, dss_act, worker_idx, config_dict)


def _extract_overrides(config_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
	"""从 config_dict 提取覆盖参数"""
	if not config_dict:
		return {}

	overrides = {}

	# 从 env_args 提取
	env_args = config_dict.get('env_args', {})
	overrides.update(env_args)

	# 从顶级提取
	if 'dss_file' in config_dict:
		overrides['dss_file'] = config_dict['dss_file']

	# 从 train 提取
	if 'train' in config_dict:
		train_cfg = config_dict['train']
		if 'episode_length' in train_cfg:
			overrides['max_episode_steps'] = train_cfg['episode_length']

	# 从 environment_specific.devices 提取
	env_specific = config_dict.get('environment_specific', {})
	devices = env_specific.get('devices', {})

	if 'regulators' in devices:
		overrides['reg_act_num'] = devices['regulators'].get('action_num', 33)

	if 'batteries' in devices:
		bat_cfg = devices['batteries']
		if bat_cfg.get('action_space') == 'continuous':
			overrides['bat_act_num'] = float('inf')
		else:
			overrides['bat_act_num'] = bat_cfg.get('action_num', 33)

	if 'pv_systems' in devices:
		pv_cfg = devices['pv_systems']
		overrides['pv_control'] = pv_cfg.get('control_enabled', False)
		if pv_cfg.get('action_space') == 'continuous':
			overrides['pv_act_num'] = float('inf')
		elif 'action_num' in pv_cfg:
			overrides['pv_act_num'] = pv_cfg['action_num']

	return overrides


def _setup_worker_files(
	folder_path: str,
	system_name: str,
	dss_file: str,
	worker_idx: int
) -> None:
	"""为指定 worker 设置文件

	创建 worker 特定的 DSS 文件副本和数据目录
	"""
	# 确定系统目录
	if 'node_systems/' in system_name:
		system_dir = os.path.join(folder_path, system_name)
	else:
		system_dir = os.path.join(folder_path, 'node_systems', system_name)

	# 创建 worker 特定的 DSS 文件
	base_dss = os.path.join(system_dir, dss_file)
	if not os.path.exists(base_dss):
		return

	worker_dss = os.path.join(system_dir, f"{dss_file[:-4]}_{worker_idx}.dss")

	with open(base_dss, 'r') as fin:
		content = fin.read()

	# 替换文件引用
	content = content.replace('redirect loadshape.dss', f'redirect loadshape_{worker_idx}.dss')
	content = content.replace('redirect pv_data.dss', f'redirect pv_data_{worker_idx}.dss')

	with open(worker_dss, 'w') as fout:
		fout.write(content)

	# 创建 loadshape 文件
	_create_loadshape_file(system_dir, worker_idx)

	# 创建 pv_data 文件
	_create_pv_data_file(system_dir, worker_idx)


def _create_loadshape_file(system_dir: str, worker_idx: int) -> None:
	"""创建 worker 特定的 loadshape 文件"""
	base_file = os.path.join(system_dir, 'loadshape.dss')
	target_file = os.path.join(system_dir, f'loadshape_{worker_idx}.dss')

	if not os.path.exists(base_file):
		return

	# 创建数据目录
	base_data_dir = os.path.join(system_dir, 'loadshape', '000')
	target_data_dir = os.path.join(system_dir, 'loadshape', f'{worker_idx:03d}')

	if not os.path.exists(target_data_dir) and os.path.exists(base_data_dir):
		import shutil
		os.makedirs(os.path.dirname(target_data_dir), exist_ok=True)
		shutil.copytree(base_data_dir, target_data_dir)

	# 创建 loadshape 文件
	with open(base_file, 'r') as fin:
		content = fin.read()

	content = content.replace('./loadshape/000/', f'./loadshape/{worker_idx:03d}/')

	with open(target_file, 'w') as fout:
		fout.write(content)


def _create_pv_data_file(system_dir: str, worker_idx: int) -> None:
	"""创建 worker 特定的 PV 数据文件"""
	target_file = os.path.join(system_dir, f'pv_data_{worker_idx}.dss')

	# 检查是否已由 ConfigGenerator 生成
	if os.path.exists(target_file):
		try:
			with open(target_file, 'r') as f:
				if '自动生成时间' in f.read(200):
					return
		except Exception:
			pass

	base_file = os.path.join(system_dir, 'pv_data.dss')
	if not os.path.exists(base_file):
		return

	# 创建数据目录
	for data_type in ['irradiation', 'temperature']:
		base_data_dir = os.path.join(system_dir, data_type, '000')
		target_data_dir = os.path.join(system_dir, data_type, f'{worker_idx:03d}')

		if not os.path.exists(target_data_dir) and os.path.exists(base_data_dir):
			import shutil
			os.makedirs(os.path.dirname(target_data_dir), exist_ok=True)
			shutil.copytree(base_data_dir, target_data_dir)

	# 创建 pv_data 文件
	with open(base_file, 'r') as fin:
		content = fin.read()

	content = content.replace('./irradiation/000/', f'./irradiation/{worker_idx:03d}/')
	content = content.replace('./temperature/000/', f'./temperature/{worker_idx:03d}/')

	with open(target_file, 'w') as fout:
		fout.write(content)


def remove_parallel_dss(env_name: str, num_workers: int) -> None:
	"""删除特定 worker 的临时 DSS 文件"""
	config = load_config(env_name)
	folder_path = str(get_data_root())

	if 'node_systems/' in config.system_name:
		system_dir = os.path.join(folder_path, config.system_name)
	else:
		system_dir = os.path.join(folder_path, 'node_systems', config.system_name)

	# 删除临时文件
	patterns = [
		f"{config.dss_file[:-4]}_{num_workers}.dss",
		f"loadshape_{num_workers}.dss",
		f"pv_data_{num_workers}.dss",
	]

	for pattern in patterns:
		filepath = os.path.join(system_dir, pattern)
		if os.path.exists(filepath):
			os.remove(filepath)


def cleanup_all_parallel_dss(node_systems_path: Optional[str] = None) -> Tuple[int, int]:
	"""批量清理所有节点系统目录下的临时 DSS 文件

	Args:
		node_systems_path: node_systems 目录路径，默认使用 get_data_root()

	Returns:
		(成功删除数, 失败数)
	"""
	if node_systems_path is None:
		node_systems_path = str(get_node_systems_path())

	if not os.path.exists(node_systems_path):
		print(f"警告: 目录不存在 {node_systems_path}")
		return 0, 0

	# 系统目录列表
	system_dirs = ['13Bus', '34Bus', '123Bus', '8500-Node', '9500-Node',
				   '34Bus_PV', '34Bus_PV_Aggressive', '34Bus_PV_Conservative', '34Bus_PV_Optimized']

	total_cleaned = 0
	total_failed = 0

	# 临时文件匹配模式
	temp_pattern = re.compile(r'^.+_\d+\.dss$')

	for system_name in system_dirs:
		system_dir = os.path.join(node_systems_path, system_name)
		if not os.path.exists(system_dir):
			continue

		# 查找临时文件
		dss_files = glob.glob(os.path.join(system_dir, '*.dss'))
		temp_files = [f for f in dss_files if temp_pattern.match(os.path.basename(f))]

		if not temp_files:
			continue

		print(f"清理 {system_name}: 发现 {len(temp_files)} 个临时文件")

		for temp_file in temp_files:
			try:
				os.remove(temp_file)
				total_cleaned += 1
				print(f"  ✓ 已删除: {os.path.basename(temp_file)}")
			except Exception as e:
				total_failed += 1
				print(f"  ✗ 删除失败 {os.path.basename(temp_file)}: {e}")

	print(f"\n清理完成: 成功删除 {total_cleaned} 个文件, 失败 {total_failed} 个文件")
	return total_cleaned, total_failed


# === 向后兼容的辅助函数 ===

def get_info_and_folder(
	env_name: str,
	config_dict: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], str]:
	"""获取环境信息和文件夹路径 - 向后兼容接口

	新代码应直接使用 load_config()
	"""
	# 处理缩放后缀
	is_scaled = re.match(r'.*(_s)([0-9]*[.])?[0-9]+?', env_name)
	scale = 1.0
	if is_scaled:
		matched_str = is_scaled.group(0)
		idx = matched_str.rfind('_s')
		env_name = matched_str[:idx]
		scale = float(matched_str[idx + 2:])

	# 提取覆盖参数
	overrides = _extract_overrides(config_dict) if config_dict else {}

	# 加载配置
	config = load_config(env_name, overrides)

	# 转换为旧格式
	info = config.to_info_dict()

	# 处理缩放
	if scale != 1.0:
		info['scale'] = scale
		info['soc_w'] = info.get('soc_w', 0) * (scale ** 2)

	folder_path = str(get_data_root())
	return info, os.path.abspath(folder_path)


def get_info_from_config(
	env_name: str,
	config_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
	"""从配置文件获取环境信息 - 向后兼容接口

	新代码应直接使用 load_config()
	"""
	overrides = _extract_overrides(config_dict) if config_dict else {}
	config = load_config(env_name, overrides)
	return config.to_info_dict()
