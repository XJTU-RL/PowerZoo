# -*- coding: utf-8 -*-
"""
模型检查点加载器

扫描训练结果目录中的检查点文件，解析目录结构和模型元数据。
支持从 checkpoint_episode_* 和 best_model 两种目录格式中加载。
"""

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# 检查点目录名的正则模式
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint_episode_(\d+)$")
# 训练运行目录名的正则模式: {algorithm}_{env_name}_{YYYYMMDD}_{HHMMSS}
_RUN_PATTERN = re.compile(
	r"^([a-zA-Z0-9_]+?)_([a-zA-Z0-9_]+?)_(\d{8})_(\d{6})$"
)


class CheckpointLoader:
	"""模型检查点加载器

	扫描结果目录中的检查点，解析模型文件结构。
	不负责实际的模型实例化，仅解析目录和元数据。
	"""

	@staticmethod
	def scan_checkpoints(results_dir: str) -> List[Dict[str, Any]]:
		"""扫描结果目录中所有可用检查点

		遍历 models/ 子目录，识别 checkpoint_episode_* 和 best_model
		格式的检查点目录，提取智能体数量、训练状态等元数据。

		参数:
			results_dir: 训练结果根目录
				(e.g. results/happo_district_dispatch_xxx/models/)

		返回:
			检查点信息字典列表，按 episode 降序排列。每个字典包含:
			- name: 检查点目录名
			- path: 检查点绝对路径
			- episode: 回合数 (best_model 为 -1)
			- n_agents: 智能体数量
			- has_critic: 是否包含 critic 模型
			- has_value_norm: 是否包含值归一化器
			- timestamp: 文件修改时间
			- training_state: 训练状态字典 (若有)
		"""
		if not os.path.isdir(results_dir):
			logger.warning(f"结果目录不存在: {results_dir}")
			return []

		checkpoints: List[Dict[str, Any]] = []

		for entry in os.listdir(results_dir):
			entry_path = os.path.join(results_dir, entry)
			if not os.path.isdir(entry_path):
				continue

			# 匹配 checkpoint_episode_N 或 best_model
			episode = -1
			match = _CHECKPOINT_PATTERN.match(entry)
			if match:
				episode = int(match.group(1))
			elif entry != "best_model":
				continue

			# 统计 actor 文件数量
			n_agents = _count_actor_files(entry_path)
			if n_agents == 0:
				logger.debug(f"跳过无 actor 文件的目录: {entry}")
				continue

			# 检查 critic 和 value_normalizer
			has_critic = os.path.isfile(
				os.path.join(entry_path, "critic_agent.pt")
			)
			has_value_norm = os.path.isfile(
				os.path.join(entry_path, "value_normalizer.pt")
			)

			# 目录修改时间
			mtime = os.path.getmtime(entry_path)
			timestamp = datetime.fromtimestamp(mtime).isoformat()

			# 训练状态
			training_state = CheckpointLoader.load_training_state(entry_path)

			checkpoints.append({
				"name": entry,
				"path": os.path.abspath(entry_path),
				"episode": episode,
				"n_agents": n_agents,
				"has_critic": has_critic,
				"has_value_norm": has_value_norm,
				"timestamp": timestamp,
				"training_state": training_state,
			})

		# 按 episode 降序排列; best_model (episode=-1) 排在最后
		checkpoints.sort(
			key=lambda c: (c["episode"] != -1, c["episode"]),
			reverse=True,
		)
		return checkpoints

	@staticmethod
	def scan_training_runs(base_dir: str = "results") -> List[Dict[str, Any]]:
		"""扫描所有训练运行

		遍历 base_dir 下的训练运行目录，解析算法名、环境名、
		时间戳，并递归扫描每个运行的检查点。

		参数:
			base_dir: 训练结果根目录 (默认 "results")

		返回:
			训练运行信息列表，按时间戳降序。每个字典包含:
			- name: 运行目录名
			- path: 运行绝对路径
			- algorithm: 算法名
			- env_name: 环境名
			- timestamp: 时间戳字符串
			- checkpoints: 检查点列表
		"""
		if not os.path.isdir(base_dir):
			logger.warning(f"基础目录不存在: {base_dir}")
			return []

		runs: List[Dict[str, Any]] = []

		for entry in os.listdir(base_dir):
			entry_path = os.path.join(base_dir, entry)
			if not os.path.isdir(entry_path):
				continue

			# 解析运行目录名
			match = _RUN_PATTERN.match(entry)
			algorithm = ""
			env_name = ""
			timestamp_str = ""

			if match:
				algorithm = match.group(1)
				env_name = match.group(2)
				date_str = match.group(3)
				time_str = match.group(4)
				timestamp_str = f"{date_str}_{time_str}"
			else:
				# 无法解析的目录名，使用目录修改时间
				mtime = os.path.getmtime(entry_path)
				timestamp_str = datetime.fromtimestamp(mtime).strftime(
					"%Y%m%d_%H%M%S"
				)

			# 扫描检查点 (models/ 子目录)
			models_dir = os.path.join(entry_path, "models")
			if os.path.isdir(models_dir):
				checkpoints = CheckpointLoader.scan_checkpoints(models_dir)
			else:
				# 直接在运行目录下查找检查点
				checkpoints = CheckpointLoader.scan_checkpoints(entry_path)

			runs.append({
				"name": entry,
				"path": os.path.abspath(entry_path),
				"algorithm": algorithm,
				"env_name": env_name,
				"timestamp": timestamp_str,
				"checkpoints": checkpoints,
			})

		# 按时间戳降序
		runs.sort(key=lambda r: r["timestamp"], reverse=True)
		return runs

	@staticmethod
	def load_training_state(checkpoint_dir: str) -> Dict[str, Any]:
		"""加载训练状态

		从检查点目录中加载 training_state.pt 文件，
		提取训练回合数、总步数和最佳奖励等信息。

		参数:
			checkpoint_dir: 检查点目录路径

		返回:
			训练状态字典。包含 episode, total_num_steps, best_reward
			等字段。文件不存在时返回空字典。
		"""
		state_path = os.path.join(checkpoint_dir, "training_state.pt")
		if not os.path.isfile(state_path):
			return {}

		try:
			state = torch.load(state_path, map_location="cpu", weights_only=False)
			if isinstance(state, dict):
				return {
					"episode": state.get("episode", -1),
					"total_num_steps": state.get("total_num_steps", -1),
					"best_reward": state.get("best_reward", float("-inf")),
				}
			logger.warning(
				f"training_state.pt 格式异常 (type={type(state).__name__})"
			)
			return {}
		except Exception as exc:
			logger.warning(f"加载训练状态失败: {exc}")
			return {}

	@staticmethod
	def get_best_checkpoint(results_dir: str) -> Optional[Dict[str, Any]]:
		"""获取最佳检查点

		优先返回 best_model，其次返回 episode 最大的检查点。

		参数:
			results_dir: models/ 目录路径

		返回:
			最佳检查点信息字典，无可用检查点时返回 None
		"""
		checkpoints = CheckpointLoader.scan_checkpoints(results_dir)
		if not checkpoints:
			return None

		# 优先 best_model
		for ckpt in checkpoints:
			if ckpt["name"] == "best_model":
				return ckpt

		# 否则返回 episode 最大的
		return checkpoints[0]


def _count_actor_files(directory: str) -> int:
	"""统计目录中 actor_agent*.pt 文件数量

	参数:
		directory: 检查点目录路径

	返回:
		actor 文件数量
	"""
	count = 0
	for fname in os.listdir(directory):
		if fname.startswith("actor_agent") and fname.endswith(".pt"):
			count += 1
	return count
