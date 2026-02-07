"""
Checkpoint Loader (Common)
模型检查点加载器

扫描训练结果目录中的检查点文件，解析目录结构和模型元数据。
从 District Dispatch 提取，完全可复用。
"""

import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

_CHECKPOINT_PATTERN = re.compile(r"^checkpoint_episode_(\d+)$")
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

		Args:
			results_dir: 训练结果根目录 (e.g. results/.../models/)

		Returns:
			检查点信息字典列表，按 episode 降序排列
		"""
		if not os.path.isdir(results_dir):
			logger.warning(f"Results directory not found: {results_dir}")
			return []

		checkpoints: List[Dict[str, Any]] = []

		for entry in os.listdir(results_dir):
			entry_path = os.path.join(results_dir, entry)
			if not os.path.isdir(entry_path):
				continue

			episode = -1
			match = _CHECKPOINT_PATTERN.match(entry)
			if match:
				episode = int(match.group(1))
			elif entry != "best_model":
				continue

			n_agents = _count_actor_files(entry_path)
			if n_agents == 0:
				logger.debug(f"Skipping directory with no actor files: {entry}")
				continue

			has_critic = os.path.isfile(
				os.path.join(entry_path, "critic_agent.pt")
			)
			has_value_norm = os.path.isfile(
				os.path.join(entry_path, "value_normalizer.pt")
			)

			mtime = os.path.getmtime(entry_path)
			timestamp = datetime.fromtimestamp(mtime).isoformat()

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

		checkpoints.sort(
			key=lambda c: (c["episode"] != -1, c["episode"]),
			reverse=True,
		)
		return checkpoints

	@staticmethod
	def scan_training_runs(base_dir: str = "results") -> List[Dict[str, Any]]:
		"""扫描所有训练运行

		Args:
			base_dir: 训练结果根目录

		Returns:
			训练运行信息列表，按时间戳降序
		"""
		if not os.path.isdir(base_dir):
			logger.warning(f"Base directory not found: {base_dir}")
			return []

		runs: List[Dict[str, Any]] = []

		for entry in os.listdir(base_dir):
			entry_path = os.path.join(base_dir, entry)
			if not os.path.isdir(entry_path):
				continue

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
				mtime = os.path.getmtime(entry_path)
				timestamp_str = datetime.fromtimestamp(mtime).strftime(
					"%Y%m%d_%H%M%S"
				)

			models_dir = os.path.join(entry_path, "models")
			if os.path.isdir(models_dir):
				checkpoints = CheckpointLoader.scan_checkpoints(models_dir)
			else:
				checkpoints = CheckpointLoader.scan_checkpoints(entry_path)

			runs.append({
				"name": entry,
				"path": os.path.abspath(entry_path),
				"algorithm": algorithm,
				"env_name": env_name,
				"timestamp": timestamp_str,
				"checkpoints": checkpoints,
			})

		runs.sort(key=lambda r: r["timestamp"], reverse=True)
		return runs

	@staticmethod
	def load_training_state(checkpoint_dir: str) -> Dict[str, Any]:
		"""加载训练状态

		Args:
			checkpoint_dir: 检查点目录路径

		Returns:
			训练状态字典
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
				f"training_state.pt unexpected format (type={type(state).__name__})"
			)
			return {}
		except Exception as exc:
			logger.warning(f"Failed to load training state: {exc}")
			return {}

	@staticmethod
	def get_best_checkpoint(results_dir: str) -> Optional[Dict[str, Any]]:
		"""获取最佳检查点

		Args:
			results_dir: models/ 目录路径

		Returns:
			最佳检查点信息字典，无可用检查点时返回 None
		"""
		checkpoints = CheckpointLoader.scan_checkpoints(results_dir)
		if not checkpoints:
			return None

		for ckpt in checkpoints:
			if ckpt["name"] == "best_model":
				return ckpt

		return checkpoints[0]


def _count_actor_files(directory: str) -> int:
	"""统计目录中 actor_agent*.pt 文件数量

	Args:
		directory: 检查点目录路径

	Returns:
		actor 文件数量
	"""
	count = 0
	for fname in os.listdir(directory):
		if fname.startswith("actor_agent") and fname.endswith(".pt"):
			count += 1
	return count
