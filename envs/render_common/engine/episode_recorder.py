"""
Episode Recorder (Common)
Episode 录制器 -- 将完整 episode 的快照序列保存为压缩 NPZ 文件

从 District Dispatch 提取，完全可复用。
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from envs.render_common.engine.episode_reader import EpisodeData


class NumpyEncoder(json.JSONEncoder):
	"""JSON 编码器 -- 处理 numpy 类型的序列化"""

	def default(self, obj: Any) -> Any:
		"""将 numpy 类型转换为 Python 原生类型"""
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		if isinstance(obj, (np.integer,)):
			return int(obj)
		if isinstance(obj, (np.floating,)):
			return float(obj)
		if isinstance(obj, (np.bool_,)):
			return bool(obj)
		return super().default(obj)


class EpisodeRecorder:
	"""Episode 录制器 -- 将完整 episode 的快照序列保存到文件

	Args:
		save_dir: 录制文件保存目录
	"""

	def __init__(self, save_dir: str = "recorded_episodes") -> None:
		self.save_dir = save_dir
		os.makedirs(save_dir, exist_ok=True)

	def save_episode(
		self,
		episode_data: EpisodeData,
		filename: Optional[str] = None,
	) -> str:
		"""保存 episode 数据到 .npz 压缩文件

		Args:
			episode_data: EpisodeData 数据容器
			filename: 自定义文件名（不含扩展名）

		Returns:
			保存的文件绝对路径
		"""
		if filename is None:
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			algo = episode_data.metadata.get("algorithm", "unknown")
			seed = episode_data.metadata.get("seed", 0)
			filename = f"episode_{timestamp}_{algo}_s{seed}"

		filepath = os.path.join(self.save_dir, f"{filename}.npz")

		save_dict: Dict[str, Any] = {}

		metadata = {
			**episode_data.metadata,
			"total_reward": episode_data.total_reward,
			"episode_length": episode_data.episode_length,
			"config_summary": episode_data.config_summary,
			"save_timestamp": datetime.now().isoformat(),
		}
		save_dict["metadata"] = np.array(
			json.dumps(metadata, cls=NumpyEncoder)
		)
		save_dict["n_steps"] = np.array(len(episode_data.snapshots))

		for i, snapshot in enumerate(episode_data.snapshots):
			serialized = self._serialize_snapshot(snapshot)
			for key, value in serialized.items():
				save_dict[f"step_{i}_{key}"] = value

		np.savez_compressed(filepath, **save_dict)
		return os.path.abspath(filepath)

	@staticmethod
	def list_recordings(save_dir: str = "recorded_episodes") -> List[Dict[str, Any]]:
		"""列出指定目录下所有已录制的 episode

		Args:
			save_dir: 录制文件目录

		Returns:
			录制列表
		"""
		recordings: List[Dict[str, Any]] = []
		save_path = Path(save_dir)

		if not save_path.exists():
			return recordings

		for npz_file in sorted(save_path.glob("*.npz")):
			try:
				info = EpisodeRecorder._extract_recording_info(npz_file)
				recordings.append(info)
			except Exception:
				recordings.append({
					"filename": npz_file.name,
					"path": str(npz_file.absolute()),
					"timestamp": None,
					"n_steps": -1,
					"total_reward": float("nan"),
					"algorithm": "unknown",
					"seed": -1,
					"error": "failed to parse",
				})

		return recordings

	@staticmethod
	def _extract_recording_info(npz_path: Path) -> Dict[str, Any]:
		"""从 .npz 文件中提取录制摘要信息"""
		with np.load(str(npz_path), allow_pickle=False) as data:
			metadata_str = str(data["metadata"])
			metadata = json.loads(metadata_str)
			n_steps = int(data["n_steps"])

		return {
			"filename": npz_path.name,
			"path": str(npz_path.absolute()),
			"timestamp": metadata.get("save_timestamp"),
			"n_steps": n_steps,
			"total_reward": metadata.get("total_reward", 0.0),
			"algorithm": metadata.get("algorithm", "unknown"),
			"seed": metadata.get("seed", -1),
		}

	@staticmethod
	def _serialize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
		"""将单个时间步快照序列化为可存储格式"""
		serialized: Dict[str, Any] = {}

		dict_keys = ["buses", "lines", "devices", "circuit"]
		for key in dict_keys:
			if key in snapshot:
				json_str = json.dumps(snapshot[key], cls=NumpyEncoder)
				serialized[key] = np.array(json_str)
			else:
				serialized[key] = np.array("{}")

		array_keys = ["actions", "rewards"]
		for key in array_keys:
			if key in snapshot:
				value = snapshot[key]
				if isinstance(value, np.ndarray):
					serialized[key] = value
				else:
					serialized[key] = np.asarray(value, dtype=np.float64)
			else:
				serialized[key] = np.array([], dtype=np.float64)

		extra_keys = set(snapshot.keys()) - set(dict_keys) - set(array_keys)
		if extra_keys:
			extras = {k: snapshot[k] for k in extra_keys}
			serialized["extras"] = np.array(
				json.dumps(extras, cls=NumpyEncoder)
			)

		return serialized
