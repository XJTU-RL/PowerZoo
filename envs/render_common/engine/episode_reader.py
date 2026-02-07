"""
Episode Reader (Common)
Episode 读取器 -- 加载已录制的 episode 数据

从 District Dispatch 提取，完全可复用。
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeData:
	"""Episode 数据容器

	作为录制和回放之间的标准数据传输格式。

	Attributes:
		snapshots: 快照列表
		total_reward: episode 累计总奖励
		episode_length: episode 总时间步数
		config_summary: 环境/算法配置摘要
		metadata: 附加元数据
	"""

	snapshots: List[Dict[str, Any]] = field(default_factory=list)
	total_reward: float = 0.0
	episode_length: int = 0
	config_summary: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodeReader:
	"""Episode 读取器 -- 加载已录制的 episode 数据

	支持完整加载、单步懒加载、摘要信息、时间序列提取。
	"""

	@staticmethod
	def load_episode(filepath: str) -> EpisodeData:
		"""加载完整 episode 数据

		Args:
			filepath: .npz 文件路径

		Returns:
			EpisodeData 对象
		"""
		with np.load(filepath, allow_pickle=False) as data:
			metadata_raw = json.loads(str(data["metadata"]))
			n_steps = int(data["n_steps"])

			snapshots: List[Dict[str, Any]] = []
			for i in range(n_steps):
				snapshot = EpisodeReader._deserialize_snapshot(data, i)
				snapshots.append(snapshot)

		total_reward = metadata_raw.pop("total_reward", 0.0)
		episode_length = metadata_raw.pop("episode_length", n_steps)
		config_summary = metadata_raw.pop("config_summary", {})

		return EpisodeData(
			snapshots=snapshots,
			total_reward=total_reward,
			episode_length=episode_length,
			config_summary=config_summary,
			metadata=metadata_raw,
		)

	@staticmethod
	def load_snapshot(filepath: str, step: int) -> Dict[str, Any]:
		"""加载单个时间步的快照

		Args:
			filepath: .npz 文件路径
			step: 时间步索引

		Returns:
			快照字典
		"""
		with np.load(filepath, allow_pickle=False) as data:
			n_steps = int(data["n_steps"])
			if step < 0 or step >= n_steps:
				raise IndexError(f"Step {step} out of range [0, {n_steps})")
			return EpisodeReader._deserialize_snapshot(data, step)

	@staticmethod
	def get_episode_summary(filepath: str) -> Dict[str, Any]:
		"""获取 episode 摘要信息

		Args:
			filepath: .npz 文件路径

		Returns:
			摘要字典
		"""
		with np.load(filepath, allow_pickle=False) as data:
			metadata = json.loads(str(data["metadata"]))
			n_steps = int(data["n_steps"])

		return {
			"n_steps": n_steps,
			"total_reward": metadata.get("total_reward", 0.0),
			"algorithm": metadata.get("algorithm", "unknown"),
			"seed": metadata.get("seed", -1),
			"timestamp": metadata.get("save_timestamp"),
			"config_summary": metadata.get("config_summary", {}),
		}

	@staticmethod
	def get_timeseries(filepath: str, key_path: str) -> np.ndarray:
		"""提取时间序列数据

		Args:
			filepath: .npz 文件路径
			key_path: 点分路径

		Returns:
			时间序列 numpy 数组
		"""
		parts = key_path.split(".", maxsplit=1)
		top_key = parts[0]
		sub_path = parts[1] if len(parts) > 1 else None

		with np.load(filepath, allow_pickle=False) as data:
			n_steps = int(data["n_steps"])
			values: List[Any] = []

			for i in range(n_steps):
				value = EpisodeReader._extract_value_from_step(
					data, i, top_key, sub_path
				)
				values.append(value)

		return np.array(values)

	@staticmethod
	def _deserialize_snapshot(data: Any, step: int) -> Dict[str, Any]:
		"""反序列化单个时间步快照"""
		snapshot: Dict[str, Any] = {}

		dict_keys = ["buses", "lines", "devices", "circuit"]
		for key in dict_keys:
			npz_key = f"step_{step}_{key}"
			if npz_key in data:
				json_str = str(data[npz_key])
				snapshot[key] = json.loads(json_str)
			else:
				snapshot[key] = {}

		array_keys = ["actions", "rewards"]
		for key in array_keys:
			npz_key = f"step_{step}_{key}"
			if npz_key in data:
				snapshot[key] = np.array(data[npz_key])
			else:
				snapshot[key] = np.array([], dtype=np.float64)

		extras_key = f"step_{step}_extras"
		if extras_key in data:
			extras = json.loads(str(data[extras_key]))
			snapshot.update(extras)

		return snapshot

	@staticmethod
	def _extract_value_from_step(
		data: Any,
		step: int,
		top_key: str,
		sub_path: Optional[str],
	) -> Any:
		"""从单个时间步中提取指定路径的值"""
		npz_key = f"step_{step}_{top_key}"

		if npz_key not in data:
			raise KeyError(f"Key '{top_key}' not found at step {step}")

		if top_key in ("actions", "rewards"):
			arr = np.array(data[npz_key])
			if sub_path is None:
				return arr
			raise KeyError(
				f"Sub-path '{sub_path}' not supported for array field '{top_key}'"
			)

		obj = json.loads(str(data[npz_key]))

		if sub_path is None:
			return obj

		for part in sub_path.split("."):
			if isinstance(obj, dict) and part in obj:
				obj = obj[part]
			else:
				raise KeyError(
					f"Path '{top_key}.{sub_path}' not found at step {step}, "
					f"failed at '{part}'"
				)

		return obj
