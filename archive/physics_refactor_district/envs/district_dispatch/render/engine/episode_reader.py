"""
Episode Reader
Episode 读取器 -- 加载已录制的 episode 数据

提供完整加载、单步懒加载、摘要信息、时间序列提取等功能，
与 EpisodeRecorder 配对使用。
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeData:
	"""Episode 数据容器

	作为录制和回放之间的标准数据传输格式。
	定义在此模块以避免循环导入。

	Attributes:
		snapshots: 快照列表，每个快照是包含 buses/lines/devices/circuit/
			actions/rewards 等键的字典
		total_reward: episode 累计总奖励
		episode_length: episode 总时间步数
		config_summary: 环境/算法配置摘要
		metadata: 附加元数据（algorithm, seed, timestamp 等）
	"""

	snapshots: List[Dict[str, Any]] = field(default_factory=list)
	total_reward: float = 0.0
	episode_length: int = 0
	config_summary: Dict[str, Any] = field(default_factory=dict)
	metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodeReader:
	"""Episode 读取器 -- 加载已录制的 episode 数据

	支持四种读取模式:
	1. 完整加载 (load_episode): 加载整个 episode 到内存
	2. 单步加载 (load_snapshot): 懒加载单个时间步，节省内存
	3. 摘要信息 (get_episode_summary): 仅读取元数据
	4. 时间序列 (get_timeseries): 提取特定指标的时间序列
	"""

	@staticmethod
	def load_episode(filepath: str) -> EpisodeData:
		"""加载完整 episode 数据

		将 .npz 文件反序列化为 EpisodeData 对象，包含所有时间步快照。

		Args:
			filepath: .npz 文件路径

		Returns:
			包含完整快照序列的 EpisodeData 对象

		Raises:
			FileNotFoundError: 文件不存在
			KeyError: 文件格式不兼容
		"""
		with np.load(filepath, allow_pickle=False) as data:
			metadata_raw = json.loads(str(data["metadata"]))
			n_steps = int(data["n_steps"])

			snapshots: List[Dict[str, Any]] = []
			for i in range(n_steps):
				snapshot = EpisodeReader._deserialize_snapshot(data, i)
				snapshots.append(snapshot)

		# 从 metadata 中分离 EpisodeData 字段
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
		"""加载单个时间步的快照（懒加载，节省内存）

		仅反序列化指定时间步的数据，适合大规模 episode 的逐帧回放。

		Args:
			filepath: .npz 文件路径
			step: 时间步索引（从 0 开始）

		Returns:
			快照字典，包含 buses, lines, devices, circuit, actions, rewards 等

		Raises:
			IndexError: step 超出 episode 范围
		"""
		with np.load(filepath, allow_pickle=False) as data:
			n_steps = int(data["n_steps"])
			if step < 0 or step >= n_steps:
				raise IndexError(
					f"Step {step} out of range [0, {n_steps})"
				)
			return EpisodeReader._deserialize_snapshot(data, step)

	@staticmethod
	def get_episode_summary(filepath: str) -> Dict[str, Any]:
		"""获取 episode 摘要信息（不加载完整数据）

		仅读取元数据和步数，避免反序列化所有快照。

		Args:
			filepath: .npz 文件路径

		Returns:
			摘要字典:
			- n_steps: 时间步数
			- total_reward: 总奖励
			- algorithm: 算法名称
			- seed: 随机种子
			- timestamp: 保存时间戳
			- config_summary: 配置摘要
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

		沿时间步维度提取指定字段的标量值，形成一维时间序列。

		Args:
			filepath: .npz 文件路径
			key_path: 点分路径，支持两级嵌套。例如:
				- "circuit.total_loss_kw"
				- "devices.pv.pv_d0_1.kw_output"
				- "actions" (直接返回 actions 数组拼接)
				- "rewards" (直接返回 rewards 数组拼接)

		Returns:
			np.ndarray of shape (n_steps,) 或 (n_steps, ...) 取决于数据维度

		Raises:
			KeyError: 指定路径在快照中不存在
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
	def _deserialize_snapshot(
		data: Any,
		step: int,
	) -> Dict[str, Any]:
		"""反序列化单个时间步快照

		Args:
			data: np.load 返回的 NpzFile 对象
			step: 时间步索引

		Returns:
			反序列化后的快照字典
		"""
		snapshot: Dict[str, Any] = {}

		# JSON 字符串字段 -> dict
		dict_keys = ["buses", "lines", "devices", "circuit"]
		for key in dict_keys:
			npz_key = f"step_{step}_{key}"
			if npz_key in data:
				json_str = str(data[npz_key])
				snapshot[key] = json.loads(json_str)
			else:
				snapshot[key] = {}

		# numpy 数组字段 -> 直接读取
		array_keys = ["actions", "rewards"]
		for key in array_keys:
			npz_key = f"step_{step}_{key}"
			if npz_key in data:
				snapshot[key] = np.array(data[npz_key])
			else:
				snapshot[key] = np.array([], dtype=np.float64)

		# extras 字段
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
		"""从单个时间步中提取指定路径的值

		Args:
			data: NpzFile 对象
			step: 时间步索引
			top_key: 顶层键名（如 "circuit", "devices", "actions"）
			sub_path: 子路径（如 "total_loss_kw", "pv.pv_d0_1.kw_output"）

		Returns:
			提取的标量值或数组
		"""
		npz_key = f"step_{step}_{top_key}"

		if npz_key not in data:
			raise KeyError(
				f"Key '{top_key}' not found at step {step}"
			)

		# actions/rewards 是直接的 numpy 数组
		if top_key in ("actions", "rewards"):
			arr = np.array(data[npz_key])
			if sub_path is None:
				return arr
			# 对数组不支持子路径
			raise KeyError(
				f"Sub-path '{sub_path}' not supported for array field '{top_key}'"
			)

		# dict 字段需要先反序列化 JSON
		obj = json.loads(str(data[npz_key]))

		if sub_path is None:
			return obj

		# 沿点分路径逐级访问
		for part in sub_path.split("."):
			if isinstance(obj, dict) and part in obj:
				obj = obj[part]
			else:
				raise KeyError(
					f"Path '{top_key}.{sub_path}' not found at step {step}, "
					f"failed at '{part}'"
				)

		return obj
