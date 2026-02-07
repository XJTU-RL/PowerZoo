"""
Base CSV Exporter (Abstract)
CSV 数据导出器基类

提供分类 CSV 导出框架，子类需定义:
- EXPORT_CATEGORIES: 支持的导出类别
- 各类别的 CSV 生成函数
"""

import csv
import io
import zipfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


def _fmt(value: Any) -> str:
	"""格式化数值为字符串，保留 6 位有效数字。None 返回空字符串。"""
	if value is None:
		return ""
	if isinstance(value, float):
		return f"{value:.6g}"
	return str(value)


class BaseCsvExporter(ABC):
	"""CSV 数据导出器基类

	子类需定义 EXPORT_CATEGORIES 列表和对应的生成函数。
	"""

	# 子类覆写: 支持的导出类别列表
	EXPORT_CATEGORIES: List[str] = []

	def export_all(
		self,
		snapshots: List[Dict[str, Any]],
		output_path: Optional[str] = None,
	) -> bytes:
		"""导出所有类别

		Args:
			snapshots: 快照列表
			output_path: 可选输出文件路径

		Returns:
			zip 文件 bytes
		"""
		return self.export_selected(
			snapshots, self.EXPORT_CATEGORIES, output_path
		)

	def export_selected(
		self,
		snapshots: List[Dict[str, Any]],
		categories: List[str],
		output_path: Optional[str] = None,
	) -> bytes:
		"""选择性导出

		Args:
			snapshots: 快照列表
			categories: 要导出的类别
			output_path: 可选输出文件路径

		Returns:
			zip 文件 bytes
		"""
		buf = io.BytesIO()

		generators = self._get_generators()

		with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
			for cat in categories:
				if cat not in generators:
					continue
				csv_content = generators[cat](snapshots)
				if csv_content:
					zf.writestr(f"{cat}.csv", csv_content)

		result = buf.getvalue()

		if output_path is not None:
			with open(output_path, "wb") as f:
				f.write(result)

		return result

	@abstractmethod
	def _get_generators(self) -> Dict[str, Any]:
		"""获取类别名到生成函数的映射

		Returns:
			{类别名: 生成函数(snapshots) -> str}
		"""
		...

	# ------------------------------------------------------------------
	# 通用 CSV 生成器 (子类可复用)
	# ------------------------------------------------------------------

	@staticmethod
	def _generate_bus_voltages_csv(snapshots: List[Dict[str, Any]]) -> str:
		"""生成母线电压 CSV"""
		output = io.StringIO()
		writer = csv.writer(output)
		writer.writerow(["bus_name", "step", "phase", "v_mag_pu", "v_angle_deg", "v_mag_kv"])

		for snap in snapshots:
			step = snap.get("step", 0)
			buses = snap.get("buses", snap.get("bus_data", {}))
			for bus_name, bus_data in buses.items():
				v_pu = bus_data.get("v_mag_pu", [])
				v_ang = bus_data.get("v_angle_deg", [])
				v_kv = bus_data.get("v_mag_kv", [])
				n_phases = len(v_pu) if isinstance(v_pu, list) else 0
				for ph in range(n_phases):
					writer.writerow([
						bus_name, step, ph + 1,
						_fmt(v_pu[ph]) if ph < len(v_pu) else "",
						_fmt(v_ang[ph]) if ph < len(v_ang) else "",
						_fmt(v_kv[ph]) if ph < len(v_kv) else "",
					])

		return output.getvalue()

	@staticmethod
	def _generate_system_totals_csv(snapshots: List[Dict[str, Any]]) -> str:
		"""生成系统级汇总数据 CSV"""
		output = io.StringIO()
		writer = csv.writer(output)
		writer.writerow([
			"step", "total_loss_kw", "total_loss_kvar",
			"total_load_kw", "total_load_kvar",
			"total_gen_kw", "total_gen_kvar",
			"total_pv_kw", "total_storage_kw",
			"converged", "v_mean_pu", "v_min_pu", "v_max_pu",
		])

		for snap in snapshots:
			step = snap.get("step", 0)
			circuit = snap.get("circuit", {})
			writer.writerow([
				step,
				_fmt(circuit.get("total_loss_kw")),
				_fmt(circuit.get("total_loss_kvar")),
				_fmt(circuit.get("total_load_kw")),
				_fmt(circuit.get("total_load_kvar")),
				_fmt(circuit.get("total_gen_kw")),
				_fmt(circuit.get("total_gen_kvar")),
				_fmt(circuit.get("total_pv_kw")),
				_fmt(circuit.get("total_storage_kw")),
				circuit.get("converged", ""),
				_fmt(circuit.get("v_mean_pu")),
				_fmt(circuit.get("v_min_pu")),
				_fmt(circuit.get("v_max_pu")),
			])

		return output.getvalue()

	@staticmethod
	def _generate_agent_actions_csv(snapshots: List[Dict[str, Any]]) -> str:
		"""生成智能体动作 CSV"""
		if not snapshots:
			return ""

		max_dim = 0
		for snap in snapshots:
			actions = snap.get("actions")
			if actions is None:
				continue
			if isinstance(actions, list):
				for agent_actions in actions:
					if isinstance(agent_actions, (list, tuple)):
						max_dim = max(max_dim, len(agent_actions))
					else:
						max_dim = max(max_dim, 1)

		output = io.StringIO()
		writer = csv.writer(output)

		header = ["step", "agent_idx"]
		for d in range(max_dim):
			header.append(f"action_dim_{d}")
		writer.writerow(header)

		for snap in snapshots:
			step = snap.get("step", 0)
			actions = snap.get("actions")
			if actions is None:
				continue

			if isinstance(actions, list):
				for agent_idx, agent_actions in enumerate(actions):
					row = [step, agent_idx]
					if isinstance(agent_actions, (list, tuple)):
						row.extend([_fmt(v) for v in agent_actions])
					else:
						row.append(_fmt(agent_actions))
					while len(row) < 2 + max_dim:
						row.append("")
					writer.writerow(row)

		return output.getvalue()

	@staticmethod
	def _generate_agent_rewards_csv(snapshots: List[Dict[str, Any]]) -> str:
		"""生成智能体奖励 CSV"""
		if not snapshots:
			return ""

		component_keys: List[str] = []
		seen_keys: set = set()
		for snap in snapshots:
			rc = snap.get("reward_components", {})
			if rc:
				for key in rc:
					if key not in seen_keys:
						seen_keys.add(key)
						component_keys.append(key)

		output = io.StringIO()
		writer = csv.writer(output)

		header = ["step", "agent_idx", "reward"]
		header.extend(component_keys)
		writer.writerow(header)

		for snap in snapshots:
			step = snap.get("step", 0)
			rewards = snap.get("rewards")
			rc = snap.get("reward_components", {}) or {}

			if rewards is None:
				continue

			if isinstance(rewards, list):
				for agent_idx, reward in enumerate(rewards):
					row = [step, agent_idx, _fmt(reward)]
					for key in component_keys:
						val = rc.get(key)
						if isinstance(val, list) and agent_idx < len(val):
							row.append(_fmt(val[agent_idx]))
						elif isinstance(val, (int, float)):
							row.append(_fmt(val))
						else:
							row.append("")
					writer.writerow(row)

		return output.getvalue()
