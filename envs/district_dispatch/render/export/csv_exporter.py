"""
CSV Exporter
CSV 数据导出器 -- 将 episode 快照序列导出为 CSV zip 包

每种数据类型生成一个 CSV 文件，打包为 zip 输出到 io.BytesIO，
可供 Gradio 直接下载。
"""

import csv
import io
import zipfile
from typing import Any, Dict, List, Optional


# 支持导出的数据类别
EXPORT_CATEGORIES = [
	"bus_voltages",
	"line_data",
	"transformer_data",
	"pv_data",
	"storage_data",
	"ev_data",
	"regulator_data",
	"system_totals",
	"agent_actions",
	"agent_rewards",
]


def export_episode_csv(
	snapshots: List[Dict[str, Any]],
	output_path: Optional[str] = None,
) -> bytes:
	"""将完整 episode 数据导出为 CSV zip 包

	每种数据类型一个 CSV 文件，全部打包在一个 zip 中。

	Args:
		snapshots: 快照列表，每个快照包含 buses/lines/devices 等
		output_path: 可选输出文件路径。若提供则同时写入磁盘。

	Returns:
		zip 文件的 bytes 内容
	"""
	return export_selected_csv(snapshots, EXPORT_CATEGORIES, output_path)


def export_selected_csv(
	snapshots: List[Dict[str, Any]],
	categories: List[str],
	output_path: Optional[str] = None,
) -> bytes:
	"""选择性导出指定类别的 CSV 数据

	Args:
		snapshots: 快照列表
		categories: 要导出的类别列表，可选值见 EXPORT_CATEGORIES
		output_path: 可选输出文件路径

	Returns:
		zip 文件的 bytes 内容
	"""
	buf = io.BytesIO()

	# 类别名称到生成函数的映射
	generators = {
		"bus_voltages": _generate_bus_voltages_csv,
		"line_data": _generate_line_data_csv,
		"transformer_data": _generate_transformer_data_csv,
		"pv_data": _generate_pv_data_csv,
		"storage_data": _generate_storage_data_csv,
		"ev_data": _generate_ev_data_csv,
		"regulator_data": _generate_regulator_data_csv,
		"system_totals": _generate_system_totals_csv,
		"agent_actions": _generate_agent_actions_csv,
		"agent_rewards": _generate_agent_rewards_csv,
	}

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


# ======================================================================
# CSV 生成函数
# ======================================================================

def _generate_bus_voltages_csv(snapshots: List[Dict[str, Any]]) -> str:
	"""生成母线电压 CSV: bus_name, step, phase, v_mag_pu, v_angle_deg, v_mag_kv"""
	output = io.StringIO()
	writer = csv.writer(output)
	writer.writerow(["bus_name", "step", "phase", "v_mag_pu", "v_angle_deg", "v_mag_kv"])

	for snap in snapshots:
		step = snap.get("step", 0)
		buses = snap.get("buses", {})
		for bus_name, bus_data in buses.items():
			v_pu = bus_data.get("v_mag_pu", [])
			v_ang = bus_data.get("v_angle_deg", [])
			v_kv = bus_data.get("v_mag_kv", [])
			n_phases = len(v_pu)
			for ph in range(n_phases):
				writer.writerow([
					bus_name,
					step,
					ph + 1,
					_fmt(v_pu[ph]) if ph < len(v_pu) else "",
					_fmt(v_ang[ph]) if ph < len(v_ang) else "",
					_fmt(v_kv[ph]) if ph < len(v_kv) else "",
				])

	return output.getvalue()


def _generate_line_data_csv(snapshots: List[Dict[str, Any]]) -> str:
	"""生成线路数据 CSV"""
	output = io.StringIO()
	writer = csv.writer(output)
	writer.writerow([
		"line_name", "step", "from_bus", "to_bus", "p_from_kw", "q_from_kvar",
		"p_to_kw", "q_to_kvar", "loss_kw", "loss_kvar", "i_max_a", "normal_amps", "loading_pct",
	])

	for snap in snapshots:
		step = snap.get("step", 0)
		lines = snap.get("lines", {})
		for line_name, line_data in lines.items():
			writer.writerow([
				line_name,
				step,
				line_data.get("from_bus", ""),
				line_data.get("to_bus", ""),
				_fmt(line_data.get("p_from_kw")),
				_fmt(line_data.get("q_from_kvar")),
				_fmt(line_data.get("p_to_kw")),
				_fmt(line_data.get("q_to_kvar")),
				_fmt(line_data.get("loss_kw")),
				_fmt(line_data.get("loss_kvar")),
				_fmt(line_data.get("i_max_a")),
				_fmt(line_data.get("normal_amps")),
				_fmt(line_data.get("loading_pct")),
			])

	return output.getvalue()


def _generate_transformer_data_csv(snapshots: List[Dict[str, Any]]) -> str:
	"""生成变压器数据 CSV"""
	output = io.StringIO()
	writer = csv.writer(output)
	writer.writerow([
		"xfm_name", "step", "tap_pu", "min_tap", "max_tap",
		"loss_kw", "loss_kvar", "i_max_a", "loading_pct", "kva_rating",
	])

	for snap in snapshots:
		step = snap.get("step", 0)
		xfms = snap.get("transformers", {})
		for xfm_name, xfm_data in xfms.items():
			writer.writerow([
				xfm_name,
				step,
				_fmt(xfm_data.get("tap_pu")),
				_fmt(xfm_data.get("min_tap")),
				_fmt(xfm_data.get("max_tap")),
				_fmt(xfm_data.get("loss_kw")),
				_fmt(xfm_data.get("loss_kvar")),
				_fmt(xfm_data.get("i_max_a")),
				_fmt(xfm_data.get("loading_pct")),
				_fmt(xfm_data.get("kva_rating")),
			])

	return output.getvalue()


def _generate_pv_data_csv(snapshots: List[Dict[str, Any]]) -> str:
	"""生成 PV 数据 CSV"""
	output = io.StringIO()
	writer = csv.writer(output)
	writer.writerow([
		"pv_name", "step", "bus", "kw_output", "kvar_output",
		"pmpp", "pct_pmpp", "irradiance", "temperature", "kw_rated", "pf",
	])

	for snap in snapshots:
		step = snap.get("step", 0)
		devices = snap.get("devices", {})
		pvs = devices.get("pv", {})
		for pv_name, pv_data in pvs.items():
			writer.writerow([
				pv_name,
				step,
				pv_data.get("bus", ""),
				_fmt(pv_data.get("kw_output")),
				_fmt(pv_data.get("kvar_output")),
				_fmt(pv_data.get("pmpp")),
				_fmt(pv_data.get("pct_pmpp")),
				_fmt(pv_data.get("irradiance")),
				_fmt(pv_data.get("temperature")),
				_fmt(pv_data.get("kw_rated")),
				_fmt(pv_data.get("pf")),
			])

	return output.getvalue()


def _generate_storage_data_csv(snapshots: List[Dict[str, Any]]) -> str:
	"""生成储能数据 CSV"""
	output = io.StringIO()
	writer = csv.writer(output)
	writer.writerow([
		"storage_name", "step", "bus", "soc_pct", "kw_output",
		"kvar_output", "state", "kw_rated", "kwh_rated", "kwh_stored",
		"charge_eff", "discharge_eff",
	])

	for snap in snapshots:
		step = snap.get("step", 0)
		devices = snap.get("devices", {})
		storages = devices.get("storage", {})
		for name, data in storages.items():
			writer.writerow([
				name,
				step,
				data.get("bus", ""),
				_fmt(data.get("soc_pct")),
				_fmt(data.get("kw_output")),
				_fmt(data.get("kvar_output")),
				data.get("state", ""),
				_fmt(data.get("kw_rated")),
				_fmt(data.get("kwh_rated")),
				_fmt(data.get("kwh_stored")),
				_fmt(data.get("charge_eff")),
				_fmt(data.get("discharge_eff")),
			])

	return output.getvalue()


def _generate_ev_data_csv(snapshots: List[Dict[str, Any]]) -> str:
	"""生成 EV 充电桩数据 CSV"""
	output = io.StringIO()
	writer = csv.writer(output)
	writer.writerow([
		"ev_name", "step", "bus", "kw", "kvar",
		"kw_rated", "connected",
	])

	for snap in snapshots:
		step = snap.get("step", 0)
		devices = snap.get("devices", {})
		evs = devices.get("ev", {})
		for ev_name, ev_data in evs.items():
			writer.writerow([
				ev_name,
				step,
				ev_data.get("bus", ""),
				_fmt(ev_data.get("kw")),
				_fmt(ev_data.get("kvar")),
				_fmt(ev_data.get("kw_rated")),
				ev_data.get("connected", ""),
			])

	return output.getvalue()


def _generate_regulator_data_csv(snapshots: List[Dict[str, Any]]) -> str:
	"""生成调压器数据 CSV"""
	output = io.StringIO()
	writer = csv.writer(output)
	writer.writerow([
		"reg_name", "step", "transformer", "tap_number",
		"v_reg", "bandwidth", "forward_vreg", "forward_band",
		"reverse_vreg", "reverse_band", "pt_ratio", "delay",
	])

	for snap in snapshots:
		step = snap.get("step", 0)
		regs = snap.get("regulators", {})
		for reg_name, reg_data in regs.items():
			writer.writerow([
				reg_name,
				step,
				reg_data.get("transformer", ""),
				reg_data.get("tap_number", ""),
				_fmt(reg_data.get("v_reg")),
				_fmt(reg_data.get("bandwidth")),
				_fmt(reg_data.get("forward_vreg")),
				_fmt(reg_data.get("forward_band")),
				_fmt(reg_data.get("reverse_vreg")),
				_fmt(reg_data.get("reverse_band")),
				_fmt(reg_data.get("pt_ratio")),
				_fmt(reg_data.get("delay")),
			])

	return output.getvalue()


def _generate_system_totals_csv(snapshots: List[Dict[str, Any]]) -> str:
	"""生成系统级汇总数据 CSV"""
	output = io.StringIO()
	writer = csv.writer(output)
	writer.writerow([
		"step", "timestamp_h",
		"total_loss_kw", "total_loss_kvar",
		"total_load_kw", "total_load_kvar",
		"total_gen_kw", "total_gen_kvar",
		"total_pv_kw", "total_storage_kw",
		"converged", "iterations",
		"v_mean_pu", "v_min_pu", "v_max_pu",
		"v_below_095_count", "v_above_105_count",
	])

	for snap in snapshots:
		step = snap.get("step", 0)
		circuit = snap.get("circuit", {})
		writer.writerow([
			step,
			snap.get("timestamp_h", step * 0.25),
			_fmt(circuit.get("total_loss_kw")),
			_fmt(circuit.get("total_loss_kvar")),
			_fmt(circuit.get("total_load_kw")),
			_fmt(circuit.get("total_load_kvar")),
			_fmt(circuit.get("total_gen_kw")),
			_fmt(circuit.get("total_gen_kvar")),
			_fmt(circuit.get("total_pv_kw")),
			_fmt(circuit.get("total_storage_kw")),
			circuit.get("converged", ""),
			circuit.get("iterations", ""),
			_fmt(circuit.get("v_mean_pu")),
			_fmt(circuit.get("v_min_pu")),
			_fmt(circuit.get("v_max_pu")),
			circuit.get("v_below_095_count", ""),
			circuit.get("v_above_105_count", ""),
		])

	return output.getvalue()


def _generate_agent_actions_csv(snapshots: List[Dict[str, Any]]) -> str:
	"""生成智能体动作 CSV (动态列数取决于动作维度)"""
	if not snapshots:
		return ""

	# 检测最大动作维度
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

	if max_dim == 0:
		# 尝试作为扁平数组处理
		for snap in snapshots:
			actions = snap.get("actions")
			if actions is not None and isinstance(actions, list):
				if len(actions) > 0 and not isinstance(actions[0], (list, tuple)):
					max_dim = 1
					break

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
				# 补齐列数
				while len(row) < 2 + max_dim:
					row.append("")
				writer.writerow(row)

	return output.getvalue()


def _generate_agent_rewards_csv(snapshots: List[Dict[str, Any]]) -> str:
	"""生成智能体奖励 CSV (含奖励分量)"""
	if not snapshots:
		return ""

	# 收集所有出现过的奖励分量键名
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


def _fmt(value: Any) -> str:
	"""格式化数值为字符串，保留 6 位有效数字。None 返回空字符串。"""
	if value is None:
		return ""
	if isinstance(value, float):
		return f"{value:.6g}"
	return str(value)
