"""
SmartGrid Line Data Extractor
线路数据提取器

从 SmartGrid Circuit 对象中提取线路和变压器的
连接关系、负载率、损耗等数据。
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def extract_line_data(env) -> Dict[str, Dict[str, Any]]:
	"""从 SmartGrid 环境提取所有线路数据

	Args:
		env: SmartGrid Env 实例

	Returns:
		{line_name: {bus1, bus2, length_km, phases, loading_pct, loss_kw, ...}}
	"""
	line_data: Dict[str, Dict[str, Any]] = {}

	try:
		circuit = env.circuit
		dss = circuit.dss

		# 线路
		dss.ActiveCircuit.Lines.First
		while True:
			name = dss.ActiveCircuit.Lines.Name.lower()
			bus1 = dss.ActiveCircuit.Lines.Bus1.split(".", 1)[0].lower()
			bus2 = dss.ActiveCircuit.Lines.Bus2.split(".", 1)[0].lower()
			length = float(dss.ActiveCircuit.Lines.Length)
			phases = int(dss.ActiveCircuit.Lines.Phases)

			# 通过 CktElement 获取损耗
			dss.ActiveCircuit.SetActiveElement(f"Line.{name}")
			losses = dss.ActiveCircuit.ActiveElement.Losses
			loss_kw = float(losses[0]) / 1000.0 if len(losses) > 0 else 0.0
			loss_kvar = float(losses[1]) / 1000.0 if len(losses) > 1 else 0.0

			# 获取额定电流和实际电流来计算负载率
			loading_pct = _compute_line_loading(dss)

			line_data[name] = {
				"name": name,
				"type": "line",
				"bus1": bus1,
				"bus2": bus2,
				"length_km": length,
				"phases": phases,
				"loss_kw": loss_kw,
				"loss_kvar": loss_kvar,
				"loading_pct": loading_pct,
			}

			if dss.ActiveCircuit.Lines.Next == 0:
				break

	except Exception as exc:
		logger.error(f"Line data extraction failed: {exc}")

	return line_data


def extract_transformer_data(env) -> Dict[str, Dict[str, Any]]:
	"""从 SmartGrid 环境提取变压器数据

	Args:
		env: SmartGrid Env 实例

	Returns:
		{xfmr_name: {bus1, bus2, kva, phases, loading_pct, ...}}
	"""
	xfmr_data: Dict[str, Dict[str, Any]] = {}

	try:
		circuit = env.circuit
		dss = circuit.dss

		xfmr_names = dss.ActiveCircuit.Transformers.AllNames
		for xfmr_name in xfmr_names:
			dss.ActiveCircuit.SetActiveElement(f"Transformer.{xfmr_name}")
			buses = dss.ActiveCircuit.ActiveElement.BusNames
			bus1 = buses[0].split(".", 1)[0].lower() if len(buses) > 0 else ""
			bus2 = buses[1].split(".", 1)[0].lower() if len(buses) > 1 else ""

			losses = dss.ActiveCircuit.ActiveElement.Losses
			loss_kw = float(losses[0]) / 1000.0 if len(losses) > 0 else 0.0

			dss.ActiveCircuit.Transformers.Name = xfmr_name
			kva = float(dss.ActiveCircuit.Transformers.kVA)

			xfmr_data[xfmr_name] = {
				"name": xfmr_name,
				"type": "transformer",
				"bus1": bus1,
				"bus2": bus2,
				"kva": kva,
				"loss_kw": loss_kw,
			}

	except Exception as exc:
		logger.error(f"Transformer data extraction failed: {exc}")

	return xfmr_data


def extract_edge_list(env) -> List[Tuple[str, str, str]]:
	"""提取所有边 (bus1, bus2, type) 列表

	Args:
		env: SmartGrid Env 实例

	Returns:
		[(bus1, bus2, 'line'|'transformer'), ...]
	"""
	edges: List[Tuple[str, str, str]] = []

	lines = getattr(env, "lines", {})
	for name, (bus1, bus2) in lines.items():
		edges.append((bus1, bus2, "line"))

	transformers = getattr(env, "transformers", {})
	for name, (bus1, bus2) in transformers.items():
		edges.append((bus1, bus2, "transformer"))

	return edges


def _compute_line_loading(dss) -> float:
	"""计算当前活跃线路元件的负载率百分比"""
	try:
		currents = dss.ActiveCircuit.ActiveElement.CurrentsMagAng
		if not currents or len(currents) < 2:
			return 0.0
		# 取各相电流幅值的最大值
		phase_currents = [float(currents[i * 2]) for i in range(len(currents) // 2)]
		max_current = max(phase_currents) if phase_currents else 0.0

		normal_amps = float(dss.ActiveCircuit.ActiveElement.NormalAmps)
		if normal_amps > 0:
			return (max_current / normal_amps) * 100.0
	except Exception:
		pass
	return 0.0
