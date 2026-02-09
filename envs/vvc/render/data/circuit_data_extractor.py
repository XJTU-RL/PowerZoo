# -*- coding: utf-8 -*-
"""
系统级电路数据提取器

从 OpenDSS 提取全系统级别的汇总数据：
总损耗、总负荷、总发电、电压统计、潮流收敛状态。
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class CircuitDataExtractor:
	"""系统级电路数据提取器

	提取 OpenDSS 电路的全局汇总指标。

	Args:
		dss: dss-python DSS 引擎实例
	"""

	def __init__(self, dss: Any) -> None:
		self._dss = dss

	def extract(self) -> Dict[str, Any]:
		"""提取系统级汇总数据。

		Returns:
			{total_loss_kw, total_loss_kvar, total_load_kw, total_load_kvar,
			 total_gen_kw, total_gen_kvar, total_pv_kw, total_storage_kw,
			 v_mean_pu, v_min_pu, v_max_pu, converged}
		"""
		circuit = self._dss.ActiveCircuit

		result: Dict[str, Any] = {
			"total_loss_kw": 0.0,
			"total_loss_kvar": 0.0,
			"total_load_kw": 0.0,
			"total_load_kvar": 0.0,
			"total_gen_kw": 0.0,
			"total_gen_kvar": 0.0,
			"total_pv_kw": 0.0,
			"total_storage_kw": 0.0,
			"v_mean_pu": 1.0,
			"v_min_pu": 1.0,
			"v_max_pu": 1.0,
			"converged": False,
		}

		try:
			# 潮流收敛状态
			solution = self._dss.ActiveCircuit.Solution
			result["converged"] = bool(solution.Converged)
		except Exception as exc:
			logger.debug(f"Failed to check convergence: {exc}")

		# 总损耗
		try:
			losses = list(circuit.Losses)
			if len(losses) >= 2:
				# Losses 返回 [W, Var]，需转换为 kW, kVar
				result["total_loss_kw"] = losses[0] / 1000.0
				result["total_loss_kvar"] = losses[1] / 1000.0
		except Exception as exc:
			logger.debug(f"Failed to get losses: {exc}")

		# 总负荷
		try:
			result["total_load_kw"] = float(circuit.TotalPower[0]) * (-1)
			result["total_load_kvar"] = float(circuit.TotalPower[1]) * (-1)
		except Exception:
			try:
				# 备用方法: 遍历负荷元素
				loads = circuit.Loads
				total_kw = 0.0
				total_kvar = 0.0
				idx = loads.First
				while idx > 0:
					total_kw += loads.kW
					total_kvar += loads.kvar
					idx = loads.Next
				result["total_load_kw"] = total_kw
				result["total_load_kvar"] = total_kvar
			except Exception as exc:
				logger.debug(f"Failed to get total load: {exc}")

		# 总发电量
		try:
			gens = circuit.Generators
			total_gen_kw = 0.0
			total_gen_kvar = 0.0
			idx = gens.First
			while idx > 0:
				total_gen_kw += gens.kW
				total_gen_kvar += gens.kvar
				idx = gens.Next
			result["total_gen_kw"] = total_gen_kw
			result["total_gen_kvar"] = total_gen_kvar
		except Exception as exc:
			logger.debug(f"Failed to get generators: {exc}")

		# PV 出力
		try:
			pvs = circuit.PVSystems
			total_pv = 0.0
			idx = pvs.First
			while idx > 0:
				circuit.SetActiveElement(f"PVSystem.{pvs.Name}")
				powers = list(circuit.ActiveCktElement.Powers)
				if powers:
					total_pv += abs(powers[0])
				idx = pvs.Next
			result["total_pv_kw"] = total_pv
		except Exception as exc:
			logger.debug(f"Failed to get PV output: {exc}")

		# 储能功率
		try:
			storages = circuit.Storages
			total_storage = 0.0
			idx = storages.First
			while idx > 0:
				circuit.SetActiveElement(f"Storage.{storages.Name}")
				powers = list(circuit.ActiveCktElement.Powers)
				if powers:
					total_storage += powers[0]  # 正=放电, 负=充电
				idx = storages.Next
			result["total_storage_kw"] = total_storage
		except Exception as exc:
			logger.debug(f"Failed to get storage output: {exc}")

		# 电压统计
		try:
			all_v_pu = list(circuit.AllBusVmagPu)
			if all_v_pu:
				# 过滤掉零值 (未连接的相)
				valid_v = [v for v in all_v_pu if v > 0.01]
				if valid_v:
					result["v_mean_pu"] = sum(valid_v) / len(valid_v)
					result["v_min_pu"] = min(valid_v)
					result["v_max_pu"] = max(valid_v)
		except Exception as exc:
			logger.debug(f"Failed to get voltage statistics: {exc}")

		return result
