# -*- coding: utf-8 -*-
"""
District Dispatch Circuit Adapter
区域调度 OpenDSS 电路适配器

封装 dss-python 底层操作，提供台区级别的电路控制接口。
参考: envs/smartgrid/circuit_system/circuit.py 的编译保护模式。
"""

import dss as opendss
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 线程安全: DSS 编译锁，防止多 worker 并发编译时的竞态条件
_dss_compile_lock = threading.Lock()

logger = logging.getLogger(__name__)


class DistrictCircuitAdapter:
	"""区域调度 OpenDSS 电路适配器

	封装 dss-python 的底层操作，提供台区级别的电路控制接口。
	线程安全：compile() 使用全局锁保护。

	参数:
		dss_file: DSS 主文件路径
		worker_idx: worker 索引，用于多进程环境区分日志
	"""

	def __init__(self, dss_file: str, worker_idx: Optional[int] = None):
		self.dss = opendss.DSS
		self.dss_file = dss_file
		self.worker_idx = worker_idx
		self._is_closed = False
		self.logger = logging.getLogger(
			f"DistrictCircuit.W{worker_idx}"
		)

	def compile(self) -> None:
		"""编译 DSS 文件（线程安全）

		保存当前工作目录 -> 切换到 DSS 文件所在目录 ->
		在全局锁保护下执行编译 -> 恢复原工作目录。
		"""
		current_dir = os.getcwd()
		dss_dir = os.path.dirname(os.path.abspath(self.dss_file))
		dss_filename = os.path.basename(self.dss_file)

		with _dss_compile_lock:
			try:
				os.chdir(dss_dir)
				self.dss.Text.Command = f"compile {dss_filename}"
				self.dss.Text.Command = "Set Maxiterations=50"
				self.dss.Text.Command = "Set Maxcontroliter=100"
				self.dss.Text.Command = "Set ControlMode=off"
				self.dss.Text.Command = "CalcVoltageBases"
				self.logger.debug(
					f"DSS 编译完成: {dss_filename}"
				)
			except Exception as e:
				self.logger.error(f"DSS 编译失败: {e}")
				raise
			finally:
				os.chdir(current_dir)

	def solve(self) -> bool:
		"""求解潮流，返回是否收敛"""
		self.dss.ActiveCircuit.Solution.SolveNoControl()
		converged = self.dss.ActiveCircuit.Solution.Converged
		if not converged:
			self.logger.warning("潮流求解未收敛")
		return converged

	def get_bus_voltages(
		self, bus_names: List[str]
	) -> Dict[str, List[float]]:
		"""获取指定母线的 pu 电压

		参数:
			bus_names: 母线名称列表

		返回:
			{bus_name: [v_pu_phase1, v_pu_phase2, ...]}
		"""
		voltages = {}
		for bus_name in bus_names:
			try:
				bus = self.dss.ActiveCircuit.Buses(bus_name)
				nodes = list(bus.Nodes)
				pu_v_angle = list(bus.puVmagAngle)
				# puVmagAngle 格式: [V1, theta1, V2, theta2, ...]
				mags = [pu_v_angle[2 * i] for i in range(len(nodes))]
				# 仅保留相导体 (1, 2, 3)，排除中性线
				phase_mags = [
					v for v, n in zip(mags, nodes) if n in (1, 2, 3)
				]
				voltages[bus_name] = phase_mags
			except Exception as e:
				self.logger.warning(
					f"获取母线 {bus_name} 电压失败: {e}"
				)
				voltages[bus_name] = [1.0]
		return voltages

	def get_all_bus_voltages(self) -> Dict[str, List[float]]:
		"""获取所有母线电压

		返回:
			{bus_name: [v_pu_phase1, v_pu_phase2, ...]}
		"""
		all_names = list(self.dss.ActiveCircuit.AllBusNames)
		return self.get_bus_voltages(all_names)

	def get_total_losses(self) -> Tuple[float, float]:
		"""获取总网损

		返回:
			(loss_kw, loss_kvar)
		"""
		losses = self.dss.ActiveCircuit.Losses
		if hasattr(losses, "__len__") and len(losses) > 0:
			real_w = float(losses[0]) if len(losses) > 0 else 0.0
			imag_w = float(losses[1]) if len(losses) > 1 else 0.0
		else:
			real_w = float(losses.real)
			imag_w = float(losses.imag)
		return abs(real_w) / 1000.0, abs(imag_w) / 1000.0

	def get_total_load(self) -> Tuple[float, float]:
		"""获取总负荷

		返回:
			(total_kw, total_kvar)
		"""
		total_kw = 0.0
		total_kvar = 0.0
		dss_load = self.dss.ActiveCircuit.Loads
		if dss_load.First != 0:
			while True:
				total_kw += abs(dss_load.kW)
				total_kvar += abs(dss_load.kvar)
				if dss_load.Next == 0:
					break
		return total_kw, total_kvar

	def get_total_generation(self) -> Tuple[float, float]:
		"""获取总发电（含 PV 和 Generator）

		返回:
			(total_kw, total_kvar)
		"""
		power = self.dss.ActiveCircuit.TotalPower
		if hasattr(power, "__len__") and len(power) > 0:
			real_kw = float(power[0]) if len(power) > 0 else 0.0
			imag_kvar = float(power[1]) if len(power) > 1 else 0.0
		else:
			real_kw = float(power.real)
			imag_kvar = float(power.imag)
		# TotalPower 返回的是从电源流入的功率（负值表示发电）
		return abs(real_kw), abs(imag_kvar)

	def set_pv_output(self, pv_name: str, pct_pmpp: float) -> None:
		"""设置 PV 出力百分比

		参数:
			pv_name: PV 系统名称（不含 pvsystem. 前缀）
			pct_pmpp: 出力百分比 (0-100)
		"""
		pct_pmpp = max(0.0, min(100.0, pct_pmpp))
		self.dss.Text.Command = (
			f"edit pvsystem.{pv_name} %Pmpp={pct_pmpp:.2f}"
		)

	def get_pv_output(self, pv_name: str) -> Tuple[float, float]:
		"""获取 PV 实际出力

		参数:
			pv_name: PV 系统名称（不含 pvsystem. 前缀）

		返回:
			(kw, kvar)
		"""
		try:
			self.dss.ActiveCircuit.SetActiveElement(
				f"pvsystem.{pv_name}"
			)
			powers = list(
				self.dss.ActiveCircuit.ActiveElement.Powers
			)
			# Powers 格式: [P1, Q1, P2, Q2, ...]，负值表示发电
			kw = sum(powers[i] for i in range(0, len(powers), 2))
			kvar = sum(powers[i] for i in range(1, len(powers), 2))
			return abs(kw), abs(kvar)
		except Exception as e:
			self.logger.warning(
				f"获取 PV {pv_name} 出力失败: {e}"
			)
			return 0.0, 0.0

	def set_storage_power(
		self, storage_name: str, kw: float
	) -> None:
		"""设置储能充放电功率

		参数:
			storage_name: 储能名称（不含 storage. 前缀）
			kw: 功率值（正=放电，负=充电）
		"""
		if kw >= 0:
			state = "DISCHARGING"
		else:
			state = "CHARGING"
		self.dss.Text.Command = (
			f"edit storage.{storage_name} "
			f"State={state} kW={abs(kw):.2f}"
		)

	def get_storage_soc(self, storage_name: str) -> float:
		"""获取储能 SOC

		参数:
			storage_name: 储能名称（不含 storage. 前缀）

		返回:
			SOC 值 (0-1)
		"""
		try:
			self.dss.ActiveCircuit.SetActiveElement(
				f"storage.{storage_name}"
			)
			# 通过 Text 命令查询 %stored
			self.dss.Text.Command = (
				f"? storage.{storage_name}.%stored"
			)
			result = self.dss.Text.Result
			return float(result) / 100.0
		except Exception as e:
			self.logger.warning(
				f"获取储能 {storage_name} SOC 失败: {e}"
			)
			return 0.5

	def set_exchange_power(
		self,
		from_bus: str,
		to_bus: str,
		tie_name: str,
		p_kw: float,
		q_kvar: float,
	) -> None:
		"""设置虚拟功率交换元素

		通过在源侧放置虚拟 Load、目标侧放置虚拟 Generator 实现
		台区间功率交换。

		参数:
			from_bus: 源侧边界母线
			to_bus: 目标侧边界母线
			tie_name: 联络线名称
			p_kw: 有功功率 (kW)
			q_kvar: 无功功率 (kvar)
		"""
		# 源侧: Load 消耗功率
		self.dss.Text.Command = (
			f"edit Load.exchange_{tie_name}_from "
			f"bus1={from_bus} kW={p_kw:.2f} kvar={q_kvar:.2f} "
			f"Vminpu=0.8"
		)
		# 目标侧: Generator 注入功率
		self.dss.Text.Command = (
			f"edit Generator.exchange_{tie_name}_to "
			f"bus1={to_bus} kW={p_kw:.2f} kvar={q_kvar:.2f}"
		)

	def set_load_power(
		self, load_name: str, kw: float, kvar: Optional[float] = None
	) -> None:
		"""设置负荷功率

		参数:
			load_name: 负荷名称（不含 load. 前缀）
			kw: 有功功率 (kW)
			kvar: 无功功率 (kvar)，None 则不修改
		"""
		cmd = f"edit load.{load_name} kW={kw:.2f}"
		if kvar is not None:
			cmd += f" kvar={kvar:.2f}"
		self.dss.Text.Command = cmd

	def advance_step(self) -> bool:
		"""推进仿真一步（duty mode）

		返回:
			是否收敛
		"""
		self.dss.Text.Command = "Set Number=1"
		self.dss.ActiveCircuit.Solution.SolveNoControl()
		return self.dss.ActiveCircuit.Solution.Converged

	def reset(self) -> None:
		"""重置电路到初始状态"""
		self.compile()
		self.dss.ActiveCircuit.Solution.SolveNoControl()
		self.logger.debug("电路已重置")

	def create_exchange_elements(
		self, tie_name: str, from_bus: str, to_bus: str
	) -> None:
		"""创建功率交换所需的虚拟元素

		在首次使用前调用，为每条联络线创建虚拟 Load 和 Generator 对。

		参数:
			tie_name: 联络线名称
			from_bus: 源侧边界母线
			to_bus: 目标侧边界母线
		"""
		self.dss.Text.Command = (
			f"New Load.exchange_{tie_name}_from "
			f"bus1={from_bus} phases=3 conn=wye "
			f"kW=0 kvar=0 model=1 Vminpu=0.8"
		)
		self.dss.Text.Command = (
			f"New Generator.exchange_{tie_name}_to "
			f"bus1={to_bus} phases=3 "
			f"kW=0 kvar=0 model=1"
		)
		self.logger.debug(
			f"创建交换元素: {tie_name} ({from_bus} -> {to_bus})"
		)

	def close(self) -> None:
		"""释放 DSS 资源"""
		if self._is_closed:
			return
		try:
			if hasattr(self, "dss") and self.dss:
				self.dss.ClearAll()
				self.logger.debug(
					f"Worker {self.worker_idx}: DSS 资源已清理"
				)
		except Exception as e:
			self.logger.warning(
				f"Worker {self.worker_idx}: DSS 资源清理出错: {e}"
			)
		finally:
			self._is_closed = True

	def __del__(self):
		"""析构函数 -- 确保资源被释放"""
		self.close()
