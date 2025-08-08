"""
主电路管理类

负责电力系统电路的初始化、编译、求解和元件管理
"""

import networkx as nx
import numpy as np
import os
from pathlib import Path
import pandas as pd
import re

import dss as opendss
import logging

from .components.edge_components import Line, Transformer, Regulator
from .components.node_components import Load, Capacitor, PVSystem, Battery

# 获取日志记录器
try:
    from ..utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class Circuits:
	"""电力系统电路主管理类"""
	
	def __init__(self, dss_file, 
				batt_file='Battery.csv', 
				RB_act_num=(33, 33), 
				dss_act=False,
				worker_idx=None):
		# DSS
		self.dss = opendss.DSS  # the dss simulator object
		self.dss_file = dss_file  # path to the dss file for the whole circuit
		self.dss_act = dss_act  # whether to use OpenDSS controllers defined in the circuit file
		self.worker_idx = worker_idx  # worker index for multi-worker environments

		self.batt_file = os.path.join(Path(self.dss_file).parent, batt_file)
		if not os.path.exists(self.batt_file): 
			self.batt_file = ''
		
		self.topology = nx.Graph()
		self.edge_obj = dict()  # map from frozenset({bus1, bus2}) to the (active) object on the edge
		self.dup_edges = dict()  # map from duplicate edge (if any) to the objects on the edge
		self.edge_weight = dict()  # map from edge to Ymatrix. Ymatrix is symmetric/tall if the number of phases is equal/different 
		self.bus_phase = dict()  # map from bus name to the number of phases.
		self.bus_obj = dict()  # map from bus name to the objects(load,capacitor,batteries) on the bus
		
		# circuit element
		self.lines = dict()
		self.transformers = dict()
		self.regulators = dict()
		self.loads = dict()
		self.capacitors = dict()
		self.batteries = dict()
		
		self.pvs = dict()

		# regulator and battery action dim
		self.reg_act_num, self.bat_act_num = RB_act_num
		
		# initialization
		self.initialize()
	
	def set_regulator_parameters(self, tap=1.1, mintap=0.9, maxtap=1.1):
		'''
		将所有调压器参数设置为相同的预定义值

		参数:
			tap: 抽头值，介于mintap和maxtap之间
			maxtap: 最大抽头值
			mintap: 最小抽头值
		
		返回值: 无
		'''
		# run this after Text.Command = "compile" and before ActiveCircuit.Solution.Solve()
		
		# numtaps: number of tap values between mintap and maxtap.
		#          reg_act_num = numtaps + 1
		numtaps = self.reg_act_num - 1
		fea = [mintap, maxtap, numtaps]
		transet = set()
		for regname in self.regulators.keys():
			self.regulators[regname].tap_feature = fea.copy()
			self.regulators[regname].tap = tap
			transet.add(regname[10:])
		
		dssTrans = self.dss.ActiveCircuit.Transformers
		if dssTrans.First == 0: 
			return  # no such kind of object
		while True:
			if dssTrans.Name in transet:
				dssTrans.Tap = tap
				dssTrans.MinTap = mintap
				dssTrans.MaxTap = maxtap 
				dssTrans.NumTaps = numtaps
			if dssTrans.Next == 0: 
				break
	
	def compile(self, disable=False):
		'''
		编译主DSS文件

		参数:
			disable: 在求解过程中禁用电源和负载。
					这在计算导纳矩阵(Ymat)时使用

		返回值: 无
		'''
		# 保存当前工作目录
		current_dir = os.getcwd()
		
		# 获取DSS文件的目录并切换到该目录
		dss_dir = os.path.dirname(os.path.abspath(self.dss_file))
		dss_filename = os.path.basename(self.dss_file)
		
		try:
			# 切换到DSS文件所在目录，确保相对路径正确
			os.chdir(dss_dir)
			
			# 创建临时的编译文件，包含正确的数据文件引用
			temp_compile_file = self._create_temp_compile_file(dss_filename)
			
			# 使用临时文件编译
			self.dss.Text.Command = f"compile {temp_compile_file}"
			self.dss.Text.Command = "Set Maxiterations=50"
			self.dss.Text.Command = "Set Maxcontroliter=100"
			
			# 删除临时文件
			if os.path.exists(temp_compile_file):
				os.remove(temp_compile_file)
			
			if disable:
				self.dss.Text.Command = 'vsource.source.enabled=no'
				self.dss.Text.Command = 'batchedit load..* enabled=no'
			else:
				self.dss.Text.Command = 'vsource.source.enabled=yes'
				self.dss.Text.Command = 'batchedit load..* enabled=yes'
		finally:
			# 恢复原工作目录
			os.chdir(current_dir) 
		
		if not self.dss_act:
			self.dss.Text.Command = "Set ControlMode = off"
	
	def _create_temp_compile_file(self, dss_filename):
		'''
		创建临时编译文件，包含worker特定的数据文件引用
		
		参数:
			dss_filename: 原始DSS文件名
		
		返回值:
			临时文件名
		'''
		# 确定worker特定的文件名
		if self.worker_idx is not None:
			loadshape_file = f"loadshape_{self.worker_idx}.dss"
			pv_data_file = f"pv_data_{self.worker_idx}.dss"
			temp_filename = f"temp_compile_{self.worker_idx}.dss"
		else:
			loadshape_file = "loadshape.dss"
			pv_data_file = "pv_data.dss"
			temp_filename = "temp_compile.dss"
		
		# 检查文件存在性并选择备选
		if not os.path.exists(loadshape_file) and os.path.exists("loadshape.dss"):
			loadshape_file = "loadshape.dss"
			logger.debug(f"使用默认负荷文件: loadshape.dss")
		
		if not os.path.exists(pv_data_file):
			if os.path.exists("pv_systems_base.dss"):
				pv_data_file = "pv_systems_base.dss"
			elif os.path.exists("pv_data.dss"):
				pv_data_file = "pv_data.dss"
			logger.debug(f"使用备选PV文件: {pv_data_file}")
		
		# 创建临时文件内容
		temp_content = ["Clear\n"]
		
		# 添加负荷曲线文件（如果存在）
		if os.path.exists(loadshape_file):
			temp_content.append(f"redirect {loadshape_file}\n")
			logger.info(f"Worker {self.worker_idx}: 加载负荷文件 {loadshape_file}")
		
		# 添加主DSS文件内容（排除之前的redirect语句）
		temp_content.append(f"redirect {dss_filename}\n")
		
		# 添加PV数据文件（如果存在）
		if os.path.exists(pv_data_file):
			temp_content.append(f"redirect {pv_data_file}\n")
			logger.info(f"Worker {self.worker_idx}: 加载PV文件 {pv_data_file}")
		
		# 写入临时文件
		with open(temp_filename, 'w') as f:
			f.writelines(temp_content)
		
		return temp_filename
	
	
	def reset(self):
		'''
		将电路重置为初始状态
		
		参数: 无
		返回值: 无
		'''
		self.compile()  # this include resetting regulators in dss
						# capacitors in dss and batteries in dss
		# reset regulator, capacitors and batteries in objects
		self.set_regulator_parameters()
		self.set_all_capacitor_statuses([1] * len(self.capacitors), change_dss=False)
		for bat in self.batteries.keys():
			self.batteries[bat].reset()

		# Solve()会执行所有控制器(如调压器、电容器等)的控制逻辑
		# SolveNoControl()只求解电路,不执行控制器逻辑
		# 这里使用SolveNoControl()因为我们要手动控制这些设备
		# self.dss.ActiveCircuit.Solution.Solve()
		self.dss.ActiveCircuit.Solution.SolveNoControl()

	def initialize(self, noWei=True):
		'''
		编译并生成所有数据成员。

		参数:
			noWei: 不计算导纳矩阵(Ymat)，存储为self.edge_weight

		返回值: 无
		'''
		# 根据noWei参数决定是否计算导纳矩阵
        # 如果需要计算导纳矩阵(noWei=False):
        # 1. 先禁用电源和负载进行编译
        # 2. 计算导纳矩阵和母线相数
        # 3. 重新启用电源和负载进行编译
        # 如果不需要计算导纳矩阵(noWei=True):
        # 1. 直接启用电源和负载进行编译
        # 2. 只计算母线相数
        
		if not noWei:
			self.compile(disable=True)
			self.__cal_edgeWei_busPhase(noWei=noWei)
			self.compile(disable=False)
		else:
			self.compile(disable=False)
			self.__cal_edgeWei_busPhase(noWei=noWei)

		# edges
		regulators, valid_trans2edge, line2edge = self._get_edge_name()
		self._gen_reg_obj(regulators)
		self._gen_trans_obj(valid_trans2edge)
		self._gen_line_obj(line2edge)
		self.set_regulator_parameters()
		

		# nodes
		self._gen_load_cap_obj()
		if self.batt_file != '':
			self._gen_bat_obj()
		
		# self.dss.ActiveCircuit.Solution.Solve()
		self.dss.ActiveCircuit.Solution.SolveNoControl()
		
	def get_all_capacitor_statuses(self):
		'''
		获取所有电容器的状态

		返回值: 
			电容器状态 (字典)
		'''
		states = dict()
		dssCap = self.dss.ActiveCircuit.Capacitors
		if dssCap.First == 0: 
			return  # no such object 
		while True:
			states['Capacitor.' + dssCap.Name] = dssCap.States[0]
			if dssCap.Next == 0: 
				break
		return states

	def set_all_capacitor_statuses(self, statuses, change_dss=True):
		'''
		设置所有电容器的状态

		参数:
			statuses: 0-1整数数组
		
		返回值:
			状态变化的绝对值
		'''
		assert len(statuses) > 0 and len(statuses) == len(self.capacitors), '电容器状态数量与电容器数量不一致'
		statuses = np.array(statuses, dtype=int)

		# set capacitor objects
		diff = np.zeros(len(statuses))
		cap2st = dict()
		for i, cap in enumerate(self.capacitors.keys()):
			capa = self.capacitors[cap]
			old_status = capa.status
			diff[i] = abs(capa.status - statuses[i])
			capa.status = statuses[i]
			cap2st[capa.name[10:]] = statuses[i]
			
			# log capacitor status change
			if diff[i] > 0:
				logger.debug(f"电容器状态变化: {cap} | {old_status} -> {statuses[i]} | Diff: {diff[i]}")
		
		# set dss object
		if change_dss:
			dssCap = self.dss.ActiveCircuit.Capacitors
			if dssCap.First == 0: 
				logger.debug("未找到电容器DSS对象")
				return diff  # no such object 
			while True:
				if dssCap.Name in cap2st:
					dssCap.States = [cap2st[dssCap.Name]]
					logger.debug(f"DSS电容器状态设置: {dssCap.Name} -> {cap2st[dssCap.Name]}")
				if dssCap.Next == 0: 
					break
		
		logger.debug(f"电容器状态设置完成 - 总变化: {np.sum(diff)}")
		return diff

	def get_all_regulator_tapnums(self):
		'''
		获取所有调压器的抽头编号

		返回值: 
			调压器抽头编号 (字典)
		'''
		mintap, maxtap, numtaps = self.regulators[next(iter(self.regulators))].tap_feature
		step = (maxtap - mintap) / numtaps

		# get trans name
		trans = {regname[10:] for regname in self.regulators.keys()}

		# get taps from dss
		tapnums = dict()
		dssTrans = self.dss.ActiveCircuit.Transformers
		if dssTrans.First == 0: 
			return  # no such kind of object
		while True:
			if dssTrans.Name in trans:
				tapnums['Regulator.' + dssTrans.Name] = int((dssTrans.Tap - mintap) / step)
			if dssTrans.Next == 0: 
				break
		
		return tapnums

	def set_all_regulator_tappings(self, tapnums, change_dss=True):
		'''
		设置所有调压器的抽头值

		参数:
			tapnums: 整数数组,取值范围[0, numtaps]
		
		返回值:
			抽头值变化的绝对值
		'''
		assert len(tapnums) > 0 and len(tapnums) == len(self.regulators), 'inconsistent tapnums'
		mintap, maxtap, numtaps = self.regulators[next(iter(self.regulators))].tap_feature
		step = (maxtap - mintap) / numtaps
		tapnums = np.maximum(0, np.minimum(numtaps, np.array(tapnums, dtype=int)))
		taps = tapnums * step + mintap

		# set regulator objects
		diff = np.zeros(len(taps))
		trans2tap = dict()
		for i, reg in enumerate(self.regulators.keys()):
			regu = self.regulators[reg]
			old_tap = regu.tap
			diff[i] = abs((regu.tap - taps[i]) / step)
			regu.tap = taps[i]
			# remove 'Regulator.' from the head of the name
			trans2tap[reg[10:]] = (taps[i], tapnums[i])
			
			# log regulator tap change
			if diff[i] > 0:
				logger.debug(f"调压器抽头变化: {reg} | {old_tap:.3f} -> {taps[i]:.3f} | Diff: {diff[i]:.3f}")

		# set dss
		if change_dss:
			dssTrans = self.dss.ActiveCircuit.Transformers
			if dssTrans.First == 0: 
				logger.debug("未找到变压器DSS对象")
				return diff  # no such kind of object
			while True:
				if dssTrans.Name in trans2tap:
					tap, tapnum = trans2tap[dssTrans.Name]
					dssTrans.NumTaps = tapnum
					dssTrans.Tap = tap
					logger.debug(f"DSS调压器设置: {dssTrans.Name} | Tap: {tap:.3f} | TapNum: {tapnum}")
				if dssTrans.Next == 0: 
					break 
		
		logger.debug(f"调压器抽头设置完成 - 总变化: {np.sum(diff):.3f}")
		return diff

	def set_all_batteries_before_solve(self, nkws_or_states, change_dss=True):
		'''
		设置所有电池的状态
		(在每次solve()之前运行此函数)

		参数:
			nkws_or_states: nkws或states的数组
				nkw: 连续电池的标准化放电功率,范围[-1, 1]
				state: 离散电池的放电状态,范围[0, len(avail_kw)-1]
		
		返回值:
			状态变化的绝对值(整数)
		'''
		
		assert len(nkws_or_states) > 0 and len(nkws_or_states) == len(self.batteries), 'inconsistent states'
		if self.bat_act_num == np.inf:
			nkws_or_states = np.array(nkws_or_states, dtype=np.float32)
		else:
			nkws_or_states = np.array(nkws_or_states, dtype=int)

		# set battery object
		bat2kwkvar = dict()
		for i, bat in enumerate(self.batteries.keys()):
			batt = self.batteries[bat]
			old_kw = getattr(batt, 'kw', 0)
			kw = batt.state_projection(nkws_or_states[i])  # projection
			kvar = kw / batt.pf
			bat2kwkvar[batt.name[8:]] = (kw, kvar)  # remove the header 'Battery.'
			
			# log battery state change
			if abs(old_kw - kw) > 0.001:  # 阈值避免微小变化的日志
				logger.debug(f"电池状态变化: {bat} | {old_kw:.3f}kW -> {kw:.3f}kW | kvar: {kvar:.3f}")
	   
		# change kw in dss
		if change_dss:
			dssGen = self.dss.ActiveCircuit.Generators
			if dssGen.First == 0: 
				return  # no such kind of object
			while True:
				if dssGen.Name in bat2kwkvar:
					kw, kvar = bat2kwkvar[dssGen.Name]
					dssGen.kW = kw
					dssGen.kvar = kvar
				if dssGen.Next == 0: 
					break
		
		# run solve() afterward
		
	def set_all_pvs_before_solve(self, actions, change_dss=True):
		'''
		设置所有光伏系统的状态
		(在每次solve()之前运行此函数)

		参数:
			actions: 光伏动作数组
					对于离散控制: 单个标准化功率值数组，范围[0, 1]
					对于连续控制: [power_ratio, power_factor] 数组的列表
		
		返回值:
			状态变化的绝对值数组
		'''
		
		assert len(actions) > 0 and len(actions) == len(self.pvs), 'inconsistent actions'
		actions = np.array(actions)
		
		# set pv system objects
		pv2params = dict()
		diffs = np.zeros(len(actions))
		
		for i, pv_name in enumerate(self.pvs.keys()):
			pv_system = self.pvs[pv_name]
			
			# Get action for this PV system
			action = actions[i]
			
			# Handle both single value and multi-dimensional actions
			if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
				# Continuous control: [power_ratio, power_factor]
				p_ratio = action[0]
				pf = action[1]
				kw = pv_system.state_projection(p_ratio)
				# Update power factor
				pf = max(0.8, min(1.0, pf))  # Limit PF between 0.8 and 1.0
			else:
				# Discrete control: single value
				if isinstance(action, (list, np.ndarray)):
					nkw = action[0] if len(action) > 0 else action
				else:
					nkw = action
				kw = pv_system.state_projection(nkw)
				pf = pv_system.pf  # Use existing power factor
			
			# Record state difference
			diffs[i] = abs(pv_system.kW - kw)
			pv_system.kW = kw  # Update stored value
			
			# Calculate percentage of Pmpp
			pct_pmpp = (kw / pv_system.pmpp * 100) if pv_system.pmpp > 0 else 0
			pct_pmpp = max(0, min(100, pct_pmpp))  # Limit to 0-100%
			
			# Remove 'pv.' prefix from name if present
			pvname = pv_system.name
			if pvname.startswith('pv.'):
				pvname = pvname[3:]
			
			pv2params[pvname] = (pct_pmpp, pf)
		
		# Change DSS objects
		if change_dss:
			for pvname, (pct_pmpp, pf) in pv2params.items():
				# Use Text command to set PV parameters
				self.dss.Text.Command = f"edit pvsystem.{pvname} %Pmpp={pct_pmpp:.2f} pf={pf:.3f}"
		
		return diffs
		
		# run solve() afterward
		
	def set_all_batteries_after_solve(self):
		'''
		根据dss对象中显示的实际功率更新kwh和soc。
		(在每次solve()之后运行此函数)

		参数: 无
		返回值: soc误差和放电误差
		'''
		soc_errs, discharge_errs = np.zeros(len(self.batteries)), np.zeros(len(self.batteries))
		for i, bat in enumerate(self.batteries):
			batt = self.batteries[bat]
			batt.kwh += batt.actual_power() * batt.duration
			# enforce capacity constraint and round to integer
			batt.kwh = round(max(0.0, min(batt.max_kwh, batt.kwh)))
			batt.soc = batt.kwh / batt.max_kwh
			soc_errs[i] = abs(batt.soc - batt.initial_soc)
		   
			if self.bat_act_num == np.inf:
				discharge_errs[i] = max(0.0, batt.kw) / batt.max_kw
			else:
				discharge_errs[i] = max(0.0, batt.avail_kw[batt.state]) / batt.max_kw
		return soc_errs, discharge_errs

	# 其他方法保持不变...
	def _get_edge_name(self):
		'''
		计算所有边上的对象名称。

		参数: 无

		返回值:
			regulators [dict]: 从边到该调压器的所有变压器和调压控制器的映射。
			valid_trans2edge [dict]: 从非调压器变压器到边的映射
			line2edge [dict]: 从线路到边的映射
		'''
		def get_edge(type_name, ignore_duplicate=False):
			self.dss.ActiveCircuit.SetActiveElement(type_name)
			buses = self.dss.ActiveCircuit.ActiveElement.BusNames

			# transformer may have >2 buses
			if type_name.startswith('Transformer'):
				assert len(buses) in [2, 3], type_name + ' has invalid number of terminals'
			else:
				assert len(buses) == 2, type_name + ' has more than two terminals'

			if len(buses) == 2:
				bus1, bus2 = map(lambda x: x.lower().split('.'), buses)
				bus1, bus2 = bus1[0], bus2[0]
				edge = frozenset({bus1, bus2})
			else:
				bus1, bus2, bus3 = map(lambda x: x.lower().split('.'), buses)
				bus1, bus2, bus3 = bus1[0], bus2[0], bus3[0]
				edge = frozenset({bus1, bus2, bus3})
			
			if ignore_duplicate:
				return edge, False
			else:
				return edge, (edge in self.edge_obj)
			
		# deal with RegControls as an exception first        
		regulators = set()  # names of transformers acting as a regulator
		dssReg = self.dss.ActiveCircuit.RegControls
		if dssReg.First == 0: 
			return  # no such kind of object
		while True:
			# reg_name = dssReg.Name
			trans_name = dssReg.Transformer
			regulators.add(trans_name)
			if dssReg.Next == 0: 
				break
		
		# run for line and transformer
		valid_trans2edge = dict()
		line2edge = dict()
		for type in ['Transformer', 'Line']:
			if type == 'Line':
				names = self.dss.ActiveCircuit.Lines.AllNames
			elif type == 'Transformer':
				names = self.dss.ActiveCircuit.Transformers.AllNames
		  
			for name in names:
				# ignore regulators since they have been processed
				if type == 'Transformer' and name in regulators: 
					continue
				type_name = type + '.' + name
				edge, has_dup = get_edge(type_name)
				
				# handle duplicate
				if not has_dup:
					self.edge_obj[edge] = type_name
				else:
					if type == 'Line':  # allow jump circuit
						dup_name = self.edge_obj[edge]
						# assert not dup_name.startswith('Line'), 'Duplicated lines: {} {}'.format(dup_name, type_name)
						self.edge_obj[edge] = type_name
				 
					if edge not in self.dup_edges:
						self.dup_edges[edge] = [dup_name, type_name]
					else:
						self.dup_edges[edge].append(type_name)
				
				if type == 'Transformer':
					valid_trans2edge[name] = edge
				else:
					line2edge[name] = edge
		return regulators, valid_trans2edge, line2edge
	
	def _gen_reg_obj(self, regulators):
		'''
		生成所有调压器对象

		参数:
			regulators (set): 作为调压器的调压器名称集合

		返回值: 无
		'''
		if len(regulators) == 0: 
			return
		dssTrans = self.dss.ActiveCircuit.Transformers
		if dssTrans.First == 0: 
			return  # no such kind of object
		while True:
			name = dssTrans.Name
			if name in regulators:
				tap = [dssTrans.Tap, dssTrans.MinTap, dssTrans.MaxTap, dssTrans.NumTaps]
				fea = [dssTrans.Xhl, dssTrans.R]
				for wdg in range(1, 1 + dssTrans.NumWindings):
					dssTrans.Wdg = wdg
					fea = fea + [dssTrans.kV, dssTrans.kVA]
				if dssTrans.NumWindings == 3:
					fea = fea + [dssTrans.Xht, dssTrans.Xlt]
				
				# get the correct bus order
				self.dss.ActiveCircuit.SetActiveElement('Transformer.' + name)
				buses = self.dss.ActiveCircuit.ActiveElement.BusNames

				self.add_regulators('Regulator.' + name, buses, fea, tap)
			if dssTrans.Next == 0: 
				break
				
	def _gen_trans_obj(self, valid_trans2edge):
		'''
		生成所有变压器对象:

		参数: 
			valid_trans2edge [dict]: self._get_edge_name()的返回对象之一

		返回值: 无
		'''
		dssTrans = self.dss.ActiveCircuit.Transformers
		if dssTrans.First == 0: 
			return  # no such kind of object
		while True:
			name = dssTrans.Name
			if name in valid_trans2edge:
				# tap = [dssTrans.Tap, dssTrans.MinTap, dssTrans.MaxTap, dssTrans.NumTaps]
				fea = [dssTrans.Xhl, dssTrans.R]
				for wdg in range(1, 1 + dssTrans.NumWindings):
					dssTrans.Wdg = wdg
					fea = fea + [dssTrans.kV, dssTrans.kVA]
				if dssTrans.NumWindings == 3:
					fea = fea + [dssTrans.Xht, dssTrans.Xlt]
				
				# get the correct bus order
				self.dss.ActiveCircuit.SetActiveElement('Transformer.' + name)
				buses = self.dss.ActiveCircuit.ActiveElement.BusNames

				self.add_transformers('Transformer.' + name, buses, fea)
			if dssTrans.Next == 0: 
				break
	
	def _gen_line_obj(self, line2edge):
		'''
		生成所有线路对象:

		参数: 
			line2edge [dict]: self._get_edge_name()的返回对象之一

		返回值: 无
		'''
		dssLine = self.dss.ActiveCircuit.Lines
		if dssLine.First == 0: 
			return  # no such kind of object
		while True:
			name = dssLine.Name
			assert name in line2edge, 'missing line for Line.{}'.format(name)
			self.dss.ActiveCircuit.SetActiveElement('Line.' + name)
			buses = self.dss.ActiveCircuit.ActiveElement.BusNames
			mats = [dssLine.Rmatrix, dssLine.Xmatrix, dssLine.Cmatrix]
			self.add_lines('Line.' + name, buses, mats)
			if dssLine.Next == 0: 
				break
	
	
	def _gen_load_cap_obj(self):
		'''
		生成所有负载和电容器对象

		参数: 无

		返回值: 无
		'''
		for type in ['Load', 'Capacitor', 'PVSystem']:
			if type == 'Load':
				dssObj = self.dss.ActiveCircuit.Loads
			elif type == 'Capacitor':
				dssObj = self.dss.ActiveCircuit.Capacitors
			elif type == 'PVSystem':
				dssObj = self.dss.ActiveCircuit.PVSystems
			if dssObj.First == 0: 
				break  # no such kind of object
			while True:
				objname = self.dss.ActiveCircuit.CktElements.Name
				BusNames = self.dss.ActiveCircuit.CktElements.BusNames[0].split('.')
				bus = BusNames[0]
				if len(BusNames) > 1:
					phases = BusNames[1:]
				else: 
				# if not specifying the phases, use all phases at the bus
					phases = self.bus_phase[bus]
				
				if type == 'Load':
					fea = [dssObj.kV, dssObj.kW, dssObj.kvar]
					self.add_loads(objname, bus, phases, fea)
				elif type == 'Capacitor':
					fea = [dssObj.States[0], dssObj.kV, dssObj.kvar]
					self.add_capacitors(objname, bus, phases, fea)
				elif type == 'PVSystem':
					fea = [dssObj.PF, dssObj.IrradianceNow, dssObj.Irradiance, dssObj.Name, dssObj.Pmpp, dssObj.kW, dssObj.kvar, dssObj.kVArated]
					self.add_pvsystems(objname, bus, phases, fea)
				if dssObj.Next == 0: 
					break
				
	def _gen_bat_obj(self):
		'''生成self.batt_file中定义的所有电池对象

		参数: 无

		返回值: 无
		'''
		batt = pd.read_csv(self.batt_file, sep=',', header=0)
		batt = batt.set_index('name')
		batts = {name: feature for name, feature in batt.iterrows()}
		
		dssGen = self.dss.ActiveCircuit.Generators
		if dssGen.First == 0: 
			return  # no such kind of object
		while True:
			name = dssGen.Name
			if name in batts:
				feature = batts[name]
				dssGen.kW = 0.0  # initialize in disconnected mode
				dssGen.PF = feature.pf
				dssGen.kvar = feature.max_kw / feature.pf
				
				BusNames = self.dss.ActiveCircuit.CktElements.BusNames[0].split('.')
				bus = BusNames[0]
				if len(BusNames) > 1:
					phases = BusNames[1:]
				else: 
				# if not specifying the phases, use all phases at the bus
					phases = self.bus_phase[bus]
				self.add_batteries('Battery.' + name, bus, phases, feature)
			if dssGen.Next == 0: 
				break
	
	def __cal_edgeWei_busPhase(self, noWei=True):
		'''
		计算边权重(导纳矩阵，Ymat)和每个母线上的相数

		参数:
			noWei: 不计算边权重

		返回值: 无
		'''
		# sort y nodes in alphabetical order
        # 获取电路中所有节点的名称顺序
		YNodeOrder = np.array(self.dss.Circuits.YNodeOrder)
		# 对节点名称进行排序,返回排序后的索引数组
		order = np.argsort(YNodeOrder)
		# 根据排序索引重新排列节点名称,使其按小到大顺序排列
		YNodeOrder = YNodeOrder[order]
		
		# find the range and bus phase
		bus_range = dict()
		for i, node_name in enumerate(YNodeOrder):
			bus_name = node_name.split('.', 1)[0].lower()
			if bus_name not in bus_range:
				bus_range[bus_name] = [i, i + 1]
			else:
				bus_range[bus_name][1] = i + 1

		for bus_name in bus_range.keys():
			self.dss.Circuits.SetActiveBus(bus_name)
			self.bus_phase[bus_name] = [str(i) for i in self.dss.Circuits.Buses.Nodes]
		# self.bus_phase = {bus:r[1]-r[0] for bus, r in bus_range.items()}
		
		if noWei: 
			return
		bus_length = len(YNodeOrder)
		Y = self.dss.Circuits.SystemY
		Y = Y.reshape((bus_length, 2 * bus_length))
		Y = Y[:, ::2] + 1j * Y[:, 1::2]
		Y = Y[order, :][:, order]
			
		# extract the submatrix of edge weight from Y
		for edge in self.edge_obj.keys():
			bus1, bus2 = tuple(edge)
			range1, range2 = bus_range[bus1], bus_range[bus2]
			nphase1, nphase2 = range1[1] - range1[0], range2[1] - range2[0]
			
			if nphase1 == nphase2:
				# symmetric matrix if same number of phases
				self.edge_weight[edge] = Y[range1[0]:range1[1], range2[0]:range2[1]]
			else:
				# store as a tall matrix
				if nphase1 < nphase2:
					self.edge_weight[edge] = Y[range2[0]:range2[1], range1[0]:range1[1]]
				else:
					self.edge_weight[edge] = Y[range1[0]:range1[1], range2[0]:range2[1]]
	
	def bus_voltage(self, bus_name):
		"""
		返回：该母线各相(不含中性线)的 pu 电压幅值列表，例如 [1.01, 0.99]。
		"""
		bus = self.dss.ActiveCircuit.Buses(bus_name)
		nodes = list(bus.Nodes)                  # e.g. [1,2,3,4]，4 往往是中性线
		puVA  = list(bus.puVmagAngle)            # [V1,θ1,V2,θ2,...]
		mags_all = [puVA[2*i] for i in range(len(nodes))]
		# 仅保留相导体
		mags_phase = [v for v, n in zip(mags_all, nodes) if n in (1, 2, 3)]
		return mags_phase

	
	def get_Y_matrix(self):
		'''
		获取电路的Y矩阵
		
		返回值:
			Y矩阵
		'''
		Y_order = np.array(self.dss.Circuits.YNodeOrder)
		bus_length = len(Y_order)
		Yorder1 = np.argsort(Y_order)  # 把序号最小的节点放前面了
		Y_order = Y_order[Yorder1]  # 按从小到大的顺序排序,要统一电压与y之间的关系
		Y = self.dss.Circuits.SystemY  # opendss可以直接获得导纳阵
		Y = Y.reshape((bus_length, 2 * bus_length))
		Y = Y[:, ::2] + 1j * Y[:, 1::2]
		Y = Y[Yorder1, :][:, Yorder1]
		return Y
	
	def get_agent_bus_dict(self):
		'''
		获取智能体到母线的映射
		
		返回值:
			智能体到母线的映射
		'''
		reg_BUS_dict = {}
		cap_BUS_dict = {}
		bat_BUS_dict = {}

		for key, value in self.regulators.items():
			value_str = str(value)
			# s = "Edge Regulator.reg1 at (650, rg60),phases:(['1','2'],['1','2'])"
			match = re.search(r"at \((\w+), (\w+)\),phases:\(\[([^\[\]]*)\],\[([^\[\]]*)\]\)", value_str)  # 搞定！
			# phase的处理可以以逗号分隔
			result = []
			result1 = []  # 初始化空列表
			bus = [''] * 2  # 初始化包含两个空字符串的列表
			if match:
				bus[0] = match.group(1).upper()
				bus[1] = match.group(2).upper()
				for i in range(3, len(match.groups()) + 1):
					phases = match.group(i)
					if ',' in phases:
						result_list = phases.split(',')
					else:
						result_list = [phases]
					# result[i-3] = [bus[i-3] + '.' + item.strip("'") for item in result_list]
					result.append([bus[i - 3] + '.' + item.strip("'") for item in result_list])
				for sublist in result:
					result1.extend(sublist)
				reg_BUS_dict[key] = result1
				# print(reg_BUS_dict)
			# 至此，提取出了调压器电压和相数的字符串数字
			
		for key, value in self.capacitors.items():
			# 提取出电压部分
			value_str = str(value)
			match1 = re.search(r"Bus: '(\w+)'", value_str)
			match2 = re.search(r"phases:\[(.*?)\]", value_str)
			# phase的处理可以以逗号分隔
			result = []
			bus = [''] * 1  # 初始化包含一个空字符串的列表
			if match1:
				bus[0] = match1.group(1).upper()
				phases = match2.group(1)
				if ',' in phases:
					result_list = phases.replace(" ", "").split(',')
				else:
					result_list = [phases]
					# result[i-3] = [bus[i-3] + '.' + item.strip("'") for item in result_list]
				result.append([bus[0] + '.' + item.strip("'") for item in result_list])
				#    for sublist in result:
				#        result1.extend(sublist)
				cap_BUS_dict[key] = result[0]
				# print(cap_BUS_dict)
			# 至此，提取出了调压器电压和相数的字符串数字
		  
			'''
			{
				'Capacitor.cap1': '675.1','675.2','675.3',
				'Capacitor.cap2': '611.3'
			}
			'''
		for key, value in self.batteries.items():
			value_str = str(value)
			match1 = re.search(r"Bus: '(\w+)'", value_str)
			match2 = re.search(r"phases:\[(.*?)\]", value_str)
			# phase的处理可以以逗号分隔
			result = []
			bus = [''] * 1  # 初始化包含一个空字符串的列表
			if match1:
				bus[0] = match1.group(1).upper()
				phases = match2.group(1)
				if ',' in phases:
					result_list = phases.replace(" ", "").split(',')
				else:
					result_list = [phases]
				result.append([bus[0] + '.' + item.strip("'") for item in result_list])
				bat_BUS_dict[key] = result[0]
		agent_bus_dict = reg_BUS_dict.copy()
		agent_bus_dict.update(cap_BUS_dict)
		agent_bus_dict.update(bat_BUS_dict)
		return agent_bus_dict
	

	def edge_current(self, edge_obj_name):
		'''
		获取边对象上的电流

		参数:
			edge_obj_name [str]: 边对象的名称

		返回值:
			以实部和虚部表示的电流值。
		'''
		return self.dss.ActiveCircuit.CktElements(edge_obj_name).Currents
	
	def total_loss(self):
		'''
		获取电路的总损耗
		
		返回值:
			总损耗(kW, kvar) - (实部：有功损耗, 虚部：无功损耗)
		'''
		losses = self.dss.ActiveCircuit.Losses  # complex 类型，单位：W + jVar
		
		# 处理可能的数组类型
		if hasattr(losses, '__len__') and len(losses) > 0:
			real_W = float(losses[0]) if len(losses) > 0 else 0.0
			imag_W = float(losses[1]) if len(losses) > 1 else 0.0
		else:
			real_W = float(losses.real)
			imag_W = float(losses.imag)
		
		real_kW = abs(real_W) / 1000  # 损耗总是正数
		reactive_kvar = abs(imag_W) / 1000  # 损耗总是正数
		return real_kW, reactive_kvar
	
	def total_power(self):
		'''
		获取来自电源的总功率（净功率）
		注意：在有PV系统时，这可能是负载减去PV发电的净值
		
		返回值:
			来自主电源的总功率(kW, kvar)
		'''
		power = self.dss.ActiveCircuit.TotalPower  # complex类型，单位：W + jVar
		
		# 处理可能的数组类型
		if hasattr(power, '__len__') and len(power) > 0:
			real_W = float(power[0]) if len(power) > 0 else 0.0
			imag_W = float(power[1]) if len(power) > 1 else 0.0
		else:
			real_W = float(power.real)
			imag_W = float(power.imag)
		
		return real_W / 1000, imag_W / 1000
	
	def total_load_power(self):
		'''
		计算系统总负载功率
		
		返回值:
			总负载功率(kW, kvar)
		'''
		total_load_kw = 0.0
		total_load_kvar = 0.0
		
		# 遍历所有负载
		dssLoad = self.dss.ActiveCircuit.Loads
		if dssLoad.First != 0:
			while True:
				total_load_kw += abs(dssLoad.kW)
				total_load_kvar += abs(dssLoad.kvar)
				if dssLoad.Next == 0:
					break
		
		return total_load_kw, total_load_kvar
	
	def calculate_loss_percentage(self):
		'''
		计算正确的功率损失百分比
		
		返回值:
			功率损失百分比 (%)
		'''
		try:
			# 获取系统损耗
			loss_kw, _ = self.total_loss()
			
			# 获取总负载功率
			load_kw, _ = self.total_load_power()
			
			# 如果负载功率很小，使用来自电源的功率作为备选
			if load_kw < 10.0:  # 小于10kW时使用电源功率
				source_kw, _ = self.total_power()
				denominator = max(abs(source_kw), abs(load_kw), 1.0)
			else:
				denominator = load_kw
			
			# 计算损失百分比
			loss_percentage = (loss_kw / denominator) * 100.0
			
			logger.debug(f"功率损失计算: 损耗={loss_kw:.3f}kW, 负载={load_kw:.3f}kW, 百分比={loss_percentage:.3f}%")
			
			return loss_percentage
			
		except Exception as e:
			logger.error(f"计算功率损失百分比时出错: {e}")
			return 0.0
	
	# object addition functions called by  
	#          _gen_reg_obj()
	#          _gen_trans_obj()
	#          _gen_line_obj()
	#          _gen_load_cap_obj()
	#          _gen_bat_obj()
	def add_lines(self, linename, buses, mats):
		'''
		向电路添加线路
		
		参数:
			linename: 线路名称
			buses: 母线列表
			mats: 矩阵参数
		
		返回值: 无
		'''
		self.lines[linename] = Line(linename, buses, mats)
	
	def add_transformers(self, transname, buses, feature):
		'''
		向电路添加变压器
		
		参数:
			transname: 变压器名称
			buses: 母线列表
			feature: 变压器特性
		
		返回值: 无
		'''
		self.transformers[transname] = Transformer(transname, buses, feature)

	def add_regulators(self, regname, buses, feature, tap):
		'''
		向电路添加调压器
		
		参数:
			regname: 调压器名称
			buses: 母线列表
			feature: 调压器特性
			tap: 抽头参数
		
		返回值: 无
		'''
		self.regulators[regname] = Regulator(self.dss, regname, buses, feature, tap)

	def add_capacitors(self, capname, bus, phases, feature):
		'''
		向电路添加电容器
		
		参数:
			capname: 电容器名称
			bus: 母线
			phases: 相位
			feature: 电容器特性
		
		返回值: 无
		'''
		self.capacitors[capname] = Capacitor(self.dss, capname, bus, phases, feature)
		if bus not in self.bus_obj:
			self.bus_obj[bus] = [capname]
		else:
			self.bus_obj[bus].append(capname)
			
	def add_loads(self, loadname, bus, phases, feature):
		'''
		向电路添加负载
		
		参数:
			loadname: 负载名称
			bus: 母线
			phases: 相位
			feature: 负载特性
		
		返回值: 无
		'''
		self.loads[loadname] = Load(loadname, bus, phases, feature)
		if bus not in self.bus_obj:
			self.bus_obj[bus] = [loadname]
		else:
			self.bus_obj[bus].append(loadname)
	
	
	def add_pvsystems(self, pvname, bus, phases, feature):
		'''
		向电路添加光伏系统
		
		参数:
			pvname: 光伏系统名称
			bus: 母线
			phases: 相位
			feature: 光伏系统特性
		
		返回值: 无
		'''
		self.pvs[pvname] = PVSystem(self.dss, pvname, bus, phases, feature)
		if bus not in self.bus_obj:
			self.bus_obj[bus] = [pvname]
		else:
			self.bus_obj[bus].append(pvname)

	def add_batteries(self, batname, bus, phases, feature):
		'''
		向电路添加电池
		
		参数:
			batname: 电池名称
			bus: 母线
			phases: 相位
			feature: 电池特性
		
		返回值: 无
		'''
		self.batteries[batname] = Battery(self.dss, batname, bus, phases, feature,
											bat_act_num=self.bat_act_num)
		if bus not in self.bus_obj:
			self.bus_obj[bus] = [batname]
		else:
			self.bus_obj[bus].append(batname)