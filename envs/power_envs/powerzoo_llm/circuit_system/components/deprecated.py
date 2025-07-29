"""
已废弃的类定义

包含不再使用的类，保留用于兼容性
"""

import numpy as np
from .base import Edge


class MergedRegulator(Edge):
	"""已废弃的合并调压器类"""
	
	def __init__(self, dss, name, ori_names, edge, feature):
		bus1, bus2 = tuple(edge)
		super().__init__(name, bus1, bus2)
		self.dss = dss                   # the circuit's dss simulator object
		self.tap = feature[0][0]         # tap value
		self.tap_feature = feature[0][1:]# [mintap, maxtap, numtaps]
		self.trans_feature = feature[1]  # [xhl, r, kv_wdg1, kva_wdg1, kv_wdg2, kva_wdg2]
		self.regctr_feature = feature[2] # [ForwardR, ForwardX, ForwardBand, ForwardVreg, CTPrimary, PTratio]
		self.ori_trans = []              # the transformer names associated with this regulator
		self.ori_regctr = []             # the regulator names associated with this regulator
		for trans, regctr in ori_names:
			self.ori_trans.append(trans)
			self.ori_regctr.append(regctr)

	def __repr__(self):
		return f'Reg Current Tapping: {self.tap!r}, Reg(mintap, maxtap, numtaps): {self.tap_feature!r} \
		Voltage at Bus: {self.bus1, self.dss.ActiveCircuit.Buses[self.bus1].puVmagAngle!r}, \
		Voltage at Bus: {self.bus2, self.dss.ActiveCircuit.Buses[self.bus2].puVmagAngle!r}'
	
	def set_tapping(self, numtap):
		'''
		将此调压器的抽头值设置为 mintap + tapnum * (maxtap-mintap)/numtaps

		参数:
			numtap: 整数值，范围[0, numtaps]
		
		返回值:
			抽头编号变化的绝对值(整数)
		'''
		numtap = min(self.tap_feature[2], max(0, numtap))
		step = (self.tap_feature[1] - self.tap_feature[0]) / self.tap_feature[2]
		new_tap = numtap * step + self.tap_feature[0]
		diff = abs((self.tap - new_tap) / step)  # record tap difference
		self.tap = new_tap
		
		dssTrans = self.dss.ActiveCircuit.Transformers
		dssTrans.First
		while True:
			if dssTrans.Name in self.ori_trans:
				dssTrans.NumTaps = numtap
				dssTrans.Tap = self.tap
			if dssTrans.Next == 0: 
				break
		
		return diff
		# should re-solve self.dss later