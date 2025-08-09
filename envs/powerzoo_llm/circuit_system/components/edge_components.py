"""
边元件类定义

包含线路、变压器、调压器等边连接元件
"""

from .base import Edge


class Line(Edge):
	"""输电线路类"""
	
	def __init__(self, name, buses, mats):
		bus1, bus2 = map(lambda x: x.lower().split('.'), buses)
		self.phase1, self.phase2 = map(lambda b: b[1:] if len(b) > 1 else ['1', '2', '3'], [bus1, bus2])
		bus1, bus2 = bus1[0], bus2[0]
		super().__init__(name, bus1, bus2)
		
		# The matrices are symmetric if both buses are of the same number of phases; 
		# Otherwise, the matrices are represented as a tall matrix.
		self.rmat = mats[0]  # resistance matrix
		self.xmat = mats[1]  # reactance matrix
		self.cmat = mats[2]  # capacitance matrix


class Transformer(Edge):
	"""变压器类"""
	
	def __init__(self, name, buses, feature):
		if len(buses) == 2:
			bus1, bus2 = map(lambda x: x.lower().split('.'), buses)
			phase1, phase2 = map(lambda b: b[1:] if len(b) > 1 else ['1', '2', '3'], [bus1, bus2])
			bus1, bus2 = bus1[0], bus2[0]
		else:
			bus1, bus2, bus3 = map(lambda x: x.lower().split('.'), buses)
			phase1, phase2, phase3 = map(lambda b: b[1:] if len(b) > 1 else ['1', '2', '3'], 
										  [bus1, bus2, bus3])
			bus1, bus2, bus3 = bus1[0], bus2[0], bus3[0]
			self.bus3 = bus3
			self.phase3 = phase3
		super().__init__(name, bus1, bus2)
		self.phase1 = phase1
		self.phase2 = phase2

		# 2 windings: [xhl, r, kv_wdg1, kva_wdg1, kv_wdg2, kva_wdg2]
		# 3 windings: [xhl, r, kv_wdg1, kva_wdg1, kv_wdg2, kva_wdg2, kv_wdg3, kva_wdg3, xht, xlt]
		self.trans_feature = feature


class Regulator(Edge):
	"""调压器类"""
	
	def __init__(self, dss, name, buses, feature, tapfea):
		assert len(buses) == 2, 'invalid number of buses for ' + name
		bus1, bus2 = map(lambda x: x.lower().split('.'), buses)
		phase1, phase2 = map(lambda b: b[1:] if len(b) > 1 else ['1', '2', '3'], [bus1, bus2])
		bus1, bus2 = bus1[0], bus2[0]
		super().__init__(name, bus1, bus2)
		self.phase1 = phase1
		self.phase2 = phase2

		self.trans_feature = feature     # [xhl, r, kv_wdg1, kva_wdg1, kv_wdg2, kva_wdg2]
		self.dss = dss                   # the circuit's dss simulator object
		self.tap = tapfea[0]             # tap value
		self.tap_feature = tapfea[1:]    # [mintap, maxtap, numtaps]
		
	def __repr__(self):
		return f"Edge {self.name} at ({self.bus1}, {self.bus2}),phases:({self.phase1},{self.phase2})"