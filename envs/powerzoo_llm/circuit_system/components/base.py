"""
电力系统基础组件类定义

包含所有电力系统元件的基类
"""

class Edge:
	"""边元件基类，表示电力系统中连接两个节点的元件"""
	
	def __init__(self, name, bus1, bus2):
		self.name = name
		self.bus1 = bus1
		self.bus2 = bus2
	
	def __repr__(self):
		return f"Edge {self.name} at ({self.bus1}, {self.bus2}),"


class Node:
	"""节点元件基类，表示电力系统中的节点设备"""
	
	def __init__(self, name, bus1, phases):
		self.name = name
		self.bus1 = bus1
		self.phases = phases  # names of the active phases; e.g., ['1','2','3']
	
	def __repr__(self):
		return f"Node {self.name} at {self.bus1}"