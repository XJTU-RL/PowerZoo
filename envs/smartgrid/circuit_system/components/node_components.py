"""
节点元件类定义

包含负载、电容器、光伏系统、电池等节点设备
"""

import math
import numpy as np
import logging
from .base import Node

# 获取日志记录器
try:
    from ...utils import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class Load(Node):
	"""负载类"""
	
	def __init__(self, loadname, bus1, phases, feature):
		super().__init__(loadname, bus1, phases)
		self.feature = feature  # [kV, kW, kvar]


class Capacitor(Node):
	"""电容器类"""
	
	def __init__(self, dss, capname, bus1, phases, feature):
		super().__init__(capname, bus1, phases)
		self.dss = dss             # the circuit's dss simulator object
		self.status = feature[0]   # close: 1,  open: 0
		self.feature = feature[1:] # [kV, kvar]
		self.bus = bus1
		self.phases = phases

	def __repr__(self):
		return f'Capacitor status: {self.status!r},\
			   Voltage at Bus: {self.bus1!r}, {self.dss.ActiveCircuit.Buses[self.bus1].puVmagAngle},\
			   phases:{self.phases!r}'
	
	def set_status(self, status):
		'''
		设置此电容器的状态

		参数:
			status: 要设置的状态
		
		返回值:
			状态变化的绝对值(整数)
		'''
		old_status = self.status
		diff = abs(self.status - status)  # record state difference
		
		dssCap = self.dss.ActiveCircuit.Capacitors
		if dssCap.First == 0: 
			logger.warning(f"未找到电容器DSS对象: {self.name}")
			return diff  # no such object 
		while True:
			if self.name.endswith(dssCap.Name):
				self.status = status
				dssCap.States = [self.status]
				logger.debug(f"电容器状态设置: {self.name} | {old_status} -> {status} | Diff: {diff}")
				break
			if dssCap.Next == 0: 
				break
		return diff
		# should re-solve self.dss later


class PVSystem(Node):
	"""
	光伏系统类
	用以获得光伏系统的参数，包括有功、无功、当前的辐照度(光伏的随机性)，光伏设备所在的节点，以及光伏设备最大的有功输出量
	"""
	
	def __init__(self, dss, pvname, bus1, phases, feature, pv_act_num=np.inf):
		super().__init__(pvname, bus1, phases)
		self.dss = dss             # the circuit's dss simulator object
		self.pf = feature[0]       # 功率因数
		self.kW = feature[5]       # 实际功率 (kW)
		# to calculate the max active power for the pv system
		self.pmpp = feature[4]     # 最大功率点 (kW)
		self.kvar = feature[6]     # 无功功率 (kvar)
		self.irradiance = feature[2]  # 最大辐照度
		# here the max active power is calculated
		# self.pdc=self.pmpp*self.irradianceNow#实际可以发出的最大功率
		self.pv_act_num = pv_act_num  # 控制光伏有功输出大小的量
		self.pctpmpp = 1
		self.max_output = 100
		self.bus = bus1
		self.phases = phases

	def __repr__(self):
		return f'PV kW: {self.kW!r},\
			   irradiance: {self.irradiance!r},\
			   phases:{self.phases!r}'
			   
	def state_projection(self, nkw):
		'''
		投影到有效状态#把光伏输出限制在可靠的区间范围内。

		参数:
			nkw: 标准化功率输出，范围[0, 1]，0表示不发电，1表示满功率发电
		返回值:
			有效的发电功率(kw)，范围[0, pmpp]
		'''
		if self.pv_act_num == np.inf:  # 连续控制模式
			# 确保nkw在[0, 1]范围内，然后乘以pmpp得到实际功率
			normalized_power = max(0.0, min(1.0, float(nkw)))
			kw = normalized_power * self.pmpp
			
			# 确保不超过最大功率点
			kw = min(kw, self.pmpp)
			
			logger.debug(f"PV系统 {self.name}: 标准化功率={normalized_power:.3f}, 实际功率={kw:.3f}kW, Pmpp={self.pmpp:.3f}kW")
			return kw
		else:
			# 离散控制模式 - 需要根据实际需求实现
			logger.warning(f"PV系统 {self.name} 使用离散控制模式，但未实现相应逻辑")
			return 0.0

	def step_before_solve(self, action):
		"""
		在每次求解前设置此光伏系统的状态
		(在每次solve()之前运行此函数)

		参数:
			action: 离散控制的单个值(nkw)
				   或连续控制的列表/数组[p_ratio, pf]
		
		返回值: diff (状态差异)
		"""
		
		# Handle both single value and list/array inputs
		if isinstance(action, (list, np.ndarray)) and len(action) >= 2:
			# Continuous control: [power_ratio, power_factor]
			p_ratio = action[0]
			pf = action[1]
			kw = self.state_projection(p_ratio)
			# Update power factor
			self.pf = max(0.8, min(1.0, pf))  # Limit PF between 0.8 and 1.0
		else:
			# Discrete control: single value
			if isinstance(action, (list, np.ndarray)):
				nkw = action[0] if len(action) > 0 else action
			else:
				nkw = action
			kw = self.state_projection(nkw)
			pf = self.pf  # Use existing power factor

		# change kw in dss
		diff = abs(self.kW - kw)  # record state difference
		self.kW = kw  # Update stored value
		
		# Use Text command to control PV output by adjusting %Pmpp
		# Since we can't directly set kW, we control the percentage of Pmpp
		pct_pmpp = (kw / self.pmpp * 100) if self.pmpp > 0 else 0
		pct_pmpp = max(0, min(100, pct_pmpp))  # Limit to 0-100%
		
		# Calculate kvar based on power factor
		if pf < 1.0:
			kvar = (kw / pf) * math.sqrt(1 - pf ** 2)
		else:
			kvar = 0.0
			
		# Remove 'pv.' prefix from name if present
		pvname = self.name
		if pvname.startswith('pv.'):
			pvname = pvname[3:]
			
		# Use Text command to set PV parameters
		self.dss.Text.Command = f"edit pvsystem.{pvname} %Pmpp={pct_pmpp:.2f} pf={pf:.3f}"
		
		return diff
		
		# run solve() afterward
	
	def get_status(self):
		'''
		获取此光伏系统的当前状态
		
		返回值:
			[power_ratio, power_factor]: 当前功率输出比率(0-1)和功率因数
		'''
		# 从OpenDSS获取实际功率输出
		try:
			# 移除pvsystem.前缀获取实际PV名称
			pv_name = self.name
			if pv_name.lower().startswith('pvsystem.'):
				pv_name = pv_name[9:]  # 移除'pvsystem.'前缀
				
			# 设置当前PV系统并获取实际功率
			self.dss.ActiveCircuit.SetActiveElement(f'PVSystem.{pv_name}')
			actual_powers = self.dss.ActiveCircuit.ActiveElement.Powers  # [P1, Q1, P2, Q2, ...]
			
			if actual_powers is not None and len(actual_powers) >= 2:
				# 计算总有功功率 (三相系统取前3对P值的和)
				actual_kw = sum(actual_powers[i] for i in range(0, min(len(actual_powers), 6), 2))
				actual_kw = abs(actual_kw) / 1000.0  # 转换为kW
				
				# 基于实际功率和pmpp计算功率比率
				if self.pmpp > 0:
					power_ratio = min(max(actual_kw / self.pmpp, 0.0), 1.0)
				else:
					power_ratio = 0.0
				
				# 更新内部kW值
				self.kW = actual_kw
				
				logger.debug(f"PV系统 {pv_name}: 实际功率={actual_kw:.3f}kW, Pmpp={self.pmpp:.3f}kW, 利用率={power_ratio:.3f}")
				
			else:
				# 如果无法获取实际功率，使用存储的kW值
				if self.pmpp > 0:
					power_ratio = min(max(self.kW / self.pmpp, 0.0), 1.0)
				else:
					power_ratio = 0.0
				logger.warning(f"无法获取PV系统 {pv_name} 的实际功率，使用存储值 {self.kW:.3f}kW")
			
		except Exception as e:
			logger.warning(f"获取PV系统 {self.name} 状态时出错: {e}，使用备用计算")
			# 备用方案：使用存储的kW值
			if self.pmpp > 0:
				power_ratio = min(max(self.kW / self.pmpp, 0.0), 1.0)
			else:
				power_ratio = 0.0
		
		return [power_ratio, self.pf]


class Battery(Node):
	"""电池类"""
	
	def __init__(self, dss, batname, bus1, phases, feature, bat_act_num=33):
		super().__init__(batname, bus1, phases)
		self.dss = dss                     # the circuit's dss simulator object
		self.max_kw = feature.max_kw       # maximum power magnitude
		self.pf = feature.pf               # power factor
		self.max_kwh = feature.max_kwh     # capacity
		self.kwh = feature.initial_kwh     # current charge
		self.soc = self.kwh / self.max_kwh # state of charge
		self.initial_soc = self.soc
		self.duration = self.dss.ActiveCircuit.Solution.StepSize / 3600.0  # time step in hour
		if self.duration < 1e-5: 
			self.duration = 1.0
		
		# battery states
		self.bat_act_num = bat_act_num
		if bat_act_num == np.inf:
			# continuous discharge state
			self.kw = 0.0
		else:
			# finite discharge state
			# current kw = avail_kw[state]
			# kw > 0 means discharging
			mode_num = bat_act_num // 2
			diff = self.max_kw / mode_num
			# avail_kw: a discrete range from -max_kw to max_kw
			self.avail_kw = [n * diff for n in range(-mode_num, mode_num + 1)]
			self.state = len(self.avail_kw) // 2  # initialize as disconnected mode
		 
	def __repr__(self):
		return f'Battery Available kW: {self.avail_kw!r}, \
		Status: {self.state!r}, kWh: {self.kwh!r}, SOC: {self.soc!r}, \
		Actual kW: {-self.actual_power()!r}, Voltage at Bus: {self.bus1!r},phases:{self.phases!r}'
   
	def state_projection(self, nkw_or_state):
		'''
		投影到有效状态

		参数:
			nkw_or_state: nkw: 连续电池的标准化放电功率，范围[-1, 1]
						  state: 离散电池的放电状态，范围[0, len(avail_kw)-1]
		返回值:
			有效的放电功率(kw)
		'''
		if self.bat_act_num == np.inf:
			kw = max(-1.0, min(1.0, nkw_or_state)) * self.max_kw
			if kw > 0:
				kw = min(self.kwh / self.duration, kw)
			else:
				kw = max((self.kwh - self.max_kwh) / self.duration, kw)
			self.kw = kw
			return kw
		else:
			state = max(0, min(len(self.avail_kw) - 1, nkw_or_state))
			mid = len(self.avail_kw) // 2
			if state > mid:  # discharging
				allowed_kw = self.kwh / self.duration  # max kw
				if self.avail_kw[state] > allowed_kw:
					state = int(state - np.ceil((self.avail_kw[state] - allowed_kw) / 
											   (self.avail_kw[1] - self.avail_kw[0]) - 1e-8))
			elif state < mid:  # charging
				allowed_kw = (self.kwh - self.max_kwh) / self.duration  # min kw
				if self.avail_kw[state] < allowed_kw:
					state = int(state + np.ceil((allowed_kw - self.avail_kw[state]) / 
											   (self.avail_kw[1] - self.avail_kw[0]) - 1e-8))
			self.state = state
			return self.avail_kw[state]

	def step_before_solve(self, nkw_or_state):
		'''
		设置此电池的状态
		(在每次solve()之前运行此函数)

		参数:
			nkw_or_state: nkw: 连续电池的标准化放电功率，范围[-1, 1]
						  state: 离散电池的放电状态，范围[0, len(avail_kw)-1]
		
		返回值: 无
		'''
		
		kw = self.state_projection(nkw_or_state)

		# change kw in dss
		name = self.name[8:]  # remove the header 'Battery.'
		dssGen = self.dss.ActiveCircuit.Generators
		if dssGen.First == 0: 
			return  # no such kind of object
		while True:
			if dssGen.Name == name:          
				dssGen.kW = kw
				dssGen.kvar = kw / self.pf
				break
			if dssGen.Next == 0: 
				break
		
		# run solve() afterward
		
	def step_after_solve(self):
		'''
		根据dss对象中显示的实际功率更新kwh和soc。
		(在每次solve()之后运行此函数)

		参数: 无
		返回值: soc误差和放电误差
		'''
		self.kwh += self.actual_power() * self.duration
		# enforce capacity constraint - 移除round()以保留精度
		self.kwh = max(0.0, min(self.max_kwh, self.kwh))
		self.soc = self.kwh / self.max_kwh
		soc_err = abs(self.soc - self.initial_soc)
		
		if self.bat_act_num == np.inf:
			discharge_err = max(0.0, self.kw) / self.max_kw
		else:
			discharge_err = max(0.0, self.avail_kw[self.state]) / self.max_kw
		return soc_err, discharge_err

	def actual_power(self):
		'''
		获取此电池的实际功率
		
		返回值:
			实际功率(kw)
		'''
		# get the actual power computed by dss in [-kw, -kvar]
		# in dss, the minus means power generation
		# so actual_power < 0 means discharging
		name = 'Generator.' + self.name[8:]
		return self.dss.ActiveCircuit.CktElements(name).TotalPowers[0]
	
	def reset(self):
		'''
		将电池重置为初始状态
		
		参数: 无
		返回值: 无
		'''
		# reset the charge
		self.soc = self.initial_soc
		self.kwh = self.soc * self.max_kwh

		# reset to zero discharge mode
		if self.bat_act_num == np.inf:
			self.kw = 0.0
		else:
			self.state = len(self.avail_kw) // 2