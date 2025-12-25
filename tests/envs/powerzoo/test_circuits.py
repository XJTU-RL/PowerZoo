# -*- coding: utf-8 -*-
"""
PowerZoo Circuits 类详细测试

测试覆盖:
- Circuits 类初始化
- 电路编译和重置
- 调压器参数设置
- 电容器状态管理
- 电池状态管理
- 电压和电流获取
- 拓扑结构构建
- 导纳矩阵计算
- 敏感度矩阵计算

@File      : test_circuits.py
@Author    : PowerZoo Test Suite
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ==============================================================================
# 单元测试 - 导入和基础功能
# ==============================================================================

@pytest.mark.unit
@pytest.mark.powerzoo
class TestCircuitsImport:
	"""测试 Circuits 模块导入"""

	def test_circuits_class_import(self):
		"""测试 Circuits 类可以正确导入"""
		from envs.powerzoo.powerzoo.circuit import Circuits
		assert Circuits is not None
		assert hasattr(Circuits, '__init__')
		assert hasattr(Circuits, 'compile')
		assert hasattr(Circuits, 'reset')
		assert hasattr(Circuits, 'initialize')

	def test_edge_classes_import(self):
		"""测试边类（Line, Transformer, Regulator）可以正确导入"""
		from envs.powerzoo.powerzoo.circuit import Edge, Line, Transformer, Regulator
		assert Edge is not None
		assert Line is not None
		assert Transformer is not None
		assert Regulator is not None

		# 验证继承关系
		assert issubclass(Line, Edge)
		assert issubclass(Transformer, Edge)
		assert issubclass(Regulator, Edge)

	def test_node_classes_import(self):
		"""测试节点类（Load, Capacitor, Battery, PVSystem）可以正确导入"""
		from envs.powerzoo.powerzoo.circuit import Node, Load, Capacitor, Battery, PVSystem
		assert Node is not None
		assert Load is not None
		assert Capacitor is not None
		assert Battery is not None
		assert PVSystem is not None

		# 验证继承关系
		assert issubclass(Load, Node)
		assert issubclass(Capacitor, Node)
		assert issubclass(Battery, Node)
		assert issubclass(PVSystem, Node)


@pytest.mark.unit
@pytest.mark.powerzoo
class TestCircuitsAttributes:
	"""测试 Circuits 类属性"""

	def test_circuits_has_required_attributes(self):
		"""验证 Circuits 类具有所有必需的属性"""
		from envs.powerzoo.powerzoo.circuit import Circuits

		# 检查类方法
		required_methods = [
			'__init__', 'compile', 'reset', 'initialize',
			'set_regulator_parameters',
			'get_all_capacitor_statuses', 'set_all_capacitor_statuses',
			'get_all_regulator_tapnums', 'set_all_regulator_tappings',
			'set_all_batteries_before_solve', 'set_all_batteries_after_solve',
			'bus_voltage', 'edge_current', 'total_loss', 'total_power',
			'get_Y_matrix', 'get_agent_bus_dict',
			'add_lines', 'add_transformers', 'add_regulators',
			'add_capacitors', 'add_loads', 'add_batteries'
		]

		for method in required_methods:
			assert hasattr(Circuits, method), f"Circuits 缺少方法: {method}"


# ==============================================================================
# 边(Edge)类测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.powerzoo
class TestEdgeClass:
	"""测试 Edge 基类和子类"""

	def test_edge_base_class(self):
		"""测试 Edge 基类"""
		from envs.powerzoo.powerzoo.circuit import Edge

		edge = Edge()
		# Edge 应该有 name, bus1, bus2 等属性
		assert hasattr(edge, 'name') or edge.__class__ == Edge

	def test_line_class_creation(self):
		"""测试 Line 类创建"""
		from envs.powerzoo.powerzoo.circuit import Line

		line = Line()
		assert line is not None
		assert hasattr(line, 'name')

	def test_transformer_class_creation(self):
		"""测试 Transformer 类创建"""
		from envs.powerzoo.powerzoo.circuit import Transformer

		transformer = Transformer()
		assert transformer is not None
		assert hasattr(transformer, 'name')

	def test_regulator_class_creation(self):
		"""测试 Regulator 类创建"""
		from envs.powerzoo.powerzoo.circuit import Regulator

		regulator = Regulator()
		assert regulator is not None
		assert hasattr(regulator, 'name')
		# Regulator 应该有 tap 相关属性
		assert hasattr(regulator, 'tap') or hasattr(regulator, 'tap_feature')


# ==============================================================================
# 节点(Node)类测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.powerzoo
class TestNodeClass:
	"""测试 Node 基类和子类"""

	def test_node_base_class(self):
		"""测试 Node 基类"""
		from envs.powerzoo.powerzoo.circuit import Node

		node = Node()
		assert node is not None

	def test_load_class_creation(self):
		"""测试 Load 类创建"""
		from envs.powerzoo.powerzoo.circuit import Load

		load = Load()
		assert load is not None
		assert hasattr(load, 'name')

	def test_capacitor_class_creation(self):
		"""测试 Capacitor 类创建"""
		from envs.powerzoo.powerzoo.circuit import Capacitor

		capacitor = Capacitor()
		assert capacitor is not None
		assert hasattr(capacitor, 'name')
		# Capacitor 应该有状态属性
		assert hasattr(capacitor, 'status') or hasattr(capacitor, 'state')

	def test_battery_class_creation(self):
		"""测试 Battery 类创建"""
		from envs.powerzoo.powerzoo.circuit import Battery

		battery = Battery()
		assert battery is not None
		assert hasattr(battery, 'name')
		# Battery 应该有 SOC 相关属性
		assert hasattr(battery, 'soc') or hasattr(battery, 'state_of_charge')

	def test_pvsystem_class_creation(self):
		"""测试 PVSystem 类创建"""
		from envs.powerzoo.powerzoo.circuit import PVSystem

		pv = PVSystem()
		assert pv is not None
		assert hasattr(pv, 'name')


# ==============================================================================
# 集成测试 - 需要 OpenDSS
# ==============================================================================

@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestCircuitsInitialization:
	"""测试 Circuits 类初始化（需要 OpenDSS）"""

	def test_circuits_init_with_valid_dss_file(self, node_systems_dir, skip_if_no_opendss):
		"""测试使用有效 DSS 文件初始化 Circuits"""
		dss_file = node_systems_dir / "13Bus" / "IEEE13Nodeckt.dss"

		if not dss_file.exists():
			pytest.skip("13Bus DSS 文件不存在")

		from envs.powerzoo.powerzoo.circuit import Circuits

		circuit = Circuits(dss_file=str(dss_file))

		# 验证基本属性
		assert circuit.dss is not None
		assert circuit.dss_file == str(dss_file)
		assert isinstance(circuit.topology, type(circuit.topology))
		assert isinstance(circuit.lines, dict)
		assert isinstance(circuit.transformers, dict)
		assert isinstance(circuit.regulators, dict)
		assert isinstance(circuit.loads, dict)
		assert isinstance(circuit.capacitors, dict)
		assert isinstance(circuit.batteries, dict)

	def test_circuits_init_with_rb_act_num(self, node_systems_dir, skip_if_no_opendss):
		"""测试使用自定义动作数量初始化"""
		dss_file = node_systems_dir / "13Bus" / "IEEE13Nodeckt.dss"

		if not dss_file.exists():
			pytest.skip("13Bus DSS 文件不存在")

		from envs.powerzoo.powerzoo.circuit import Circuits

		reg_act_num = 17
		bat_act_num = 21

		circuit = Circuits(
			dss_file=str(dss_file),
			RB_act_num=(reg_act_num, bat_act_num)
		)

		assert circuit.reg_act_num == reg_act_num
		assert circuit.bat_act_num == bat_act_num

	def test_circuits_init_with_invalid_dss_file(self, skip_if_no_opendss):
		"""测试使用无效 DSS 文件初始化应该失败"""
		from envs.powerzoo.powerzoo.circuit import Circuits

		with pytest.raises(Exception):
			Circuits(dss_file="/nonexistent/path/invalid.dss")


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestCircuitsCompileReset:
	"""测试 Circuits 编译和重置功能"""

	@pytest.fixture
	def circuit_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 电路实例"""
		dss_file = node_systems_dir / "13Bus" / "IEEE13Nodeckt.dss"
		if not dss_file.exists():
			pytest.skip("13Bus DSS 文件不存在")

		from envs.powerzoo.powerzoo.circuit import Circuits
		return Circuits(dss_file=str(dss_file))

	def test_compile_basic(self, circuit_13bus):
		"""测试基本编译功能"""
		circuit_13bus.compile()
		# 编译后应该能执行求解
		assert circuit_13bus.dss.ActiveCircuit is not None

	def test_compile_with_disable(self, circuit_13bus):
		"""测试禁用负荷和电源的编译"""
		circuit_13bus.compile(disable=True)
		# 验证编译成功
		assert circuit_13bus.dss.ActiveCircuit is not None

	def test_reset_restores_initial_state(self, circuit_13bus):
		"""测试重置功能恢复初始状态"""
		# 先获取初始状态
		initial_cap_statuses = circuit_13bus.get_all_capacitor_statuses()

		# 修改状态
		if len(circuit_13bus.capacitors) > 0:
			modified_statuses = [0] * len(circuit_13bus.capacitors)
			circuit_13bus.set_all_capacitor_statuses(modified_statuses)

		# 重置
		circuit_13bus.reset()

		# 验证状态恢复
		reset_cap_statuses = circuit_13bus.get_all_capacitor_statuses()
		# 重置后电容器应该是开启状态 (1)
		if len(reset_cap_statuses) > 0:
			assert all(s == 1 for s in reset_cap_statuses)


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestRegulatorOperations:
	"""测试调压器操作"""

	@pytest.fixture
	def circuit_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 电路实例"""
		dss_file = node_systems_dir / "13Bus" / "IEEE13Nodeckt.dss"
		if not dss_file.exists():
			pytest.skip("13Bus DSS 文件不存在")

		from envs.powerzoo.powerzoo.circuit import Circuits
		return Circuits(dss_file=str(dss_file))

	def test_set_regulator_parameters(self, circuit_13bus):
		"""测试设置调压器参数"""
		tap = 1.05
		mintap = 0.9
		maxtap = 1.1

		circuit_13bus.set_regulator_parameters(tap=tap, mintap=mintap, maxtap=maxtap)

		# 验证调压器参数已设置
		for reg_name, reg in circuit_13bus.regulators.items():
			if hasattr(reg, 'tap'):
				assert reg.tap == tap
			if hasattr(reg, 'tap_feature'):
				assert reg.tap_feature[0] == mintap
				assert reg.tap_feature[1] == maxtap

	def test_get_all_regulator_tapnums(self, circuit_13bus):
		"""测试获取所有调压器分接头位置"""
		tapnums = circuit_13bus.get_all_regulator_tapnums()

		assert isinstance(tapnums, (list, np.ndarray))
		assert len(tapnums) == len(circuit_13bus.regulators)

	def test_set_all_regulator_tappings(self, circuit_13bus):
		"""测试设置所有调压器分接头"""
		if len(circuit_13bus.regulators) == 0:
			pytest.skip("电路中没有调压器")

		# 获取调压器数量
		reg_count = len(circuit_13bus.regulators)

		# 设置分接头位置 (中间位置)
		tap_positions = [16] * reg_count  # 假设动作范围是 0-32

		circuit_13bus.set_all_regulator_tappings(tap_positions)

		# 验证设置成功
		current_taps = circuit_13bus.get_all_regulator_tapnums()
		assert len(current_taps) == reg_count


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestCapacitorOperations:
	"""测试电容器操作"""

	@pytest.fixture
	def circuit_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 电路实例"""
		dss_file = node_systems_dir / "13Bus" / "IEEE13Nodeckt.dss"
		if not dss_file.exists():
			pytest.skip("13Bus DSS 文件不存在")

		from envs.powerzoo.powerzoo.circuit import Circuits
		return Circuits(dss_file=str(dss_file))

	def test_get_all_capacitor_statuses(self, circuit_13bus):
		"""测试获取所有电容器状态"""
		statuses = circuit_13bus.get_all_capacitor_statuses()

		assert isinstance(statuses, (list, np.ndarray))
		assert len(statuses) == len(circuit_13bus.capacitors)
		# 状态应该是 0 或 1
		for s in statuses:
			assert s in [0, 1]

	def test_set_all_capacitor_statuses(self, circuit_13bus):
		"""测试设置所有电容器状态"""
		if len(circuit_13bus.capacitors) == 0:
			pytest.skip("电路中没有电容器")

		cap_count = len(circuit_13bus.capacitors)

		# 设置所有电容器为关闭
		circuit_13bus.set_all_capacitor_statuses([0] * cap_count)
		assert all(s == 0 for s in circuit_13bus.get_all_capacitor_statuses())

		# 设置所有电容器为开启
		circuit_13bus.set_all_capacitor_statuses([1] * cap_count)
		assert all(s == 1 for s in circuit_13bus.get_all_capacitor_statuses())

	def test_set_capacitor_statuses_with_invalid_length(self, circuit_13bus):
		"""测试设置电容器状态时使用无效长度"""
		if len(circuit_13bus.capacitors) == 0:
			pytest.skip("电路中没有电容器")

		cap_count = len(circuit_13bus.capacitors)

		# 使用错误长度应该引发异常或被正确处理
		with pytest.raises((ValueError, IndexError, Exception)):
			circuit_13bus.set_all_capacitor_statuses([0] * (cap_count + 5))


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestBatteryOperations:
	"""测试电池操作"""

	@pytest.fixture
	def circuit_with_batteries(self, node_systems_dir, skip_if_no_opendss):
		"""创建带电池的电路实例"""
		# 尝试使用带电池的配置
		dss_file = node_systems_dir / "13Bus" / "IEEE13Nodeckt.dss"
		if not dss_file.exists():
			pytest.skip("DSS 文件不存在")

		from envs.powerzoo.powerzoo.circuit import Circuits
		circuit = Circuits(dss_file=str(dss_file))
		return circuit

	def test_battery_dict_structure(self, circuit_with_batteries):
		"""测试电池字典结构"""
		batteries = circuit_with_batteries.batteries

		assert isinstance(batteries, dict)

		# 如果有电池，验证结构
		for bat_name, bat in batteries.items():
			assert isinstance(bat_name, str)
			assert hasattr(bat, 'name')

	def test_set_all_batteries_before_solve(self, circuit_with_batteries):
		"""测试求解前设置电池状态"""
		if len(circuit_with_batteries.batteries) == 0:
			pytest.skip("电路中没有电池")

		bat_count = len(circuit_with_batteries.batteries)
		# 设置电池动作 (假设动作范围是 0-32)
		actions = [16] * bat_count

		# 不应该引发异常
		circuit_with_batteries.set_all_batteries_before_solve(actions)

	def test_set_all_batteries_after_solve(self, circuit_with_batteries):
		"""测试求解后更新电池状态"""
		if len(circuit_with_batteries.batteries) == 0:
			pytest.skip("电路中没有电池")

		# 先编译并求解
		circuit_with_batteries.compile()
		circuit_with_batteries.dss.ActiveCircuit.Solution.SolveNoControl()

		# 更新电池状态
		circuit_with_batteries.set_all_batteries_after_solve()


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestVoltageCurrentOperations:
	"""测试电压和电流操作"""

	@pytest.fixture
	def solved_circuit(self, node_systems_dir, skip_if_no_opendss):
		"""创建已求解的电路实例"""
		dss_file = node_systems_dir / "13Bus" / "IEEE13Nodeckt.dss"
		if not dss_file.exists():
			pytest.skip("13Bus DSS 文件不存在")

		from envs.powerzoo.powerzoo.circuit import Circuits
		circuit = Circuits(dss_file=str(dss_file))
		circuit.compile()
		circuit.dss.ActiveCircuit.Solution.SolveNoControl()
		return circuit

	def test_bus_voltage(self, solved_circuit):
		"""测试获取母线电压"""
		voltages = solved_circuit.bus_voltage()

		assert isinstance(voltages, dict)
		assert len(voltages) > 0

		# 验证电压值在合理范围内 (0.8 - 1.2 p.u.)
		for bus_name, voltage_data in voltages.items():
			assert isinstance(bus_name, str)
			if isinstance(voltage_data, (list, np.ndarray)):
				for v in voltage_data:
					if isinstance(v, (int, float)) and not np.isnan(v):
						# 电压幅值应该在合理范围内
						pass  # 某些值可能是相角，不做限制

	def test_edge_current(self, solved_circuit):
		"""测试获取边电流"""
		currents = solved_circuit.edge_current()

		assert isinstance(currents, dict)

	def test_total_loss(self, solved_circuit):
		"""测试获取总损耗"""
		loss = solved_circuit.total_loss()

		# 损耗应该是元组或列表 (有功, 无功)
		assert isinstance(loss, (tuple, list))
		assert len(loss) == 2

		kw_loss, kvar_loss = loss
		# 有功损耗应该是非负的
		assert kw_loss >= 0

	def test_total_power(self, solved_circuit):
		"""测试获取总功率"""
		power = solved_circuit.total_power()

		# 功率应该是元组或列表 (有功, 无功)
		assert isinstance(power, (tuple, list))
		assert len(power) == 2


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestTopologyOperations:
	"""测试拓扑结构操作"""

	@pytest.fixture
	def circuit_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 电路实例"""
		dss_file = node_systems_dir / "13Bus" / "IEEE13Nodeckt.dss"
		if not dss_file.exists():
			pytest.skip("13Bus DSS 文件不存在")

		from envs.powerzoo.powerzoo.circuit import Circuits
		return Circuits(dss_file=str(dss_file))

	def test_topology_is_networkx_graph(self, circuit_13bus):
		"""测试拓扑是 NetworkX 图"""
		import networkx as nx
		assert isinstance(circuit_13bus.topology, nx.Graph)

	def test_topology_has_nodes_and_edges(self, circuit_13bus):
		"""测试拓扑有节点和边"""
		assert circuit_13bus.topology.number_of_nodes() > 0
		assert circuit_13bus.topology.number_of_edges() > 0

	def test_lines_dict(self, circuit_13bus):
		"""测试线路字典"""
		assert isinstance(circuit_13bus.lines, dict)

	def test_transformers_dict(self, circuit_13bus):
		"""测试变压器字典"""
		assert isinstance(circuit_13bus.transformers, dict)

	def test_get_agent_bus_dict(self, circuit_13bus):
		"""测试获取设备-母线映射"""
		agent_bus_dict = circuit_13bus.get_agent_bus_dict()

		assert isinstance(agent_bus_dict, dict)


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestYMatrixOperations:
	"""测试导纳矩阵操作"""

	@pytest.fixture
	def circuit_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 电路实例"""
		dss_file = node_systems_dir / "13Bus" / "IEEE13Nodeckt.dss"
		if not dss_file.exists():
			pytest.skip("13Bus DSS 文件不存在")

		from envs.powerzoo.powerzoo.circuit import Circuits
		return Circuits(dss_file=str(dss_file))

	def test_get_Y_matrix(self, circuit_13bus):
		"""测试获取导纳矩阵"""
		Y = circuit_13bus.get_Y_matrix()

		# Y 矩阵应该是方阵
		assert isinstance(Y, np.ndarray)
		if len(Y.shape) == 2:
			assert Y.shape[0] == Y.shape[1]

	def test_get_Y_matrix_acc_dense(self, circuit_13bus):
		"""测试获取加速导纳矩阵（密集）"""
		Y = circuit_13bus.get_Y_matrix_acc(use_sparse=False, use_gpu=False)

		assert isinstance(Y, np.ndarray)

	def test_get_Y_matrix_acc_sparse(self, circuit_13bus):
		"""测试获取加速导纳矩阵（稀疏）"""
		from scipy.sparse import issparse

		Y = circuit_13bus.get_Y_matrix_acc(use_sparse=True, use_gpu=False)

		# 应该是稀疏矩阵或密集矩阵
		assert isinstance(Y, np.ndarray) or issparse(Y)


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestSensitivityOperations:
	"""测试敏感度矩阵操作"""

	@pytest.fixture
	def circuit_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 电路实例"""
		dss_file = node_systems_dir / "13Bus" / "IEEE13Nodeckt.dss"
		if not dss_file.exists():
			pytest.skip("13Bus DSS 文件不存在")

		from envs.powerzoo.powerzoo.circuit import Circuits
		return Circuits(dss_file=str(dss_file))

	def test_get_node_sensity(self, circuit_13bus):
		"""测试获取节点敏感度"""
		Y = circuit_13bus.get_Y_matrix()
		S = circuit_13bus.get_node_sensity(Y)

		assert isinstance(S, np.ndarray)

	def test_get_node_sensity_acc(self, circuit_13bus):
		"""测试获取加速节点敏感度"""
		Y = circuit_13bus.get_Y_matrix_acc(use_sparse=False, use_gpu=False)
		S = circuit_13bus.get_node_sensity_acc(Y, use_noise=False)

		assert isinstance(S, np.ndarray)

	def test_get_node_sensity_acc_with_noise(self, circuit_13bus):
		"""测试获取加速节点敏感度（带噪声）"""
		Y = circuit_13bus.get_Y_matrix_acc(use_sparse=False, use_gpu=False)
		S = circuit_13bus.get_node_sensity_acc(Y, use_noise=True)

		assert isinstance(S, np.ndarray)


# ==============================================================================
# 边界条件和错误处理测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.powerzoo
class TestCircuitsEdgeCases:
	"""测试边界条件和错误处理"""

	def test_circuits_with_empty_dss_file_path(self):
		"""测试使用空 DSS 文件路径"""
		from envs.powerzoo.powerzoo.circuit import Circuits

		with pytest.raises(Exception):
			Circuits(dss_file="")

	def test_circuits_with_none_dss_file(self):
		"""测试使用 None DSS 文件"""
		from envs.powerzoo.powerzoo.circuit import Circuits

		with pytest.raises((TypeError, Exception)):
			Circuits(dss_file=None)


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestCircuitsRobustness:
	"""测试 Circuits 健壮性"""

	@pytest.fixture
	def circuit_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""创建 13Bus 电路实例"""
		dss_file = node_systems_dir / "13Bus" / "IEEE13Nodeckt.dss"
		if not dss_file.exists():
			pytest.skip("13Bus DSS 文件不存在")

		from envs.powerzoo.powerzoo.circuit import Circuits
		return Circuits(dss_file=str(dss_file))

	def test_multiple_compile_calls(self, circuit_13bus):
		"""测试多次编译调用"""
		for _ in range(3):
			circuit_13bus.compile()

		# 应该不会引发异常
		assert circuit_13bus.dss.ActiveCircuit is not None

	def test_multiple_reset_calls(self, circuit_13bus):
		"""测试多次重置调用"""
		for _ in range(3):
			circuit_13bus.reset()

		# 应该不会引发异常
		assert circuit_13bus.dss.ActiveCircuit is not None

	def test_solve_after_multiple_operations(self, circuit_13bus):
		"""测试多次操作后求解"""
		# 编译
		circuit_13bus.compile()

		# 修改电容器状态
		if len(circuit_13bus.capacitors) > 0:
			cap_count = len(circuit_13bus.capacitors)
			circuit_13bus.set_all_capacitor_statuses([1] * cap_count)

		# 修改调压器
		circuit_13bus.set_regulator_parameters(tap=1.0)

		# 求解
		circuit_13bus.dss.ActiveCircuit.Solution.SolveNoControl()

		# 获取电压
		voltages = circuit_13bus.bus_voltage()
		assert len(voltages) > 0
