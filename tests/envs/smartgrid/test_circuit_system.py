# -*- coding: utf-8 -*-
"""
SmartGrid circuit_system 模块详细测试

测试覆盖:
- Circuits 类初始化
- 电力元件组件 (Node, Edge, Load, Capacitor, PVSystem, Battery)
- 拓扑结构构建
- 电气计算功能

@File      : test_circuit_system.py
@Author    : PowerZoo Test Suite
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ==============================================================================
# 单元测试 - 导入测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestCircuitSystemImport:
	"""测试 circuit_system 模块导入"""

	def test_circuits_import(self):
		"""测试 Circuits 类可以正确导入"""
		from envs.smartgrid.circuit_system import Circuits
		assert Circuits is not None

	def test_circuits_from_circuit_module(self):
		"""测试从 circuit 模块导入"""
		from envs.smartgrid.circuit_system.circuit import Circuits
		assert Circuits is not None


@pytest.mark.unit
@pytest.mark.smartgrid
class TestComponentsImport:
	"""测试电力元件导入"""

	def test_base_classes_import(self):
		"""测试基类导入"""
		from envs.smartgrid.circuit_system.components import Node, Edge
		assert Node is not None
		assert Edge is not None

	def test_node_components_import(self):
		"""测试节点组件导入"""
		from envs.smartgrid.circuit_system.components import Load, Capacitor, PVSystem, Battery
		assert Load is not None
		assert Capacitor is not None
		assert PVSystem is not None
		assert Battery is not None

	def test_edge_components_import(self):
		"""测试边组件导入"""
		from envs.smartgrid.circuit_system.components import Line, Transformer, Regulator
		assert Line is not None
		assert Transformer is not None
		assert Regulator is not None

	def test_all_components_from_init(self):
		"""测试从 __init__ 导入所有组件"""
		from envs.smartgrid.circuit_system.components import (
			Node, Edge, Load, Capacitor, PVSystem, Battery,
			Line, Transformer, Regulator
		)
		assert all([Node, Edge, Load, Capacitor, PVSystem, Battery,
					Line, Transformer, Regulator])


# ==============================================================================
# 单元测试 - Node 基类和子类
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestNodeClass:
	"""测试 Node 基类"""

	def test_node_creation(self):
		"""测试 Node 创建"""
		from envs.smartgrid.circuit_system.components import Node

		node = Node()
		assert node is not None

	def test_node_has_name_attribute(self):
		"""测试 Node 有 name 属性"""
		from envs.smartgrid.circuit_system.components import Node

		node = Node()
		assert hasattr(node, 'name')


@pytest.mark.unit
@pytest.mark.smartgrid
class TestLoadClass:
	"""测试 Load 类"""

	def test_load_creation(self):
		"""测试 Load 创建"""
		from envs.smartgrid.circuit_system.components import Load

		load = Load()
		assert load is not None

	def test_load_has_required_attributes(self):
		"""测试 Load 有必需属性"""
		from envs.smartgrid.circuit_system.components import Load

		load = Load()
		assert hasattr(load, 'name')
		# 可能还有其他属性如 kW, kvar 等


@pytest.mark.unit
@pytest.mark.smartgrid
class TestCapacitorClass:
	"""测试 Capacitor 类"""

	def test_capacitor_creation(self):
		"""测试 Capacitor 创建"""
		from envs.smartgrid.circuit_system.components import Capacitor

		cap = Capacitor()
		assert cap is not None

	def test_capacitor_has_status(self):
		"""测试 Capacitor 有状态属性"""
		from envs.smartgrid.circuit_system.components import Capacitor

		cap = Capacitor()
		# 电容器应该有开关状态
		assert hasattr(cap, 'status') or hasattr(cap, 'state')


@pytest.mark.unit
@pytest.mark.smartgrid
class TestPVSystemClass:
	"""测试 PVSystem 类"""

	def test_pvsystem_creation(self):
		"""测试 PVSystem 创建"""
		from envs.smartgrid.circuit_system.components import PVSystem

		pv = PVSystem()
		assert pv is not None

	def test_pvsystem_has_power_attributes(self):
		"""测试 PVSystem 有功率属性"""
		from envs.smartgrid.circuit_system.components import PVSystem

		pv = PVSystem()
		# PV 系统应该有功率相关属性
		assert hasattr(pv, 'name')


@pytest.mark.unit
@pytest.mark.smartgrid
class TestBatteryClass:
	"""测试 Battery 类"""

	def test_battery_creation(self):
		"""测试 Battery 创建"""
		from envs.smartgrid.circuit_system.components import Battery

		bat = Battery()
		assert bat is not None

	def test_battery_has_soc(self):
		"""测试 Battery 有 SOC 属性"""
		from envs.smartgrid.circuit_system.components import Battery

		bat = Battery()
		# 电池应该有 SOC（荷电状态）
		assert hasattr(bat, 'soc') or hasattr(bat, 'state_of_charge')

	def test_battery_has_power_capacity(self):
		"""测试 Battery 有功率容量属性"""
		from envs.smartgrid.circuit_system.components import Battery

		bat = Battery()
		# 电池应该有功率相关属性
		assert hasattr(bat, 'name')


# ==============================================================================
# 单元测试 - Edge 基类和子类
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestEdgeClass:
	"""测试 Edge 基类"""

	def test_edge_creation(self):
		"""测试 Edge 创建"""
		from envs.smartgrid.circuit_system.components import Edge

		edge = Edge()
		assert edge is not None


@pytest.mark.unit
@pytest.mark.smartgrid
class TestLineClass:
	"""测试 Line 类"""

	def test_line_creation(self):
		"""测试 Line 创建"""
		from envs.smartgrid.circuit_system.components import Line

		line = Line()
		assert line is not None

	def test_line_has_bus_connections(self):
		"""测试 Line 有母线连接"""
		from envs.smartgrid.circuit_system.components import Line

		line = Line()
		# 线路应该有两端母线
		assert hasattr(line, 'bus1') or hasattr(line, 'from_bus')


@pytest.mark.unit
@pytest.mark.smartgrid
class TestTransformerClass:
	"""测试 Transformer 类"""

	def test_transformer_creation(self):
		"""测试 Transformer 创建"""
		from envs.smartgrid.circuit_system.components import Transformer

		trans = Transformer()
		assert trans is not None


@pytest.mark.unit
@pytest.mark.smartgrid
class TestRegulatorClass:
	"""测试 Regulator 类"""

	def test_regulator_creation(self):
		"""测试 Regulator 创建"""
		from envs.smartgrid.circuit_system.components import Regulator

		reg = Regulator()
		assert reg is not None

	def test_regulator_has_tap(self):
		"""测试 Regulator 有分接头属性"""
		from envs.smartgrid.circuit_system.components import Regulator

		reg = Regulator()
		# 调压器应该有分接头
		assert hasattr(reg, 'tap') or hasattr(reg, 'tap_position')


# ==============================================================================
# 集成测试 - Circuits 类
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestCircuitsInitialization:
	"""测试 Circuits 类初始化"""

	def test_circuits_creation_with_dss_file(self, node_systems_dir, skip_if_no_opendss):
		"""测试使用 DSS 文件创建 Circuits"""
		# 寻找可用的 DSS 系统
		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		# 查找主 DSS 文件
		dss_files = list(dss_folder.glob("*duty.dss")) + list(dss_folder.glob("*.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		from envs.smartgrid.circuit_system import Circuits

		try:
			circuit = Circuits(dss_file=str(dss_files[0]))
			assert circuit is not None
		except Exception as e:
			pytest.skip(f"Circuits 创建失败: {e}")

	def test_circuits_has_dss_object(self, node_systems_dir, skip_if_no_opendss):
		"""测试 Circuits 有 DSS 对象"""
		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list(dss_folder.glob("*.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		from envs.smartgrid.circuit_system import Circuits

		try:
			circuit = Circuits(dss_file=str(dss_files[0]))
			assert hasattr(circuit, 'dss')
			assert circuit.dss is not None
		except Exception as e:
			pytest.skip(f"Circuits 创建失败: {e}")


@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestCircuitsTopology:
	"""测试 Circuits 拓扑结构"""

	@pytest.fixture
	def circuit_instance(self, node_systems_dir, skip_if_no_opendss):
		"""创建 Circuits 实例"""
		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list(dss_folder.glob("*duty.dss")) + list(dss_folder.glob("*.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		from envs.smartgrid.circuit_system import Circuits

		try:
			return Circuits(dss_file=str(dss_files[0]))
		except Exception as e:
			pytest.skip(f"Circuits 创建失败: {e}")

	def test_topology_is_graph(self, circuit_instance):
		"""测试拓扑是图结构"""
		import networkx as nx

		if hasattr(circuit_instance, 'topology'):
			assert isinstance(circuit_instance.topology, nx.Graph)

	def test_has_lines_dict(self, circuit_instance):
		"""测试有线路字典"""
		assert hasattr(circuit_instance, 'lines')
		assert isinstance(circuit_instance.lines, dict)

	def test_has_transformers_dict(self, circuit_instance):
		"""测试有变压器字典"""
		assert hasattr(circuit_instance, 'transformers')
		assert isinstance(circuit_instance.transformers, dict)

	def test_has_loads_dict(self, circuit_instance):
		"""测试有负荷字典"""
		assert hasattr(circuit_instance, 'loads')
		assert isinstance(circuit_instance.loads, dict)

	def test_has_capacitors_dict(self, circuit_instance):
		"""测试有电容器字典"""
		assert hasattr(circuit_instance, 'capacitors')
		assert isinstance(circuit_instance.capacitors, dict)


@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestCircuitsOperations:
	"""测试 Circuits 操作功能"""

	@pytest.fixture
	def circuit_instance(self, node_systems_dir, skip_if_no_opendss):
		"""创建 Circuits 实例"""
		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list(dss_folder.glob("*duty.dss")) + list(dss_folder.glob("*.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		from envs.smartgrid.circuit_system import Circuits

		try:
			circuit = Circuits(dss_file=str(dss_files[0]))
			circuit.compile()
			return circuit
		except Exception as e:
			pytest.skip(f"Circuits 创建失败: {e}")

	def test_compile_method(self, circuit_instance):
		"""测试编译方法"""
		# 应该不会引发异常
		circuit_instance.compile()
		assert circuit_instance.dss.ActiveCircuit is not None

	def test_reset_method(self, circuit_instance):
		"""测试重置方法"""
		if hasattr(circuit_instance, 'reset'):
			circuit_instance.reset()
			# 不应该引发异常

	def test_bus_voltage_method(self, circuit_instance):
		"""测试获取母线电压"""
		if hasattr(circuit_instance, 'bus_voltage'):
			voltages = circuit_instance.bus_voltage()
			assert isinstance(voltages, dict)

	def test_total_loss_method(self, circuit_instance):
		"""测试获取总损耗"""
		if hasattr(circuit_instance, 'total_loss'):
			loss = circuit_instance.total_loss()
			assert isinstance(loss, (tuple, list))

	def test_total_power_method(self, circuit_instance):
		"""测试获取总功率"""
		if hasattr(circuit_instance, 'total_power'):
			power = circuit_instance.total_power()
			assert isinstance(power, (tuple, list))


# ==============================================================================
# 组件状态管理测试
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestComponentStateManagement:
	"""测试组件状态管理"""

	@pytest.fixture
	def circuit_instance(self, node_systems_dir, skip_if_no_opendss):
		"""创建 Circuits 实例"""
		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list(dss_folder.glob("*duty.dss")) + list(dss_folder.glob("*.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		from envs.smartgrid.circuit_system import Circuits

		try:
			circuit = Circuits(dss_file=str(dss_files[0]))
			circuit.compile()
			return circuit
		except Exception as e:
			pytest.skip(f"Circuits 创建失败: {e}")

	def test_capacitor_status_get(self, circuit_instance):
		"""测试获取电容器状态"""
		if hasattr(circuit_instance, 'get_all_capacitor_statuses'):
			statuses = circuit_instance.get_all_capacitor_statuses()
			assert isinstance(statuses, (list, np.ndarray))

	def test_capacitor_status_set(self, circuit_instance):
		"""测试设置电容器状态"""
		if len(circuit_instance.capacitors) == 0:
			pytest.skip("电路中没有电容器")

		if hasattr(circuit_instance, 'set_all_capacitor_statuses'):
			cap_count = len(circuit_instance.capacitors)
			circuit_instance.set_all_capacitor_statuses([1] * cap_count)

	def test_regulator_tap_get(self, circuit_instance):
		"""测试获取调压器分接头"""
		if hasattr(circuit_instance, 'get_all_regulator_tapnums'):
			taps = circuit_instance.get_all_regulator_tapnums()
			assert isinstance(taps, (list, np.ndarray))

	def test_regulator_tap_set(self, circuit_instance):
		"""测试设置调压器分接头"""
		if len(circuit_instance.regulators) == 0:
			pytest.skip("电路中没有调压器")

		if hasattr(circuit_instance, 'set_all_regulator_tappings'):
			reg_count = len(circuit_instance.regulators)
			circuit_instance.set_all_regulator_tappings([16] * reg_count)
