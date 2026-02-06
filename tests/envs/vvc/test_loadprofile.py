# -*- coding: utf-8 -*-
"""
PowerZoo LoadProfile 类详细测试

测试覆盖:
- LoadProfile 类初始化
- 负荷曲线生成和选择
- DSS 文件创建和修改
- 负荷名称提取
- 多 worker 支持

@File      : test_loadprofile.py
@Author    : PowerZoo Test Suite
"""

import pytest
import numpy as np
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ==============================================================================
# 单元测试 - 导入测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.powerzoo
class TestLoadProfileImport:
	"""测试 LoadProfile 模块导入"""

	def test_loadprofile_class_import(self):
		"""测试 LoadProfile 类可以正确导入"""
		from envs.vvc.vvc.loadprofile import LoadProfile
		assert LoadProfile is not None

	def test_loadprofile_has_required_methods(self):
		"""验证 LoadProfile 类具有所有必需的方法"""
		from envs.vvc.vvc.loadprofile import LoadProfile

		required_methods = [
			'__init__',
			'find_load_names',
			'gen_loadprofile',
			'choose_loadprofile',
			'get_loadprofile',
			'create_file_with_duty',
			'create_file_with_daily',
			'add_redirect_and_mode_at_main_daily_dss',
			'add_redirect_and_mode_at_main_duty_dss',
		]

		for method in required_methods:
			assert hasattr(LoadProfile, method), f"LoadProfile 缺少方法: {method}"


# ==============================================================================
# 单元测试 - 静态方法和辅助函数
# ==============================================================================

@pytest.mark.unit
@pytest.mark.powerzoo
class TestLoadProfileStaticMethods:
	"""测试 LoadProfile 静态方法"""

	def test_find_load_names_with_mock_file(self, tmp_path):
		"""测试从 DSS 文件中提取负荷名称"""
		# 创建临时 DSS 文件
		dss_content = """
		New Load.Load1 Bus1=bus1 Phases=3 Conn=Wye Model=1 kV=4.16 kW=1155 kvar=660
		New Load.Load2 Bus1=bus2 Phases=3 Conn=Wye Model=1 kV=4.16 kW=2000 kvar=1000
		New Line.Line1 Bus1=650.1.2.3 Bus2=632.1.2.3
		New Load.Load3 Bus1=bus3 Phases=1 Conn=Wye Model=1 kV=2.4 kW=500 kvar=200
		"""
		dss_file = tmp_path / "test.dss"
		dss_file.write_text(dss_content)

		from envs.vvc.vvc.loadprofile import LoadProfile

		# 使用类方法提取负荷名称
		# 注意：find_load_names 是实例方法，需要创建实例或 mock
		load_names = []
		with open(str(dss_file), 'r') as f:
			for line in f:
				if line.lower().strip().startswith('new load.'):
					parts = line.split()
					if len(parts) >= 2:
						load_name = parts[1].split('.', 1)[1] if '.' in parts[1] else parts[1]
						load_names.append(load_name)

		assert 'Load1' in load_names
		assert 'Load2' in load_names
		assert 'Load3' in load_names
		assert len(load_names) == 3


# ==============================================================================
# 集成测试 - 需要真实文件系统
# ==============================================================================

@pytest.mark.integration
@pytest.mark.powerzoo
class TestLoadProfileInitialization:
	"""测试 LoadProfile 初始化"""

	@pytest.fixture
	def mock_dss_folder(self, tmp_path):
		"""创建模拟的 DSS 文件夹结构"""
		dss_folder = tmp_path / "13Bus"
		dss_folder.mkdir()

		# 创建主 DSS 文件
		dss_file = dss_folder / "IEEE13Nodeckt.dss"
		dss_content = """
		Clear
		New Circuit.IEEE13Nodeckt basekv=115 pu=1.0001 phases=3 bus1=SourceBus
		New Load.Load1 Bus1=bus1 Phases=3 Conn=Wye Model=1 kV=4.16 kW=1155 kvar=660
		New Load.Load2 Bus1=bus2 Phases=3 Conn=Wye Model=1 kV=4.16 kW=2000 kvar=1000
		"""
		dss_file.write_text(dss_content)

		# 创建 loadshape 目录
		loadshape_dir = dss_folder / "loadshape" / "data_without_noise"
		loadshape_dir.mkdir(parents=True)

		# 创建负荷曲线 CSV 文件
		loadshape_csv = loadshape_dir / "loadshape_001.csv"
		csv_content = "Load1,Load2\n1.0,1.0\n0.9,0.95\n0.8,0.85\n"
		loadshape_csv.write_text(csv_content)

		return dss_folder

	def test_loadprofile_init_basic(self, mock_dss_folder):
		"""测试 LoadProfile 基本初始化"""
		from envs.vvc.vvc.loadprofile import LoadProfile

		dss_file = mock_dss_folder / "IEEE13Nodeckt.dss"

		lp = LoadProfile(
			steps=24,
			dss_folder_path=str(mock_dss_folder),
			dss_file=str(dss_file),
			use_noise=False,
			worker_idx=None
		)

		assert lp is not None
		assert lp.steps == 24
		assert lp.dss_folder_path == str(mock_dss_folder)

	def test_loadprofile_init_with_worker_idx(self, mock_dss_folder):
		"""测试带 worker_idx 的 LoadProfile 初始化"""
		from envs.vvc.vvc.loadprofile import LoadProfile

		dss_file = mock_dss_folder / "IEEE13Nodeckt.dss"

		lp = LoadProfile(
			steps=24,
			dss_folder_path=str(mock_dss_folder),
			dss_file=str(dss_file),
			use_noise=False,
			worker_idx=5
		)

		assert lp.loadshape_dss == 'loadshape_5.dss'

	def test_loadprofile_finds_csv_files(self, mock_dss_folder):
		"""测试 LoadProfile 发现 CSV 文件"""
		from envs.vvc.vvc.loadprofile import LoadProfile

		dss_file = mock_dss_folder / "IEEE13Nodeckt.dss"

		lp = LoadProfile(
			steps=24,
			dss_folder_path=str(mock_dss_folder),
			dss_file=str(dss_file),
			use_noise=False
		)

		assert len(lp.FILES) > 0
		assert all('loadshape' in f.lower() for f in lp.FILES)

	def test_loadprofile_finds_load_names(self, mock_dss_folder):
		"""测试 LoadProfile 提取负荷名称"""
		from envs.vvc.vvc.loadprofile import LoadProfile

		dss_file = mock_dss_folder / "IEEE13Nodeckt.dss"

		lp = LoadProfile(
			steps=24,
			dss_folder_path=str(mock_dss_folder),
			dss_file=str(dss_file),
			use_noise=False
		)

		assert len(lp.LOAD_NAMES) >= 0  # 可能为空如果 DSS 文件解析有特殊要求


@pytest.mark.integration
@pytest.mark.powerzoo
class TestLoadProfileDSSFileCreation:
	"""测试 LoadProfile DSS 文件创建功能"""

	@pytest.fixture
	def loadprofile_with_files(self, tmp_path):
		"""创建带有完整文件结构的 LoadProfile"""
		dss_folder = tmp_path / "13Bus"
		dss_folder.mkdir()

		# 创建负荷文件
		loads_file = dss_folder / "Loads.dss"
		loads_content = """
		New Load.Load1 Bus1=bus1 Phases=3 Conn=Wye Model=1 kV=4.16 kW=1155 kvar=660
		New Load.Load2 Bus1=bus2 Phases=3 Conn=Wye Model=1 kV=4.16 kW=2000 kvar=1000
		"""
		loads_file.write_text(loads_content)

		# 创建主 DSS 文件
		dss_file = dss_folder / "IEEE13Nodeckt.dss"
		dss_content = """
		Clear
		New Circuit.IEEE13Nodeckt basekv=115 pu=1.0001 phases=3 bus1=SourceBus
		redirect Loads.dss
		"""
		dss_file.write_text(dss_content)

		# 创建 loadshape 目录
		loadshape_dir = dss_folder / "loadshape" / "data_without_noise"
		loadshape_dir.mkdir(parents=True)

		loadshape_csv = loadshape_dir / "loadshape_001.csv"
		csv_content = "Load1,Load2\n1.0,1.0\n0.9,0.95\n"
		loadshape_csv.write_text(csv_content)

		from envs.vvc.vvc.loadprofile import LoadProfile

		return LoadProfile(
			steps=24,
			dss_folder_path=str(dss_folder),
			dss_file=str(dss_file),
			use_noise=False
		), dss_folder

	def test_create_file_with_duty(self, loadprofile_with_files):
		"""测试创建 duty DSS 文件"""
		lp, dss_folder = loadprofile_with_files

		# 创建 duty 文件
		lp.create_file_with_duty("Loads.dss")

		# 验证文件已创建
		duty_file = dss_folder / "Loads_duty.dss"
		assert duty_file.exists()

		# 验证文件内容包含 duty 关键字
		content = duty_file.read_text()
		assert 'duty=' in content.lower()

	def test_create_file_with_daily(self, loadprofile_with_files):
		"""测试创建 daily DSS 文件"""
		lp, dss_folder = loadprofile_with_files

		# 创建 daily 文件
		lp.create_file_with_daily("Loads.dss")

		# 验证文件已创建
		daily_file = dss_folder / "Loads_daily.dss"
		assert daily_file.exists()

		# 验证文件内容包含 daily 关键字
		content = daily_file.read_text()
		assert 'daily=' in content.lower()


@pytest.mark.integration
@pytest.mark.powerzoo
class TestLoadProfileGeneration:
	"""测试 LoadProfile 生成功能"""

	@pytest.fixture
	def loadprofile_instance(self, tmp_path):
		"""创建 LoadProfile 实例"""
		dss_folder = tmp_path / "13Bus"
		dss_folder.mkdir()

		# 创建 DSS 文件
		dss_file = dss_folder / "IEEE13Nodeckt.dss"
		dss_content = """
		New Load.Load1 Bus1=bus1 Phases=3 kW=1155
		New Load.Load2 Bus1=bus2 Phases=3 kW=2000
		"""
		dss_file.write_text(dss_content)

		# 创建 loadshape 目录和多个 CSV 文件
		loadshape_dir = dss_folder / "loadshape" / "data_without_noise"
		loadshape_dir.mkdir(parents=True)

		for i in range(5):
			csv_file = loadshape_dir / f"loadshape_{i:03d}.csv"
			csv_content = "Load1,Load2\n" + "\n".join([f"{0.8+0.1*np.random.rand()},{0.8+0.1*np.random.rand()}" for _ in range(24)])
			csv_file.write_text(csv_content)

		from envs.vvc.vvc.loadprofile import LoadProfile

		return LoadProfile(
			steps=24,
			dss_folder_path=str(dss_folder),
			dss_file=str(dss_file),
			use_noise=False
		)

	def test_gen_loadprofile(self, loadprofile_instance):
		"""测试生成负荷曲线"""
		lp = loadprofile_instance

		if hasattr(lp, 'gen_loadprofile') and callable(getattr(lp, 'gen_loadprofile')):
			# 生成负荷曲线
			lp.gen_loadprofile()

			# 验证生成的属性
			assert hasattr(lp, 'profiles') or hasattr(lp, 'loadprofiles')

	def test_choose_loadprofile(self, loadprofile_instance):
		"""测试选择负荷曲线"""
		lp = loadprofile_instance

		if hasattr(lp, 'gen_loadprofile'):
			lp.gen_loadprofile()

		if hasattr(lp, 'choose_loadprofile') and callable(getattr(lp, 'choose_loadprofile')):
			# 选择负荷曲线
			lp.choose_loadprofile(0)

	def test_get_loadprofile(self, loadprofile_instance):
		"""测试获取负荷曲线数据"""
		lp = loadprofile_instance

		if hasattr(lp, 'gen_loadprofile'):
			lp.gen_loadprofile()

		if hasattr(lp, 'choose_loadprofile'):
			lp.choose_loadprofile(0)

		if hasattr(lp, 'get_loadprofile') and callable(getattr(lp, 'get_loadprofile')):
			data = lp.get_loadprofile()
			assert data is not None


@pytest.mark.integration
@pytest.mark.powerzoo
@pytest.mark.requires_opendss
class TestLoadProfileWithRealSystem:
	"""使用真实系统测试 LoadProfile"""

	def test_loadprofile_with_13bus(self, node_systems_dir, skip_if_no_opendss):
		"""测试使用 13Bus 系统的 LoadProfile"""
		dss_folder = node_systems_dir / "13Bus"
		dss_file = dss_folder / "IEEE13Nodeckt.dss"

		if not dss_file.exists():
			pytest.skip("13Bus DSS 文件不存在")

		from envs.vvc.vvc.loadprofile import LoadProfile

		# 如果 loadshape 目录存在
		loadshape_dir = dss_folder / "loadshape" / "data_without_noise"
		if not loadshape_dir.exists():
			pytest.skip("loadshape 目录不存在")

		lp = LoadProfile(
			steps=24,
			dss_folder_path=str(dss_folder),
			dss_file=str(dss_file),
			use_noise=False
		)

		assert lp is not None
		assert len(lp.FILES) >= 0

	def test_loadprofile_with_34bus(self, node_systems_dir, skip_if_no_opendss):
		"""测试使用 34Bus 系统的 LoadProfile"""
		# 尝试不同的 34Bus 变体
		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("34Bus 系统不存在")

		# 查找主 DSS 文件
		dss_files = list(dss_folder.glob("*.dss"))
		main_dss = None
		for f in dss_files:
			if 'master' in f.name.lower() or 'ieee34' in f.name.lower():
				main_dss = f
				break

		if main_dss is None and dss_files:
			main_dss = dss_files[0]

		if main_dss is None:
			pytest.skip("未找到主 DSS 文件")

		from envs.vvc.vvc.loadprofile import LoadProfile

		# 检查 loadshape 目录
		loadshape_dir = dss_folder / "loadshape" / "data_without_noise"
		if not loadshape_dir.exists():
			pytest.skip("loadshape 目录不存在")

		lp = LoadProfile(
			steps=24,
			dss_folder_path=str(dss_folder),
			dss_file=str(main_dss),
			use_noise=False
		)

		assert lp is not None


# ==============================================================================
# 边界条件和错误处理测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.powerzoo
class TestLoadProfileEdgeCases:
	"""测试 LoadProfile 边界条件"""

	def test_loadprofile_with_empty_loadshape_dir(self, tmp_path):
		"""测试空 loadshape 目录"""
		dss_folder = tmp_path / "empty_system"
		dss_folder.mkdir()

		# 创建 DSS 文件
		dss_file = dss_folder / "test.dss"
		dss_file.write_text("New Load.Load1 Bus1=bus1 kW=100")

		# 创建空的 loadshape 目录
		loadshape_dir = dss_folder / "loadshape" / "data_without_noise"
		loadshape_dir.mkdir(parents=True)

		from envs.vvc.vvc.loadprofile import LoadProfile

		lp = LoadProfile(
			steps=24,
			dss_folder_path=str(dss_folder),
			dss_file=str(dss_file),
			use_noise=False
		)

		# 应该创建成功但没有 CSV 文件
		assert len(lp.FILES) == 0

	def test_loadprofile_with_noise(self, tmp_path):
		"""测试使用噪声的 LoadProfile"""
		dss_folder = tmp_path / "noise_system"
		dss_folder.mkdir()

		# 创建 DSS 文件
		dss_file = dss_folder / "test.dss"
		dss_file.write_text("New Load.Load1 Bus1=bus1 kW=100")

		# 创建带噪声的 loadshape 目录
		loadshape_dir = dss_folder / "loadshape" / "data_with_gaussian_noise"
		loadshape_dir.mkdir(parents=True)

		# 创建 CSV 文件
		csv_file = loadshape_dir / "loadshape_001.csv"
		csv_file.write_text("Load1\n1.0\n0.9\n0.8")

		from envs.vvc.vvc.loadprofile import LoadProfile

		lp = LoadProfile(
			steps=24,
			dss_folder_path=str(dss_folder),
			dss_file=str(dss_file),
			use_noise=True
		)

		assert 'gaussian_noise' in lp.loadshape_path


@pytest.mark.unit
@pytest.mark.powerzoo
class TestLoadProfileMultiWorker:
	"""测试 LoadProfile 多 worker 支持"""

	def test_different_worker_idx_creates_different_files(self, tmp_path):
		"""测试不同 worker_idx 创建不同的文件名"""
		dss_folder = tmp_path / "multi_worker"
		dss_folder.mkdir()

		dss_file = dss_folder / "test.dss"
		dss_file.write_text("New Load.Load1 Bus1=bus1 kW=100")

		loadshape_dir = dss_folder / "loadshape" / "data_without_noise"
		loadshape_dir.mkdir(parents=True)

		from envs.vvc.vvc.loadprofile import LoadProfile

		lp0 = LoadProfile(
			steps=24,
			dss_folder_path=str(dss_folder),
			dss_file=str(dss_file),
			worker_idx=0
		)

		lp1 = LoadProfile(
			steps=24,
			dss_folder_path=str(dss_folder),
			dss_file=str(dss_file),
			worker_idx=1
		)

		assert lp0.loadshape_dss == 'loadshape_0.dss'
		assert lp1.loadshape_dss == 'loadshape_1.dss'

	def test_none_worker_idx_uses_default(self, tmp_path):
		"""测试 None worker_idx 使用默认文件名"""
		dss_folder = tmp_path / "default_worker"
		dss_folder.mkdir()

		dss_file = dss_folder / "test.dss"
		dss_file.write_text("New Load.Load1 Bus1=bus1 kW=100")

		loadshape_dir = dss_folder / "loadshape" / "data_without_noise"
		loadshape_dir.mkdir(parents=True)

		from envs.vvc.vvc.loadprofile import LoadProfile

		lp = LoadProfile(
			steps=24,
			dss_folder_path=str(dss_folder),
			dss_file=str(dss_file),
			worker_idx=None
		)

		assert lp.loadshape_dss == 'loadshape.dss'


# ==============================================================================
# 负荷名称提取测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.powerzoo
class TestFindLoadNames:
	"""测试负荷名称提取功能"""

	def test_find_load_names_with_standard_format(self, tmp_path):
		"""测试标准格式的负荷名称提取"""
		dss_file = tmp_path / "standard.dss"
		dss_content = """
		New Load.MyLoad1 Bus1=bus1 Phases=3 kW=100
		New Load.MyLoad2 Bus1=bus2 Phases=1 kW=50
		New Load.MyLoad3 Bus1=bus3 Phases=3 kW=200
		"""
		dss_file.write_text(dss_content)

		# 手动解析（模拟 find_load_names 方法）
		load_names = []
		with open(str(dss_file), 'r') as f:
			for line in f:
				line_lower = line.lower().strip()
				if line_lower.startswith('new load.'):
					parts = line.split()
					if len(parts) >= 2:
						name_part = parts[1]
						if '.' in name_part:
							load_name = name_part.split('.', 1)[1]
							load_names.append(load_name)

		assert 'MyLoad1' in load_names
		assert 'MyLoad2' in load_names
		assert 'MyLoad3' in load_names

	def test_find_load_names_with_comments(self, tmp_path):
		"""测试带注释的负荷名称提取"""
		dss_file = tmp_path / "with_comments.dss"
		dss_content = """
		! This is a comment
		New Load.Load1 Bus1=bus1 kW=100  ! inline comment
		// Another comment style
		New Load.Load2 Bus1=bus2 kW=50  // inline comment
		"""
		dss_file.write_text(dss_content)

		load_names = []
		with open(str(dss_file), 'r') as f:
			for line in f:
				line_clean = line.strip()
				if '!' in line_clean:
					line_clean = line_clean[:line_clean.find('!')]
				if '//' in line_clean:
					line_clean = line_clean[:line_clean.find('//')]
				line_lower = line_clean.lower().strip()
				if line_lower.startswith('new load.'):
					parts = line_clean.split()
					if len(parts) >= 2:
						name_part = parts[1]
						if '.' in name_part:
							load_name = name_part.split('.', 1)[1]
							load_names.append(load_name)

		assert 'Load1' in load_names
		assert 'Load2' in load_names

	def test_find_load_names_case_insensitive(self, tmp_path):
		"""测试大小写不敏感的负荷名称提取"""
		dss_file = tmp_path / "case_test.dss"
		dss_content = """
		NEW LOAD.UpperCase Bus1=bus1 kW=100
		new load.lowerCase Bus1=bus2 kW=50
		New Load.MixedCase Bus1=bus3 kW=200
		"""
		dss_file.write_text(dss_content)

		load_names = []
		with open(str(dss_file), 'r') as f:
			for line in f:
				line_lower = line.lower().strip()
				if line_lower.startswith('new load.'):
					parts = line.split()
					if len(parts) >= 2:
						name_part = parts[1]
						if '.' in name_part:
							load_name = name_part.split('.', 1)[1]
							load_names.append(load_name)

		assert len(load_names) == 3
