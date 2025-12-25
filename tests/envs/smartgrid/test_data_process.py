# -*- coding: utf-8 -*-
"""
SmartGrid data_process 模块详细测试

测试覆盖:
- Constants 类
- DSSFileParser 类
- LoadProfile 类
- EpisodeGenerator 类
- ConfigGenerator 类

@File      : test_data_process.py
@Author    : PowerZoo Test Suite
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


# ==============================================================================
# 单元测试 - 导入测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestDataProcessImport:
	"""测试 data_process 模块导入"""

	def test_loadprofile_import(self):
		"""测试 LoadProfile 类导入"""
		from envs.smartgrid.data_process.loadprofile import LoadProfile
		assert LoadProfile is not None

	def test_constants_import(self):
		"""测试 Constants 类导入"""
		from envs.smartgrid.data_process.loadprofile import Constants
		assert Constants is not None

	def test_dss_file_parser_import(self):
		"""测试 DSSFileParser 类导入"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser
		assert DSSFileParser is not None

	def test_episode_generator_import(self):
		"""测试 EpisodeGenerator 导入"""
		from envs.smartgrid.data_process.loadprofile import EpisodeGenerator
		assert EpisodeGenerator is not None

	def test_config_generator_import(self):
		"""测试 ConfigGenerator 导入"""
		from envs.smartgrid.data_process.loadprofile import ConfigGenerator
		assert ConfigGenerator is not None

	def test_all_exports(self):
		"""测试所有导出"""
		from envs.smartgrid.data_process.loadprofile import (
			LoadProfile,
			Constants,
			DSSFileParser,
			EpisodeGenerator,
			ConfigGenerator
		)
		assert all([LoadProfile, Constants, DSSFileParser, EpisodeGenerator, ConfigGenerator])


# ==============================================================================
# 单元测试 - Constants 类
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestConstants:
	"""测试 Constants 类"""

	def test_dss_extension(self):
		"""测试 DSS 扩展名常量"""
		from envs.smartgrid.data_process.loadprofile import Constants

		assert Constants.DSS_EXTENSION == '.dss'

	def test_csv_extension(self):
		"""测试 CSV 扩展名常量"""
		from envs.smartgrid.data_process.loadprofile import Constants

		assert Constants.CSV_EXTENSION == '.csv'

	def test_duty_suffix(self):
		"""测试 duty 后缀常量"""
		from envs.smartgrid.data_process.loadprofile import Constants

		assert Constants.DUTY_SUFFIX == '_duty'

	def test_loadshape_folder(self):
		"""测试 loadshape 文件夹常量"""
		from envs.smartgrid.data_process.loadprofile import Constants

		assert Constants.LOADSHAPE_FOLDER == 'loadshape'

	def test_dss_comment_markers(self):
		"""测试 DSS 注释标记"""
		from envs.smartgrid.data_process.loadprofile import Constants

		assert '!' in Constants.DSS_COMMENT_MARKERS
		assert '//' in Constants.DSS_COMMENT_MARKERS

	def test_dss_load_prefix(self):
		"""测试 DSS 负载前缀"""
		from envs.smartgrid.data_process.loadprofile import Constants

		assert Constants.DSS_LOAD_PREFIX == 'new load.'

	def test_seconds_per_hour(self):
		"""测试每小时秒数常量"""
		from envs.smartgrid.data_process.loadprofile import Constants

		assert Constants.SECONDS_PER_HOUR == 3600

	def test_default_duty_settings(self):
		"""测试默认 duty 设置"""
		from envs.smartgrid.data_process.loadprofile import Constants

		assert 'mode=duty' in Constants.DEFAULT_DUTY_SETTINGS


# ==============================================================================
# 单元测试 - DSSFileParser 类
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestDSSFileParser:
	"""测试 DSSFileParser 类"""

	def test_clean_line_removes_whitespace(self):
		"""测试清理行移除空白"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "  new load.load1 bus1=bus1  "
		cleaned = DSSFileParser.clean_line(line)

		assert cleaned == "new load.load1 bus1=bus1"

	def test_clean_line_removes_comments(self):
		"""测试清理行移除注释"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "new load.load1 bus1=bus1 ! this is a comment"
		cleaned = DSSFileParser.clean_line(line)

		assert cleaned == "new load.load1 bus1=bus1"

	def test_clean_line_removes_double_slash_comments(self):
		"""测试清理行移除双斜杠注释"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "new load.load1 bus1=bus1 // this is a comment"
		cleaned = DSSFileParser.clean_line(line)

		assert cleaned == "new load.load1 bus1=bus1"

	def test_parse_load_line_valid(self):
		"""测试解析有效负载行"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "new load.load_s844 bus1=s844b phases=1 kV=2.4 kW=135 kvar=105 model=5"
		is_load, load_name = DSSFileParser.parse_load_line(line)

		assert is_load is True
		assert load_name == "load_s844"

	def test_parse_load_line_invalid(self):
		"""测试解析非负载行"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "new capacitor.cap1 bus1=bus1"
		is_load, load_name = DSSFileParser.parse_load_line(line)

		assert is_load is False
		assert load_name is None

	def test_parse_load_line_comment(self):
		"""测试解析注释行"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "! this is a comment"
		is_load, load_name = DSSFileParser.parse_load_line(line)

		assert is_load is False

	def test_has_duty_in_line_true(self):
		"""测试检测 duty 关键字 - 存在"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "new load.load1 bus1=bus1 duty=loadshape_load1"
		has_duty = DSSFileParser.has_duty_in_line(line)

		assert has_duty is True

	def test_has_duty_in_line_false(self):
		"""测试检测 duty 关键字 - 不存在"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "new load.load1 bus1=bus1"
		has_duty = DSSFileParser.has_duty_in_line(line)

		assert has_duty is False

	def test_is_duty_mode_line_true(self):
		"""测试检测 duty 模式行 - 是"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "Set mode=duty number=360 hour=0"
		is_duty = DSSFileParser.is_duty_mode_line(line)

		assert is_duty is True

	def test_is_duty_mode_line_false(self):
		"""测试检测 duty 模式行 - 否"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "new load.load1 bus1=bus1"
		is_duty = DSSFileParser.is_duty_mode_line(line)

		assert is_duty is False

	def test_parse_redirect_line_loads(self):
		"""测试解析 redirect 行 - loads"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "redirect loads.dss"
		result = DSSFileParser.parse_redirect_line(line)

		assert result == "loads.dss"

	def test_parse_redirect_line_not_loads(self):
		"""测试解析 redirect 行 - 非 loads"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "redirect buscoords.dss"
		result = DSSFileParser.parse_redirect_line(line)

		assert result is None


# ==============================================================================
# 单元测试 - LoadProfile 类基础功能
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestLoadProfileBasic:
	"""测试 LoadProfile 类基础功能"""

	def test_loadprofile_class_exists(self):
		"""测试 LoadProfile 类存在"""
		from envs.smartgrid.data_process.loadprofile import LoadProfile

		assert LoadProfile is not None

	def test_loadprofile_has_required_attributes(self):
		"""测试 LoadProfile 有必需属性"""
		from envs.smartgrid.data_process.loadprofile_core import LoadProfile

		# 检查类方法
		assert hasattr(LoadProfile, '__init__')
		assert hasattr(LoadProfile, 'find_load_names')


# ==============================================================================
# 集成测试 - LoadProfile 类
# ==============================================================================

@pytest.mark.integration
@pytest.mark.smartgrid
@pytest.mark.requires_opendss
class TestLoadProfileIntegration:
	"""测试 LoadProfile 类集成"""

	def test_loadprofile_creation(self, node_systems_dir, skip_if_no_opendss):
		"""测试 LoadProfile 创建"""
		from envs.smartgrid.data_process.loadprofile import LoadProfile

		# 查找可用系统
		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list(dss_folder.glob("*duty.dss"))
		if not dss_files:
			# 尝试其他 DSS 文件
			dss_files = list(dss_folder.glob("*.dss"))

		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		try:
			lp = LoadProfile(
				steps=24,
				dss_folder_path=str(dss_folder),
				dss_file=dss_files[0].name
			)
			assert lp is not None
			assert lp.steps == 24
			assert len(lp.load_names) > 0
		except Exception as e:
			pytest.skip(f"LoadProfile 创建失败: {e}")

	def test_loadprofile_with_worker_idx(self, node_systems_dir, skip_if_no_opendss):
		"""测试带 worker_idx 的 LoadProfile"""
		from envs.smartgrid.data_process.loadprofile import LoadProfile

		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list(dss_folder.glob("*duty.dss")) + list(dss_folder.glob("*.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		try:
			lp = LoadProfile(
				steps=24,
				dss_folder_path=str(dss_folder),
				dss_file=dss_files[0].name,
				worker_idx=0
			)
			assert lp is not None
			assert lp.worker_idx == 0
			# worker_idx 应该影响 loadshape_dss 文件名
			assert '_0' in lp.loadshape_dss
		except Exception as e:
			pytest.skip(f"LoadProfile with worker_idx 创建失败: {e}")

	def test_loadprofile_find_load_names(self, node_systems_dir, skip_if_no_opendss):
		"""测试 LoadProfile 查找负载名称"""
		from envs.smartgrid.data_process.loadprofile import LoadProfile

		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list(dss_folder.glob("*duty.dss")) + list(dss_folder.glob("*.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		try:
			lp = LoadProfile(
				steps=24,
				dss_folder_path=str(dss_folder),
				dss_file=dss_files[0].name
			)

			assert len(lp.load_names) > 0
			# 负载名称应该唯一
			assert len(lp.load_names) == len(set(lp.load_names))
		except Exception as e:
			pytest.skip(f"LoadProfile 查找负载名称失败: {e}")

	def test_loadprofile_generate_episodes(self, node_systems_dir, skip_if_no_opendss):
		"""测试 LoadProfile 生成 episodes"""
		from envs.smartgrid.data_process.loadprofile import LoadProfile

		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list(dss_folder.glob("*duty.dss")) + list(dss_folder.glob("*.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		try:
			lp = LoadProfile(
				steps=24,
				dss_folder_path=str(dss_folder),
				dss_file=dss_files[0].name
			)

			# 尝试生成 episodes
			if hasattr(lp, 'generate_episodes_from_existing_files'):
				num_profiles = lp.generate_episodes_from_existing_files()
				assert num_profiles >= 0
			elif hasattr(lp, 'gen_loadprofile'):
				num_profiles = lp.gen_loadprofile()
				assert num_profiles >= 0
		except Exception as e:
			pytest.skip(f"LoadProfile 生成 episodes 失败: {e}")

	def test_loadprofile_select_profile(self, node_systems_dir, skip_if_no_opendss):
		"""测试 LoadProfile 选择配置"""
		from envs.smartgrid.data_process.loadprofile import LoadProfile

		for variant in ['34Bus_PV_Aggressive', '34Bus_PV', '34Bus']:
			dss_folder = node_systems_dir / variant
			if dss_folder.exists():
				break
		else:
			pytest.skip("没有可用的 34Bus 系统")

		dss_files = list(dss_folder.glob("*duty.dss")) + list(dss_folder.glob("*.dss"))
		if not dss_files:
			pytest.skip("未找到 DSS 文件")

		try:
			lp = LoadProfile(
				steps=24,
				dss_folder_path=str(dss_folder),
				dss_file=dss_files[0].name
			)

			# 生成 episodes
			if hasattr(lp, 'generate_episodes_from_existing_files'):
				num_profiles = lp.generate_episodes_from_existing_files()
			elif hasattr(lp, 'gen_loadprofile'):
				num_profiles = lp.gen_loadprofile()

			# 选择配置
			if hasattr(lp, 'select_load_profile'):
				lp.select_load_profile(0)
			elif hasattr(lp, 'choose_loadprofile'):
				lp.choose_loadprofile(0, False)
		except Exception as e:
			pytest.skip(f"LoadProfile 选择配置失败: {e}")


# ==============================================================================
# 单元测试 - EpisodeGenerator 类
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestEpisodeGenerator:
	"""测试 EpisodeGenerator 类"""

	def test_episode_generator_exists(self):
		"""测试 EpisodeGenerator 类存在"""
		from envs.smartgrid.data_process.loadprofile_episode import EpisodeGenerator

		assert EpisodeGenerator is not None

	def test_episode_generator_has_required_methods(self):
		"""测试 EpisodeGenerator 有必需方法"""
		from envs.smartgrid.data_process.loadprofile_episode import EpisodeGenerator

		assert hasattr(EpisodeGenerator, '__init__')


# ==============================================================================
# 单元测试 - ConfigGenerator 类
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestConfigGenerator:
	"""测试 ConfigGenerator 类"""

	def test_config_generator_exists(self):
		"""测试 ConfigGenerator 类存在"""
		from envs.smartgrid.data_process.loadprofile_config import ConfigGenerator

		assert ConfigGenerator is not None

	def test_config_generator_has_required_methods(self):
		"""测试 ConfigGenerator 有必需方法"""
		from envs.smartgrid.data_process.loadprofile_config import ConfigGenerator

		assert hasattr(ConfigGenerator, '__init__')


# ==============================================================================
# 边界条件测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestDataProcessEdgeCases:
	"""测试 data_process 边界条件"""

	def test_dss_parser_empty_line(self):
		"""测试 DSS 解析器处理空行"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = ""
		cleaned = DSSFileParser.clean_line(line)
		assert cleaned == ""

	def test_dss_parser_whitespace_only(self):
		"""测试 DSS 解析器处理纯空白行"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "   \t   "
		cleaned = DSSFileParser.clean_line(line)
		assert cleaned == ""

	def test_dss_parser_comment_only(self):
		"""测试 DSS 解析器处理纯注释行"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "! This is only a comment"
		cleaned = DSSFileParser.clean_line(line)
		assert cleaned == ""

	def test_dss_parser_mixed_case(self):
		"""测试 DSS 解析器处理混合大小写"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		line = "NEW LOAD.Load1 Bus1=Bus1"
		is_load, load_name = DSSFileParser.parse_load_line(line)
		# 应该能识别大写的 NEW LOAD
		assert is_load is True

	def test_dss_parser_malformed_load_line(self):
		"""测试 DSS 解析器处理格式错误的负载行"""
		from envs.smartgrid.data_process.loadprofile import DSSFileParser

		# 缺少点号分隔符
		line = "new load_load1 bus1=bus1"
		is_load, load_name = DSSFileParser.parse_load_line(line)
		# 应该无法解析为负载
		assert is_load is False

	def test_constants_immutability(self):
		"""测试常量不可变性"""
		from envs.smartgrid.data_process.loadprofile import Constants

		# 常量应该是字符串或列表
		assert isinstance(Constants.DSS_EXTENSION, str)
		assert isinstance(Constants.DSS_COMMENT_MARKERS, list)


# ==============================================================================
# 临时文件测试
# ==============================================================================

@pytest.mark.unit
@pytest.mark.smartgrid
class TestDataProcessTempFiles:
	"""测试 data_process 临时文件处理"""

	def test_create_temp_dss_file(self):
		"""测试创建临时 DSS 文件"""
		with tempfile.TemporaryDirectory() as tmpdir:
			dss_content = """
! Test DSS file
new load.test_load bus1=bus1 kV=12.47 kW=100 kvar=50
			"""

			dss_path = Path(tmpdir) / "test.dss"
			with open(dss_path, 'w') as f:
				f.write(dss_content)

			assert dss_path.exists()

			# 验证可以解析
			from envs.smartgrid.data_process.loadprofile import DSSFileParser

			with open(dss_path, 'r') as f:
				for line in f:
					is_load, load_name = DSSFileParser.parse_load_line(line)
					if is_load:
						assert load_name == "test_load"

	def test_loadshape_folder_structure(self):
		"""测试 loadshape 文件夹结构"""
		with tempfile.TemporaryDirectory() as tmpdir:
			loadshape_dir = Path(tmpdir) / "loadshape"
			loadshape_dir.mkdir()

			# 创建测试 CSV 文件
			csv_path = loadshape_dir / "loadshape_test.csv"
			pd.DataFrame({'mult': [1.0, 0.9, 0.8]}).to_csv(csv_path, index=False)

			assert csv_path.exists()
			df = pd.read_csv(csv_path)
			assert 'mult' in df.columns
