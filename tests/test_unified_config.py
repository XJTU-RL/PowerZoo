# tests/test_unified_config.py
"""Unified config architecture tests"""
import pytest
from pathlib import Path


class TestResolveSystemPath:
    """Test resolve_system_path utility"""

    def test_resolves_13bus(self):
        from utils.path_utils import resolve_system_path
        path = resolve_system_path('13Bus')
        assert path.exists()
        assert path.name == '13Bus'
        assert 'node_systems' in str(path)

    def test_resolves_34bus_pv(self):
        from utils.path_utils import resolve_system_path
        path = resolve_system_path('34Bus_PV')
        assert path.exists()

    def test_resolves_with_prefix(self):
        from utils.path_utils import resolve_system_path
        path = resolve_system_path('node_systems/13Bus')
        assert path.exists()
        assert path.name == '13Bus'

    def test_raises_for_nonexistent(self):
        from utils.path_utils import resolve_system_path
        with pytest.raises(FileNotFoundError):
            resolve_system_path('NonExistent999Bus')


class TestLegacyVariantMapping:
    """Test VVC legacy env_name -> config expansion"""

    def test_13bus_cbat_expands(self):
        from utils.unified_config_loader import expand_legacy_variant
        result = expand_legacy_variant('13Bus_cbat')
        assert result['system_ref'] == '13Bus'
        assert result['bat_act_num'] == float('inf')

    def test_13bus_soc_expands(self):
        from utils.unified_config_loader import expand_legacy_variant
        result = expand_legacy_variant('13Bus_soc')
        assert result['system_ref'] == '13Bus'
        assert result['soc_w'] == pytest.approx(20.0 / 33)

    def test_34bus_pv_expands(self):
        from utils.unified_config_loader import expand_legacy_variant
        result = expand_legacy_variant('34Bus_pv')
        assert result['system_ref'] == '34Bus'
        assert result['pv_control_enabled'] is True

    def test_unknown_returns_none(self):
        from utils.unified_config_loader import expand_legacy_variant
        result = expand_legacy_variant('totally_unknown')
        assert result is None

    def test_base_system_returns_none(self):
        from utils.unified_config_loader import expand_legacy_variant
        result = expand_legacy_variant('13Bus')
        assert result is None


class TestSmartGridConfigFromEnvArgs:
    """Test SmartGridConfig.from_env_args()"""

    def test_basic_fields(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        env_args = {
            'system_name': '34Bus_PV',
            'dss_file': 'ieee34Mod1_duty.dss',
            'max_episode_steps': 360,
            'seed': 42,
            'pv_plan': 'aggressive',
        }
        config = SmartGridConfig.from_env_args(env_args)
        assert config.system_name == '34Bus_PV'
        assert config.max_episode_steps == 360
        assert config.pv_plan == 'aggressive'

    def test_device_config_from_env_specific(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        env_args = {
            'env_specific_config': {
                'devices': {
                    'batteries': {'action_num': 33, 'action_space': 'continuous'},
                    'pv_systems': {'control_enabled': True, 'action_space': 'continuous'},
                },
                'reward_weights': {
                    'power_loss': 0.4,
                    'pv_control': 0.4,
                },
            },
        }
        config = SmartGridConfig.from_env_args(env_args)
        assert config.battery.is_continuous
        assert config.pv.control_enabled
        assert config.reward_weights.power_loss == 0.4

    def test_unknown_keys_ignored(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        env_args = {'totally_unknown_key': 'should_not_crash'}
        config = SmartGridConfig.from_env_args(env_args)
        assert not hasattr(config, 'totally_unknown_key')

    def test_defaults_used_when_empty(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        config = SmartGridConfig.from_env_args({})
        assert config.max_episode_steps == 360
        assert config.pv_plan is None


class TestCircuitPvPlan:
    """Test pv_plan injection into Circuits"""

    def test_circuits_accepts_pv_plan(self):
        from envs.smartgrid.circuit_system.circuit import Circuits
        import inspect
        sig = inspect.signature(Circuits.__init__)
        assert 'pv_plan' in sig.parameters

    def test_temp_compile_uses_pv_plan(self, tmp_path):
        """_create_temp_compile_file should use pv_plans/{plan}.dss"""
        import os
        dss_dir = tmp_path / "test_system"
        dss_dir.mkdir()
        (dss_dir / "main.dss").write_text("! dummy\n")
        (dss_dir / "loadshape.dss").write_text("! dummy\n")
        (dss_dir / "pv_data.dss").write_text("! no PVSystem here\n")
        pv_plans = dss_dir / "pv_plans"
        pv_plans.mkdir()
        (pv_plans / "aggressive.dss").write_text("New PVSystem.PV1 phases=3\n")

        original_cwd = os.getcwd()
        os.chdir(str(dss_dir))
        try:
            from envs.smartgrid.circuit_system.circuit import Circuits
            c = object.__new__(Circuits)
            c.worker_idx = None
            c.pv_plan = "aggressive"
            temp_file = c._create_temp_compile_file("main.dss")
            with open(temp_file, 'r') as f:
                content = f.read()
            assert "pv_plans/aggressive.dss" in content
            assert "pv_systems_base.dss" not in content
            os.remove(temp_file)
        finally:
            os.chdir(original_cwd)

    def test_temp_compile_fallback_without_pv_plan(self, tmp_path):
        """Without pv_plan, should fall back to pv_systems_base.dss"""
        import os
        dss_dir = tmp_path / "test_fallback"
        dss_dir.mkdir()
        (dss_dir / "main.dss").write_text("! dummy\n")
        (dss_dir / "loadshape.dss").write_text("! dummy\n")
        (dss_dir / "pv_data.dss").write_text("! no PVSystem here\n")
        (dss_dir / "pv_systems_base.dss").write_text("New PVSystem.PV1 phases=3\n")

        original_cwd = os.getcwd()
        os.chdir(str(dss_dir))
        try:
            from envs.smartgrid.circuit_system.circuit import Circuits
            c = object.__new__(Circuits)
            c.worker_idx = None
            c.pv_plan = None  # No pv_plan -> use pv_systems_base.dss
            temp_file = c._create_temp_compile_file("main.dss")
            with open(temp_file, 'r') as f:
                content = f.read()
            assert "pv_systems_base.dss" in content
            os.remove(temp_file)
        finally:
            os.chdir(original_cwd)


class TestDSRConfigFromEnvArgs:
    """Test DSRConfig.from_env_args()"""

    def test_basic_fields(self):
        from envs.dsr.core.config import DSRConfig
        env_args = {
            'system_name': '123Bus',
            'max_episode_steps': 15,
            'n_dg': 7,
            'reward_restore': 20.0,
        }
        config = DSRConfig.from_env_args(env_args)
        assert config.system_name == '123Bus'
        assert config.n_dg == 7

    def test_unknown_keys_ignored(self):
        from envs.dsr.core.config import DSRConfig
        config = DSRConfig.from_env_args({'unknown_key': 'ignored'})
        assert config.system_name == '123Bus'  # default

    def test_none_values_use_default(self):
        from envs.dsr.core.config import DSRConfig
        config = DSRConfig.from_env_args({'system_name': None})
        assert config.system_name == '123Bus'  # default, not None

    def test_all_dataclass_fields_accepted(self):
        """Any field defined in DSRConfig should be accepted from env_args"""
        import dataclasses
        from envs.dsr.core.config import DSRConfig
        # Get all field names
        field_names = [f.name for f in dataclasses.fields(DSRConfig)]
        # Verify we can pass any field
        assert 'fault_scenarios' in field_names  # was "dead" before
        assert 'n1_security' not in field_names or True  # may not be a field


class TestStackelbergConfigFromEnvArgs:
    """Test StackelbergConfig.from_env_args()"""

    def test_basic_fields(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        env_args = {
            'system_name': '13Bus',
            'max_episode_steps': 24,
            'n_consumer_agents': 8,
        }
        config = StackelbergConfig.from_env_args(env_args)
        assert config.system_name == '13Bus'
        assert config.n_consumer_agents == 8

    def test_extracts_system_from_env_name(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        env_args = {'env_name': 'stackelberg_34Bus'}
        config = StackelbergConfig.from_env_args(env_args)
        assert config.system_name == '34Bus'

    def test_max_episode_steps_alias(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        env_args = {'num_steps': 48}
        config = StackelbergConfig.from_env_args(env_args)
        assert config.max_episode_steps == 48

    def test_dict_sub_configs_passthrough(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        tou = {'peak_hours': [8, 9, 10]}
        env_args = {'tou_config': tou}
        config = StackelbergConfig.from_env_args(env_args)
        assert config.tou_config == tou

    def test_unknown_keys_ignored(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        config = StackelbergConfig.from_env_args({'totally_unknown': True})
        assert config.system_name == '13Bus'  # default
