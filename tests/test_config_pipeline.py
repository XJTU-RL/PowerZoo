"""End-to-end integration tests: YAML -> Config -> env validation

Tests the complete config pipeline:
1. Load YAML via get_defaults_yaml_args()
2. Build typed Config dataclass via from_env_args()
3. Validate key fields match expected YAML values
"""
import pytest

from utils.configs_tools import get_defaults_yaml_args


class TestSmartGridPipeline:
    """SmartGrid: smartgrid.yaml -> SmartGridConfig"""

    def test_yaml_to_config(self):
        algo_args, env_args = get_defaults_yaml_args('happo', 'smartgrid')
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        config = SmartGridConfig.from_env_args(env_args)
        assert config.system_name == '34Bus_PV'
        assert config.pv_plan == 'aggressive'
        assert config.max_episode_steps == 360

    def test_env_args_has_system_name(self):
        _, env_args = get_defaults_yaml_args('happo', 'smartgrid')
        assert env_args.get('system_name') == '34Bus_PV'

    def test_env_args_has_max_episode_steps(self):
        _, env_args = get_defaults_yaml_args('happo', 'smartgrid')
        assert env_args.get('max_episode_steps') == 360


class TestVVCPipeline:
    """VVC: vvc.yaml -> VVCConfig"""

    def test_yaml_to_config(self):
        algo_args, env_args = get_defaults_yaml_args('happo', 'vvc')
        from envs.vvc.vvc.vvc_config import VVCConfig
        config = VVCConfig.from_env_args(env_args)
        assert config.system_name == '13Bus'
        assert config.max_episode_steps == 24

    def test_env_args_has_system_name(self):
        _, env_args = get_defaults_yaml_args('happo', 'vvc')
        assert env_args.get('system_name') == '13Bus'

    def test_episode_length_mapped(self):
        """episode_length in YAML -> max_episode_steps in env_args"""
        _, env_args = get_defaults_yaml_args('happo', 'vvc')
        assert env_args.get('max_episode_steps') == 24


class TestDSRPipeline:
    """DSR: dsr.yaml -> DSRConfig"""

    def test_yaml_to_config(self):
        algo_args, env_args = get_defaults_yaml_args('happo', 'dsr')
        from envs.dsr.core.config import DSRConfig
        config = DSRConfig.from_env_args(env_args)
        assert config.system_name == '123Bus'
        assert config.max_episode_steps == 15

    def test_env_args_has_system_name(self):
        _, env_args = get_defaults_yaml_args('happo', 'dsr')
        assert env_args.get('system_name') == '123Bus'


class TestStackelbergPipeline:
    """Stackelberg: stackelberg_13bus.yaml -> StackelbergConfig"""

    def test_yaml_to_config(self):
        algo_args, env_args = get_defaults_yaml_args('happo', 'stackelberg_13bus')
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        config = StackelbergConfig.from_env_args(env_args)
        assert config.system_name == '13Bus'
        assert config.max_episode_steps == 24

    def test_env_args_has_system_name(self):
        _, env_args = get_defaults_yaml_args('happo', 'stackelberg_13bus')
        assert env_args.get('system_name') == '13Bus'


class TestAllConfigsHaveFromEnvArgs:
    """Verify all Config classes have from_env_args classmethod"""

    def test_smartgrid(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig
        assert hasattr(SmartGridConfig, 'from_env_args')
        assert callable(getattr(SmartGridConfig, 'from_env_args'))

    def test_vvc(self):
        from envs.vvc.vvc.vvc_config import VVCConfig
        assert hasattr(VVCConfig, 'from_env_args')
        assert callable(getattr(VVCConfig, 'from_env_args'))

    def test_dsr(self):
        from envs.dsr.core.config import DSRConfig
        assert hasattr(DSRConfig, 'from_env_args')
        assert callable(getattr(DSRConfig, 'from_env_args'))

    def test_stackelberg(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig
        assert hasattr(StackelbergConfig, 'from_env_args')
        assert callable(getattr(StackelbergConfig, 'from_env_args'))


class TestCrossEnvironmentConsistency:
    """Cross-environment consistency checks"""

    def test_all_envs_produce_system_name(self):
        """All environments must produce a system_name in env_args"""
        env_pairs = [
            ('happo', 'smartgrid'),
            ('happo', 'vvc'),
            ('happo', 'dsr'),
            ('happo', 'stackelberg_13bus'),
        ]
        for algo, env in env_pairs:
            _, env_args = get_defaults_yaml_args(algo, env)
            assert 'system_name' in env_args, f"{env} missing system_name"

    def test_all_envs_produce_max_episode_steps(self):
        """All environments must produce max_episode_steps in env_args"""
        env_pairs = [
            ('happo', 'smartgrid', 360),
            ('happo', 'vvc', 24),
            ('happo', 'dsr', 15),
            ('happo', 'stackelberg_13bus', 24),
        ]
        for algo, env, expected in env_pairs:
            _, env_args = get_defaults_yaml_args(algo, env)
            assert env_args.get('max_episode_steps') == expected, (
                f"{env}: expected max_episode_steps={expected}, "
                f"got {env_args.get('max_episode_steps')}"
            )
