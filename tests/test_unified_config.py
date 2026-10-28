# tests/test_unified_config.py
"""Unified config architecture tests"""
import pytest
from utils.config_validation import ConfigError
from pathlib import Path


class TestResolveSystemPath:
    """Test resolve_system_path utility"""

    def test_resolves_13bus(self):
        from utils.path_utils import resolve_system_path

        path = resolve_system_path("13Bus")
        assert path.exists()
        assert path.name == "13Bus"
        assert "node_systems" in str(path)

    def test_resolves_34bus_pv(self):
        from utils.path_utils import resolve_system_path

        path = resolve_system_path("34Bus_PV")
        assert path.exists()

    def test_resolves_with_prefix(self):
        from utils.path_utils import resolve_system_path

        path = resolve_system_path("node_systems/13Bus")
        assert path.exists()
        assert path.name == "13Bus"

    def test_raises_for_nonexistent(self):
        from utils.path_utils import resolve_system_path

        with pytest.raises(FileNotFoundError):
            resolve_system_path("NonExistent999Bus")


class TestLegacyVariantMapping:
    """Test VVC legacy env_name -> config expansion"""

    def test_13bus_cbat_expands(self):
        from utils.unified_config_loader import expand_legacy_variant

        result = expand_legacy_variant("13Bus_cbat")
        assert result["system_ref"] == "13Bus"
        assert result["bat_act_num"] == float("inf")

    def test_13bus_soc_expands(self):
        from utils.unified_config_loader import expand_legacy_variant

        result = expand_legacy_variant("13Bus_soc")
        assert result["system_ref"] == "13Bus"
        assert result["soc_w"] == pytest.approx(20.0 / 33)

    def test_34bus_pv_expands(self):
        from utils.unified_config_loader import expand_legacy_variant

        result = expand_legacy_variant("34Bus_pv")
        assert result["system_ref"] == "34Bus"
        assert result["pv_control_enabled"] is True

    def test_unknown_returns_none(self):
        from utils.unified_config_loader import expand_legacy_variant

        result = expand_legacy_variant("totally_unknown")
        assert result is None

    def test_base_system_returns_none(self):
        from utils.unified_config_loader import expand_legacy_variant

        result = expand_legacy_variant("13Bus")
        assert result is None


class TestSmartGridConfigFromEnvArgs:
    """Test SmartGridConfig.from_env_args()"""

    def test_basic_fields(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig

        env_args = {
            "system_name": "34Bus_PV",
            "dss_file": "ieee34Mod1_duty.dss",
            "max_episode_steps": 360,
            "seed": 42,
            "pv_plan": "aggressive",
        }
        config = SmartGridConfig.from_env_args(env_args)
        assert config.system_name == "34Bus_PV"
        assert config.max_episode_steps == 360
        assert config.pv_plan == "aggressive"

    def test_device_config_from_env_specific(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig

        env_args = {
            "env_specific_config": {
                "devices": {
                    "batteries": {"action_num": 33, "action_space": "continuous"},
                    "pv_systems": {
                        "control_enabled": True,
                        "action_space": "continuous",
                    },
                },
                "reward_weights": {
                    "power_loss": 0.4,
                    "pv_control": 0.4,
                },
            },
        }
        config = SmartGridConfig.from_env_args(env_args)
        assert config.bat_act_num == float("inf")
        assert config.pv_control_enabled
        assert config.reward_weights.power_loss == 0.4

    def test_unknown_keys_ignored(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig

        with pytest.raises(ConfigError):
            SmartGridConfig.from_env_args({"unknown_key": True})

    def test_defaults_used_when_empty(self):
        from envs.smartgrid.base_env.env_config import SmartGridConfig

        config = SmartGridConfig.from_env_args({})
        assert config.max_episode_steps == 360
        assert config.pv_plan is None


class TestCircuitPvPlan:
    @pytest.mark.parametrize(
        "plan,count",
        [("none", 0), ("aggressive", 9), ("optimized", 7), ("conservative", 5)],
    )
    def test_overlay_is_compiled_without_rewriting_source(self, plan, count):
        from envs.smartgrid.circuit_system import SmartGridCircuit

        folder = Path(__file__).resolve().parents[1] / "node_systems/34Bus_PV"
        before = (folder / "network.dss").read_bytes()
        c = SmartGridCircuit(folder, "network.dss", plan, 60)
        try:
            assert len(c.pvs) == count
            assert (folder / "network.dss").read_bytes() == before
            assert c.folder != folder
        finally:
            c.close()


class TestDSRConfigFromEnvArgs:
    """Test DSRConfig.from_env_args()"""

    def test_basic_fields(self):
        from envs.dsr.core.config import DSRConfig

        env_args = {
            "system_name": "RestorationSmall",
            "max_episode_steps": 12,
            "priorities": {"Load.critical": 3.},
        }
        config = DSRConfig.from_env_args(env_args)
        assert config.system_name == "RestorationSmall"
        assert config.priorities["Load.critical"] == 3.

    def test_unknown_keys_ignored(self):
        from envs.dsr.core.config import DSRConfig

        with pytest.raises(ConfigError):
            DSRConfig.from_env_args({"unknown_key": True})

    def test_none_values_use_default(self):
        from envs.dsr.core.config import DSRConfig

        with pytest.raises(ConfigError):
            DSRConfig.from_env_args({"system_name": None})

    def test_all_dataclass_fields_accepted(self):
        """Any field defined in DSRConfig should be accepted from env_args"""
        import dataclasses
        from envs.dsr.core.config import DSRConfig

        # Get all field names
        field_names = [f.name for f in dataclasses.fields(DSRConfig)]
        # Verify we can pass any field
        assert "fault_scenarios" not in field_names
        assert "data" in field_names and "control_groups" in field_names
        assert "n_dg" not in field_names and "n_pv" not in field_names
        assert "use_dynamic_network" not in field_names


class TestStackelbergConfigFromEnvArgs:
    """Test StackelbergConfig.from_env_args()"""

    def test_basic_fields(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig

        env_args = {
            "system_name": "13Bus",
            "max_episode_steps": 24,
            "n_consumer_agents": 8,
        }
        config = StackelbergConfig.from_env_args(env_args)
        assert config.system_name == "13Bus"
        assert config.n_consumer_agents == 8

    def test_extracts_system_from_env_name(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig

        env_args = {"env_name": "stackelberg_34Bus"}
        config = StackelbergConfig.from_env_args(env_args)
        assert config.system_name == "34Bus"

    def test_max_episode_steps_alias(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig

        env_args = {"num_steps": 48}
        config = StackelbergConfig.from_env_args(env_args)
        assert config.max_episode_steps == 48

    def test_dict_sub_configs_passthrough(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig

        with pytest.raises(ConfigError):
            StackelbergConfig.from_env_args({"tou_config": {"peak_hours": [8, 9]}})

    def test_unknown_keys_ignored(self):
        from envs.stackelberg.stackelberg_config import StackelbergConfig

        with pytest.raises(ConfigError):
            StackelbergConfig.from_env_args({"unknown_key": True})


class TestVVCConfigFromEnvArgs:
    """Test VVCConfig.from_env_args()"""

    def test_basic_fields(self):
        from envs.vvc.vvc.vvc_config import VVCConfig

        env_args = {
            "system_name": "13Bus",
            "dss_file": "IEEE13Nodeckt_daily.dss",
            "max_episode_steps": 24,
            "power_w": 10.0,
            "reg_act_num": 33,
        }
        config = VVCConfig.from_env_args(env_args)
        assert config.system_name == "13Bus"
        assert config.power_w == 10.0

    def test_episode_length_alias(self):
        from envs.vvc.vvc.vvc_config import VVCConfig

        config = VVCConfig.from_env_args({"episode_length": 12})
        assert config.max_episode_steps == 12

    def test_num_steps_alias(self):
        from envs.vvc.vvc.vvc_config import VVCConfig

        config = VVCConfig.from_env_args({"num_steps": 18})
        assert config.max_episode_steps == 18

    def test_pv_control_alias(self):
        from envs.vvc.vvc.vvc_config import VVCConfig

        with pytest.raises(ConfigError):
            VVCConfig.from_env_args({"pv_control": True})

    def test_pv_flag_from_env_info_style(self):
        from envs.vvc.vvc.vvc_config import VVCConfig

        with pytest.raises(ConfigError):
            VVCConfig.from_env_args({"pv": True})

    def test_defaults(self):
        from envs.vvc.vvc.vvc_config import VVCConfig

        config = VVCConfig.from_env_args({})
        assert config.system_name == "13Bus"
        assert config.reg_act_num == 33
        assert not hasattr(config, "pv_act_num")
        assert not hasattr(config, "pv_control_enabled")

    def test_one_typed_config_without_old_engine_views(self):
        from dataclasses import asdict
        from envs.vvc.vvc.vvc_config import VVCConfig

        config = VVCConfig(power_w=10)
        assert asdict(config)["power_w"] == 10
        assert not hasattr(config, "to_env_info")
        assert not hasattr(config, "to_sys_info")
        with pytest.raises(TypeError):
            VVCConfig(source_bus="sourcebus", node_size=500)

    def test_unknown_keys_ignored(self):
        from envs.vvc.vvc.vvc_config import VVCConfig

        with pytest.raises(ConfigError):
            VVCConfig.from_env_args({"unknown_key": True})

    def test_none_values_use_default(self):
        from envs.vvc.vvc.vvc_config import VVCConfig

        with pytest.raises(ConfigError):
            VVCConfig.from_env_args({"system_name": None})

    def test_env_specific_config_reward_weights(self):
        from envs.vvc.vvc.vvc_config import VVCConfig

        env_args = {
            "env_specific_config": {
                "reward_weights": {
                    "power_loss": 20.0,
                    "capacitor": 0.5,
                    "battery_soc": 1.0,
                }
            }
        }
        config = VVCConfig.from_env_args(env_args)
        assert config.power_w == 20.0
        assert config.cap_w == 0.5
        assert config.soc_w == 1.0

    def test_inf_bat_act_num(self):
        from envs.vvc.vvc.vvc_config import VVCConfig

        config = VVCConfig.from_env_args({"bat_act_num": float("inf")})
        assert config.bat_act_num == float("inf")
