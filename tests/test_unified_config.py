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
