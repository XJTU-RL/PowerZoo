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
