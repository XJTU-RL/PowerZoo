# -*- coding: utf-8 -*-
"""
PyTest configuration and fixtures for PowerZoo tests.

This file ensures the project root is in the Python path.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: mark test as slow")
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "stackelberg: mark test for Stackelberg environment")
    config.addinivalue_line("markers", "opendss: mark test requiring OpenDSS")
