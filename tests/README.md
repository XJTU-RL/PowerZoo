# PowerZoo Test Suite

Comprehensive test suite for PowerZoo and PowerZoo_LLM environments.

## Installation

Install test dependencies:

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (includes pytest)
pip install -e ".[dev]"
```

## Running Tests

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest -m unit

# Integration tests
pytest -m integration

# PowerZoo environment tests
pytest -m vvc

# PowerZoo_LLM environment tests (SmartGrid)
pytest -m smartgrid

# Skip slow tests
pytest -m "not slow"

# Skip tests requiring OpenDSS
pytest -m "not requires_opendss"
```

### Run Specific Test Files

```bash
# PowerZoo (VVC) physics tests
pytest tests/test_vvc_physics.py

# PowerZoo_LLM environment tests (SmartGrid)
pytest tests/envs/smartgrid/

# LFC (OHASAC) physics tests — MATLAB-free layers only; the co-simulation
# layer runs via `python scripts/check_lfc.py --matlab`
pytest tests/test_lfc_physics.py

# Specific test function
pytest tests/test_lfc_physics.py::test_ohasac_algorithm_registry
```

### Code Coverage

```bash
# Run tests with coverage
pytest --cov=envs --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Parallel Execution

```bash
# Run tests in parallel (requires pytest-xdist)
pytest -n auto
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures, configuration, marker registration
├── envs/
│   ├── vvc/                       # VVC environment registration tests
│   └── smartgrid/                 # SmartGrid env config / rewards / logging / single-agent
├── test_algorithms.py             # Algorithm registry and actor contracts
├── test_common.py / test_models.py / test_runners.py / test_utils.py
├── test_config_pipeline.py / test_unified_config.py / test_configuration_*.py
│                                  # Config parsing, provenance, contract and behavioral effects
├── test_*_physics.py              # Physics contracts per task family:
│                                  #   vvc / smartgrid / dsr / district / pricing / lfc / powergrid
└── test_training_entrypoints.py / test_windows_runtime.py / test_hf_shared_assets.py
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit`: Fast unit tests, no external dependencies
- `@pytest.mark.integration`: Integration tests, may require system setup
- `@pytest.mark.slow`: Slow tests, full environment simulation
- `@pytest.mark.vvc`: Tests specific to PowerZoo environment
- `@pytest.mark.smartgrid`: Tests specific to PowerZoo_LLM environment
- `@pytest.mark.requires_opendss`: Tests requiring OpenDSS installation
- `@pytest.mark.requires_gpu`: Tests requiring GPU/CUDA

## Writing New Tests

### Test File Naming

- Test files must start with `test_` or end with `_test.py`
- Place tests in the appropriate directory under `tests/`

### Example Test

```python
import pytest

@pytest.mark.unit
@pytest.mark.smartgrid
def test_example():
    """Test description."""
    from envs.smartgrid import VVCEnv
    assert VVCEnv is not None
```

### Using Fixtures

Common fixtures are defined in `tests/conftest.py`:

```python
def test_with_config(smartgrid_config):
    """Use predefined configuration fixture."""
    assert smartgrid_config["num_agents"] > 0
```

## Continuous Integration

Tests are automatically run on CI/CD pipeline (if configured).

## Troubleshooting

### OpenDSS Not Found

If you see errors related to OpenDSS:

```bash
# Install OpenDSS Python bindings
pip install dss-python==0.15.7
```

Or skip tests requiring OpenDSS:

```bash
pytest -m "not requires_opendss"
```

### Missing Test Data

If tests fail due to missing IEEE test systems:

1. Ensure `node_systems/` directory exists in project root
2. Download IEEE test systems if necessary
3. Check paths in `conftest.py` fixtures

### Import Errors

If you encounter import errors:

```bash
# Install PowerZoo in development mode
pip install -e .
```

## Coverage Goals

- **Unit tests**: 80%+ coverage
- **Integration tests**: Cover critical paths
- **Overall**: 70%+ coverage target

## Contact

For test-related issues, please open an issue on GitHub.
