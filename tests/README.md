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
pytest -m powerzoo

# PowerZoo_LLM environment tests
pytest -m smartgrid

# Skip slow tests
pytest -m "not slow"

# Skip tests requiring OpenDSS
pytest -m "not requires_opendss"
```

### Run Specific Test Files

```bash
# PowerZoo environment tests
pytest tests/envs/vvc/

# PowerZoo_LLM environment tests
pytest tests/envs/smartgrid/

# Specific test file
pytest tests/envs/vvc/test_vvc_env.py

# Specific test function
pytest tests/envs/vvc/test_vvc_env.py::TestVVCEnvBasics::test_import_powerzoo
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
├── conftest.py                    # Shared fixtures and configuration
├── pytest.ini                     # Pytest configuration (in project root)
├── envs/
│   ├── powerzoo/
│   │   └── test_vvc_env.py   # PowerZoo environment tests
│   └── smartgrid/
│       └── test_smartgrid_env.py  # PowerZoo_LLM environment tests
└── README.md                      # This file
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.unit`: Fast unit tests, no external dependencies
- `@pytest.mark.integration`: Integration tests, may require system setup
- `@pytest.mark.slow`: Slow tests, full environment simulation
- `@pytest.mark.powerzoo`: Tests specific to PowerZoo environment
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
