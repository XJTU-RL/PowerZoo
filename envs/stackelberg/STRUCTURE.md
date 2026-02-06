# Stackelberg Environment Structure

## Directory Layout

```
envs/stackelberg/
├── __init__.py                           # Package initialization
├── README.md                             # Main user-facing documentation
├── README_STACKELBERG.md                 # Detailed technical documentation
├── OPTIMIZATION_SUMMARY.md               # Summary of implementation optimizations
├── STRUCTURE.md                          # This file, explaining the structure
├── stackelberg_vvc_env.py           # Main environment entry point, compatible with PowerZoo
├── test_stackelberg_integration.py       # Integration tests for the environment
└── stackelberg_game/                     # Core game logic module
    ├── __init__.py                       # Core module initialization
    ├── stackelberg_base_env.py           # Base environment with core mechanics
    ├── async_wrapper.py                  # Asynchronous execution wrapper
    ├── stackelberg_monitor.py            # Monitoring and visualization system
    ├── circuit_adapter.py                # Interface with PowerZoo circuits
    ├── load_aggregator.py                # Intelligent load aggregation logic
    └── env_factory.py                    # Factory functions for environment creation
```

## Key Components

### 1. **stackelberg_vvc_env.py**
- Main entry point for the Stackelberg environment.
- Ensures compatibility with the broader PowerZoo framework.
- Handles environment creation and registration.

### 2. **stackelberg_base_env.py**
- Implements the core logic of the Stackelberg game.
- Manages agent interactions, state transitions, and reward calculations.
- Contains the mathematical formulations from the reference paper.

### 3. **async_wrapper.py**
- Wraps the base environment to manage the asynchronous, bi-level nature of the game.
- Ensures the leader (UC) acts before the followers (consumers).

### 4. **stackelberg_monitor.py**
- A comprehensive system for tracking and logging metrics.
- Monitors Nash gap convergence, agent rewards, and system stability.
- Provides visualization capabilities.

### 5. **circuit_adapter.py**
- Acts as a bridge to the underlying VVC circuit simulation.
- Translates high-level actions into low-level OpenDSS commands.
- Fetches system state data from the simulation.

### 6. **load_aggregator.py**
- Implements various strategies to map physical loads to consumer agents.
- Supports zone-based, priority-based, and graph-based aggregation.

### 7. **env_factory.py**
- Provides convenient factory functions for creating instances of the Stackelberg environment.
- Manages configuration loading and simplifies environment setup.

## Usage Examples

### Basic Usage
```python
# Recommended way using the main entry point
from envs.stackelberg import StackelbergVVCEnv
env = StackelbergVVCEnv(system_name='13Bus')

# Or using the factory for more control
from envs.stackelberg.stackelberg_game.env_factory import make_stackelberg_env
env = make_stackelberg_env('stackelberg_13bus')
```

### With Monitoring
```python
config = {
    'monitoring_config': {
        'enable': True,
        'log_dir': 'logs/my_experiment'
    }
}
env = make_stackelberg_env('stackelberg_13bus', config=config)
```

### Training Integration
```python
# In training script
from runners.on_policy_ha_runner import OnPolicyHARunner

# Runner will handle SN-MAPPO algorithm
runner = OnPolicyHARunner(args)
runner.run()  # Uses 'sn_mappo' algorithm
```

## Configuration

Configurations are stored in `configs/envs_cfgs/`:
- `stackelberg_13bus.yaml`
- `stackelberg_34bus.yaml` (to be created)
- `stackelberg_123bus.yaml` (to be created)

## Integration Points

1. **Algorithm**: `algorithms/actors/sn_mappo.py`
2. **Runner**: `runners/on_policy_ha_runner.py`
3. **Config**: `configs/algos_cfgs/sn_mappo.yaml`
4. **Registration**: Added to `env_register.py` for compatibility

## Testing

Test script: `test_stackelberg_integration.py`
Example script: `examples/stackelberg_example.py`

## Key Features

1. **Hierarchical Game Structure**
   - UC as Stackelberg leader
   - Consumers as Nash followers
   - Asynchronous action execution

2. **Advanced Dynamics**
   - ESS state-of-charge tracking
   - DER curtailment control
   - Time-of-use tiered pricing
   - Carbon emission monitoring

3. **Comprehensive Monitoring**
   - Nash gap convergence
   - System stability metrics
   - Agent behavior analysis
   - Episode summaries

4. **Flexibility**
   - Supports multiple bus systems
   - Configurable agent mappings
   - Extensible reward functions
   - Modular design

## Future Extensions

1. Add support for:
   - Multi-timescale coordination
   - Advanced N-1 security constraints
   - Distributed training
   - Real-time visualization

2. Implement additional features:
   - Action masking for constraints
   - Curriculum learning schedules
   - Advanced opponent modeling
   - Communication protocols