# Stackelberg Environment Structure

## Directory Layout

```
envs/stackelberg/
├── __init__.py                    # Package initialization
├── README.md                      # Documentation
├── STRUCTURE.md                   # This file
└── stackelberg_game/
    ├── __init__.py               # Core module initialization
    ├── stackelberg_base_env.py   # Main environment implementation
    ├── async_wrapper.py          # Asynchronous execution wrapper
    ├── stackelberg_monitor.py    # Monitoring and visualization
    ├── circuit_adapter.py        # Interface with PowerZoo circuits
    └── env_factory.py            # Factory functions for environment creation
```

## Key Components

### 1. **stackelberg_base_env.py**
- Core Stackelberg game environment
- Implements UC-consumer hierarchical game
- Reward functions based on paper equations
- ESS and DER dynamics
- TUTT pricing implementation

### 2. **async_wrapper.py**
- Manages temporal relationships between agents
- UC acts first (leader phase)
- Consumers respond (follower phase)
- Supports action delays and persistence

### 3. **stackelberg_monitor.py**
- Comprehensive monitoring system
- Nash gap tracking
- Convergence analysis
- Real-time visualization
- Episode logging and reporting

### 4. **circuit_adapter.py**
- Interfaces with PowerZoo circuit module
- Handles power flow calculations
- N-1 security checking
- Carbon emission tracking
- Bus and load management

### 5. **env_factory.py**
- Convenient factory functions
- Configuration loading
- Parallel environment creation
- Default parameter management

## Usage Examples

### Basic Usage
```python
from envs.stackelberg.stackelberg_game import StackelbergBaseEnv, AsyncMultiAgentWrapper

# Create environment
env = StackelbergBaseEnv(config)
wrapped_env = AsyncMultiAgentWrapper(env)

# Or use factory
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