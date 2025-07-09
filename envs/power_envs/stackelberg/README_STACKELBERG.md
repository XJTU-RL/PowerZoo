# Stackelberg Game-Based Demand Response Environment

This document describes the implementation of a bi-level non-cooperative Stackelberg-Nash game framework for demand response in PowerZoo, based on the paper "Asynchronous multi-agent reinforcement learning-based framework for bi-level noncooperative game-theoretic demand response".

## Overview

The Stackelberg environment models the interaction between a Utility Company (UC) as the leader and multiple consumers as followers in a hierarchical decision-making framework. The UC sets price signals and DR incentives, while consumers respond by adjusting their loads, DER output, and storage usage.

### Key Features

1. **Hierarchical Decision Making**: UC acts first, consumers respond based on UC signals
2. **Asynchronous Execution**: Temporal relationships between agents are explicitly modeled
3. **Multi-System Support**: Supports 13Bus, 34Bus, and 123Bus power systems
4. **Intelligent Load Aggregation**: Maps physical loads to consumer agents using zone/priority/graph methods
5. **Comprehensive Monitoring**: Real-time tracking of system metrics, convergence, and Nash equilibrium
6. **SN-MAPPO Algorithm**: Specialized algorithm for Stackelberg-Nash games

## Architecture

### Core Components

#### 1. **StackelbergBaseEnv** (`envs/powerzoo/powerzoo/stackelberg_base_env.py`)
- Base environment implementing the Stackelberg game mechanics
- Does NOT inherit from existing Env class for maximum flexibility
- Manages UC-consumer interactions and system state

#### 2. **AsyncMultiAgentWrapper** (`envs/powerzoo/powerzoo/async_wrapper.py`)
- Handles asynchronous execution of UC and consumer actions
- Maintains action history and temporal relationships
- Supports configurable delays and partial observability

#### 3. **IntelligentLoadAggregator** (`envs/powerzoo/powerzoo/load_aggregator.py`)
- Maps physical loads to consumer agents
- Supports multiple aggregation methods:
  - **Zone-based**: Groups loads by electrical proximity
  - **Priority-based**: Groups by load criticality
  - **Graph-based**: Uses network topology for clustering
  - **Adaptive**: Dynamic re-aggregation based on load patterns

#### 4. **StackelbergMonitor** (`envs/powerzoo/powerzoo/stackelberg_monitor.py`)
- Comprehensive monitoring and visualization system
- Tracks convergence, Nash equilibrium, and system metrics
- Supports TensorBoard integration
- Generates plots and analysis reports

#### 5. **SN-MAPPO Algorithm** (`algorithms/actors/sn_mappo.py`)
- Extension of MAPPO for Stackelberg-Nash games
- Separate learning rates for UC and consumers
- Best response tracking and opponent modeling
- Nash equilibrium seeking

## Configuration

### Environment Configurations

Three system-specific configurations are provided:

1. **13Bus System** (`configs/envs_cfgs/stackelberg_13bus.yaml`)
   - 5 consumer agents
   - Zone-based aggregation
   - Suitable for testing and development

2. **34Bus System** (`configs/envs_cfgs/stackelberg_34bus.yaml`)
   - 10 consumer agents
   - Priority-based aggregation
   - Medium-scale experiments

3. **123Bus System** (`configs/envs_cfgs/stackelberg_123bus.yaml`)
   - 20 consumer agents
   - Graph-based aggregation
   - Large-scale experiments with zone coordination

### Algorithm Configuration

The SN-MAPPO algorithm configuration (`configs/algos_cfgs/sn_mappo.yaml`) includes:
- Hierarchical learning parameters
- Nash equilibrium solver settings
- Opponent modeling configuration
- Curriculum learning stages

## Usage

### Basic Training

```bash
# Train on 13Bus system
python examples/train_stackelberg.py --env stackelberg_13bus --algo sn_mappo

# Train on 34Bus system with CUDA
python examples/train_stackelberg.py --env stackelberg_34bus --algo sn_mappo --cuda

# Train on 123Bus system with custom settings
python examples/train_stackelberg.py --env stackelberg_123bus --algo sn_mappo \
    --num_env_steps 50000000 --n_rollout_threads 4
```

### Using the Original train.py

The environment is also compatible with the original PowerZoo training script:

```bash
# Using the general training script
python examples/train.py --algo sn_mappo --env stackelberg_13bus --exp_name my_experiment
```

### Custom Configuration

Create a custom config file combining environment and algorithm settings:

```yaml
# custom_config.yaml
env: stackelberg_34bus
algo: sn_mappo
seed: 42
num_env_steps: 20000000

# Override specific parameters
n_consumer_agents_34bus: 15
load_aggregation:
  method: adaptive
  adaptation_threshold: 0.05

# Custom reward weights
reward_weights:
  power_loss_weight: 20.0
  carbon_penalty_weight: 5.0
```

Then train with:
```bash
python examples/train_stackelberg.py --config custom_config.yaml
```

## Monitoring and Analysis

### Real-time Monitoring

The environment provides comprehensive monitoring during training:

1. **Episode Metrics**: UC/consumer rewards, social welfare, power loss
2. **Convergence Tracking**: Nash gap, price stability, action variance
3. **System Metrics**: Voltage violations, line loading, carbon emissions
4. **Agent Behavior**: Price signals, load adjustments, coordination

### Visualization

Monitor generates various plots:
- Episode reward progression
- System performance metrics
- Nash equilibrium convergence
- Agent behavior heatmaps

Plots are saved to `logs/stackelberg_<system>/plots/`

### TensorBoard Integration

If TensorboardX is installed, metrics are logged for real-time visualization:

```bash
tensorboard --logdir logs/stackelberg_13bus/tensorboard
```

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
pytest tests/test_stackelberg_env.py -v

# Run specific test
pytest tests/test_stackelberg_env.py::TestStackelbergBaseEnv::test_env_creation -v

# Quick integration test
python tests/test_stackelberg_env.py
```

## Implementation Details

### Action Spaces

**UC Actions** (continuous):
- Price signal multiplier: [0.5, 2.0]
- DR incentive: [0.0, 0.5]
- Capacity allocation: [0.0, 1.0]

**Consumer Actions** (continuous):
- Load adjustment: [-1.0, 1.0] (fraction of base load)
- DER output: [0.0, 1.0] (fraction of capacity)
- Storage action: [-1.0, 1.0] (charge/discharge)

### Observation Spaces

**UC Observations**:
- System-wide metrics (load, generation, losses)
- Bus voltages and violations
- Time encoding
- Carbon intensity

**Consumer Observations**:
- UC signals (price, DR incentive)
- Local voltage and load
- Historical actions
- Time encoding

### Reward Structure

**UC Rewards**:
- Minimize power losses
- Minimize voltage violations
- Maximize revenue
- Minimize carbon emissions

**Consumer Rewards**:
- Minimize electricity cost
- Maintain comfort (minimize load changes)
- DR participation incentives
- Service reliability bonus

## Advanced Features

### Multi-Timescale Coordination (Planned)

The framework supports day-ahead, intraday, and real-time coordination:
- Day-ahead: 24-hour planning
- Intraday: 4-hour rolling updates
- Real-time: 5-minute dispatch

### N-1 Security Constraints (Planned)

Future enhancement to check system security under contingencies:
- Automatic critical line identification
- Security-constrained optimization
- Emergency rating considerations

### Zone Coordination

For large systems (123Bus), hierarchical coordination:
- Zone coordinators aggregate consumer responses
- Reduces communication overhead
- Improves scalability

## Performance Optimization

### Computational Efficiency

- Sparse matrix operations for large systems
- Parallel power flow calculations
- Cached network data
- Batch observation processing

### Memory Management

- Circular buffers for action history
- Periodic garbage collection
- Compressed data logging
- Selective metric tracking

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PowerZoo is properly installed and PYTHONPATH includes the project root
2. **OpenDSS Errors**: Verify DSS files exist in `envs/powerzoo/systems/<system>/`
3. **Memory Issues**: Reduce buffer sizes or disable some monitoring metrics for large systems
4. **Convergence Issues**: Adjust learning rates or use curriculum learning

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this environment in your research, please cite:

```bibtex
@article{nie2024asynchronous,
  title={Asynchronous multi-agent reinforcement learning-based framework for bi-level noncooperative game-theoretic demand response},
  author={Nie, Yongxin and Liu, Jun and others},
  journal={Journal Name},
  year={2024}
}
```

## Future Work

Planned enhancements include:
- Multi-timescale coordinator implementation
- N-1 security constraint checking
- Carbon emission tracking by generation source
- Distributed training support
- Real-world data integration

## Contributors

- Implementation based on the paper by Yongxin Nie, Jun Liu, et al.
- Integrated into PowerZoo framework by the development team

For questions or issues, please open an issue on the PowerZoo GitHub repository.