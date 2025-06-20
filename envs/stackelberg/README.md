# Stackelberg Game Environment

## Overview

This package implements a bi-level non-cooperative Stackelberg-Nash game framework for demand response in power distribution networks, based on the paper "Asynchronous multi-agent reinforcement learning-based framework for bi-level noncooperative game-theoretic demand response" (2024).

## Key Features

- **Hierarchical Game Structure**: UC (Utility Company) as leader, consumers as followers
- **Asynchronous Execution**: Temporal relationships between UC decisions and consumer responses
- **Advanced Reward Functions**: Based on paper equations for both UC and consumer utilities
- **ESS Integration**: Energy Storage System dynamics with state-of-charge tracking
- **DER Management**: Distributed Energy Resources with curtailment control
- **TUTT Pricing**: Time-of-Use Tiered Tariff implementation
- **Comprehensive Monitoring**: Nash gap tracking, convergence analysis, and visualization

## Architecture

```
envs/stackelberg/
├── __init__.py
├── README.md
└── stackelberg_game/
    ├── __init__.py
    ├── stackelberg_base_env.py    # Core environment implementation
    ├── async_wrapper.py           # Asynchronous multi-agent wrapper
    ├── stackelberg_monitor.py     # Monitoring and visualization
    └── circuit_adapter.py         # Interface with PowerZoo circuits
```

## Components

### 1. StackelbergBaseEnv

The core environment implementing the Stackelberg game dynamics:

- **UC Actions** (5D):
  - Price signal multiplier (0.5-2.0)
  - DR incentive (0-0.5)
  - Capacity allocation (0-1)
  - ESS charge/discharge (-1 to 1)
  - DER curtailment (0-1)

- **Consumer Actions** (2D):
  - Load adjustment (-30% to +10%)
  - DER output control (0-1)

### 2. AsyncMultiAgentWrapper

Manages the temporal relationship between UC and consumer actions:

- UC acts first (leader phase)
- Consumers respond based on UC signals (follower phase)
- Supports action persistence and delays

### 3. StackelbergMonitor

Comprehensive monitoring system tracking:

- Nash gap convergence
- Agent rewards and behaviors
- System stability metrics
- Carbon emissions
- Visualization and logging

### 4. CircuitAdapter

Interfaces with PowerZoo circuit simulation:

- Power flow analysis
- N-1 security checking
- Carbon emission calculations
- Bus and load management

## Usage

### Basic Example

```python
from envs.stackelberg import StackelbergBaseEnv, AsyncMultiAgentWrapper
import yaml

# Load configuration
with open('configs/envs_cfgs/stackelberg_13bus.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create base environment
base_env = StackelbergBaseEnv(config)

# Wrap with async execution
env = AsyncMultiAgentWrapper(base_env, config['async_config'])

# Reset environment
obs = env.reset()

# UC phase
uc_action = {0: np.array([1.0, 0.2, 0.5, 0.0, 0.1])}
obs, rewards, done, infos = env.step(uc_action)

# Consumer phase
consumer_actions = {
    1: np.array([-0.1, 0.5]),
    2: np.array([-0.2, 0.3]),
    # ... more consumers
}
obs, rewards, done, infos = env.step(consumer_actions)
```

### Training with SN-MAPPO

```python
from algorithms.actors.sn_mappo import SN_MAPPO

# Create UC agent
uc_agent = SN_MAPPO(uc_config, obs_space, act_space)

# Create consumer agents
consumer_agents = [
    SN_MAPPO(consumer_config, obs_space, act_space) 
    for _ in range(n_consumers)
]

# Training loop handles hierarchical updates
```

## Configuration

See `configs/envs_cfgs/stackelberg_13bus.yaml` for detailed configuration options:

- Agent mappings to buses
- Reward function weights
- Physical constraints
- Market parameters
- ESS specifications
- DER configurations
- Monitoring settings

## Mathematical Formulation

The environment implements the following key equations from the paper:

- **UC Utility**: J_u = C_t^s + C_t^m + C_t^g + C_t^r (Eq. 5)
- **Consumer Utility**: J_c,i = -(U_i,t^s + U_i,t^c - U_i,t^r) (Eq. 20)
- **ESS Dynamics**: E_t+1 = η_s·E_t + η_c·P_c·Δt - P_d·Δt/η_d (Eq. 19)
- **TUTT Pricing**: Combines time-of-use and tiered pricing structures

## Monitoring and Analysis

The monitoring system provides:

- Real-time Nash gap tracking
- Convergence analysis plots
- Agent behavior heatmaps
- System stability metrics
- Episode summaries and reports

Logs and visualizations are saved to `logs/stackelberg/`.

## Extensions

The environment supports:

- Multiple power system topologies (13Bus, 34Bus, 123Bus)
- N-1 security constraints
- Carbon emission tracking
- Curriculum learning
- Prioritized Experience Replay (PER)

## Citation

If you use this environment, please cite:

```
Nie, Y., Liu, J., Liu, X., Zhao, Y., Ren, K., & Chen, C. (2024). 
Asynchronous multi-agent reinforcement learning-based framework for 
bi-level noncooperative game-theoretic demand response. 
IEEE Transactions on Smart Grid.
```