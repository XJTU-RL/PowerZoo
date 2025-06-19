# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Environment Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate PowerZoo

# Install package in development mode
pip install -e .
```

### Training Commands
```bash
# Basic training with default settings (SHOM on PowerZoo)
python examples/train.py

# Custom training with specific algorithm and environment
python examples/train.py --algo <algorithm> --env <environment> --exp_name <experiment_name>

# Available algorithms: happo, hatrpo, haa2c, mappo, shom, haddpg, hatd3, hasac, maddpg, matd3, qmix, had3qn
# Available environments: powerzoo, dsr, smac, mamujoco, pettingzoo_mpe, football, dexhands, lag, mujoco, rware

# Using shell script
./examples/train.sh
```

### Code Quality
```bash
# Run pre-commit hooks
pre-commit run --all-files
```

## Architecture Overview

PowerZoo is a multi-agent reinforcement learning framework focused on power system applications, built on OpenDSS simulation.

### Core Components

**Runners** (`runners/`): Training orchestration for different RL paradigms
- `OnPolicyHARunner` - Heterogeneous agent on-policy training
- `OnPolicyMARunner` - Multi-agent on-policy training  
- `OffPolicyHARunner` - Heterogeneous agent off-policy training
- `QMIXRunner` - QMIX algorithm training

**Algorithms** (`algorithms/`): RL algorithm implementations
- On-policy: HAPPO, HATRPO, HAA2C, MAPPO, SHOM
- Off-policy: HADDPG, HATD3, HASAC, MADDPG, MATD3
- Value-based: QMIX, HAD3QN

**Environments** (`envs/`): Multi-agent simulation environments
- Primary: PowerZoo (power system VVC using OpenDSS)
- Power Systems: DSR (Distribution System Restoration using OpenDSS)
- Secondary: SMAC, MuJoCo, PettingZoo, Football, DexHands

**Models** (`models/`): Neural network architectures
- `policy_models/` - Actor networks for different algorithms
- `value_function_models/` - Critic networks (Q-critics, V-critics)
- `base/` - Shared network components

### PowerZoo Environment Architecture

Located in `envs/powerzoo/`, this is the core power system environment:
- Integrates with OpenDSS for circuit simulation
- Supports multiple power system topologies (13Bus, 34Bus, 123Bus, 8500Node)
- Implements Voltage-Var Control (VVC) optimization
- 24-step episodes representing 24-hour power system operation

### Configuration System

YAML-based configuration in `configs/`:
- `algos_cfgs/` - Algorithm hyperparameters and network architecture
- `envs_cfgs/` - Environment specifications and episode settings
- `tuned_configs/` - Pre-optimized parameters for different scenarios

### Entry Points

**Main Training**: `examples/train.py`
- Parses command line arguments or JSON config files
- Maps algorithm names to runner classes via `runners/__init__.py`
- Supports all algorithm-environment combinations

**Algorithm-Runner Mapping** (in `runners/__init__.py`):
- On-policy algorithms → `OnPolicyHARunner` or `OnPolicyMARunner`
- Off-policy algorithms → `OffPolicyHARunner`
- QMIX/HAD3QN → `QMIXRunner`

## Development Notes

### Missing Test Infrastructure
- No pytest configuration or test directory
- Use manual testing via `examples/train.py` with short episodes
- Pre-commit hooks provide basic code quality checks

### Dependencies
- Primary: PyTorch 1.7.1+, OpenDSS integration (`dss-python`, `opendssdirect-py`)
- Environment: Gym/Gymnasium, PettingZoo
- Optional: CUDA libraries for GPU acceleration

### Power System Specifics

**PowerZoo Environment:**
- Episodes are 24 steps (representing 24-hour operation cycles)
- Default environment is 13Bus power system
- Agents control capacitor banks and voltage regulators for VVC
- Observation space includes bus voltages, power flows, and control states

**DSR Environment:**
- Episodes are 15 steps (representing restoration time periods)
- Default environment is 123Bus power system with random faults
- 95 heterogeneous agents: 1 switch controller + 9 PV agents + 85 load agents
- Objective: restore power supply after distribution system faults
- Key constraints: voltage limits, line capacity, load priority