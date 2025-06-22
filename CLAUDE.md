# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the PowerZoo Stackelberg game environment branch.

## Branch Purpose

This branch is dedicated to developing the Stackelberg game-theoretic environment for power system demand response, implementing the bi-level non-cooperative framework described in the referenced paper.

## Common Commands

### Environment Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate PowerZoo

# Install package in development mode
pip install -e .
```

### Training Commands for Stackelberg Environment
```bash
# Basic training with SN-MAPPO on Stackelberg environment
python examples/stackelberg_example.py

# Custom training with specific configurations
python examples/train.py --algo sn_mappo --env stackelberg_13bus --exp_name stackelberg_experiment

# Available Stackelberg environments: stackelberg_13bus, stackelberg_34bus, stackelberg_123bus
```

### Testing
```bash
# Run Stackelberg integration tests
python envs/stackelberg/test_stackelberg_integration.py
```

## Stackelberg Environment Architecture

### Core Components

**Environment Structure** (`envs/stackelberg/`):
- `stackelberg_powerzoo_env.py` - Main environment wrapper integrating with PowerZoo
- `stackelberg_game/` - Core game-theoretic implementation
  - `stackelberg_base_env.py` - Base class for Stackelberg game environment
  - `circuit_adapter.py` - Adapter for OpenDSS circuit simulation
  - `load_aggregator.py` - Load aggregation for consumer agents
  - `env_factory.py` - Factory for creating environment instances
  - `async_wrapper.py` - Asynchronous execution support
  - `stackelberg_monitor.py` - Monitoring and logging utilities

### Game Structure

**Hierarchical Agents**:
- **Leader (UC)**: Utility Company that sets electricity prices and DR incentives
- **Followers (Consumers)**: Multiple consumers responding to UC's pricing signals

**Key Features**:
- Bi-level optimization framework
- Non-cooperative game dynamics
- Time-varying electricity pricing (TUTT)
- Demand response mechanisms
- Distributed energy resources (DER) integration

### Observation and Action Spaces

**UC (Leader) Observations**:
- System load profile
- DER generation forecast
- Consumer response history
- Grid operational constraints

**UC (Leader) Actions**:
- Time-of-use pricing signals
- DR incentive levels
- DER curtailment decisions

**Consumer (Follower) Observations**:
- UC price signals
- Own load requirements
- Comfort preferences
- ESS state of charge

**Consumer (Follower) Actions**:
- Load shifting decisions
- ESS charging/discharging
- DR participation level

## SN-MAPPO Algorithm

The Stackelberg-Nash Multi-Agent PPO (`algorithms/actors/sn_mappo.py`) implements:
- Asynchronous policy updates for leader and followers
- Total derivative computation for bi-level optimization (Equation 39)
- Nash equilibrium seeking
- Best response tracking
- Opponent modeling capabilities

### Key Algorithm Parameters
- `is_leader`: Boolean flag for UC agent
- `hierarchy_level`: 0 for UC, 1 for consumers
- `leader_lr_scale`: Learning rate scaling for UC
- `follower_lr_scale`: Learning rate scaling for consumers
- `nash_iterations`: Iterations for equilibrium computation
- `equilibrium_threshold`: Convergence threshold

## Configuration System

### Environment Configs (`configs/envs_cfgs/`):
- `stackelberg_13bus.yaml` - 13-bus system configuration
- `stackelberg_34bus.yaml` - 34-bus system configuration
- `stackelberg_123bus.yaml` - 123-bus system configuration

### Key Configuration Parameters:
- `num_uc_agents`: Number of UC agents (typically 1)
- `num_consumer_agents`: Number of consumer agents
- `episode_length`: 24 (representing 24-hour operation)
- `price_update_frequency`: How often UC updates prices
- `dr_response_delay`: Consumer response delay to price changes

## Development Guidelines

### When Modifying the Environment:
1. Maintain separation between UC and consumer logic
2. Ensure time-consistency in the bi-level game
3. Validate equilibrium convergence metrics
4. Test with different numbers of consumer agents

### When Working with SN-MAPPO:
1. Verify leader-follower update ordering
2. Monitor policy convergence for both levels
3. Check gradient flow in bi-level optimization
4. Validate Nash equilibrium conditions

## Testing Requirements

Before committing changes:
1. Run integration tests: `python envs/stackelberg/test_stackelberg_integration.py`
2. Verify environment reset and step functions
3. Check observation and action space consistency
4. Test with both single and multiple consumer configurations
5. Validate SN-MAPPO convergence on simple scenarios

## Paper Reference

Implementation based on:
"Asynchronous multi-agent reinforcement learning-based framework for bi-level noncooperative game-theoretic demand response" (2024)

Key equations implemented:
- UC utility function (Eq. 12-14)
- Consumer utility function (Eq. 24-25)
- Total derivative for bi-level optimization (Eq. 39)
- KL divergence constraints (Eq. 57-58)

# Code Style
- Use async python where appropriate for game dynamics
- Use tabs for indentation in all python code
- Use modern python >=3.10 typing
- Use pydantic v2 for configuration validation

# 代码可修改范围

开发一个基于Stackelberg博弈理论的PowerZoo模拟环境及相关算法。具体要求如下：

1. 环境开发：
   - 主环境文件: @stackelberg_powerzoo_env.py
   - 基础环境文件： @stackelberg_base_env.py
   - 保持这两个核心环境文件唯一性 

2. 代码修改范围：仅限envs/stackelberg/目录及其子目录下的文件
   - 其他目录文件保持原状

3. 开发目标：
   - 实现Stackelberg博弈环境的核心功能
   - 开发配套算法支持环境运行
   - 确保环境具备可扩展性和稳定性

我需要你修改当前环境以适配sn_mappo算法和同目录下的其他算法。具体要求如下：
1. 保持sn_mappo算法的现有功能不变，该文件位于algorithms/actors/sn_mappo.py
2. 不允许对同级算法进行修改
3. 修改范围仅限于环境适配部分，不得改动算法本身的实现细节
4. 所有修改必须经过严格测试验证，确保不会引入新的兼容性问题
5. 修改后的环境应能同时支持sn_mappo.py和其他同级算法的正常运行

* 参数文件管理要求：
  - 仅修改configs/envs_cfgs文件夹下的以下三个YAML文件：
    - stackelberg_13bus.yaml
    - stackelberg_34bus.yaml  
    - stackelberg_123bus.yaml
  - 允许对上述文件进行参数优化调整
  - 新增参数文件必须存放在同一目录下，命名需遵循"stackelberg_*bus.yaml"格式

* 示例文件管理规范：
  - 保持examples文件夹下仅保留一个示例文件
  - 示例文件功能需满足：
    - 展示stackelberg环境基本用法
    - 代码行数不超过300行
    - 不包含非必要功能模块

* 代码修改限制：
  - 严格禁止修改/envs目录下非stackelberg环境的任何代码文件
  - 如需使用其他环境代码，必须：
    1. 完整复制到/envs/stackelberg目录下
    2. 或通过import方式引用

* 节点系统文件操作规范：
  - 允许使用/node_systems/下所有标准节点文件
  - 新增.dss文件要求：
    - 文件名需明确描述其用途
    - 必须附带同名markdown说明文档
    - 说明文档需包含：
      - 文件用途
      - 修改记录
      - 兼容性说明
  - 严禁删除或修改现有任何.dss文件