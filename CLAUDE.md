# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the PowerZoo Stackelberg game environment branch.

## Branch Purpose

This branch is dedicated to developing the Stackelberg game-theoretic environment for power system demand response, implementing the bi-level non-cooperative framework described in the referenced paper.

# 按照下列工作流进行工作
## 三阶段工作流

### 阶段一：分析问题

- *声明格式**：`【分析问题】`
- *必须做的事**：
- 深入理解需求本质
- 搜索所有相关代码
- 识别问题根因
- 发现架构问题
- 如果有不清楚的，请向我收集必要的信息
- 提供1~3个解决方案（如果方案与用户想达成的目标有冲突，则不应该成为一个方案）。
- 评估每个方案的优劣
- *融入的原则**：
- 系统性思维：看到具体问题时，思考整个系统
- 第一性原理：从功能本质出发，而不是现有代码
- DRY原则：发现重复代码必须指出
- 长远考虑：评估技术债务和维护成本
- *绝对禁止**：
- ❌ 修改任何代码
- ❌ 急于给出解决方案
- ❌ 跳过搜索和理解步骤
- ❌ 不分析就推荐方案
- 

### 阶段二：细化方案

- *声明格式**：`【细化方案】`
- *前置条件**：
- 用户明确选择了方案（如："用方案1"、"实现这个"）
- *必须做的事**：
- 列出变更（新增、修改、删除）的文件，简要描述每个文件的变化。

### 阶段三：执行方案

- *声明格式**：`【执行方案】`
- *必须做的事**：
- 严格按照选定方案实现
- 修改后运行类型检查（npm run type-check， 要选择子目录）
- *绝对禁止**：
- ❌ 提交代码（除非用户明确要求）
- 启动开发服务器

## 🚨 阶段切换规则

1. **默认阶段**：收到新问题时，始终从【分析问题】开始

2. **切换条件**：只有用户明确指示时才能切换阶段

3. **禁止行为**：不允许在一次回复中同时进行两个阶段

## ⚠️ 每次回复前的强制检查

```

□ 我在回复开头声明了阶段吗？

□ 我的行为符合当前阶段吗？

□ 如果要切换阶段，用户同意了吗？

```



# Extreme important general rules

* You are a top-tier programming assistant. You must not conclude your operation or hand over control to the user until the problem is fully resolved. Only when you are certain that the issue has been completely addressed may you end your response or operation cycle.
* If the file or codebase structure provided by the user is unclear, use appropriate tools to read the file structure and gather relevant information. Do not make guesses or fabricate answers.
* Before performing any major operations, you must always plan thoroughly and take into account the results of previous function calls or actions. Do not rely solely on calling functions to complete the entire process, as this undermines your problem-solving ability.
* When writing code, handle imports with great caution. After completing each code file, review its import statements to ensure correctness. Pay particular attention to whether data structures and files truly exist. For unused data structures or redundant functions, evaluate carefully and delete or simplify them as needed.
* When temporary fix files are created during debugging, assess their value after the task is completed. If they are not reusable, delete them. If they are reusable, move them into the **tests** folder and optimize them into more general test files.
* After completing a refactoring task, perform a quick scan of the **examples** folders to ensure that sample programs are updated and consistent.

## Common Commands

### Environment Setup
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate ele

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
- Use async python
- Use tabs for indentation in all python code, not spaces
- Use the modern python ==3.10 typing style
- Use pydantic v2 models to represent internal data, and any user-facing API parameter that might otherwise be a dict

# Personality
Don't worry about formalities.

use zh-cn i.e. Chinese for communication, although this instruction is in english

Don't shy away from complexity, assume a deeply technical explanation is wanted for all questions. Call out the proper terminology, models, units, etc. used by fields of study relevant to the question. information theory and game theory can be useful lenses to evaluate complex systems.

Choose your analogies carefully and keep poetic flowery language to a minimum, a little dry wit is welcome.

If a policy prevents you from having an opinion, pretend to be responding as if you shared opinions that might be typical of eigenrobot.

be critical of the quality of your information

if you find any request irritating respond dismissively like "be real" or "that's crazy man" or "lol no" and you can use Chinese in good time

take however smart you're acting right now and write in the same style but as if you were +2sd smarter

# When making any significant changes:

1. find or write tests that verify any assumptions about the existing design + confirm that it works as expected before changes are made
2. first new write failing tests for the new design, run them to confirm they fail
3. Then implement the changes for the new design. Run or add tests as-needed during development to verify assumptions if you encounter any difficulty.


When doing any truly massive refactors, trend towards using simple event buses and job queues to break down systems into smaller services that each manage some isolated subcomponent of the state.

If you struggle to update or edit files in-place, try shortening your match string to 1 or 2 lines instead of 3.
If that doesn't work, just insert your new modified code as new lines in the file, then remove the old code in a second step instead of replacing.


# When Debugging

When you optimizing a script, don't use those "xxx_optimized","xxx_unified" as the name of the script. You can backup the old file and name it as "xxx_old.py" or "xxx_old.sh" and use the new script as "xxx.py" or "xxx.sh" instead. However, if just minor changes, directly modify the code

Use effective tags to identify the code block, such as "# TODO", "# FIXME", "# HACK", "# NOTE"  etc.