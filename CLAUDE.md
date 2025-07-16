# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# Code Style
- Use async python
- Use tabs for indentation in all python code, not spaces
- Use the modern python ==3.10 typing style
- Use pydantic v2 models to represent internal data, and any user-facing API parameter that might otherwise be a dict

# Personality
Don't worry about formalities.

use zh-cn i.e. Chinese for communication, although this instruction is in english


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

