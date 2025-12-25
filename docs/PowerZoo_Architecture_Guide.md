# PowerZoo 项目架构说明

## 📋 项目概述

**PowerZoo** 是一个基于 Python 和 OpenDSS 的电力系统强化学习仿真平台，专注于配电网的智能控制与优化。

### 核心定位
- **领域**：电力系统 + 强化学习
- **目标**：为电力系统控制问题提供标准化的多智能体强化学习（MARL）实验平台
- **特色**：支持混合动作空间、异构智能体、多种控制设备的协同优化

### 技术栈
- **仿真引擎**：OpenDSS（电力系统潮流计算）
- **RL框架**：自研MARL算法库 + Stable-Baselines3（单智能体）
- **深度学习**：PyTorch
- **环境接口**：Gymnasium/Gym API

---

## 🏗️ 整体架构设计

### 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│                     Training Layer (训练层)                   │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐ │
│  │  Examples     │  │   Runners     │  │  Tensorboard    │ │
│  │  (启动脚本)   │  │  (训练引擎)    │  │   (监控可视化)   │ │
│  └───────────────┘  └───────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Algorithm Layer (算法层)                    │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  14种MARL算法                                         │   │
│  │  • On-Policy: HAPPO, MAPPO, HAA2C, HATRPO, SHOM     │   │
│  │  • Off-Policy: HADDPG, HASAC, HATD3, MADDPG, MATD3  │   │
│  │  • Value-Based: QMIX, HAD3QN                        │   │
│  │  • Special: 2TS-VVC (Two-Timescale VVC)             │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Layer (模型层)                       │
│  ┌──────────────────┐  ┌──────────────────────────────┐    │
│  │  Policy Models   │  │  Value Function Models       │    │
│  │  • Actor网络     │  │  • Critic网络                │    │
│  │  • 混合动作处理   │  │  • 中心化价值函数             │    │
│  └──────────────────┘  └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                Environment Layer (环境层)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │  PowerZoo    │  │ PowerZoo_LLM │  │  Stackelberg    │  │
│  │  (基础环境)   │  │  (完整环境)   │  │   (博弈环境)     │  │
│  └──────────────┘  └──────────────┘  └─────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DSR (Demand Side Response - 需求侧响应环境)          │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                 Simulation Layer (仿真层)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  OpenDSS电力系统仿真引擎                              │   │
│  │  • 潮流计算                                          │   │
│  │  • 设备建模 (CRBP: Cap/Reg/Bat/PV)                   │   │
│  │  • 网络拓扑 (13/34/123/8500节点系统)                 │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 核心模块详解

### 1. 环境模块 (`envs/`)

#### 1.1 PowerZoo 基础环境
- **路径**：`envs/powerzoo/`
- **特点**：轻量级、快速迭代
- **适用**：算法原型验证、快速实验

#### 1.2 PowerZoo_LLM 完整环境
- **路径**：`envs/powerzoo_llm/`
- **特点**：完整功能、详细监控
- **核心文件**：
  - `base_env.py`：多智能体环境基类
  - `single_agent.py`：单智能体包装器（Gymnasium API）
  - `action_space.py`：混合动作空间管理
  - `observation_space.py`：观测空间构建
  - `reward_calculator.py`：奖励函数计算
  - `dss_engine.py`：OpenDSS仿真引擎封装

#### 1.3 Stackelberg 博弈环境
- **路径**：`envs/stackelberg/`
- **特点**：多层博弈、主从结构
- **应用**：电力市场、需求响应博弈

#### 1.4 DSR 需求侧响应环境
- **路径**：`envs/dsr/`
- **特点**：需求侧管理、负荷调度

#### CRBP 设备架构
```
┌──────────────────────────────────────────────────┐
│  CRBP 控制设备体系                                │
│                                                  │
│  C - Capacitor (电容器)    → 无功补偿            │
│  R - Regulator (调压器)    → 电压调节            │
│  B - Battery (电池)        → 有功功率/储能       │
│  P - Photovoltaic (光伏)   → 分布式发电          │
└──────────────────────────────────────────────────┘
```

### 2. 算法模块 (`algorithms/`)

#### 2.1 Actor-Critic 架构
- **Actor**：`algorithms/actors/`
  - 策略网络实现
  - 支持离散、连续、混合动作空间
  - 异构智能体actor

- **Critic**：`algorithms/critics/`
  - 价值函数网络
  - 中心化训练（CTDE）支持
  - 状态-动作价值评估

#### 2.2 算法分类

**On-Policy 算法**
- `HAPPO` (Heterogeneous-Agent PPO)：异构智能体近端策略优化
- `MAPPO` (Multi-Agent PPO)：多智能体PPO
- `SHOM` (Sensitivity-based Heterogeneous Ordered MARL)：敏感性异构有序MARL
- `HAA2C` (Heterogeneous-Agent A2C)
- `HATRPO` (Heterogeneous-Agent TRPO)

**Off-Policy 算法**
- `HADDPG` (Heterogeneous-Agent DDPG)
- `HASAC` (Heterogeneous-Agent SAC)
- `HATD3` (Heterogeneous-Agent TD3)
- `MADDPG` (Multi-Agent DDPG)
- `MATD3` (Multi-Agent TD3)

**Value-Based 算法**
- `QMIX`：值分解网络
- `HAD3QN`：异构双重延迟Q网络

**特殊算法**
- `2TS-VVC` (Two-Timescale VVC)：双时间尺度电压-无功控制

### 3. 运行器模块 (`runners/`)

#### 3.1 On-Policy 运行器
- **基类**：`on_policy_base_runner.py`
  - 数据收集、策略更新、评估流程
  - 支持异构智能体训练

- **异构**：`on_policy_ha_runner.py`
  - HAPPO/HATRPO 专用运行器

- **同构**：`on_policy_ma_runner.py`
  - MAPPO 专用运行器

#### 3.2 Off-Policy 运行器
- **基类**：`off_policy_base_runner.py`
  - Replay Buffer 管理
  - 异步数据采样与训练

- **异构**：`off_policy_ha_runner.py`
- **同构**：`off_policy_ma_runner.py`

#### 3.3 特殊运行器
- `Qmix_runner.py`：QMIX 算法专用
- `two_ts_runner.py`：双时间尺度算法专用

### 4. 模型模块 (`models/`)

```
models/
├── base/                        # 基础网络组件
│   └── distributions.py         # 概率分布定义
├── policy_models/               # 策略网络
│   ├── gaussian_policy.py       # 高斯策略（连续动作）
│   ├── categorical_policy.py    # 分类策略（离散动作）
│   └── hybrid_policy.py         # 混合策略（离散+连续）
└── value_function_models/       # 价值函数网络
    ├── v_critic.py              # 状态价值函数 V(s)
    └── q_critic.py              # 动作价值函数 Q(s,a)
```

### 5. 配置模块 (`configs/`)

```
configs/
├── envs_cfgs/          # 环境配置
│   ├── powerzoo.yaml
│   ├── powerzoo_llm.yaml
│   └── stackelberg.yaml
├── algos_cfgs/         # 算法超参数配置
│   ├── happo.yaml
│   ├── mappo.yaml
│   └── ... (14种算法配置)
├── sys_cfgs/           # 系统配置
│   └── run_config.yaml
└── single_agent_cfgs/  # 单智能体配置
    └── ppo_config.yaml
```

### 6. 工具模块 (`utils/`)

**核心工具**
- `configs_tools.py`：配置文件加载与解析
- `envs_tools.py`：环境工具函数
- `models_tools.py`：模型构建工具
- `single_agent_tools.py`：单智能体训练工具
- `tensorboard_callback.py`：TensorBoard 回调与可视化
- `happo_diagnostics.py`：HAPPO 算法诊断工具
- `happo_monitor.py`：HAPPO 训练监控

### 7. 公共模块 (`common/`)

```
common/
├── buffers/                 # 经验回放缓冲区
│   ├── shared_buffer.py    # On-Policy 共享缓冲区
│   └── replay_buffer.py    # Off-Policy 回放缓冲区
├── base_logger.py          # 日志系统
└── valuenorm.py            # 价值归一化
```

### 8. 示例模块 (`examples/`)

```
examples/
├── multi_agent/
│   ├── launchers/          # 训练启动脚本 (.sh)
│   ├── scripts/            # Python训练脚本 (.py)
│   └── README.md
└── single_agent/
    ├── launchers/
    ├── scripts/
    └── README.md
```

### 9. 测试模块 (`tests/`)

```
tests/
├── envs/                   # 环境单元测试
│   ├── test_powerzoo.py
│   ├── test_powerzoo_llm.py
│   └── test_stackelberg.py
├── pytest.ini              # pytest配置
└── README.md
```

### 10. 节点系统 (`node_systems/`)

**标准测试系统**
- `13Bus/`：IEEE 13节点系统
- `34Bus/`：IEEE 34节点系统
- `123Bus/`：IEEE 123节点系统
- `8500-Node/`：大规模8500节点系统
- `9500-Node/`：大规模9500节点系统

**特殊配置**
- `34Bus_PV_Aggressive/`：激进PV控制配置
- `34Bus_PV_Conservative/`：保守PV控制配置
- `34Bus_PV_Optimized/`：优化PV控制配置

---

## 🔄 数据流与训练流程

### 训练流程图

```
┌─────────────┐
│ 1. 初始化    │
│   - 加载配置 │
│   - 创建环境 │
│   - 构建模型 │
└──────┬──────┘
       ↓
┌─────────────────────────────────────┐
│ 2. 数据收集循环                      │
│   ┌──────────────────────────────┐ │
│   │  for episode in episodes:    │ │
│   │    obs = env.reset()         │ │
│   │    for step in steps:        │ │
│   │      actions = policy(obs)   │ │
│   │      next_obs, reward, done  │ │
│   │         = env.step(actions)  │ │
│   │      buffer.insert(...)      │ │
│   └──────────────────────────────┘ │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 3. 策略更新                          │
│   On-Policy:                        │
│     - 计算优势函数 (GAE)             │
│     - 多轮PPO更新                   │
│   Off-Policy:                       │
│     - 从Replay Buffer采样           │
│     - 单步TD更新                    │
└──────────┬──────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 4. 评估与记录                        │
│   - TensorBoard记录指标             │
│   - 保存模型检查点                   │
│   - 诊断分析（HAPPO专用）            │
└──────────┬──────────────────────────┘
           ↓
        重复 2-4 直至收敛
```

### 关键数据流

```
环境输出 → Buffer → 模型输入 → 策略输出 → 环境输入
   ↓          ↓         ↓           ↓          ↓
Observation  存储    神经网络     Action    执行控制
  Reward    采样     前向传播    Sampling   仿真更新
   Info     更新     反向传播    Clipping   状态转移
```

---

## 🎯 关键技术特性

### 1. 混合动作空间支持
```python
# 离散动作（电容器、调压器）+ 连续动作（电池、光伏）
action_space = Tuple([
    MultiDiscrete([2, 2, ..., 33, 33, ...]),  # 离散部分
    Box(-1, 1, shape=(n_continuous,))          # 连续部分
])
```

### 2. 中心化训练-去中心化执行 (CTDE)
```
训练阶段：
  Critic 接收全局状态 → 中心化价值评估
  Actor 仅接收局部观测 → 去中心化策略

执行阶段：
  仅使用 Actor → 完全去中心化决策
```

### 3. 异构智能体支持
- **异构观测空间**：不同类型智能体观测不同特征
- **异构动作空间**：电容器(离散2) vs 电池(连续)
- **异构网络架构**：不同智能体使用不同网络结构
- **策略排序**：基于敏感性的智能体执行顺序优化（SHOM算法）

### 4. 奖励函数设计
```python
total_reward = (
    + ctrl_reward           # 控制成本（减少频繁动作）
    + voltage_reward        # 电压质量（约束满足）
    + powerloss_reward      # 功率损耗（系统效率）
    + power_balance_reward  # 功率平衡
    + pv_optimization_reward # PV优化奖励
)
```

### 5. 诊断与监控系统
- **HAPPO 诊断**：`utils/happo_diagnostics.py`
  - 梯度范数监控
  - KL散度跟踪
  - 策略熵分析
  - 价值损失监控

- **TensorBoard 集成**：实时可视化训练指标
  - 奖励曲线
  - 损失函数
  - 电压质量
  - 功率损耗

---

## 🔌 扩展性设计

### 1. 新环境接入
```python
# 1. 继承基类
class NewEnv(PowerZooBaseEnv):
    def __init__(self, config):
        super().__init__(config)

    def _get_observation(self):
        # 自定义观测逻辑
        pass

    def _calculate_reward(self):
        # 自定义奖励逻辑
        pass

# 2. 添加配置文件
# configs/envs_cfgs/new_env.yaml

# 3. 注册环境
# envs/__init__.py
```

### 2. 新算法接入
```python
# 1. 实现Actor/Critic
# algorithms/actors/new_actor.py
# algorithms/critics/new_critic.py

# 2. 继承运行器
class NewRunner(OnPolicyBaseRunner):
    def algorithm_specific_update(self):
        # 算法特定的更新逻辑
        pass

# 3. 添加配置
# configs/algos_cfgs/new_algo.yaml
```

### 3. 新设备类型支持
```python
# 1. 扩展 CRBP 架构
# envs/powerzoo_llm/devices/new_device.py

class NewDeviceController:
    def get_action_space(self):
        pass

    def get_observation(self):
        pass

    def execute_action(self, action):
        pass

# 2. 在环境中注册
# envs/powerzoo_llm/base_env.py
```

---

## 📊 项目统计

- **支持算法**：14种MARL算法
- **环境数量**：4类环境（PowerZoo基础/完整、Stackelberg、DSR）
- **测试系统**：5种标准IEEE节点系统（13/34/123/8500/9500）
- **控制设备**：4类CRBP设备（电容器/调压器/电池/光伏）
- **代码规模**：约20,000+ 行Python代码
- **测试覆盖**：完整的pytest测试套件

---

## 🚀 使用场景

### 研究场景
1. **配电网电压调控**：Volt-Var优化问题
2. **分布式能源管理**：PV+储能协调控制
3. **需求侧响应**：负荷调度与优化
4. **电力市场博弈**：Stackelberg博弈、多层博弈

### 教学场景
1. **MARL算法教学**：提供标准化benchmark环境
2. **电力系统控制**：电网控制问题建模与求解
3. **强化学习实践**：从单智能体到多智能体进阶

### 工程场景
1. **算法原型验证**：快速测试新算法性能
2. **控制策略优化**：实际电网控制策略开发
3. **系统仿真分析**：配电网运行状态模拟

---

## 📚 参考文献

项目核心论文：
```
X. Zheng, S. Yu, H. Cao, T. Shi, S. Xue, and T. Ding,
"Sensitivity-Based Heterogeneous Ordered Multi-Agent Reinforcement Learning
for Distributed Volt-Var Control in Active Distribution Network,"
IEEE Transactions on Smart Grid, Feb. 2025.
```

---

## 🔗 相关文档

- [PowerZoo环境说明](../PowerZoo环境说明.md)
- [示例脚本说明](../examples/README.md)
- [测试文档](../tests/README.md)
- [HAPPO诊断总结](../HAPPO_DIAGNOSTICS_SUMMARY.md)

---

**文档版本**：v1.0
**最后更新**：2025-12-02
**维护者**：PowerZoo开发团队
