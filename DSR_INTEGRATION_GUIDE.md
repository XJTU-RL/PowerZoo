# DSR环境集成指南

## 概述

DSR (Distribution System Restoration) 环境已成功集成到PowerZoo框架中。这是一个基于OpenDSS的配电网恢复多智能体强化学习环境，原本使用PandaPower，现已完全迁移到OpenDSS以保持与PowerZoo的一致性。

## 主要特性

### 🔄 核心功能
- **配电网故障恢复**: 基于IEEE 123节点系统的配电网恢复场景
- **多智能体协作**: 95个智能体协同工作（1个开关控制器 + 9个PV智能体 + 85个负荷智能体）
- **动态故障生成**: 随机生成3-5个故障场景，模拟真实配电网故障
- **智能恢复策略**: 考虑负荷优先级、电压约束和线路容量的恢复决策

### 🏗️ 技术架构
- **仿真引擎**: 从PandaPower迁移到OpenDSS，与PowerZoo保持一致
- **智能体类型**:
  - 开关智能体：控制配电网开关操作
  - PV智能体：控制分布式光伏发电输出
  - 负荷智能体：决定负荷投切优先级
- **动作掩码**: 智能动作掩码，防止不可行动作
- **观测空间**: 包含母线电压、设备状态、通电信息等

### 🎯 奖励机制
- **负荷恢复奖励**: 基于负荷优先级的加权恢复率
- **电压约束惩罚**: 母线电压越限惩罚
- **线路过载惩罚**: 线路容量违规惩罚
- **失败惩罚**: 潮流不收敛或操作失败惩罚

## 文件结构

```
envs/dsr/
├── __init__.py              # DSR环境包初始化
├── dsr_env.py              # DSR环境包装器，对接PowerZoo框架
├── dsr_logger.py           # DSR专用日志记录器
└── core/
    ├── __init__.py         # 核心组件初始化
    ├── config.py           # DSR配置类和预定义配置
    └── dsr_core.py         # DSR核心环境实现（基于OpenDSS）

configs/envs_cfgs/
└── dsr.yaml                # DSR环境默认配置文件
```

## 使用方法

### 基本训练命令

```bash
# 使用SHOM算法训练DSR环境
python examples/train.py --algo shom --env dsr --exp_name dsr_test

# 使用HAPPO算法训练
python examples/train.py --algo happo --env dsr --exp_name dsr_happo

# 使用QMIX算法训练
python examples/train.py --algo qmix --env dsr --exp_name dsr_qmix
```

### 支持的算法

DSR环境支持PowerZoo框架中的所有多智能体强化学习算法：

**On-policy算法**:
- HAPPO (Heterogeneous Agent Proximal Policy Optimization)
- HATRPO (Heterogeneous Agent Trust Region Policy Optimization)
- HAA2C (Heterogeneous Agent Advantage Actor-Critic)
- MAPPO (Multi-Agent Proximal Policy Optimization)
- SHOM (Shared Experience Actor-Critic)

**Off-policy算法**:
- HADDPG (Heterogeneous Agent Deep Deterministic Policy Gradient)
- HATD3 (Heterogeneous Agent Twin Delayed DDPG)
- HASAC (Heterogeneous Agent Soft Actor-Critic)
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- MATD3 (Multi-Agent Twin Delayed DDPG)

**Value-based算法**:
- QMIX (Q-Value Mixing)
- HAD3QN (Heterogeneous Agent Dueling Deep Q-Network)

### 配置自定义

可以通过修改 `configs/envs_cfgs/dsr.yaml` 来自定义环境参数：

```yaml
env_args:
  max_episode_steps: 15      # 最大恢复步数
  n_dg: 7                   # 黑启动DG数量
  n_pv: 9                   # PV智能体数量
  n_switch: 20              # 可控开关数量
  reward_restore: 20.0      # 负荷恢复奖励权重
  reward_voltage: 1.0       # 电压越限惩罚权重
  use_action_mask: True     # 启用动作掩码
```

## 环境规格

### 智能体信息
- **总智能体数**: 95个
- **智能体类型**: 异构智能体（3种类型）
- **协作模式**: 完全协作，共享奖励

### 动作空间
- **开关智能体**: Discrete(21) - 选择操作的开关（0=不操作，1-20=开关ID）
- **PV智能体**: Discrete(11) - 功率输出等级（0-10，对应0%-100%）
- **负荷智能体**: Discrete(2) - 负荷状态（0=断开，1=恢复）

### 观测空间
- **维度**: 每个智能体约150维观测
- **内容**: 母线电压、设备状态、通电信息、时间步、智能体特定信息

### Episode设置
- **最大步数**: 15步（代表15个恢复时段）
- **成功条件**: 负荷恢复率达到95%以上
- **失败条件**: 潮流不收敛或达到最大步数

## 依赖要求

### 必需依赖
- OpenDSS: PowerZoo的电力系统仿真引擎
- NetworkX: 图论分析
- NumPy: 数值计算
- Gym: 强化学习环境接口

### 可选依赖
- CuPy: GPU加速计算（用于大型网络）
- WandB: 实验跟踪和可视化
- TensorboardX: 训练可视化

## 测试验证

运行集成测试：
```bash
python test_dsr_env.py
```

测试内容包括：
- ✓ DSR配置验证
- ✓ 核心环境功能（如果OpenDSS可用）
- ✓ 环境包装器接口
- ✓ 框架集成验证

## 技术细节

### 从PandaPower到OpenDSS的迁移

1. **网络初始化**: 使用PowerZoo的123Bus OpenDSS模型替代PandaPower网络
2. **潮流计算**: `pp.runpp()` → `dss.ActiveCircuit.Solution.Solve()`
3. **设备控制**: 通过OpenDSS API控制开关、PV、负荷
4. **拓扑分析**: 使用OpenDSS拓扑信息构建NetworkX图

### 智能体设计

**开关智能体（全局）**:
- 观测：全网拓扑、开关状态、通电区域
- 动作：选择操作的开关进行投切

**PV智能体（分布式）**:
- 观测：本地母线电压、PV状态、周边负荷
- 动作：调节PV输出功率（0-100%）

**负荷智能体（分布式）**:
- 观测：负荷状态、母线电压、优先级
- 动作：决定是否恢复负荷

### 故障场景

- **故障类型**: 线路故障（断开）
- **故障数量**: 3-5个随机故障
- **故障位置**: 随机选择非关键线路
- **恢复目标**: 通过开关操作和资源调度恢复供电

## 性能指标

### 训练指标
- `dsr/restored_load_ratio`: 负荷恢复率
- `dsr/energized_buses_ratio`: 通电母线比例
- `dsr/convergence_rate`: 潮流收敛率
- `dsr/restoration_steps`: 平均恢复步数

### 评估指标
- `eval/restored_load_ratio`: 评估负荷恢复率
- `eval/success_rate`: 成功恢复比例（>90%恢复率）
- `eval/convergence_rate`: 评估收敛率

## 未来扩展

1. **更多电网拓扑**: 支持34Bus、8500Node等其他PowerZoo系统
2. **故障类型扩展**: 发电机故障、母线故障等
3. **动态负荷**: 时变负荷需求和优先级
4. **不确定性**: 可再生能源功率预测不确定性
5. **实时约束**: 电压稳定、频率控制等动态约束

## 贡献和反馈

DSR环境现已完全集成到PowerZoo框架中，支持所有现有的多智能体强化学习算法。如有问题或改进建议，请参考PowerZoo的贡献指南。

---

**注意**: 在没有完整OpenDSS环境的情况下，某些功能测试会被跳过，但框架集成是完整的。在生产环境中请确保安装所有必需依赖。