# 单智能体PowerZoo环境

本模块提供了专门针对单智能体强化学习的PowerZoo环境实现，基于OpenDSS电力系统仿真，专注于电压-无功控制(VVC)优化问题。

## 概述

单智能体PowerZoo环境将多智能体的电力系统控制问题转换为单智能体问题，通过统一的动作空间和观测空间，让单个智能体同时控制系统中的所有可控设备，包括：

- **电容器组**: 无功补偿设备
- **调压器**: 电压调节设备  
- **储能系统**: 有功/无功功率调节
- **光伏系统**: 功率因数控制

## 主要特性

### 🎯 统一控制策略
- 单个智能体控制所有设备，避免多智能体协调复杂性
- 全局优化视角，更好的系统级性能
- 简化的训练流程和算法选择

### 🔧 灵活配置
- 可选择启用/禁用不同类型的控制设备
- 可调节的奖励函数权重
- 多种预定义配置模板

### 📊 完整日志系统
- 详细的训练过程记录
- 性能指标统计和分析
- 可视化数据导出

### ⚡ 高效实现
- 优化的状态表示和动作编码
- 快速的环境重置和步进
- 内存友好的数据处理

## 文件结构

```
single_agent/
├── __init__.py                 # 模块初始化
├── single_agent_env.py         # 主环境类
├── single_agent_config.py      # 配置管理
├── single_agent_logger.py      # 日志记录器
└── README.md                   # 本文档
```

## 快速开始

### 基本使用

```python
from smartgrid.single_agent import SingleAgentPowerZooEnv, SingleAgentConfig

# 创建环境配置
config = SingleAgentConfig(
    circuit_name="13Bus",
    max_episode_steps=24,
    voltage_penalty_weight=1.0
)

# 创建环境
env = SingleAgentPowerZooEnv(config=config)

# 训练循环
obs = env.reset()
for step in range(24):
    action = env.action_space.sample()  # 随机动作
    obs, reward, done, info = env.step(action)
    if done:
        break

env.close()
```

### 使用预定义配置

```python
from smartgrid.single_agent import SingleAgentPowerZooEnv
from smartgrid.single_agent.single_agent_config import TRAINING_CONFIG

# 使用训练配置
env = SingleAgentPowerZooEnv(config=TRAINING_CONFIG)
```

## 环境详细说明

### 观测空间

观测空间包含以下信息（可通过配置选择）：

1. **母线电压** (Bus Voltages)
   - 所有母线的电压幅值（标幺值）
   - 维度: [num_buses]

2. **设备状态** (Device States)
   - 电容器开关状态
   - 调压器抽头位置
   - 储能系统功率输出和SOC
   - 光伏系统功率输出

3. **系统信息** (System Info)
   - 总有功功率
   - 总无功功率
   - 系统损耗

### 动作空间

动作空间为多维离散空间，包含：

1. **电容器动作**: 每个电容器的开关控制 {0: 关闭, 1: 开启}
2. **调压器动作**: 每个调压器的抽头调节 {-1: 降低, 0: 保持, 1: 提高}
3. **储能动作**: 功率设定值的离散化 {0-10: 对应不同功率水平}
4. **光伏动作**: 功率因数调节 {0-4: 对应不同功率因数}

### 奖励函数

奖励函数设计考虑多个目标：

```python
reward = -(voltage_penalty_weight * voltage_penalty + 
          power_loss_weight * power_loss + 
          discharge_penalty_weight * discharge_penalty)
```

- **电压偏差惩罚**: 母线电压偏离额定值的程度
- **功率损耗惩罚**: 系统总功率损耗
- **放电惩罚**: 储能系统过度放电的惩罚

## 配置参数

### 基础配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `circuit_name` | str | "13Bus" | 电路拓扑名称 |
| `max_episode_steps` | int | 24 | 最大步数 |
| `seed` | int | 42 | 随机种子 |

### 奖励配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `voltage_penalty_weight` | float | 1.0 | 电压偏差权重 |
| `power_loss_weight` | float | 0.1 | 功率损耗权重 |
| `discharge_penalty_weight` | float | 0.5 | 放电惩罚权重 |
| `voltage_target` | float | 1.0 | 目标电压 |
| `voltage_tolerance` | float | 0.05 | 电压容差 |

### 设备控制配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_capacitors` | bool | True | 启用电容器控制 |
| `enable_regulators` | bool | True | 启用调压器控制 |
| `enable_batteries` | bool | True | 启用储能控制 |
| `enable_pv_systems` | bool | True | 启用光伏控制 |

## 预定义配置模板

### DEFAULT_CONFIG
标准配置，适用于一般训练和测试。

### TRAINING_CONFIG
训练优化配置：
- 增强的奖励权重
- 启用详细日志
- 保存回合数据

### TESTING_CONFIG
测试配置：
- 标准奖励权重
- 简化日志
- 不保存详细数据

### FAST_CONFIG
快速测试配置：
- 减少回合步数
- 关闭数据标准化
- 最小日志记录

## 日志系统

### 日志级别
- **DEBUG**: 详细的步骤信息
- **INFO**: 回合和训练进度
- **WARNING**: 警告信息
- **ERROR**: 错误信息

### 保存的数据
- `training.log`: 训练日志文件
- `config.json`: 环境配置
- `episodes.csv`: 回合级数据
- `steps.csv`: 步骤级数据
- `metrics.json`: 性能指标

## 性能优化建议

### 训练效率
1. **合理设置回合长度**: 根据问题复杂度调整`max_episode_steps`
2. **选择性启用设备**: 只启用必要的控制设备类型
3. **优化观测空间**: 关闭不必要的观测信息
4. **批量处理**: 使用向量化环境进行并行训练

### 收敛性能
1. **调节奖励权重**: 根据具体目标调整各项权重
2. **设置合适的容差**: 避免过于严格的约束
3. **使用预训练**: 从简单配置开始逐步增加复杂度

## 故障排除

### 常见问题

1. **环境重置失败**
   - 检查OpenDSS安装和电路文件
   - 确认电路拓扑名称正确

2. **动作空间维度错误**
   - 检查设备启用配置
   - 确认电路中设备数量

3. **奖励值异常**
   - 检查奖励权重设置
   - 确认电压目标值合理

4. **训练不收敛**
   - 调整学习率和网络结构
   - 检查奖励函数设计
   - 尝试不同的算法

### 调试技巧

1. **启用详细日志**: 设置`log_level="DEBUG"`
2. **保存回合数据**: 设置`save_episode_data=True`
3. **使用快速配置**: 先用`FAST_CONFIG`验证环境
4. **检查动作有效性**: 确认动作在合理范围内

## 扩展开发

### 自定义奖励函数

```python
class CustomSingleAgentEnv(SingleAgentPowerZooEnv):
    def _calculate_reward(self, info):
        # 自定义奖励计算逻辑
        custom_reward = ...
        return custom_reward
```

### 添加新的观测信息

```python
def _get_observation(self):
    obs = super()._get_observation()
    # 添加自定义观测
    custom_obs = ...
    return np.concatenate([obs, custom_obs])
```

### 自定义动作空间

```python
def _create_action_space(self):
    # 定义自定义动作空间
    custom_action_space = ...
    return custom_action_space
```

## 相关资源

- [PowerZoo主项目](../../../README.md)
- [OpenDSS文档](https://www.epri.com/pages/sa/opendss)
- [强化学习算法参考](../../../../algorithms/)
- [训练示例](../../../../examples/)

## 许可证

本项目遵循PowerZoo主项目的许可证协议。