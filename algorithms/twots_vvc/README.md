# 两时间尺度VVC算法实现文档

## 概述

两时间尺度VVC（Two-Timescale Volt-VAR Control）算法是一种分层强化学习方法，用于电力系统的电压无功控制。该算法将控制问题分解为快层和慢层两个时间尺度：

- **快层**：高频控制连续型设备（PV逆变器、BESS储能）
- **慢层**：低频控制离散型设备（电容器组、变压器分接头）

## 算法架构

### 1. 快层 - DDPG算法

快层使用深度确定性策略梯度（DDPG）算法控制连续动作空间的设备。

**核心特性**：
- 连续动作空间输出，范围[-1, 1]
- Ornstein-Uhlenbeck噪声探索，带自适应衰减
- 软目标网络更新（tau=0.005）
- 梯度裁剪（max_norm=1.0）

**网络架构**：
- Actor网络：[obs_dim] → [256] → [128] → [action_dim]
- Critic网络：[obs_dim + action_dim] → [256] → [128] → [1]
- 每层包含ReLU激活和LayerNorm

**探索策略**：
- OU噪声：theta=0.15, sigma初始0.2，衰减至0.05
- 噪声衰减率：0.99995（每次更新）

### 2. 慢层 - SAC-Discrete算法

慢层使用离散版本的软演员评论家（SAC-D）算法，支持多头离散动作输出。

**核心特性**：
- 多头离散动作空间（每个设备独立动作头）
- 自动熵调节机制
- 双Q网络防止过估计
- LSTM序列编码器处理快层轨迹
- Epsilon-greedy辅助探索

**网络架构**：
- Actor网络：共享层[256, 128] + 多个动作头
- Critic网络（CTDE）：[state + seq_embed + action_onehot] → [256, 256, 128] → [1]
- 序列编码器：LSTM(64, 2层) → Linear(128)

**探索策略**：
- 熵正则化（自动调节alpha）
- Epsilon-greedy：初始0.3，衰减至0.01

### 3. 协调器

协调器负责管理两个层级的交互和训练流程。

**训练阶段**：
1. **预训练阶段**（前200个interval）：仅训练快层，慢层使用随机动作
2. **联合训练阶段**：同时训练快层和慢层

**更新频率控制**：
- Critic更新频率：每个step
- Actor更新频率：每2个step
- 可通过配置独立控制快层和慢层的更新频率

## 关键改进

### 1. 算法正确性修复
- **慢层SAC更新逻辑**：Actor更新时重新采样动作，而非使用批次中的旧动作
- **Alpha更新**：使用重采样的log概率更新熵系数

### 2. 训练稳定性增强
- **梯度裁剪统一**：所有网络统一使用1.0的梯度裁剪
- **LSTM显式初始化**：防止序列编码不稳定
- **输入验证**：所有维度和超参数范围检查

### 3. 探索能力优化
- **噪声自适应衰减**：OU噪声sigma随训练进度衰减
- **双重探索策略**：慢层同时使用熵正则化和epsilon-greedy
- **探索参数监控**：返回噪声水平和epsilon值用于调试

### 4. 性能优化
- **更新频率控制**：Actor和Critic独立更新频率
- **动作格式统一**：确保返回值格式一致性
- **批处理优化**：支持条件更新减少计算

## 使用示例

```python
from algorithms.twots_vvc import TwoTSVVC

# 配置
cfg = {
    "train": {
        "pretrain_fast_intervals": 200,
        "joint_intervals": 1000,
        "fast_critic_update_freq": 1,
        "fast_actor_update_freq": 2,
    },
    "fast": {
        "ddpg": {
            "lr": 1e-3,
            "tau": 0.005,
            "gamma": 0.99,
            "noise": {
                "theta": 0.15,
                "sigma": 0.2,
                "sigma_min": 0.05,
                "sigma_decay": 0.99995
            }
        }
    },
    "slow": {
        "sacd": {
            "lr": 3e-4,
            "tau": 0.005,
            "gamma": 0.99,
            "alpha": 0.2,
            "epsilon_start": 0.3,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.9999
        },
        "seq_embed_dim": 128,
        "lstm_hidden": 64,
        "lstm_layers": 2
    }
}

# 初始化
algo = TwoTSVVC(cfg)

# 环境交互后初始化agents
algo.init_agents(
    fast_obs_dim=10,
    fast_action_dim=4,
    slow_obs_dim=20,
    slow_action_dims=[3, 3, 5],  # 3个离散设备
    seq_input_dim=15
)

# 训练循环
for interval in range(1000):
    # 收集经验...
    
    # 快层动作
    fast_action = algo.act_fast(fast_obs, explore=True)
    
    # 慢层动作
    slow_actions = algo.act_slow(slow_state, deterministic=False)
    
    # 序列编码
    seq_embed = algo.encode_sequence(fast_trajectory)
    
    # 更新
    fast_metrics = algo.update_fast(fast_batch)
    slow_metrics = algo.update_slow(slow_batch)
    
    algo.increment_interval()
```

## 超参数建议

### 快层DDPG
- 学习率：1e-3 到 1e-4
- 软更新系数tau：0.001 到 0.01
- 折扣因子gamma：0.95 到 0.99
- 噪声参数：根据动作范围调整

### 慢层SAC-D
- 学习率：3e-4 到 1e-4
- 初始熵系数alpha：0.1 到 0.3
- Epsilon初始值：0.2 到 0.5（取决于探索需求）

### 训练策略
- 预训练快层：100-500个interval
- 更新频率比：Actor更新频率为Critic的1/2到1/4
- 批大小：256 到 1024

## 常见问题

### Q: 为什么需要两个时间尺度？
A: 电力系统中不同设备有不同的操作特性。PV逆变器可以快速调节，而电容器组切换次数有限。分层控制可以更好地匹配设备特性。

### Q: 如何调试训练不稳定？
A: 
1. 监控噪声水平和epsilon值
2. 检查Q值是否发散
3. 降低学习率
4. 增加更新频率比

### Q: 序列编码器的作用？
A: LSTM编码器将快层的历史轨迹压缩为固定维度的特征向量，为慢层决策提供时序信息。

## 版本记录

### v1.1.0 (2024-01)
- 修复慢层SAC更新逻辑
- 添加探索噪声衰减
- 增强输入验证
- 添加更新频率控制
- 改进探索策略

### v1.0.0 (2023-12)
- 初始实现
- 基础DDPG + SAC-D架构
- 两阶段训练流程