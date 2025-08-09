# 多智能体强化学习训练

本目录包含PowerZoo环境的多智能体强化学习训练脚本和启动器。

## 📁 目录结构

```
multi_agent/
├── scripts/              # Python训练脚本
│   ├── train.py         # 通用多智能体训练脚本
│   ├── train_powerzoo.py # PowerZoo专用训练脚本
│   ├── train_dsr_aggregation.py # DSR聚合训练
│   ├── train_stackelberg.py # Stackelberg博弈训练
│   ├── train_dan_happo.py # DAN-HAPPO训练
│   └── auto_train.py    # 自动化训练脚本
└── launchers/           # Shell启动脚本
    ├── quick_train_happo_pv.sh # 快速HAPPO训练
    ├── train_happo_pv_full.sh  # 完整参数HAPPO训练
    ├── train_marl_pv_.sh        # MARL PV训练
    ├── train_powerzoo.sh        # PowerZoo通用训练
    └── run_dsr_aggregation.sh  # DSR聚合运行
```

## 🚀 快速开始

### 1. PowerZoo专用训练（推荐）

```bash
# 使用Python脚本直接训练
python scripts/train_powerzoo.py --pv_plan optimized --algo happo

# 参数说明
--pv_plan: [conservative|optimized|aggressive] # PV方案选择
--algo: [happo|mappo|hatrpo|maddpg|matd3]     # 算法选择
--train_steps: 训练步数（默认1000000）
--gpu: GPU设备ID（-1表示CPU）
--use_wandb: 启用WandB日志记录
```

### 2. 快速HAPPO训练

```bash
# 使用Shell脚本快速启动
cd launchers
./quick_train_happo_pv.sh aggressive  # 使用激进方案

# 可选参数
./quick_train_happo_pv.sh [conservative|optimized|aggressive]
```

### 3. 完整参数配置

```bash
# 使用完整参数脚本
cd launchers
./train_happo_pv_full.sh \
  --plan optimized \
  --gpu 0 \
  --threads 32 \
  --batch-size 2048 \
  --lr 5e-4 \
  --steps 10000000 \
  --wandb
```

## 📊 支持的算法

### HAPPO（异构智能体PPO）
- **特点**: 支持异构动作空间，适合电力系统控制
- **配置文件**: `configs/algos_cfgs/happo.yaml`
- **推荐场景**: PV控制、调压器协同

### MAPPO（多智能体PPO）
- **特点**: 集中式训练，分布式执行
- **配置文件**: `configs/algos_cfgs/mappo.yaml`
- **推荐场景**: 同构智能体协作

### HATRPO（异构智能体TRPO）
- **特点**: 保证单调改进，训练稳定
- **配置文件**: `configs/algos_cfgs/hatrpo.yaml`
- **推荐场景**: 需要稳定训练的场景

### MADDPG/MATD3
- **特点**: 确定性策略，连续动作
- **配置文件**: `configs/algos_cfgs/maddpg.yaml`
- **推荐场景**: 连续控制任务

## 🎯 训练参数建议

### 基础设置
```python
# 环境参数
episode_length = 360      # 6小时仿真，2分钟间隔
n_rollout_threads = 8     # 并行环境数

# 训练参数
batch_size = 2048         # 批次大小
learning_rate = 5e-4      # 学习率
gamma = 0.99             # 折扣因子
ppo_epoch = 10           # PPO更新轮数
```

### PV方案特定参数

#### 保守方案
```python
voltage_penalty = -1.5
power_loss_penalty = -0.6
pv_utilization_reward = 0.2
```

#### 优化方案
```python
voltage_penalty = -2.0
power_loss_penalty = -0.5
pv_utilization_reward = 0.3
```

#### 激进方案
```python
voltage_penalty = -3.0
power_loss_penalty = -0.4
pv_utilization_reward = 0.4
enable_curtailment = True
```

## 📈 性能监控

训练过程中会自动记录以下指标：
- 平均奖励
- 电压越限率
- 功率损耗百分比
- PV利用率
- 策略熵
- 价值函数损失

## 💾 结果保存

训练结果保存在 `results/` 目录下：
```
results/
└── happo_pv_optimized_20250807_120000/
    ├── training_config.yaml  # 训练配置
    ├── training_report.md    # 训练报告
    ├── models/              # 模型检查点
    ├── logs/                # 训练日志
    ├── eval/                # 评估结果
    └── plots/               # 可视化图表
```

## ⚠️ 注意事项

1. **GPU内存**: HAPPO算法建议至少8GB显存
2. **并行环境**: 增加`n_rollout_threads`可以加速训练
3. **学习率调度**: 长时间训练建议使用学习率衰减
4. **评估频率**: 建议每25个episode评估一次

## 🔧 故障排除

### 内存不足
- 减少`batch_size`
- 减少`n_rollout_threads`
- 使用梯度累积

### 训练不稳定
- 降低学习率
- 增加PPO的`clip_param`
- 使用HATRPO算法

### 收敛缓慢
- 增加训练步数
- 调整奖励权重
- 使用预训练模型