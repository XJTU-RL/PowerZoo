# 单智能体强化学习训练

本目录包含PowerZoo环境的单智能体强化学习训练脚本和启动器。

## 📁 目录结构

```
single_agent/
├── scripts/                     # Python训练脚本
│   ├── train_single.py          # 基础单智能体训练
│   ├── train_single_agent.py    # 单智能体PowerZoo训练
│   ├── train_with_enhanced_callback.py # 增强回调训练
│   └── enhanced_tensorboard_callback.py # TensorBoard回调
└── launchers/                   # Shell启动脚本
    ├── train_single_agent_vvc_pv.sh # PowerZoo PV训练
    ├── train_single_pv_discrete.sh       # 离散PV控制训练
    ├── train_single.sh                   # 通用单智能体训练
    └── test_ddpg.sh                      # DDPG测试脚本
```

## 🚀 快速开始

### 1. 基础训练

```bash
# 使用PPO算法训练
cd launchers
./train_single_agent_vvc_pv.sh

# 使用Python脚本直接训练
python scripts/train_single_agent.py
```

### 2. 增强训练（带TensorBoard）

```bash
# 使用增强回调进行训练
python scripts/train_with_enhanced_callback.py \
  --algo PPO \
  --env SingleAgentPowerZoo \
  --timesteps 1000000 \
  --tensorboard
```

### 3. 测试不同算法

```bash
# 测试DDPG算法
cd launchers
./test_ddpg.sh

# 测试离散控制
./train_single_pv_discrete.sh
```

## 📊 支持的算法

### On-Policy算法
- **PPO** (Proximal Policy Optimization)
  - 推荐用于电力系统控制
  - 稳定性好，样本效率适中
  
- **A2C** (Advantage Actor-Critic)
  - 训练速度快
  - 适合简单控制任务

### Off-Policy算法
- **SAC** (Soft Actor-Critic)
  - 适合连续动作空间
  - 样本效率高
  
- **TD3** (Twin Delayed DDPG)
  - 连续控制
  - 减少过估计问题
  
- **DDPG** (Deep Deterministic Policy Gradient)
  - 确定性策略
  - 适合精确控制

- **DQN** (Deep Q-Network)
  - 离散动作空间
  - 简单有效

## 🎯 训练参数建议

### PPO配置（推荐）
```python
config = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
}
```

### SAC配置（连续控制）
```python
config = {
    "learning_rate": 3e-4,
    "buffer_size": 1000000,
    "learning_starts": 100,
    "batch_size": 256,
    "tau": 0.005,
    "gamma": 0.99,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto"
}
```

### DQN配置（离散控制）
```python
config = {
    "learning_rate": 1e-4,
    "buffer_size": 50000,
    "learning_starts": 50000,
    "batch_size": 32,
    "tau": 1.0,
    "gamma": 0.99,
    "train_freq": 4,
    "gradient_steps": 1,
    "target_update_interval": 10000,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1.0,
    "exploration_final_eps": 0.05
}
```

## 📈 性能监控

### TensorBoard集成

```bash
# 启动TensorBoard
tensorboard --logdir ./tensorboard_logs/

# 训练时启用TensorBoard
python scripts/train_with_enhanced_callback.py --tensorboard
```

监控指标：
- Episode Reward（回合奖励）
- Episode Length（回合长度）
- Loss（各种损失函数）
- Learning Rate（学习率）
- Entropy（策略熵）
- Value Function（价值函数）

### 自定义回调

```python
from utils.tensorboard_callback import EnhancedTensorBoardCallback

# 创建增强回调
callback = EnhancedTensorBoardCallback(
    verbose=1,
    log_freq=100,
    save_freq=10000,
    save_path="./models/",
    name_prefix="ppo_vvc"
)

# 训练时使用
model.learn(
    total_timesteps=1000000,
    callback=callback
)
```

## 🔧 环境配置

### 基础配置
```python
from smartgrid.single_agent import SingleAgentConfig

config = SingleAgentConfig(
    circuit_name="13Bus",       # 电路名称
    max_episode_steps=360,       # 最大步数
    voltage_penalty_weight=1.0, # 电压惩罚权重
    power_loss_weight=0.1,       # 功率损耗权重
    discharge_penalty_weight=0.5,# 放电惩罚权重
    log_level="INFO"            # 日志级别
)
```

### 高级配置
```python
config = SingleAgentConfig(
    circuit_name="34Bus",
    max_episode_steps=720,
    # 设备控制
    enable_capacitors=True,
    enable_regulators=True,
    enable_batteries=True,
    enable_pv_systems=True,
    # 奖励权重
    voltage_penalty_weight=2.0,
    power_loss_weight=0.2,
    discharge_penalty_weight=1.0,
    pv_utilization_weight=0.5,
    # 其他参数
    normalize_observations=True,
    normalize_rewards=True,
    clip_rewards=10.0
)
```

## 💾 模型保存与加载

### 保存模型
```python
# 训练后保存
model.save("models/ppo_vvc_final")

# 训练中定期保存（使用回调）
callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="ppo_checkpoint"
)
```

### 加载模型
```python
from stable_baselines3 import PPO

# 加载已训练模型
model = PPO.load("models/ppo_vvc_final")

# 继续训练
model.set_env(env)
model.learn(total_timesteps=500000)

# 测试模型
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

## ⚠️ 注意事项

1. **动作空间**: 单智能体使用MultiDiscrete动作空间
2. **归一化**: 建议对观察和奖励进行归一化
3. **学习率调度**: 长时间训练建议使用学习率衰减
4. **探索策略**: 初期保持较高探索率

## 🔧 故障排除

### 训练不收敛
- 降低学习率
- 增加batch size
- 调整奖励权重
- 使用更长的训练时间

### 内存问题
- 减少buffer_size（off-policy算法）
- 减少n_steps（on-policy算法）
- 使用更少的并行环境

### 性能问题
- 使用GPU加速：`device='cuda'`
- 增加并行环境数
- 使用向量化环境

## 📚 相关资源

- [Stable Baselines3文档](https://stable-baselines3.readthedocs.io/)
- [PowerZoo环境文档](../../envs/power_envs/smartgrid/single_agent/README.md)
- [强化学习算法比较](https://spinningup.openai.com/en/latest/)