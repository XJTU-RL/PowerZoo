# PowerZoo 单智能体模型保存和Checkpoint指南

## 概述

PowerZoo的单智能体训练系统现在提供了完整的模型保存和checkpoint功能，包括：

- **定期检查点保存**: 按指定步数间隔自动保存模型
- **最佳模型保存**: 基于评估结果自动保存性能最佳的模型
- **最终模型保存**: 训练结束时保存最终模型
- **增强回调模型保存**: 通过增强TensorBoard回调保存的额外模型
- **完整的训练状态保存**: 包括replay buffer和normalization参数

## 功能特性

### 1. 标准Stable-Baselines3回调

#### CheckpointCallback
- 按指定频率保存模型检查点
- 支持保存replay buffer和vecnormalize
- 文件命名格式: `{model_name}_checkpoint_{steps}_steps.zip`

#### EvalCallback
- 定期评估模型性能
- 自动保存最佳模型
- 生成详细的评估日志

### 2. 增强TensorBoard回调

- 集成模型保存功能
- 智能最佳模型检测
- 详细的训练指标记录
- 支持PowerZoo和Gym环境

## 使用方法

### 基本训练命令

```bash
# 使用默认参数训练
python examples/single_agent/train_with_enhanced_callback.py

# 自定义参数训练
python examples/single_agent/train_with_enhanced_callback.py \
    --env-type gym \
    --gym-env CartPole-v1 \
    --algorithm ppo \
    --total-timesteps 50000 \
    --save-freq 10000 \
    --eval-freq 5000 \
    --log-dir ./logs/my_experiment
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--env-type` | str | gym | 环境类型 (gym/powerzoo) |
| `--gym-env` | str | CartPole-v1 | Gym环境名称 |
| `--algorithm` | str | ppo | 算法类型 (ppo/dqn/sac/a2c) |
| `--total-timesteps` | int | 10000 | 总训练步数 |
| `--save-freq` | int | 1000 | 模型保存频率（步数） |
| `--eval-freq` | int | 500 | 模型评估频率（步数） |
| `--n-eval-episodes` | int | 10 | 评估时的回合数 |
| `--log-dir` | str | ./logs/enhanced_training | 日志保存目录 |
| `--verbose` | int | 1 | 详细程度 |

## 输出文件结构

训练完成后，会在指定的日志目录下生成带时间戳的子目录，包含以下文件：

```
logs/
└── {algorithm}_{env_type}_{timestamp}/
    ├── checkpoints/                    # 检查点文件
    │   ├── {model_name}_checkpoint_5000_steps.zip
    │   ├── {model_name}_checkpoint_10000_steps.zip
    │   └── ...
    ├── best_model/                     # 最佳模型
    │   └── best_model.zip
    ├── eval_logs/                      # 评估日志
    │   └── evaluations.npz
    ├── models/                         # 增强回调模型
    │   ├── {model_name}_5000_steps.zip
    │   ├── {model_name}_10000_steps.zip
    │   ├── {model_name}_best.zip
    │   └── ...
    ├── final_model.zip                 # 最终模型
    └── sb3_logs/                       # TensorBoard日志
        └── PPO_1/
            └── events.out.tfevents.*
```

## 模型加载和使用

### 加载最终模型

```python
from stable_baselines3 import PPO
import gymnasium as gym

# 加载模型
model = PPO.load("./logs/ppo_gym_20250806_235713/final_model.zip")

# 创建环境
env = gym.make('CartPole-v1')

# 使用模型进行预测
obs, _ = env.reset()
action, _states = model.predict(obs, deterministic=True)
```

### 加载检查点模型

```python
# 加载特定步数的检查点
checkpoint_path = "./logs/ppo_gym_20250806_235713/checkpoints/ppo_gym_checkpoint_10000_steps.zip"
model = PPO.load(checkpoint_path)
```

### 加载最佳模型

```python
# 加载评估性能最佳的模型
best_model_path = "./logs/ppo_gym_20250806_235713/best_model/best_model.zip"
model = PPO.load(best_model_path)
```

## 高级配置

### 自定义回调组合

```python
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from examples.single_agent.powerzoo_llm.enhanced_tensorboard_callback import create_enhanced_callback

# 创建增强回调
enhanced_callback = create_enhanced_callback(
    log_dir="./logs/custom_experiment",
    env_type='gym',
    enable_powerzoo_logging=False,
    name_prefix='custom_model',
    save_freq=5000
)

# 创建检查点回调
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./logs/custom_experiment/checkpoints",
    name_prefix='custom_checkpoint',
    save_replay_buffer=True,
    save_vecnormalize=True
)

# 创建评估回调
eval_callback = EvalCallback(
    env,
    best_model_save_path="./logs/custom_experiment/best_model",
    log_path="./logs/custom_experiment/eval_logs",
    eval_freq=5000,
    n_eval_episodes=10
)

# 组合所有回调
callback_list = CallbackList([enhanced_callback, checkpoint_callback, eval_callback])

# 训练模型
model.learn(total_timesteps=50000, callback=callback_list)
```

## 监控和可视化

### TensorBoard可视化

```bash
# 启动TensorBoard
tensorboard --logdir ./logs/ppo_gym_20250806_235713 --port 6006 --host 0.0.0.0

# 在浏览器中访问
# http://localhost:6006
```

### 评估结果分析

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载评估结果
eval_data = np.load("./logs/ppo_gym_20250806_235713/eval_logs/evaluations.npz")

# 绘制评估曲线
plt.figure(figsize=(10, 6))
plt.plot(eval_data['timesteps'], eval_data['results'].mean(axis=1))
plt.xlabel('Training Steps')
plt.ylabel('Mean Reward')
plt.title('Model Performance During Training')
plt.grid(True)
plt.show()
```

## 最佳实践

### 1. 频率设置建议

- **保存频率**: 设置为总训练步数的5-10%
- **评估频率**: 设置为保存频率的一半
- **评估回合数**: 对于简单环境使用5-10回合，复杂环境使用10-20回合

### 2. 存储空间管理

- 定期清理旧的检查点文件
- 只保留关键的检查点（如最佳模型、最终模型）
- 使用压缩格式存储模型

### 3. 训练监控

- 实时监控TensorBoard指标
- 关注评估奖励的趋势
- 及时发现过拟合或训练不稳定

### 4. 模型选择

- 优先使用最佳模型进行部署
- 保留多个检查点以便回滚
- 在不同数据集上验证模型泛化性能

## 故障排除

### 常见问题

1. **模型文件过大**
   - 检查是否保存了不必要的replay buffer
   - 考虑降低模型复杂度

2. **评估性能不稳定**
   - 增加评估回合数
   - 检查环境的随机性设置

3. **检查点保存失败**
   - 确保有足够的磁盘空间
   - 检查文件权限设置

4. **模型加载错误**
   - 确认模型文件完整性
   - 检查算法和环境的兼容性

## 版本兼容性

- **Stable-Baselines3**: >= 2.0.0
- **Gymnasium**: >= 0.26.0
- **PyTorch**: >= 1.7.0
- **TensorBoard**: >= 2.0.0

## 扩展功能

### 自定义评估指标

可以通过继承EvalCallback来实现自定义的评估指标和保存逻辑。

### 分布式训练支持

对于大规模训练，可以结合分布式训练框架使用checkpoint功能。

### 云存储集成

可以将模型保存到云存储服务，实现跨平台的模型管理。

---

通过这套完整的模型保存和checkpoint系统，PowerZoo为单智能体强化学习提供了可靠的训练保障和便捷的模型管理功能。