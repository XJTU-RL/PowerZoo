# 通用增强TensorBoard回调使用指南

## 概述

`EnhancedTensorBoardCallback` 是一个通用的增强TensorBoard回调类，支持多种强化学习环境（PowerZoo、Gym等），提供了丰富的日志记录、模型管理和性能监控功能。

## 主要特性

### 🔧 核心功能
- **自动环境检测**: 自动识别PowerZoo、Gym或通用环境类型
- **智能日志记录**: 根据环境类型记录相应的指标
- **模型自动保存**: 定期保存模型和最佳模型
- **TensorBoard集成**: 完整的TensorBoard日志支持
- **多算法兼容**: 支持PPO、DQN、SAC、A2C等算法

### 📊 日志记录功能
- **系统指标**: CPU、内存、GPU使用率（如果启用PowerZoo日志）
- **训练指标**: 奖励、损失、学习率等
- **环境指标**: 根据环境类型记录特定指标
- **性能指标**: 训练速度、时间统计等

### 💾 模型管理
- **定期保存**: 按指定频率保存模型检查点
- **最佳模型**: 自动保存性能最佳的模型
- **元数据记录**: 保存训练配置和性能指标

## 安装和配置

### 依赖要求
```bash
# 基础依赖
pip install stable-baselines3 tensorboard gymnasium

# PowerZoo环境（可选）
# 如果需要使用PowerZoo环境，请确保PowerZoo已正确安装
```

### 文件结构
```
PowerZoo/
├── examples/single_agent/
│   ├── powerzoo_llm/
│   │   └── enhanced_tensorboard_callback.py  # 增强回调类
│   ├── train_with_enhanced_callback.py       # 通用训练脚本
│   └── README_enhanced_callback.md           # 本文档
├── configs/single_agent_cfgs/                # PowerZoo配置文件
│   ├── ppo.yaml
│   ├── dqn.yaml
│   └── ...
└── logs/                                     # 日志输出目录
```

## 使用方法

### 1. 基础使用

```python
from enhanced_tensorboard_callback import create_enhanced_callback
from stable_baselines3 import PPO
import gymnasium as gym

# 创建环境
env = gym.make('CartPole-v1')

# 创建模型
model = PPO('MlpPolicy', env, verbose=1)

# 创建增强回调
callback = create_enhanced_callback(
    log_dir='./logs/my_experiment',
    env_type='gym',  # 或 'powerzoo', 'auto'
    save_freq=1000,
    enable_powerzoo_logging=False
)

# 开始训练
model.learn(total_timesteps=10000, callback=callback)
```

### 2. PowerZoo环境使用

```python
# 使用配置文件训练PowerZoo环境
python train_with_enhanced_callback.py \
    --env-type powerzoo \
    --config ppo.yaml \
    --algorithm ppo \
    --total-timesteps 50000 \
    --log-dir ./logs/powerzoo_experiment \
    --save-freq 2000
```

### 3. Gym环境使用

```python
# 使用Gym环境训练
python train_with_enhanced_callback.py \
    --env-type gym \
    --gym-env CartPole-v1 \
    --algorithm ppo \
    --total-timesteps 20000 \
    --log-dir ./logs/gym_experiment \
    --save-freq 1000
```

### 4. 多算法支持

```bash
# PPO算法
python train_with_enhanced_callback.py --algorithm ppo --gym-env CartPole-v1

# DQN算法
python train_with_enhanced_callback.py --algorithm dqn --gym-env CartPole-v1

# SAC算法（连续控制）
python train_with_enhanced_callback.py --algorithm sac --gym-env Pendulum-v1

# A2C算法
python train_with_enhanced_callback.py --algorithm a2c --gym-env CartPole-v1
```

## 配置参数

### EnhancedTensorBoardCallback 参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `log_dir` | str | 必需 | 日志保存目录 |
| `save_freq` | int | 1000 | 模型保存频率（步数） |
| `name_prefix` | str | "enhanced_model" | 模型文件名前缀 |
| `buffer_size` | int | 10000 | 系统日志缓冲区大小 |
| `save_interval` | int | 100 | 系统日志保存间隔 |
| `enable_powerzoo_logging` | bool | True | 是否启用PowerZoo日志 |
| `env_type` | str | "auto" | 环境类型（auto/gym/powerzoo） |
| `verbose` | int | 1 | 详细程度 |

### 训练脚本参数

| 参数 | 描述 | 示例 |
|------|------|------|
| `--env-type` | 环境类型 | gym, powerzoo |
| `--gym-env` | Gym环境名称 | CartPole-v1, Pendulum-v1 |
| `--config` | PowerZoo配置文件 | ppo.yaml, dqn.yaml |
| `--algorithm` | 强化学习算法 | ppo, dqn, sac, a2c |
| `--total-timesteps` | 总训练步数 | 10000, 50000 |
| `--log-dir` | 日志目录 | ./logs/experiment |
| `--save-freq` | 保存频率 | 1000, 2000 |
| `--verbose` | 详细程度 | 0, 1, 2 |

## 输出文件结构

训练完成后，会在指定的日志目录中生成以下文件：

```
logs/experiment_20250806_123456/
├── models/                          # 模型文件
│   ├── enhanced_model_1000_steps.zip
│   ├── enhanced_model_2000_steps.zip
│   ├── enhanced_model_best.zip
│   └── ...
├── tensorboard_logs/                # TensorBoard日志
│   ├── events.out.tfevents.*
│   └── ...
├── system_logs/                     # 系统日志（如果启用）
│   ├── system_metrics.json
│   └── ...
└── powerzoo_logs/                   # PowerZoo日志（如果启用）
    ├── training_log.json
    └── ...
```

## TensorBoard可视化

启动TensorBoard查看训练日志：

```bash
# 查看特定实验
tensorboard --logdir ./logs/experiment_20250806_123456 --port 6006

# 查看所有实验
tensorboard --logdir ./logs --port 6006

# 指定主机（用于远程访问）
tensorboard --logdir ./logs --port 6006 --host 0.0.0.0
```

然后在浏览器中访问 `http://localhost:6006`

## 高级功能

### 1. 自定义环境检测

```python
# 手动指定环境类型
callback = create_enhanced_callback(
    log_dir='./logs',
    env_type='powerzoo',  # 明确指定
    enable_powerzoo_logging=True
)

# 自动检测环境类型
callback = create_enhanced_callback(
    log_dir='./logs',
    env_type='auto',  # 自动检测
)
```

### 2. 性能监控

回调类会自动记录以下性能指标：
- 训练速度（FPS）
- 内存使用情况
- GPU利用率（如果可用）
- 最佳奖励追踪
- 模型保存统计

### 3. 错误处理

回调类具有健壮的错误处理机制：
- 自动检测缺失的依赖
- 优雅降级（如果某些功能不可用）
- 详细的错误日志记录

## 故障排除

### 常见问题

1. **PowerZoo环境不可用**
   ```
   解决方案：确保PowerZoo已正确安装，或使用env_type='gym'
   ```

2. **配置文件未找到**
   ```
   解决方案：检查配置文件路径，确保文件存在于configs/single_agent_cfgs/目录中
   ```

3. **TensorBoard无法启动**
   ```
   解决方案：检查端口是否被占用，尝试使用不同的端口号
   ```

4. **模型保存失败**
   ```
   解决方案：检查磁盘空间和写入权限
   ```

### 调试模式

启用详细日志记录：

```python
callback = create_enhanced_callback(
    log_dir='./logs',
    verbose=2  # 最详细的日志
)
```

## 扩展和自定义

### 添加自定义指标

```python
class CustomEnhancedCallback(EnhancedTensorBoardCallback):
    def _on_step(self) -> bool:
        # 调用父类方法
        result = super()._on_step()
        
        # 添加自定义指标
        if self.n_calls % 100 == 0:
            custom_metric = self.compute_custom_metric()
            self.logger.record('custom/my_metric', custom_metric)
        
        return result
    
    def compute_custom_metric(self):
        # 实现自定义指标计算
        return 42.0
```

### 支持新环境类型

```python
def _detect_environment_type(self) -> str:
    # 添加新的环境检测逻辑
    if hasattr(self.training_env, 'my_custom_attribute'):
        return 'my_custom_env'
    
    # 调用原始检测逻辑
    return super()._detect_environment_type()
```

## 最佳实践

1. **合理设置保存频率**: 根据训练时长调整save_freq，避免过于频繁的保存
2. **监控磁盘空间**: 长时间训练会产生大量日志文件
3. **使用有意义的实验名称**: 便于后续分析和比较
4. **定期清理旧日志**: 避免磁盘空间不足
5. **备份重要模型**: 保存训练好的最佳模型

## 版本历史

- **v1.0**: 初始版本，支持PowerZoo LLM环境
- **v2.0**: 重构为通用回调，支持多种环境类型
- **v2.1**: 修复日志记录器属性冲突问题
- **v2.2**: 改进配置文件处理和错误处理

## 贡献

欢迎提交问题报告和功能请求！请确保：
1. 提供详细的错误信息和复现步骤
2. 包含环境信息（Python版本、依赖版本等）
3. 遵循代码风格指南

## 许可证

本项目遵循PowerZoo项目的许可证条款。