# 单智能体PowerZoo训练示例

本文件夹包含单智能体PowerZoo环境的训练示例和演示代码。

## 文件结构

```
single_agent/
├── __init__.py                 # 模块初始化文件
├── train_single_agent.py       # 基础训练脚本
├── single_agent_example.py     # 环境演示和测试
└── README.md                   # 本说明文件
```

## 快速开始

### 1. 基础环境测试

运行环境演示脚本，验证环境是否正常工作：

```bash
cd /home/zhengxiaodong/exps/PowerZoo
python examples/single_agent/single_agent_example.py
```

这将运行以下测试：
- 基础环境创建和交互测试
- 不同配置模板测试
- 自定义配置测试
- RL算法兼容性测试

### 2. 简单训练测试

运行基础训练脚本：

```bash
python examples/single_agent/train_single_agent.py
```

这将执行一个简单的随机动作测试，验证环境的基本功能。

## 训练脚本说明

### train_single_agent.py

基础训练脚本，包含：
- 环境配置和初始化
- 简单的随机动作测试
- 基础的训练循环框架

**主要功能：**
```python
def simple_random_test():
    """简单的随机动作测试"""
    # 创建配置
    config = SingleAgentConfig(
        circuit_name="13Bus",
        max_episode_steps=10,
        voltage_penalty_weight=1.0,
        power_loss_weight=0.1,
        discharge_penalty_weight=0.5,
        log_level="INFO"
    )
    
    # 创建环境
    env = SingleAgentPowerZooEnv(config=config)
    
    # 运行测试
    for episode in range(2):
        obs = env.reset()
        for step in range(config.max_episode_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
    
    env.close()
```

### single_agent_example.py

环境演示脚本，包含：
- 环境创建和基础交互测试
- 不同配置模板的使用演示
- 自定义配置的创建和测试
- RL算法兼容性验证

## 配置使用

### 预定义配置模板

```python
from powerzoo_llm.single_agent.single_agent_config import TRAINING_CONFIG, TESTING_CONFIG

# 使用训练配置
env = SingleAgentPowerZooEnv(config=TRAINING_CONFIG)

# 使用测试配置
env = SingleAgentPowerZooEnv(config=TESTING_CONFIG)
```

### 自定义配置

```python
from powerzoo_llm.single_agent import SingleAgentConfig

# 创建自定义配置
custom_config = SingleAgentConfig(
    circuit_name="34Bus",
    max_episode_steps=24,
    voltage_penalty_weight=2.0,
    power_loss_weight=0.2,
    discharge_penalty_weight=1.0,
    enable_capacitors=True,
    enable_regulators=True,
    enable_batteries=False,
    enable_pv_systems=True,
    log_level="DEBUG"
)

env = SingleAgentPowerZooEnv(config=custom_config)
```

## 与RL算法集成

### 支持的算法

单智能体PowerZoo环境支持以下类型的RL算法：

1. **离散动作算法**：DQN、Rainbow、PPO、A2C等
2. **策略梯度算法**：REINFORCE、PPO、A3C等
3. **Actor-Critic算法**：A2C、PPO、SAC（需要动作空间适配）等

### 集成示例

```python
# 以PPO为例
import torch
import torch.nn as nn
from stable_baselines3 import PPO

# 创建环境
config = SingleAgentConfig(circuit_name="13Bus")
env = SingleAgentPowerZooEnv(config=config)

# 创建PPO模型
model = PPO("MlpPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=10000)

# 测试模型
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
```

## 性能监控

环境提供了完整的日志记录功能：

```python
# 启用详细日志
config = SingleAgentConfig(
    circuit_name="13Bus",
    log_level="DEBUG"  # 可选：DEBUG, INFO, WARNING, ERROR
)

env = SingleAgentPowerZooEnv(config=config)

# 日志将包含：
# - 环境状态变化
# - 动作执行结果
# - 奖励计算详情
# - 性能指标统计
```

## 故障排除

### 常见问题

1. **导入错误**
   ```
   ModuleNotFoundError: No module named 'powerzoo_llm'
   ```
   **解决方案**：确保在PowerZoo根目录下运行脚本

2. **OpenDSS错误**
   ```
   DSS initialization failed
   ```
   **解决方案**：检查OpenDSS安装和电路文件路径

3. **动作空间错误**
   ```
   Invalid action for MultiDiscrete space
   ```
   **解决方案**：确保动作在有效范围内，使用`env.action_space.sample()`生成有效动作

### 调试技巧

1. **启用详细日志**：设置`log_level="DEBUG"`
2. **检查环境状态**：使用`info`字典获取详细信息
3. **验证配置**：打印配置参数确保正确设置

## 扩展开发

### 添加新的训练脚本

1. 在本文件夹中创建新的Python文件
2. 导入必要的模块：
   ```python
   from powerzoo_llm.single_agent import SingleAgentPowerZooEnv, SingleAgentConfig
   ```
3. 创建配置和环境
4. 实现训练逻辑

### 自定义奖励函数

可以通过配置参数调整奖励函数权重：

```python
config = SingleAgentConfig(
    voltage_penalty_weight=2.0,    # 电压偏差惩罚权重
    power_loss_weight=0.5,         # 功率损耗惩罚权重
    discharge_penalty_weight=1.0   # 放电惩罚权重
)
```

## 相关资源

- [单智能体环境文档](../../envs/power_envs/powerzoo_llm/single_agent/README.md)
- [PowerZoo主文档](../../README.md)
- [配置参数说明](../../envs/power_envs/powerzoo_llm/single_agent/single_agent_config.py)