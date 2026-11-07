# Stackelberg博弈环境 (Stackelberg Game Environment)

## 概述 (Overview)

本环境实现了基于Stackelberg-Nash博弈理论的电力需求响应多智能体强化学习框架，基于论文"Asynchronous multi-agent reinforcement learning-based framework for bi-level noncooperative game-theoretic demand response" (2024)。

该环境模拟电力公司(UC)作为领导者和多个消费者作为跟随者的分层决策过程，支持异步执行和复杂的电力系统交互。

## 核心特性 (Key Features)

- **分层博弈结构**: 电力公司(UC)作为领导者，消费者作为跟随者
- **异步执行机制**: UC和消费者决策的时序关系建模
- **高级奖励函数**: 基于论文方程的UC和消费者效用函数
- **储能系统集成**: 包含充放电状态跟踪的储能系统动态
- **分布式能源管理**: 支持分布式能源资源的削减控制
- **分时分档电价**: 完整的TUTT定价机制实现
- **全面监控系统**: Nash间隙跟踪、收敛分析和可视化
- **多系统支持**: 支持13Bus、34Bus、123Bus电力系统
- **算法兼容性**: 与sn_mappo、happo、mappo等算法兼容

## 项目结构 (Architecture)

```
envs/stackelberg/
├── __init__.py                           # 包初始化文件
├── README.md                             # 本文档
├── README_STACKELBERG.md                 # 详细技术文档
├── OPTIMIZATION_SUMMARY.md              # 优化总结
├── STRUCTURE.md                          # 架构说明
├── stackelberg_vvc_env.py           # 主环境接口
├── test_stackelberg_integration.py      # 集成测试
└── stackelberg_game/                     # 核心游戏逻辑
    ├── __init__.py                       # 子包初始化
    ├── stackelberg_base_env.py           # 核心环境实现
    ├── async_wrapper.py                  # 异步多智能体包装器
    ├── stackelberg_monitor.py            # 监控和可视化
    ├── circuit_adapter.py                # PowerZoo电路接口
    ├── load_aggregator.py                # 智能负荷聚合器
    └── env_factory.py                    # 环境工厂类
```

## 核心组件 (Core Components)

### 1. StackelbergBaseEnv (核心环境)

实现Stackelberg博弈动态的核心环境：

**UC动作空间** (5维):
- 价格信号倍数 (0.5-2.0)
- 需求响应激励 (0-0.5)
- 容量分配 (0-1)
- 储能充放电 (-1到1)
- 分布式能源削减 (0-1)

**消费者动作空间** (2维):
- 负荷调整 (-30%到+10%)
- 分布式能源输出控制 (0-1)

### 2. AsyncMultiAgentWrapper (异步包装器)

管理UC和消费者动作的时序关系：
- UC优先行动 (领导者阶段)
- 消费者基于UC信号响应 (跟随者阶段)
- 支持动作持续性和延迟

### 3. StackelbergMonitor (监控系统)

全面的监控系统，跟踪：
- Nash间隙收敛
- 智能体奖励和行为
- 系统稳定性指标
- 碳排放
- 可视化和日志记录

### 4. LoadAggregator (负荷聚合器)

智能负荷聚合系统：
- 区域聚合：基于电气距离分组
- 优先级聚合：基于负荷重要性分组
- 图聚合：使用网络拓扑聚类
- 自适应聚合：基于负荷模式动态重组

## 快速开始 (Quick Start)

### 环境安装

```bash
# 激活conda环境
conda activate powerzoo

# 验证环境
python -c "from envs.stackelberg import StackelbergVVCEnv; print('环境导入成功')"
```

### 基础使用示例

```python
import numpy as np
from envs.stackelberg import StackelbergVVCEnv

# 创建环境
env = StackelbergVVCEnv(
    system_name="13Bus",
    n_consumer_agents=5,
    case_path="node_systems/13Bus/IEEE13Nodeckt.dss"
)

# 重置环境
obs = env.reset()

# UC阶段 (领导者)
uc_action = {0: np.array([1.2, 0.3, 0.8, 0.1, 0.05])}
obs, rewards, done, infos = env.step(uc_action)

# 消费者阶段 (跟随者)
consumer_actions = {
    1: np.array([-0.15, 0.6]),
    2: np.array([-0.08, 0.4]),
    3: np.array([-0.12, 0.5]),
    4: np.array([-0.20, 0.3]),
    5: np.array([-0.10, 0.7])
}
obs, rewards, done, infos = env.step(consumer_actions)

print(f"UC奖励: {rewards[0]:.3f}")
print(f"消费者奖励: {[f'{rewards[i]:.3f}' for i in range(1, 6)]}")
```

### 训练示例

```python
# 使用SN-MAPPO算法训练
from algorithms.actors.sn_mappo import SN_MAPPO

# 创建环境
env = StackelbergVVCEnv(system_name="13Bus")

# 创建UC智能体
uc_agent = SN_MAPPO(
    obs_space=env.uc_observation_space,
    act_space=env.uc_action_space,
    config=uc_config
)

# 创建消费者智能体
consumer_agents = [
    SN_MAPPO(
        obs_space=env.consumer_observation_space,
        act_space=env.consumer_action_space,
        config=consumer_config
    ) for _ in range(env.n_consumer_agents)
]

# 训练循环
for episode in range(num_episodes):
    obs = env.reset()
    done = False
    
    while not done:
        # UC决策
        uc_action = uc_agent.get_action(obs[0])
        
        # 消费者响应
        consumer_actions = {
            i+1: agent.get_action(obs[i+1]) 
            for i, agent in enumerate(consumer_agents)
        }
        
        # 执行动作
        actions = {0: uc_action, **consumer_actions}
        obs, rewards, done, infos = env.step(actions)
        
        # 更新智能体
        uc_agent.update(obs[0], uc_action, rewards[0])
        for i, agent in enumerate(consumer_agents):
            agent.update(obs[i+1], consumer_actions[i+1], rewards[i+1])
```

## 配置文件 (Configuration)

### 支持的系统配置

1. **13Bus系统** (`configs/envs_cfgs/stackelberg_13bus.yaml`)
   - 5个消费者智能体
   - 区域聚合方式
   - 适合测试和开发

2. **34Bus系统** (`configs/envs_cfgs/stackelberg_34bus.yaml`)
   - 10个消费者智能体
   - 优先级聚合方式
   - 中等规模实验

3. **123Bus系统** (`configs/envs_cfgs/stackelberg_123bus.yaml`)
   - 20个消费者智能体
   - 图聚合方式
   - 大规模实验

### 关键配置参数

```yaml
# 环境基础配置
system_name: "13Bus"
n_consumer_agents: 5
case_path: "node_systems/13Bus/IEEE13Nodeckt.dss"

# 奖励函数权重
reward_weights:
  power_loss_weight: 20.0
  voltage_violation_weight: 50.0
  carbon_penalty_weight: 5.0
  social_welfare_weight: 1.0

# 物理约束
constraints:
  voltage_min: 0.95
  voltage_max: 1.05
  line_loading_max: 1.0
  ess_soc_min: 0.1
  ess_soc_max: 0.9

# 市场参数
market_params:
  base_price: 0.12  # $/kWh
  peak_multiplier: 1.5
  valley_multiplier: 0.5
  tier_thresholds: [500, 1000, 2000]  # kWh
  tier_multipliers: [1.0, 1.2, 1.5]
```

## 数学公式 (Mathematical Formulation)

环境实现了论文中的关键方程：

### UC效用函数 (Equation 5)
```
J_u = C_t^s + C_t^m + C_t^g + C_t^r
```
- C_t^s: 售电收入
- C_t^m: 购电成本  
- C_t^g: DER吸纳利润
- C_t^r: DR灵活性服务成本

### 消费者效用函数 (Equation 20)
```
J_c,i = -(U_i,t^s + U_i,t^c - U_i,t^r)
```
- U_i,t^s: 电费支出
- U_i,t^c: 舒适度损失
- U_i,t^r: DR参与收入

### 储能动态 (Equation 19)
```
E_t+1 = η_s·E_t + η_c·P_c·Δt - P_d·Δt/η_d
```

## 训练命令 (Training Commands)

### 使用原有训练脚本
```bash
# 13Bus系统训练
python examples/train.py --algo sn_mappo --env stackelberg_13bus --exp_name test_13bus

# 34Bus系统训练
python examples/train.py --algo sn_mappo --env stackelberg_34bus --exp_name test_34bus --cuda

# 123Bus系统训练  
python examples/train.py --algo sn_mappo --env stackelberg_123bus --exp_name test_123bus \
    --num_env_steps 50000000 --n_rollout_threads 4
```

### 算法兼容性测试
```bash
# 测试HAPPO算法
python examples/train.py --algo happo --env stackelberg_13bus --exp_name test_happo

# 测试MAPPO算法
python examples/train.py --algo mappo --env stackelberg_13bus --exp_name test_mappo

# 测试DAN-HAPPO算法
python examples/train.py --algo dan_happo --env stackelberg_13bus --exp_name test_dan_happo
```

## 监控和分析 (Monitoring and Analysis)

### 实时监控指标

监控系统提供：
- 实时Nash间隙跟踪
- 智能体奖励和行为分析
- 系统稳定性指标
- 碳排放跟踪
- 收敛分析图表

### 日志和可视化

```python
# 启用详细监控
env = StackelbergVVCEnv(
    system_name="13Bus",
    enable_monitoring=True,
    log_level="DEBUG"
)

# 监控数据保存路径
logs_path = "logs/stackelberg_13bus/"
plots_path = "logs/stackelberg_13bus/plots/"
```

### TensorBoard集成

```bash
# 启动TensorBoard
tensorboard --logdir logs/stackelberg_13bus/tensorboard

# 在浏览器中查看: http://localhost:6006
```

## 测试验证 (Testing and Validation)

### 运行测试套件

```bash
# 运行所有测试
python tests/test_stackelberg_env.py

# 运行集成测试
python envs/stackelberg/test_stackelberg_integration.py

# 验证环境创建
python -c "
from envs.stackelberg import StackelbergVVCEnv
env = StackelbergVVCEnv('13Bus')
print('环境创建成功')
print(f'UC观测空间: {env.uc_observation_space}')
print(f'消费者观测空间: {env.consumer_observation_space}')
"
```

### 性能基准测试

```bash
# 基准测试脚本
python examples/benchmark_stackelberg.py --system 13Bus --episodes 100
```

## 扩展功能 (Extensions)

### 支持的扩展

- **多时间尺度协调**: 日前、日内、实时调度
- **N-1安全约束**: 系统安全性检查
- **碳排放跟踪**: 环境影响评估
- **课程学习**: 渐进式训练策略
- **优先经验回放**: PER算法支持

### 自定义扩展

```python
# 自定义奖励函数
class CustomStackelbergEnv(StackelbergVVCEnv):
    def _calculate_uc_reward(self):
        base_reward = super()._calculate_uc_reward()
        # 添加自定义奖励项
        custom_reward = self._calculate_custom_metric()
        return base_reward + custom_reward
```

## 故障排除 (Troubleshooting)

### 常见问题

1. **环境导入失败**
   ```bash
   # 检查Python路径
   export PYTHONPATH=$PYTHONPATH:/path/to/PowerZoo
   ```

2. **配置文件错误**
   ```bash
   # 验证YAML格式
   python -c "import yaml; yaml.safe_load(open('configs/envs_cfgs/stackelberg_13bus.yaml'))"
   ```

3. **内存不足**
   ```bash
   # 减少并行环境数量
   --n_rollout_threads 2
   ```

4. **训练不稳定**
   ```bash
   # 降低学习率
   --lr 1e-4
   ```

## 开发指南 (Development Guide)

### 代码修改范围

根据项目规范，允许修改的文件范围：
- `envs/stackelberg/` 目录及其子目录
- `configs/envs_cfgs/stackelberg_*.yaml` 配置文件
- `examples/` 目录下的示例文件（保持简洁）

### 开发最佳实践

1. **遵循命名规范**: 使用描述性的变量和函数名
2. **添加文档字符串**: 所有公共方法都应有详细说明
3. **编写单元测试**: 新功能必须包含测试用例
4. **保持向后兼容**: 不破坏现有API接口

## 引用 (Citation)

如果您在研究中使用了本环境，请引用：

```bibtex
@article{nie2024asynchronous,
  title={Asynchronous multi-agent reinforcement learning-based framework for bi-level noncooperative game-theoretic demand response},
  author={Nie, Y. and Liu, J. and Liu, X. and Zhao, Y. and Ren, K. and Chen, C.},
  journal={IEEE Transactions on Smart Grid},
  year={2024},
  publisher={IEEE}
}

@misc{powerzoo_stackelberg,
  title={PowerZoo Stackelberg Game Environment},
  author={Xiaodong Zheng},
  year={2024},
  note={Based on PowerZoo framework}
}
```

## 许可证 (License)

本项目遵循PowerZoo项目的许可证协议。

## 联系方式 (Contact)

- 作者: Xiaodong Zheng
- 邮箱: zxd_xjtu@stu.xjtu.edu.cn
- 项目地址: [PowerZoo-stackelberg-game-env](https://github.com/XJTU-RL/PowerZoo)

---

**注意**: 本环境正在持续开发中，如遇到问题请及时反馈。更多技术细节请参考 `README_STACKELBERG.md` 和 `OPTIMIZATION_SUMMARY.md`。