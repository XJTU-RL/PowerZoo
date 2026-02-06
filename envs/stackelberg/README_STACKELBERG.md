# 基于Stackelberg博弈的需求响应环境

本文档详细描述了在PowerZoo中实现的，一个用于电力需求响应的双层非合作Stackelberg-Nash博弈框架。该实现基于论文 "Asynchronous multi-agent reinforcement learning-based framework for bi-level noncooperative game-theoretic demand response"。

## 概述 (Overview)

Stackelberg环境模型模拟了电力公司（UC）作为领导者与多个消费者作为跟随者之间的分层决策框架。UC设定价格信号和需求响应（DR）激励，而消费者则通过调整其负荷、分布式能源（DER）输出和储能使用情况来响应。

### 核心特性 (Key Features)

1.  **分层决策 (Hierarchical Decision Making)**: UC首先行动，消费者根据UC的信号进行响应。
2.  **异步执行 (Asynchronous Execution)**: 智能体之间的时间关系被明确建模。
3.  **多系统支持 (Multi-System Support)**: 支持13节点、34节点和123节点电力系统。
4.  **智能负荷聚合 (Intelligent Load Aggregation)**: 使用区域/优先级/图方法将物理负荷映射到消费者智能体。
5.  **全面监控 (Comprehensive Monitoring)**: 实时跟踪系统指标、收敛性和纳什均衡。
6.  **SN-MAPPO算法 (SN-MAPPO Algorithm)**: 为Stackelberg-Nash博弈设计的专用算法。

## 架构 (Architecture)

### 核心组件 (Core Components)

#### 1. **StackelbergBaseEnv** (`envs/vvc/vvc/stackelberg_base_env.py`)
- 实现Stackelberg博弈机制的基础环境。
- 为了最大的灵活性，不继承自现有的 `Env` 类。
- 管理UC与消费者之间的交互和系统状态。

#### 2. **AsyncMultiAgentWrapper** (`envs/vvc/vvc/async_wrapper.py`)
- 处理UC和消费者动作的异步执行。
- 维护动作历史和时间关系。
- 支持可配置的延迟和部分可观察性。

#### 3. **IntelligentLoadAggregator** (`envs/vvc/vvc/load_aggregator.py`)
- 将物理负荷映射到消费者智能体。
- 支持多种聚合方法:
  - **基于区域 (Zone-based)**: 按电气邻近度对负荷进行分组。
  - **基于优先级 (Priority-based)**: 按负荷重要性进行分组。
  - **基于图 (Graph-based)**: 使用网络拓扑进行聚类。
  - **自适应 (Adaptive)**: 基于负荷模式进行动态重聚合。

#### 4. **StackelbergMonitor** (`envs/vvc/vvc/stackelberg_monitor.py`)
- 全面的监控和可视化系统。
- 跟踪收敛性、纳什均衡和系统指标。
- 支持TensorBoard集成。
- 生成图表和分析报告。

#### 5. **SN-MAPPO算法** (`algorithms/actors/sn_mappo.py`)
- 针对Stackelberg-Nash博弈的MAPPO扩展。
- UC和消费者使用独立的学习率。
- 最佳响应跟踪和对手建模。
- 纳什均衡求解。

## 配置 (Configuration)

### 环境配置

提供了三个特定于系统的配置：

1.  **13节点系统** (`configs/envs_cfgs/stackelberg_13bus.yaml`)
    - 5个消费者智能体
    - 基于区域的聚合
    - 适用于测试和开发

2.  **34节点系统** (`configs/envs_cfgs/stackelberg_34bus.yaml`)
    - 10个消费者智能体
    - 基于优先级的聚合
    - 用于中等规模的实验

3.  **123节点系统** (`configs/envs_cfgs/stackelberg_123bus.yaml`)
    - 20个消费者智能体
    - 基于图的聚合
    - 用于具有区域协调的大规模实验

### 算法配置

SN-MAPPO算法配置 (`configs/algos_cfgs/sn_mappo.yaml`) 包括：
- 分层学习参数
- 纳什均衡求解器设置
- 对手建模配置
- 课程学习阶段

## 使用方法 (Usage)

### 基础训练

```bash
# 在13节点系统上训练
python examples/train_stackelberg.py --env stackelberg_13bus --algo sn_mappo

# 使用CUDA在34节点系统上训练
python examples/train_stackelberg.py --env stackelberg_34bus --algo sn_mappo --cuda

# 在123节点系统上使用自定义设置进行训练
python examples/train_stackelberg.py --env stackelberg_123bus --algo sn_mappo \
    --num_env_steps 50000000 --n_rollout_threads 4
```

### 使用原始 `train.py`

该环境也与原始的PowerZoo训练脚本兼容：

```bash
# 使用通用训练脚本
python examples/train.py --algo sn_mappo --env stackelberg_13bus --exp_name my_experiment
```

### 自定义配置

创建一个结合环境和算法设置的自定义配置文件：

```yaml
# custom_config.yaml
env: stackelberg_34bus
algo: sn_mappo
seed: 42
num_env_steps: 20000000

# 覆盖特定参数
n_consumer_agents_34bus: 15
load_aggregation:
  method: adaptive
  adaptation_threshold: 0.05

# 自定义奖励权重
reward_weights:
  power_loss_weight: 20.0
  carbon_penalty_weight: 5.0
```

然后使用以下命令进行训练：
```bash
python examples/train_stackelberg.py --config custom_config.yaml
```

## 监控与分析 (Monitoring and Analysis)

### 实时监控

环境在训练期间提供全面的监控：

1.  **回合指标**: UC/消费者奖励、社会福利、功率损耗
2.  **收敛跟踪**: 纳什差距、价格稳定性、动作方差
3.  **系统指标**: 电压违规、线路负载、碳排放
4.  **智能体行为**: 价格信号、负荷调整、协调

### 可视化

监控器生成各种图表：
- 回合奖励进程
- 系统性能指标
- 纳什均衡收敛
- 智能体行为热力图

图表保存在 `logs/stackelberg_<system>/plots/` 目录下。

### TensorBoard集成

如果安装了TensorboardX，指标将被记录以进行实时可视化：

```bash
tensorboard --logdir logs/stackelberg_13bus/tensorboard
```

## 测试 (Testing)

运行测试套件以验证安装：

```bash
# 运行所有测试
pytest tests/test_stackelberg_env.py -v

# 运行特定测试
pytest tests/test_stackelberg_env.py::TestStackelbergBaseEnv::test_env_creation -v

# 快速集成测试
python tests/test_stackelberg_env.py
```

## 实现细节 (Implementation Details)

### 动作空间 (Action Spaces)

**UC动作** (连续):
- 价格信号乘数: [0.5, 2.0]
- DR激励: [0.0, 0.5]
- 容量分配: [0.0, 1.0]

**消费者动作** (连续):
- 负荷调整: [-1.0, 1.0] (基础负荷的比例)
- DER输出: [0.0, 1.0] (容量的比例)
- 储能动作: [-1.0, 1.0] (充电/放电)

### 观察空间 (Observation Spaces)

**UC观察**:
- 系统范围内的指标 (负荷、发电、损耗)
- 母线电压和违规情况
- 时间编码
- 碳强度

**消费者观察**:
- UC信号 (价格、DR激励)
- 本地电压和负荷
- 历史动作
- 时间编码

### 奖励结构 (Reward Structure)

**UC奖励**:
- 最小化功率损耗
- 最小化电压违规
- 最大化收益
- 最小化碳排放

**消费者奖励**:
- 最小化电费成本
- 维持舒适度 (最小化负荷变化)
- DR参与激励
- 服务可靠性奖励

## 高级特性 (Advanced Features)

### 多时间尺度协调 (计划中)

该框架支持日前、日内和实时协调：
- 日前: 24小时计划
- 日内: 4小时滚动更新
- 实时: 5分钟调度

### N-1安全约束 (计划中)

未来的增强功能，用于检查紧急情况下的系统安全性：
- 自动识别关键线路
- 安全约束优化
- 紧急额定值考虑

### 区域协调

对于大型系统 (123Bus)，采用分层协调：
- 区域协调员聚合消费者响应
- 减少通信开销
- 提高可扩展性

## 性能优化 (Performance Optimization)

### 计算效率

- 针对大型系统的稀疏矩阵运算
- 并行潮流计算
- 缓存的网络数据
- 批量观察处理

### Memory Management

- Circular buffers for action history
- Periodic garbage collection
- Compressed data logging
- Selective metric tracking

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure PowerZoo is properly installed and PYTHONPATH includes the project root
2. **OpenDSS Errors**: Verify DSS files exist in `envs/vvc/systems/<system>/`
3. **Memory Issues**: Reduce buffer sizes or disable some monitoring metrics for large systems
4. **Convergence Issues**: Adjust learning rates or use curriculum learning

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Citation

If you use this environment in your research, please cite:

```bibtex
@article{nie2024asynchronous,
  title={Asynchronous multi-agent reinforcement learning-based framework for bi-level noncooperative game-theoretic demand response},
  author={Nie, Yongxin and Liu, Jun and others},
  journal={Journal Name},
  year={2024}
}
```

## Future Work

Planned enhancements include:
- Multi-timescale coordinator implementation
- N-1 security constraint checking
- Carbon emission tracking by generation source
- Distributed training support
- Real-world data integration

## Contributors

- Implementation based on the paper by Yongxin Nie, Jun Liu, et al.
- Integrated into PowerZoo framework by the development team

For questions or issues, please open an issue on the PowerZoo GitHub repository.