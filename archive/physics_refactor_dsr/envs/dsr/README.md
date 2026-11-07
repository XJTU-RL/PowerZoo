# DSR (Distribution System Restoration) Environment

配电网恢复（DSR）多智能体强化学习环境，集成到PowerZoo框架中。

## 概述

DSR环境模拟了配电网故障后的恢复过程，多个异构智能体协同工作以恢复供电：
- **开关智能体**：控制线路开关操作
- **PV智能体**：控制光伏发电输出
- **负荷智能体**：管理负荷恢复顺序

## 主要特性

### 1. 多系统支持
- **13Bus系统**：小型测试系统（~15个负荷）
- **34Bus系统**：中型系统（~25个负荷）
- **123Bus系统**：标准测试系统（~85个负荷）
- **8500-Node系统**：大型系统（~1177个负荷）

### 2. 负荷聚合功能
针对大规模系统的可扩展性问题，DSR环境支持负荷聚合：

```python
# 配置示例
env_args:
  use_load_aggregation: True      # 启用负荷聚合
  n_load_agents: 50               # 将所有负荷聚合到50个智能体
  load_aggregation_method: "zone"  # 聚合方法
```

#### 聚合方法
- **zone**：基于网络拓扑的区域聚合（推荐）
- **priority**：基于负荷优先级的聚合
- **random**：随机聚合

#### 自动聚合策略
当不指定`n_load_agents`时，系统会自动选择合适的智能体数量：
- 小系统（≤20负荷）：不聚合，一对一映射
- 中等系统（≤50负荷）：聚合到15个智能体
- 大系统（≤200负荷）：聚合到30个智能体
- 超大系统（>200负荷）：聚合到50个智能体

### 3. 基于OpenDSS的电力系统仿真
- 准确的潮流计算
- 电压和线路过载约束
- 支持三相不平衡系统

### 4. 优先级负荷恢复
- 3级负荷优先级（重要、一般、普通）
- 加权奖励机制
- 考虑恢复顺序的优化

## 使用示例

### 基础训练
```bash
# 训练123Bus系统（默认配置）
python examples/train.py --env_name dsr --algorithm_name mappo

# 训练13Bus系统（小系统，无聚合）
python examples/train_dsr_aggregation.py --system_name 13Bus

# 训练8500-Node系统（大系统，使用聚合）
python examples/train_dsr_aggregation.py --system_name 8500-Node --use_load_aggregation --n_load_agents 50
```

### 自定义配置
```python
from envs.dsr import DSREnv, DSRConfig

# 创建自定义配置
config = DSRConfig(
    system_name="123Bus",
    use_load_aggregation=True,
    n_load_agents=30,
    load_aggregation_method="priority",
    max_episode_steps=20
)

# 创建环境
env = DSREnv({'env_args': config.__dict__})
```

### 测试负荷聚合
```bash
# 运行聚合功能测试
python test_dsr_aggregation.py
```

## 配置文件

预定义的配置文件位于 `configs/envs_cfgs/`：
- `dsr.yaml`：默认123Bus配置
- `dsr_13bus.yaml`：13Bus系统优化配置
- `dsr_8500node.yaml`：8500-Node大系统配置

## 观测和动作空间

### 观测空间
每个智能体的观测包括：
- 时间信息（当前步/最大步）
- 母线电压状态
- 设备状态（线路、负荷）
- 智能体特定信息

对于聚合负荷智能体，额外包括：
- 管理的负荷恢复比例
- 可恢复负荷比例
- 平均优先级信息

### 动作空间
- **开关智能体**：选择操作的线路
- **PV智能体**：设置输出功率等级（0-10）
- **负荷智能体**：断开(0)或恢复(1)管理的负荷

## 奖励机制

综合考虑：
- 负荷恢复率（加权）
- 电压越限惩罚
- 线路过载惩罚
- 潮流不收敛惩罚

## 注意事项

1. **大系统性能**：8500-Node系统建议使用负荷聚合以提高训练效率
2. **内存使用**：大系统可能需要较多内存，建议关闭`record_node`选项
3. **并行训练**：支持多进程并行训练，使用`n_rollout_threads`参数
4. **依赖项**：需要安装OpenDSS和相关PowerZoo依赖

## 扩展和自定义

1. **添加新系统**：在`config.py`中添加系统配置
2. **自定义聚合策略**：继承并重写`_group_loads_for_aggregation`方法
3. **修改奖励函数**：在`_calculate_rewards`中调整权重和计算方式

## 许可证

遵循PowerZoo项目许可证。