# PowerZoo LLM Logger 使用指南

## 概述

PowerZooLLMLogger 是专门为 PowerZoo LLM 环境设计的日志记录器，相比原始的 PowerZooLogger，增强了对现代电力系统组件（特别是光伏系统和电池储能系统）的监控能力。

## 主要特性

### 1. 增强的奖励追踪

除了基础的总奖励外，PowerZooLLMLogger 详细记录各种奖励组成：
- **功率损耗奖励** (`powerloss_reward`)：反映系统功率损耗的优化情况
- **电压质量奖励** (`voltage_reward`)：反映电压稳定性
- **控制成本奖励** (`ctrl_reward`)：反映控制动作的经济性
- **PV利用率奖励** (`pv_utilization_reward`)：新增，反映光伏系统的利用效率

### 2. 光伏系统专项监控

- **PV输出功率** (`pv_output_kw`)：实时记录各PV系统的有功功率输出
- **PV功率因数** (`pv_power_factor`)：监控PV逆变器的功率因数控制
- **PV利用率** (`pv_utilization`)：计算PV系统的实际利用率与理论最大值的比例

### 3. 电池储能系统监控

- **充电功率** (`battery_charge_kw`)：记录电池充电时的功率
- **放电功率** (`battery_discharge_kw`)：记录电池放电时的功率
- **SOC状态** (`battery_soc`)：追踪电池的荷电状态，对储能调度至关重要

### 4. 电能质量指标

- **电压违规次数** (`voltage_violations`)：统计电压超出允许范围的次数
- **功率损耗百分比**：计算功率损耗占总功率的百分比

### 5. 详细的TensorBoard集成

PowerZooLLMLogger 在 TensorBoard 中组织了多个仪表板：

#### 奖励分解面板 (`train/reward_breakdown`)
可视化各种奖励组成的贡献度，帮助理解奖励函数的平衡性。

#### 功率指标面板 (`train/power_metrics`)
- 有功/无功功率损耗
- 总有功/无功功率
- 功率损耗百分比

#### 可再生能源面板 (`train/renewable_metrics`)
- PV系统输出、功率因数、利用率
- 电池充放电功率、SOC状态

## 使用方法

### 1. 在环境配置中指定Logger

Logger 已经在 `envs/__init__.py` 中注册，训练时会自动使用：

```python
LOGGER_REGISTRY = {
    ...
    "powerzoo_llm": PowerZooLLMLogger,
    ...
}
```

### 2. 查看训练日志

训练过程中，Logger 会在控制台输出关键指标：

```
========== Episode 100 训练指标 ==========
平均总奖励: 85.3241
功率损耗: 125.32 kW (3.45%)
电压违规次数: 2.3
PV利用率: 92.50%
电池平均SOC: 65.80%
==========================================
```

### 3. TensorBoard 可视化

启动 TensorBoard 查看详细的训练曲线：

```bash
tensorboard --logdir=./results/powerzoo_llm/
```

### 4. 评估指标

评估阶段会记录更详细的性能指标：

```
========== 评估结果 (Episode 100) ==========
平均奖励: 87.5432
平均功率损耗: 118.65 kW
平均电压违规: 1.8
平均PV利用率: 94.20%
============================================
```

## 扩展性

PowerZooLLMLogger 的设计便于扩展，如需添加新的监控指标：

1. 在 `init()` 方法中初始化新的追踪变量
2. 在 `per_step()` 方法中更新追踪值
3. 在 `_log_episode_metrics()` 方法中计算统计值并记录到 TensorBoard
4. 在 `_extract_info_value()` 中添加新指标的提取逻辑

## 与环境的集成

确保环境的 `info` 字典返回以下关键信息：
- `power_loss_ratio`, `vol_reward`, `ctrl_reward`：基础奖励组成
- `pv_utilization_reward`：PV利用率奖励
- `pv_output_kw`, `pv_avg_power_factor`, `pv_utilization`：PV系统状态
- `battery_charge_kw`, `battery_discharge_kw`, `battery_avg_soc`：电池系统状态
- `voltage_violation_count`：电压违规统计

## 性能考虑

PowerZooLLMLogger 优化了数据处理流程：
- 使用向量化操作处理多线程数据
- 批量写入 TensorBoard 减少I/O开销
- 仅在 episode 结束时进行统计计算
- 合理的数据聚合减少存储需求

## 故障排查

如果日志记录出现问题：

1. 检查环境是否正确返回所需的 `info` 字段
2. 确认 TensorBoard writer 正确初始化
3. 验证日志目录有写入权限
4. 查看控制台是否有错误信息输出