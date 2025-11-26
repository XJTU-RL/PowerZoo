# PV方案配置文件

本目录包含PowerZoo LLM环境的三种PV（光伏）渗透率方案配置。

## 📊 方案对比

| 方案 | 文件名 | PV容量 | 渗透率 | PV系统数量 | 风险等级 |
|------|--------|--------|--------|------------|----------|
| 保守 | smartgrid_conservative.yaml | 720kW | 40.7% | 5个 | 低 |
| 优化 | smartgrid_optimized.yaml | 900kW | 50.8% | 7个 | 中 |
| 激进 | smartgrid_aggressive.yaml | 1080kW | 61% | 9个 | 高 |

## 🎯 方案详情

### 保守方案 (Conservative)
**文件**: `smartgrid_conservative.yaml`

**PV系统配置**:
- 原有系统: PV834 (180kW), PV890 (180kW), PV864 (180kW)
- 新增系统: PV844 (120kW), PV832 (60kW)
- 总容量: 720kW

**特点**:
- ✅ 风险最低，系统稳定性好
- ✅ 易于实施和管理
- ✅ 适合初期部署
- ⚠️ PV利用率相对较低

**奖励权重**:
```yaml
voltage_violation: -1.5
power_loss: -0.6
pv_utilization: 0.2
import_penalty: -0.15
```

### 优化方案 (Optimized)
**文件**: `smartgrid_optimized.yaml`

**PV系统配置**:
- 原有系统: PV834 (180kW), PV890 (180kW), PV864 (180kW)
- 新增系统: PV844 (120kW), PV832 (60kW), PV848 (90kW), PV840 (90kW)
- 总容量: 900kW

**特点**:
- ✅ 平衡风险与收益
- ✅ 良好的经济性
- ✅ 适合标准部署
- ⚠️ 需要更复杂的控制策略

**奖励权重**:
```yaml
voltage_violation: -2.0
power_loss: -0.5
pv_utilization: 0.3
import_penalty: -0.1
```

### 激进方案 (Aggressive)
**文件**: `smartgrid_aggressive.yaml`

**PV系统配置**:
- 原有系统: PV834 (180kW), PV890 (180kW), PV864 (180kW)
- 新增系统: PV844 (150kW), PV832 (90kW), PV848 (120kW), PV840 (120kW), PV822 (60kW)
- 总容量: 1080kW

**特点**:
- ✅ 最大化PV利用
- ✅ 最高的可再生能源渗透率
- ⚠️ 需要高级控制和弃光策略
- ⚠️ 可能出现反向潮流

**奖励权重**:
```yaml
voltage_violation: -3.0
power_loss: -0.4
pv_utilization: 0.4
import_penalty: -0.05
curtailment: -0.2
reverse_flow: -0.5
```

**特殊配置**:
```yaml
enable_voltage_regulation: true
enable_reactive_power_control: true
enable_curtailment: true
max_reverse_flow: 200  # kW
```

## 🔧 使用指南

### 1. 在训练脚本中使用
```python
# PowerZoo专用训练脚本
python train_powerzoo.py --pv_plan optimized

# 通用训练脚本
python train.py \
  --env smartgrid \
  --env_config configs/envs_cfgs/pv_plans/smartgrid_optimized.yaml
```

### 2. 选择建议

**初学者**: 
- 推荐使用**保守方案**
- 系统稳定，易于理解

**标准训练**:
- 推荐使用**优化方案**
- 平衡性能与稳定性

**研究探索**:
- 可尝试**激进方案**
- 探索极限性能

### 3. 参数调优

每个方案的参数都可以根据实际需求调整：

```yaml
# 调整奖励权重
reward_weights:
  voltage_violation: -2.0  # 增加电压稳定性重视程度
  power_loss: -0.5        # 调整功率损耗权重
  pv_utilization: 0.3     # 提高PV利用率奖励

# 调整训练参数
training:
  learning_rate: 5e-4
  batch_size: 2048
  gamma: 0.99
```

## 📈 性能预期

| 指标 | 保守方案 | 优化方案 | 激进方案 |
|------|----------|----------|----------|
| 电压越限率 | <3% | <5% | <8% |
| 功率损耗 | ~4.5% | ~4.0% | ~3.5% |
| PV利用率 | >80% | >85% | >75%* |
| 平均进口功率 | ~620kW | ~470kW | ~320kW |
| 训练难度 | 低 | 中 | 高 |
| 收敛速度 | 快 | 中 | 慢 |

*激进方案由于弃光策略，PV利用率可能较低

## ⚠️ 注意事项

1. **电压稳定性**: 高渗透率方案需要更严格的电压控制
2. **弃光策略**: 激进方案必须启用弃光功能
3. **反向潮流**: 激进方案可能产生反向潮流，需要特殊处理
4. **收敛性**: 渗透率越高，训练收敛越困难

## 📚 相关资源

- [主环境配置](../smartgrid.yaml)
- [DSS网络文件](../../../node_systems/)
- [训练脚本](../../../examples/multi_agent/)