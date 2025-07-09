# PowerZoo 环境重构说明

## 重构目标

为了使 PowerZoo 更专注于电力强化学习，我们将环境重组为两个主要类别：

## 目录结构

```
envs/
├── power_envs/          # 电力系统环境
│   ├── powerzoo/        # 电力配网电压-无功控制环境
│   ├── dsr/             # 配电网故障恢复环境
│   └── stackelberg/     # Stackelberg 博弈环境
├── other_envs/          # 其他多智能体环境
│   ├── dexhands/        # 机械手环境
│   ├── football/        # 足球环境
│   ├── gym/             # 通用 Gym 环境
│   ├── lag/             # 飞行器环境
│   ├── mamujoco/        # 多智能体 MuJoCo 环境
│   ├── pettingzoo_mpe/  # 多智能体粒子环境
│   ├── smac/            # 星际争霸环境
│   └── smacv2/          # 星际争霸 v2 环境
└── env_wrappers.py      # 环境封装工具
```

## 电力环境 (power_envs)

### powerzoo
- **描述**: 电力配网电压-无功控制环境
- **目标**: 通过调节电容器组和电压调节器来优化配网电压
- **支持系统**: 13Bus, 34Bus, 123Bus, 8500Node
- **特点**: 基于 OpenDSS 进行精确的电力系统仿真

### dsr
- **描述**: 配电网故障恢复环境
- **目标**: 在配电网发生故障后快速恢复电力供应
- **支持系统**: 123Bus (主要测试系统)
- **特点**: 多智能体协作，包括开关控制、负荷管理、分布式发电

### stackelberg
- **描述**: 电力市场 Stackelberg 博弈环境
- **目标**: 研究电力市场中的策略博弈行为
- **特点**: 基于 PowerZoo 扩展，支持多层次决策

## 其他环境 (other_envs)

这些环境保持原有功能，但被归类为"其他环境"，主要用于：
- 算法通用性测试
- 与其他多智能体环境的性能对比
- 向后兼容性

**注意**: 这些环境可能在未来版本中被移除或独立成单独的包。

## 向后兼容性

- 所有现有的算法和训练脚本保持不变
- 环境名称和调用方式完全相同
- 导入路径已自动更新，用户无需修改代码

## 使用示例

```bash
# 电力环境
python examples/train.py --algo happo --env powerzoo --exp_name test_powerzoo
python examples/train.py --algo happo --env dsr --exp_name test_dsr

# 其他环境（向后兼容）
python examples/train.py --algo happo --env smac --exp_name test_smac
```

## 未来规划

1. 专注于电力环境的功能增强
2. 逐步移除或独立其他环境
3. 添加更多电力系统相关的环境和场景
4. 完善电力环境的文档和示例

---

*此重构于 2025-07-09 完成，旨在使 PowerZoo 成为专业的电力强化学习框架。*