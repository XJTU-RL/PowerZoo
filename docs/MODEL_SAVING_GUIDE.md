# PowerZoo 模型保存指南

## 概述

PowerZoo项目现在支持多种模型保存策略，确保训练过程中的模型能够被正确保存和恢复。

## 保存策略

### 1. 定期检查点保存

模型会在以下时机自动保存：

- **评估间隔保存**：每 `eval_interval` 个episode保存一次（默认在配置文件中设置）
- **定期检查点**：每 `save_interval` 个episode保存一次（默认10个episode）

### 2. 最佳模型保存

系统会自动跟踪训练过程中的最佳模型：
- 当平均奖励超过历史最佳时，自动保存到 `best_model` 目录
- 最佳模型始终保持最新状态

### 3. 目录结构

训练过程中会生成以下目录结构：

```
results/
└── powerzoo_llm/
    └── 34Bus_pv/
        └── happo/
            └── experiment_name/
                └── seed-xxxxx-timestamp/
                    ├── models/                    # 最新模型文件
                    │   ├── actor_agent0.pt
                    │   ├── actor_agent1.pt
                    │   ├── ...
                    │   ├── critic_agent.pt
                    │   └── value_normalizer.pt
                    ├── checkpoint_episode_10/     # 检查点目录
                    │   ├── actor_agent0.pt
                    │   ├── ...
                    │   └── training_state.pt
                    ├── checkpoint_episode_20/
                    ├── ...
                    └── best_model/               # 最佳模型目录
                        ├── actor_agent0.pt
                        ├── ...
                        └── training_state.pt
```

## 配置参数

在算法配置文件（如 `configs/algos_cfgs/happo.yaml`）中设置：

```yaml
train:
  eval_interval: 50000    # 评估间隔（按步数计算）
  save_interval: 10       # 模型保存间隔（按episode计算）
```

## 训练状态文件

每个保存点都包含 `training_state.pt` 文件，记录：
- `episode`: 当前episode数
- `total_num_steps`: 总训练步数
- `best_reward`: 历史最佳平均奖励

## 恢复训练

要从检查点恢复训练，可以：

1. 使用 `--model_dir` 参数指定检查点目录
2. 系统会自动加载模型权重和训练状态

## 注意事项

1. **存储空间**：频繁保存检查点会占用较多磁盘空间，建议定期清理旧的检查点
2. **保存间隔**：根据训练时长和稳定性调整 `save_interval`
3. **最佳模型**：`best_model` 目录始终保存性能最好的模型，适合用于部署

## 故障排查

如果模型没有正确保存：

1. 检查 `save_interval` 设置是否合理
2. 确认训练是否达到了保存点（至少运行了 `save_interval` 个episode）
3. 检查磁盘空间是否充足
4. 查看训练日志中的保存相关信息