# PowerZoo 统一日志系统文档

## 概述

PowerZoo项目采用统一的日志管理系统，确保所有训练日志、系统参数、模型检查点等文件都存储在一个清晰的层级结构中。

## 统一目录结构

```
results/
└── {env_name}/                              # 环境名称 (如 powerzoo_llm)
    └── {system_name}/                        # 系统名称 (如 34Bus_pv)
        └── {algorithm}/                      # 算法名称 (如 happo)
            └── {experiment_name}/            # 实验名称
                └── seed-{seed}-{timestamp}/  # 具体运行实例
                    ├── logs/                  # TensorBoard日志
                    │   ├── agent0/           # 各智能体的独立日志
                    │   ├── agent1/
                    │   ├── agent2/
                    │   └── ...
                    ├── models/                # 模型检查点
                    │   ├── checkpoint_episode_5/
                    │   ├── checkpoint_episode_10/
                    │   └── ...
                    ├── plots/                 # 可视化图表
                    ├── eval/                  # 评估结果
                    ├── system_logs/           # 系统日志
                    ├── system_params/         # 系统参数记录（HDF5格式）
                    ├── training_logs/         # 训练过程日志
                    ├── training_config.yaml   # 训练配置
                    ├── config.json           # 详细配置
                    ├── progress.txt          # 训练进度
                    ├── train_info.txt        # 训练信息
                    └── eval_info.txt         # 评估信息
```

## 核心组件

### 1. UnifiedLogManager (unified_logger.py)

统一日志管理器是整个日志系统的核心，负责：
- 创建和维护统一的目录结构
- 提供路径管理接口
- 迁移现有日志文件
- 保存训练配置

主要功能：
```python
# 创建统一日志管理器
log_manager = UnifiedLogManager(
    env_name="powerzoo_llm",
    system_name="34Bus_pv", 
    algorithm="happo",
    experiment_name="pv_aggressive",
    seed=12345,
    existing_run_dir="/path/to/existing/run"  # 可选：使用现有目录
)

# 获取特定子目录路径
models_path = log_manager.get_path("models")
system_logs_path = log_manager.get_path("system_logs")
```

### 2. PowerZooLLMLogger (powerzoo_llm_logger.py)

PowerZoo LLM环境专用Logger，继承自BaseLogger，提供：
- 详细的奖励组成记录
- 物理量追踪（功率、电压等）
- PV系统和电池系统状态监控
- TensorBoard集成

### 3. SystemLogger (system_logger.py)

系统参数记录器，负责记录：
- 电力系统核心参数（电压、功率、损耗等）
- 控制设备状态（电容器、调压器、电池、PV）
- 奖励函数分解
- 系统性能指标
- 使用HDF5格式高效存储时序数据

### 4. PowerZooLoggerAdapter (logger_adapter.py)

日志适配器，连接SystemLogger和BaseLogger：
- 集成系统参数记录
- 提供统一的日志接口
- 线程安全的日志操作
- 性能监控

## 使用方法

### 在训练脚本中使用

```python
from envs.power_envs.powerzoo_llm.logging import PowerZooLLMLogger

# 在runner初始化时
logger = PowerZooLLMLogger(
    args=args,
    algo_args=algo_args,
    env_args=env_args,
    num_agents=num_agents,
    writter=tensorboard_writer,
    run_dir=run_dir  # 传入现有的运行目录
)
```

### 配置系统日志

在环境配置中设置：
```yaml
# 环境配置
enable_system_logging: true  # 启用系统参数记录
log_buffer_size: 5000        # 缓冲区大小
log_save_interval: 100       # 保存间隔（步数）
enable_realtime_log: true    # 实时日志记录
```

## 日志文件说明

### 主要日志文件

1. **training_config.yaml**: 完整的训练配置，包括环境、算法、实验参数
2. **progress.txt**: 训练进度记录，格式：`步数,平均奖励`
3. **train_info.txt**: 详细的训练信息，包括每个episode的统计
4. **eval_info.txt**: 评估结果记录

### 系统参数文件

1. **system_data_{session}.h5**: HDF5格式的系统参数时序数据
   - `/timeseries/voltages`: 电压数据
   - `/timeseries/powers`: 功率数据
   - `/timeseries/devices`: 设备状态
   - `/timeseries/rewards`: 奖励组成
   - `/statistics/episode_summary`: Episode统计

2. **metadata_{session}.yaml**: 会话元数据

3. **final_stats_{session}.json**: 最终统计信息

### TensorBoard日志

位于`logs/`目录下，可使用以下命令查看：
```bash
tensorboard --logdir results/powerzoo_llm/34Bus_pv/happo/experiment_name/seed-*/logs
```

## 数据访问

### 读取HDF5数据

```python
import h5py
import pandas as pd

# 打开HDF5文件
with h5py.File('system_params/system_data_*.h5', 'r') as f:
    # 读取时间序列数据
    timestamps = f['timeseries/timestamps'][:]
    total_rewards = f['timeseries/rewards/total_rewards'][:]
    
    # 创建DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'reward': total_rewards
    })
```

### 导出系统数据

```python
from envs.power_envs.powerzoo_llm.logging import get_system_logger

# 获取系统logger实例
system_logger = get_system_logger()

# 导出为CSV
system_logger.export_data('output.csv', format='csv')

# 导出为JSON
system_logger.export_data('output.json', format='json')
```

## 故障排除

### 问题：日志分散在多个目录

**原因**：多个logger组件独立创建目录
**解决**：确保所有logger使用UnifiedLogManager提供的路径

### 问题：找不到日志文件

**检查步骤**：
1. 确认训练配置中的`logger_path`设置
2. 查看`results/`目录下的层级结构
3. 检查权限问题

### 问题：HDF5文件过大

**解决方案**：
1. 调整`log_save_interval`减少记录频率
2. 设置合适的`compression_level`（0-9）
3. 定期导出并清理历史数据

## 最佳实践

1. **统一使用UnifiedLogManager**：所有日志组件都应通过UnifiedLogManager获取路径
2. **避免硬编码路径**：使用配置文件或参数传递路径
3. **定期备份**：重要的训练结果应定期备份
4. **监控磁盘空间**：长时间训练可能产生大量日志数据
5. **使用压缩**：HDF5文件启用压缩可节省大量空间

## 版本历史

- v1.0.0 (2025-08-07): 初始版本，实现统一日志管理系统
- v1.0.1 (2025-08-07): 修复多重日志记录问题，实现完整的并集合并

## 联系支持

如有问题或建议，请联系开发团队。