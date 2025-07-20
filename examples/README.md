# PowerZoo Examples 使用指南

本目录包含了PowerZoo框架的示例脚本和训练工具，特别是针对DSR（需求侧响应）环境的负荷聚合训练功能。

## 文件说明

- `train_stackelberg.py`: **（新增）** 专用于Stackelberg博弈环境的训练脚本。
- `train.py` - 通用训练脚本
- `train_dsr_aggregation.py` - DSR负荷聚合训练脚本
- `run_dsr_aggregation.sh` - DSR聚合训练的灵活运行脚本
- `main.py` - 主入口脚本
- `auto_train.py` - 自动训练脚本

## 1. Stackelberg 博弈环境训练 (新增)

这是为Stackelberg博弈环境设计的专用训练入口，请使用 `train_stackelberg.py` 脚本。

### 快速开始

使用以下命令启动 `sn_mappo` 算法在Stackelberg环境中的训练。

```bash
# 训练13节点系统 (默认)
python examples/train_stackelberg.py --bus 13 --exp_name "my_13bus_exp"

# 训练34节点系统
python examples/train_stackelberg.py --bus 34 --exp_name "my_34bus_exp"

# 训练123节点系统
python examples/train_stackelberg.py --bus 123 --exp_name "my_123bus_exp"
```

### 参数说明

- `--bus <13|34|123>`: **(必需)** 选择电力系统的节点数。
- `--exp_name <string>`: **(必需)** 为你的实验指定一个唯一的名称，用于日志和模型保存。
- `--load_config <path>`: (可选) 加载一个指定的配置文件来代替默认的 `yaml` 配置。

## 2. DSR负荷聚合训练

### 快速开始

#### 1. 使用Shell脚本（推荐）

```bash
# 查看帮助信息
./run_dsr_aggregation.sh --help

# 小系统快速训练
./run_dsr_aggregation.sh --small --eval

# 大系统完整训练
./run_dsr_aggregation.sh --large --wandb
```

#### 2. 直接使用Python脚本

```bash
# 基础训练
python train_dsr_aggregation.py --system_name 123Bus --use_load_aggregation

# 自定义配置
python train_dsr_aggregation.py \
    --system_name 8500-Node \
    --use_load_aggregation \
    --n_load_agents 50 \
    --load_aggregation_method zone \
    --algorithm_name mappo \
    --num_env_steps 10000000
```

### Shell脚本详细使用

#### 预设场景

| 预设 | 系统 | 聚合 | 智能体数 | 训练步数 | 适用场景 |
|------|------|------|----------|----------|----------|
| `--small` | 13Bus | 否 | - | 100万 | 小规模测试 |
| `--medium` | 123Bus | 是 | 10 | 500万 | 中等规模训练 |
| `--large` | 8500-Node | 是 | 50 | 2000万 | 大规模训练 |
| `--quick` | 默认 | 默认 | 默认 | 10万 | 快速验证 |
| `--benchmark` | 默认 | 默认 | 默认 | 5000万 | 基准测试 |

#### 系统参数

```bash
# 选择电力系统
--system 13Bus          # 13节点系统（小规模）
--system 34Bus          # 34节点系统
--system 123Bus         # 123节点系统（中等规模）
--system 8500-Node      # 8500节点系统（大规模）

# 选择算法
--algorithm mappo       # MAPPO（默认）
--algorithm maddpg      # MADDPG
--algorithm qmix        # QMIX
--algorithm hatd3       # HATD3
```

#### 聚合参数

```bash
# 启用/禁用聚合
--aggregation           # 启用负荷聚合
--no-aggregation        # 禁用负荷聚合

# 设置智能体数量
--agents 20             # 指定20个负荷智能体

# 选择聚合方法
--method zone           # 基于区域聚合（默认）
--method priority       # 基于优先级聚合
--method random         # 随机聚合
```

#### 训练参数

```bash
# 基础参数
--seed 42               # 随机种子
--steps 5000000         # 训练步数
--episode-len 20        # 回合长度
--threads 4             # 并行线程数

# 网络参数
--hidden 256            # 隐藏层大小
--lr 1e-4               # 学习率

# 评估和日志
--eval                  # 启用评估
--wandb                 # 启用wandb日志
--save-interval 50      # 保存间隔
--eval-interval 25      # 评估间隔
```

#### 实用功能

```bash
# 预览命令（不执行）
./run_dsr_aggregation.sh --large --dry-run

# 详细输出
./run_dsr_aggregation.sh --medium --verbose

# 组合使用
./run_dsr_aggregation.sh --system 123Bus --aggregation --agents 15 --method priority --eval --wandb
```

### 使用示例

#### 示例1：小规模快速验证

```bash
# 使用13Bus系统进行快速验证
./run_dsr_aggregation.sh --small --quick --eval
```

这将：
- 使用13Bus系统（无聚合）
- 训练10万步
- 启用评估
- 适合快速验证代码正确性

#### 示例2：中等规模完整训练

```bash
# 使用123Bus系统进行完整训练
./run_dsr_aggregation.sh --medium --wandb --eval
```

这将：
- 使用123Bus系统
- 启用负荷聚合（10个智能体）
- 训练500万步
- 启用wandb日志和评估

#### 示例3：大规模基准测试

```bash
# 使用8500-Node系统进行基准测试
./run_dsr_aggregation.sh --large --benchmark --agents 100 --wandb
```

这将：
- 使用8500-Node系统
- 100个负荷智能体
- 训练5000万步
- 启用wandb日志

#### 示例4：自定义配置

```bash
# 完全自定义的训练配置
./run_dsr_aggregation.sh \
    --system 123Bus \
    --aggregation \
    --agents 20 \
    --method priority \
    --algorithm maddpg \
    --steps 8000000 \
    --hidden 512 \
    --lr 3e-4 \
    --eval \
    --wandb
```

#### 示例5：算法对比实验

```bash
# MAPPO训练
./run_dsr_aggregation.sh --medium --algorithm mappo --experiment mappo_test --wandb

# MADDPG训练
./run_dsr_aggregation.sh --medium --algorithm maddpg --experiment maddpg_test --wandb

# QMIX训练
./run_dsr_aggregation.sh --medium --algorithm qmix --experiment qmix_test --wandb
```

### Python脚本参数说明

如果直接使用`train_dsr_aggregation.py`，支持以下主要参数：

#### 环境参数
- `--env_name`: 环境名称（默认：dsr）
- `--algorithm_name`: 算法名称（默认：mappo）
- `--experiment_name`: 实验名称（默认：dsr_aggregation）

#### DSR特定参数
- `--system_name`: 电力系统选择
- `--use_load_aggregation`: 启用负荷聚合
- `--n_load_agents`: 负荷智能体数量
- `--load_aggregation_method`: 聚合方法

#### 训练参数
- `--seed`: 随机种子
- `--episode_length`: 回合长度
- `--num_env_steps`: 训练步数
- `--n_rollout_threads`: 并行线程数
- `--hidden_size`: 隐藏层大小
- `--lr`: 学习率

#### 评估参数
- `--use_eval`: 启用评估
- `--eval_interval`: 评估间隔
- `--use_wandb`: 启用wandb日志

### 系统要求

1. **Python环境**：Python 3.7+
2. **依赖包**：按照项目根目录的`environment.yml`安装
3. **硬件要求**：
   - 小系统（13Bus）：2GB内存，1个CPU核心
   - 中等系统（123Bus）：4GB内存，2-4个CPU核心
   - 大系统（8500-Node）：8GB+内存，4-8个CPU核心

### 输出文件

训练过程中会生成以下文件：

- `results/`: 训练结果和日志
- `models/`: 保存的模型文件
- `logs/`: 详细的训练日志
- `wandb/`: wandb日志文件（如果启用）

### 故障排除

#### 常见问题

1. **内存不足**
   ```bash
   # 减少并行线程数
   ./run_dsr_aggregation.sh --medium --threads 1
   ```

2. **训练速度慢**
   ```bash
   # 使用GPU加速（如果可用）
   ./run_dsr_aggregation.sh --medium --cuda
   ```

3. **大系统训练失败**
   ```bash
   # 增加聚合程度，减少智能体数量
   ./run_dsr_aggregation.sh --large --agents 20
   ```

4. **查看详细错误信息**
   ```bash
   # 使用verbose模式
   ./run_dsr_aggregation.sh --medium --verbose
   ```

### 进阶使用

#### 批量实验

```bash
#!/bin/bash
# 批量运行不同配置的实验

for system in "13Bus" "123Bus"; do
    for method in "zone" "priority" "random"; do
        ./run_dsr_aggregation.sh \
            --system $system \
            --aggregation \
            --method $method \
            --experiment "${system}_${method}" \
            --wandb
    done
done
```

#### 超参数搜索

```bash
# 不同学习率实验
for lr in "1e-4" "5e-4" "1e-3"; do
    ./run_dsr_aggregation.sh \
        --medium \
        --lr $lr \
        --experiment "lr_${lr}" \
        --wandb
done
```

### 性能优化建议

1. **系统选择**：根据计算资源选择合适的系统规模
2. **聚合策略**：大系统必须使用聚合，小系统可以不用
3. **并行设置**：根据CPU核心数设置合适的线程数
4. **内存管理**：大系统训练时注意内存使用
5. **日志记录**：生产环境建议启用wandb进行实验管理

### 联系方式

如有问题或建议，请联系：
- 作者：Xiaodong Zheng
- 项目地址：PowerZoo框架

---

**注意**：首次运行前请确保已正确安装所有依赖包，并根据你的硬件配置调整相应参数。