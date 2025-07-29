# HAPPO PowerZoo PV 训练指南

本指南介绍如何使用HAPPO算法在PowerZoo环境中训练光伏系统的多智能体控制策略。

## 📋 目录

1. [环境配置](#环境配置)
2. [快速开始](#快速开始)
3. [训练配置](#训练配置)
4. [启动训练](#启动训练)
5. [监控和分析](#监控和分析)
6. [故障排除](#故障排除)

## 🔧 环境配置

### 前置要求

- Python 3.10+
- CUDA兼容的GPU（推荐）
- 足够的内存（≥16GB推荐）

### 依赖安装

```bash
# 安装conda环境
conda env create -f environment.yml
conda activate ele

# 安装项目包
pip install -e .
```

### 验证安装

```bash
# 检查环境是否正确配置
python -c "from envs.power_envs.powerzoo_llm.env import PowerZooLLMEnv; print('✅ 环境配置正确')"
```

## 🚀 快速开始

### 方法一：使用快速启动脚本

```bash
# 直接启动训练（使用默认配置）
./scripts/quick_train_happo_pv.sh
```

### 方法二：使用完整训练脚本

```bash
# 启动带监控的训练
python scripts/train_happo_powerzoo_pv.py \
  --exp_name "my_experiment" \
  --threads 32 \
  --gpu 0
```

### 方法三：使用原生train.py

```bash
# 手动启动训练
python examples/train.py \
  --algo happo \
  --env powerzoo_llm \
  --exp_name "happo_pv_test" \
  --load_config configs/algos_cfgs/happo_powerzoo_pv.yaml
```

## ⚙️ 训练配置

### 核心配置文件

**位置**: `configs/algos_cfgs/happo_powerzoo_pv.yaml`

#### 关键参数说明

```yaml
# 训练规模
num_env_steps: 2000000        # 总训练步数（2M步，约1-2天训练）
episode_length: 24            # 每个episode长度（24小时）
n_rollout_threads: 32         # 并行环境数量

# 环境配置
env_args:
  env_name: "34Bus_pv"        # 使用34总线带PV的环境
  pv_control: True            # 启用PV控制
  
# 约束感知训练
power_system:
  voltage_violation_penalty: 100.0    # 电压违规惩罚
  power_loss_weight: 10.0            # 功率损失权重
  curriculum_learning: True          # 启用课程学习
```

#### 环境变体

- `34Bus_pv`: 34总线系统 + PV控制（推荐）
- `34Bus_pv_discrete`: 34总线系统 + 离散PV控制
- `13Bus_pv`: 13总线系统 + PV控制（用于快速测试）

### 自定义配置

```bash
# 复制并修改配置文件
cp configs/algos_cfgs/happo_powerzoo_pv.yaml configs/algos_cfgs/my_config.yaml

# 使用自定义配置
python scripts/train_happo_powerzoo_pv.py --config configs/algos_cfgs/my_config.yaml
```

## 🏃 启动训练

### 训练脚本参数

```bash
python scripts/train_happo_powerzoo_pv.py --help
```

**主要参数**:

- `--exp_name`: 实验名称
- `--config`: 配置文件路径
- `--resume`: 恢复训练的检查点
- `--gpu`: GPU设备号
- `--threads`: 并行线程数
- `--dry_run`: 仅显示命令，不执行

### 示例启动命令

```bash
# 基础训练
python scripts/train_happo_powerzoo_pv.py --exp_name "baseline_test"

# 高性能训练
python scripts/train_happo_powerzoo_pv.py \
  --exp_name "high_perf" \
  --threads 64 \
  --gpu 0

# 恢复训练
python scripts/train_happo_powerzoo_pv.py \
  --exp_name "resumed_training" \
  --resume "results/happo_powerzoo_pv/checkpoint.pth"

# 试运行模式
python scripts/train_happo_powerzoo_pv.py \
  --exp_name "test" \
  --dry_run
```

## 📊 监控和分析

### 实时监控

```bash
# 监控训练日志
tail -f results/happo_powerzoo_pv/training_*.log

# 使用TensorBoard（如果启用）
tensorboard --logdir results/happo_powerzoo_pv/
```

### 训练分析

```bash
# 运行收敛分析
python utils/happo_monitor.py --log_dir results/happo_powerzoo_pv/

# 生成训练报告
python utils/happo_monitor.py \
  --log_dir results/happo_powerzoo_pv/ \
  --pattern "training_*.log"
```

### 关键指标

**收敛指标**:
- 平均奖励稳定性
- 电压违规率 < 5%
- 功率损失改善 > 15%
- 策略稳定性

**性能指标**:
- Episode奖励
- 功率损失
- 电压合规率
- PV系统利用率

## 🔍 故障排除

### 常见问题

#### 1. 内存不足

**症状**: `CUDA out of memory` 或 系统内存不足

**解决方案**:
```bash
# 减少并行线程数
--threads 16

# 减少批处理大小
# 在配置文件中修改:
num_mini_batch: 2
buffer_size: 1000
```

#### 2. 训练不收敛

**症状**: 奖励长期波动，无改善趋势

**解决方案**:
```yaml
# 降低学习率
lr: 0.0001
critic_lr: 0.0002

# 增加训练步数
num_env_steps: 5000000

# 启用课程学习
curriculum_learning: True
```

#### 3. GPU不可用

**症状**: CUDA相关错误

**解决方案**:
```bash
# 检查GPU状态
nvidia-smi

# 使用CPU训练（较慢）
# 在配置文件中设置:
cuda: False
```

#### 4. 环境初始化失败

**症状**: OpenDSS或环境相关错误

**解决方案**:
```bash
# 检查DSS文件
ls envs/power_envs/powerzoo_llm/node_systems_with_pv/34Bus/

# 重新初始化环境
python -c "from envs.power_envs.powerzoo_llm.env_register import make_env; make_env('34Bus_pv')"
```

### 性能优化

#### 1. 训练速度优化

```bash
# 使用更多线程（如果硬件支持）
--threads 64

# 启用确定性计算（略慢但可重现）
cuda_deterministic: True

# 禁用不必要的功能
use_render: False
record_node: False
```

#### 2. 内存优化

```yaml
# 配置文件优化
buffer_size: 1000        # 减少缓冲区大小
num_mini_batch: 2        # 减少批次大小
n_rollout_threads: 16    # 减少并行数
```

### 日志分析

#### 检查训练进度

```bash
# 查看最新奖励
grep "average_episode_rewards" results/happo_powerzoo_pv/training_*.log | tail -10

# 检查收敛情况
grep -E "(converged|violation)" results/happo_powerzoo_pv/training_*.log
```

#### 诊断问题

```bash
# 检查错误信息
grep -i "error\|exception\|failed" results/happo_powerzoo_pv/training_*.log

# 检查内存使用
grep -i "memory\|cuda" results/happo_powerzoo_pv/training_*.log
```

## 📈 期望结果

### 成功训练的标志

1. **奖励收敛**: 500K步后奖励趋于稳定
2. **约束满足**: 电压违规率 < 5%
3. **性能改善**: 功率损失相比基线改善 15%+
4. **策略稳定**: 最后10万步内策略变化小

### 典型训练时间

- **13Bus系统**: 2-4小时（测试用）
- **34Bus系统**: 8-16小时（推荐）
- **123Bus系统**: 24-48小时（大规模）

### 资源需求

- **CPU**: 16+ cores推荐
- **内存**: 16GB+推荐
- **GPU**: 8GB+ VRAM推荐
- **存储**: 10GB+用于结果存储

## 📝 最佳实践

1. **开始前**: 先用小系统（13Bus）验证配置
2. **监控**: 定期检查训练日志和收敛情况
3. **保存**: 定期保存检查点防止训练中断
4. **对比**: 保存基线结果用于性能对比
5. **文档**: 记录实验设置和结果

## 🔗 相关资源

- [PowerZoo环境文档](../envs/power_envs/powerzoo_llm/README.md)
- [HAPPO算法说明](../algorithms/happo/README.md)
- [训练配置参考](../configs/algos_cfgs/)
- [结果分析工具](../utils/happo_monitor.py)

---

如有问题，请查看故障排除部分或联系开发团队。