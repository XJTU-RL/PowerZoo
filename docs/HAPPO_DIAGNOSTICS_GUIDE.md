# HAPPO算法诊断工具使用指南

## 概述

本诊断工具专门用于PowerZoo项目中HAPPO算法的数值稳定性分析和调试。通过植入的诊断代码，可以实时监控训练过程中的关键指标，帮助定位和解决以下问题：

- Advantage函数恒为0的问题
- Explained Variance为0的问题
- 训练不稳定/波动大的问题
- Policy Loss和Ratio异常的问题

## 快速开始

### 1. 启动训练

诊断功能已经自动集成到训练流程中，无需额外配置：

```bash
# 使用默认配置启动训练
bash examples/multi_agent/launchers/quick_train_powerzoo_pv.sh aggressive powerzoo_llm

# 或使用Python脚本
python examples/multi_agent/scripts/train_powerzoo.py \
    --system powerzoo_llm \
    --pv_plan aggressive \
    --algo happo
```

### 2. 查看TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir results/

# 或指定具体实验目录
tensorboard --logdir results/happo_powerzoo_llm_pv_aggressive_[timestamp]/logs
```

### 3. 监控诊断指标

在TensorBoard中，查看以下新增的诊断标签：

- `gae_debug/*` - GAE计算过程监控
- `advantages_debug/*` - Advantage归一化分析
- `policy_debug/*` - Policy Loss组件分解
- `value_debug/*` - Value Loss误差分析
- `explained_var_debug/*` - Explained Variance详细分解
- `stability/*` - 数值稳定性监控

## 诊断指标详解

### 1. GAE计算诊断 (gae_debug/)

监控GAE (Generalized Advantage Estimation) 计算过程的中间值：

| 指标 | 含义 | 正常范围 | 异常提示 |
|------|------|----------|----------|
| `gae_debug/step_*/delta` | TD误差 | [-1, 1] | 恒为0表示Value预测完美或奖励设计问题 |
| `gae_debug/step_*/gae` | GAE累积值 | [-2, 2] | 恒为0可能是gamma或lambda设置问题 |
| `gae_debug/step_*/reward` | 当前奖励 | 取决于环境 | 检查奖励函数设计 |
| `gae_debug/step_*/value_curr` | 当前价值估计 | 取决于环境 | 与returns差异过大说明Critic学习失败 |
| `gae_debug/step_*/value_next` | 下一步价值估计 | 取决于环境 | 应该有合理的时序关系 |

**诊断建议**：
- 如果delta恒为0：检查奖励函数和Value网络初始化
- 如果gae恒为0：检查gamma和gae_lambda参数设置

### 2. Advantage归一化诊断 (advantages_debug/)

分析Advantage函数归一化前后的统计信息：

| 指标 | 含义 | 正常范围 | 异常提示 |
|------|------|----------|----------|
| `mean_before` | 归一化前均值 | 接近0 | 远离0表示Value函数偏差 |
| `std_before` | 归一化前标准差 | >0.01 | 接近0表示探索不足 |
| `mean_after` | 归一化后均值 | ≈0 | 应该接近0 |
| `std_after` | 归一化后标准差 | ≈1 | 应该接近1 |
| `nan_count` | NaN值数量 | 0 | >0表示数值溢出 |
| `inf_count` | Inf值数量 | 0 | >0表示除0错误 |

**诊断建议**：
- std_before < 0.01：增加探索（提高熵系数）或检查环境奖励
- nan_count > 0：降低学习率，检查网络初始化

### 3. Policy Loss诊断 (policy_debug/)

分解PPO策略损失的各个组件：

| 指标 | 含义 | 正常范围 | 异常提示 |
|------|------|----------|----------|
| `policy_loss` | 策略损失值 | [-0.1, 0] | 过大表示更新过激 |
| `imp_weights_mean` | 重要性权重均值 | [0.8, 1.2] | 偏离1过多表示策略变化过大 |
| `imp_weights_std` | 重要性权重标准差 | <0.5 | 过大表示策略不稳定 |
| `clip_fraction` | 裁剪比例 | <0.2 | >0.3表示更新步长过大 |
| `surr1_mean` | 未裁剪surrogate | - | 用于对比分析 |
| `surr2_mean` | 裁剪后surrogate | - | 应该小于等于surr1 |

**诊断建议**：
- clip_fraction > 0.3：降低学习率或减小clip_range
- imp_weights偏离1：减小更新步长，增加更新次数

### 4. Value Loss诊断 (value_debug/)

分析价值函数的预测误差：

| 指标 | 含义 | 正常范围 | 异常提示 |
|------|------|----------|----------|
| `value_loss` | 价值损失 | <1.0 | 过大表示Critic学习困难 |
| `values_mean` | 预测值均值 | 取决于环境 | 应与returns相近 |
| `returns_mean` | 目标回报均值 | 取决于环境 | 真实的累积奖励 |
| `error_mean` | 预测误差均值 | 接近0 | 偏离0表示系统性偏差 |
| `error_std` | 预测误差标准差 | <returns_std | 过大表示预测不准确 |

**诊断建议**：
- error_std过大：增加Critic网络容量或降低学习率
- values与returns差异大：检查Value网络初始化和学习率

### 5. Explained Variance诊断 (explained_var_debug/)

详细分解解释方差的计算：

| 指标 | 含义 | 正常范围 | 异常提示 |
|------|------|----------|----------|
| `explained_variance` | 解释方差 | >0.5 | <0表示预测比均值还差 |
| `returns_variance` | 回报方差 | >0 | 0表示奖励恒定 |
| `residual_variance` | 残差方差 | <returns_variance | 越小越好 |
| `correlation` | 预测与真实的相关性 | >0.7 | <0.3表示Critic失效 |

**诊断建议**：
- explained_variance < 0.1：Critic网络学习失败，需要调整架构或超参数
- correlation < 0.3：检查Critic输入是否正确，状态表示是否充分

### 6. 数值稳定性监控 (stability/)

实时检测各个张量的数值稳定性：

| 指标 | 含义 | 正常值 | 异常处理 |
|------|------|---------|----------|
| `*/nan_count` | NaN值数量 | 0 | 立即停止训练检查 |
| `*/inf_count` | Inf值数量 | 0 | 检查除0和溢出 |
| `*/nonfinite_ratio` | 非有限值比例 | 0 | >0.01需要介入 |
| `*/mean` | 张量均值 | 有限值 | 监控趋势 |
| `*/std` | 张量标准差 | >0 | 0表示梯度消失 |

## 常见问题诊断流程

### 问题1：Advantage恒为0

1. 查看 `advantages_debug/std_before`
   - 如果接近0：原始Advantage就是0，检查GAE计算
   - 如果>0.01：是归一化后的显示问题

2. 查看 `gae_debug/*/delta`
   - 如果恒为0：Value预测完美匹配或奖励问题
   - 如果不为0：检查GAE累积计算

3. 解决方案：
   - 增加探索（熵系数）
   - 检查奖励函数设计
   - 调整gamma和gae_lambda

### 问题2：Explained Variance为0

1. 查看 `explained_var_debug/correlation`
   - <0.3：Critic完全失效
   - 0.3-0.7：Critic学习不充分

2. 查看 `value_debug/error_std`
   - 过大：预测误差太大
   - 检查values_mean vs returns_mean

3. 解决方案：
   - 降低Critic学习率
   - 增加Critic更新次数
   - 检查状态表示是否充分

### 问题3：训练不稳定

1. 查看 `policy_debug/clip_fraction`
   - >0.3：策略更新过激
   - 查看imp_weights_std

2. 查看 `stability/*/nan_count`
   - >0：数值溢出问题
   - 检查学习率和梯度裁剪

3. 解决方案：
   - 降低学习率
   - 减小clip_range (如0.2→0.1)
   - 增加批次大小
   - 使用梯度裁剪

### 问题4：Policy Loss异常

1. 查看 `policy_debug/imp_weights_mean`
   - 远离1.0：新旧策略差异过大
   - 查看imp_weights_std

2. 查看 `policy_debug/policy_loss`
   - 绝对值过大：损失函数scale问题
   - 符号异常：优势函数问题

3. 解决方案：
   - 调整PPO的clip参数
   - 降低Actor学习率
   - 检查Advantage计算

## 性能影响

诊断功能的性能开销：
- 计算开销：约5-10%额外CPU时间
- 内存开销：<50MB额外内存
- 存储开销：日志文件增加20-30%

## 禁用诊断

如果需要禁用诊断功能（用于正式训练）：

```python
# 在训练脚本开始处添加
from utils import happo_diagnostics
happo_diagnostics.disable_diagnostics()
```

或者在命令行设置环境变量：

```bash
export HAPPO_DIAGNOSTICS_DISABLED=1
python train.py ...
```

## 导出诊断数据

诊断数据可以通过TensorBoard导出为CSV：

```python
# 使用tensorboard提供的工具
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

event_acc = EventAccumulator('path/to/logs')
event_acc.Reload()

# 导出特定标量
scalar_data = event_acc.Scalars('policy_debug/clip_fraction')
```

## 高级用法

### 自定义诊断点

在您的代码中添加自定义诊断：

```python
from utils import happo_diagnostics

# 检查任意张量的稳定性
stability_info = happo_diagnostics.check_tensor_stability(
    your_tensor, 
    "custom_tensor_name"
)

# 添加自定义标量
if happo_diagnostics._global_diagnostics.writer is not None:
    happo_diagnostics._global_diagnostics.writer.add_scalar(
        "custom/metric", 
        value, 
        step
    )
```

### 批量分析

使用提供的分析脚本批量处理多个实验：

```bash
python scripts/analyze_happo_diagnostics.py \
    --logdir results/ \
    --output analysis_report.html
```

## 联系支持

如有问题或需要进一步的诊断支持，请联系：
- 项目维护者：[您的联系方式]
- 问题追踪：[项目Issue页面]

---

最后更新：2024年
版本：1.0