# PowerZoo 调试指南

本指南详细说明如何调试 PowerZoo 项目中的 Python 代码，特别是通过 shell 脚本调用的训练代码。

## 1. VSCode 调试配置说明

我们在 `.vscode/launch.json` 中配置了多个调试选项：

### 1.1 Debug HAPPO PowerZoo PV Training
**用途**: 完整调试 HAPPO PowerZoo PV 训练过程
- 使用与 `quick_train_happo_pv.sh` 相同的参数
- 启用 CUDA 和确定性训练
- 适合调试完整的训练流程

### 1.2 Debug HAPPO PowerZoo PV Training (Short)
**用途**: 快速调试，缩短训练时间
- 禁用 CUDA（CPU 模式，调试更稳定）
- 减少训练步数（1000步）和 episode 长度（5步）
- 适合快速验证代码逻辑和定位问题

### 1.3 Debug Train.py with Custom Args
**用途**: 自定义参数调试
- 空参数列表，可以手动修改
- 适合测试不同的参数组合

## 2. 调试步骤

### 2.1 使用 VSCode 调试器

1. **打开 VSCode**
   ```bash
   cd /home/zhengxiaodong/exps/PowerZoo
   code .
   ```

2. **设置断点**
   - 在 `examples/scripts/train.py` 中设置断点
   - 在具体的算法文件中设置断点（如 `algorithms/actors/happo.py`）
   - 在环境文件中设置断点（如 `envs/power_envs/powerzoo_llm/`）

3. **启动调试**
   - 按 `F5` 或点击调试面板的运行按钮
   - 选择相应的调试配置

### 2.2 命令行调试

如果需要在命令行中调试：

```bash
# 进入项目根目录
cd /home/zhengxiaodong/exps/PowerZoo

# 设置环境变量
export PYTHONPATH="$PWD:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# 使用 pdb 调试
python -m pdb examples/scripts/train.py \
  --algo happo \
  --env powerzoo_llm \
  --exp_name "debug_session" \
  --load_config configs/algos_cfgs/happo_powerzoo_pv.yaml \
  --n_rollout_threads 1 \
  --cuda False \
  --num_env_steps 100
```

## 3. 常见调试场景

### 3.1 调试环境初始化问题

**目标文件**: `envs/power_envs/powerzoo_llm/powerzoo_env.py`

**关键断点位置**:
- `__init__` 方法
- `reset` 方法
- `step` 方法

### 3.2 调试算法训练问题

**目标文件**: `algorithms/actors/happo.py`

**关键断点位置**:
- `train` 方法
- `prep_training` 方法
- `prep_rollout` 方法

### 3.3 调试电路系统问题

**目标文件**: `envs/power_envs/powerzoo_llm/circuit_system/circuit.py`

**关键断点位置**:
- `__cal_edgeWei_busPhase` 方法
- `compile` 方法
- `initialize` 方法

## 4. 调试技巧

### 4.1 环境变量设置

确保正确设置以下环境变量：
```bash
# Python 路径
export PYTHONPATH="/home/zhengxiaodong/exps/PowerZoo:$PYTHONPATH"

# CUDA 设备（如果使用 GPU）
export CUDA_VISIBLE_DEVICES=0

# OpenDSS 相关（如果需要）
export DSS_EXTENSIONS_PATH="/path/to/dss/extensions"
```

### 4.2 日志调试

在代码中添加详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 在关键位置添加日志
logger.debug(f"参数值: {param_value}")
logger.info(f"执行步骤: {step_name}")
logger.error(f"错误信息: {error_msg}")
```

### 4.3 条件断点

在 VSCode 中设置条件断点：
- 右键点击断点 → "编辑断点"
- 添加条件，如 `episode_step > 10`
- 只在满足条件时停止

## 5. 性能调试

### 5.1 使用 cProfile

```bash
python -m cProfile -o profile_output.prof examples/scripts/train.py \
  --algo happo \
  --env powerzoo_llm \
  --num_env_steps 100

# 分析性能
python -c "import pstats; p=pstats.Stats('profile_output.prof'); p.sort_stats('cumulative').print_stats(20)"
```

### 5.2 内存调试

```bash
# 安装 memory_profiler
pip install memory-profiler

# 使用内存分析
python -m memory_profiler examples/scripts/train.py \
  --algo happo \
  --env powerzoo_llm \
  --num_env_steps 100
```

## 6. 常见问题解决

### 6.1 导入错误

**问题**: `ModuleNotFoundError: No module named 'xxx'`

**解决方案**:
1. 检查 `PYTHONPATH` 设置
2. 确认项目根目录在路径中
3. 检查 `__init__.py` 文件是否存在

### 6.2 CUDA 相关错误

**问题**: CUDA 内存不足或设备不可用

**解决方案**:
1. 使用 CPU 模式调试：`--cuda False`
2. 减少批次大小：`--n_rollout_threads 1`
3. 检查 GPU 状态：`nvidia-smi`

### 6.3 OpenDSS 相关错误

**问题**: OpenDSS 初始化失败

**解决方案**:
1. 检查 OpenDSS 安装
2. 验证电路文件路径
3. 检查权限设置

## 7. 调试最佳实践

1. **从简单开始**: 使用短训练配置先验证基本功能
2. **逐步增加复杂度**: 逐步增加训练步数和复杂度
3. **保存调试配置**: 将有效的调试参数保存为配置文件
4. **使用版本控制**: 调试过程中及时提交代码变更
5. **文档记录**: 记录调试过程中发现的问题和解决方案

## 8. 扩展调试配置

如需添加新的调试配置，在 `.vscode/launch.json` 中添加：

```json
{
    "name": "Debug Custom Configuration",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/examples/scripts/train.py",
    "args": [
        "--algo", "your_algo",
        "--env", "your_env",
        "--custom_param", "value"
    ],
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}",
    "env": {
        "PYTHONPATH": "${workspaceFolder}:${env:PYTHONPATH}"
    },
    "justMyCode": false,
    "stopOnEntry": false
}
```

---

**注意**: 调试时建议先使用较小的参数（如短 episode、少步数）来快速定位问题，然后再使用完整参数进行验证。