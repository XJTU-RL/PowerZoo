# 代码审查发现的健壮性问题

本文档汇总了代码审查中发现的非阻断性问题，建议逐步改进。

---

## 🟡 Major问题（影响健壮性）

### M1. Factor计算存在数值溢出风险
**文件**: `runners/on_policy_ha_runner.py` (行231-239)
**问题**: 当`new_actions_logprob - old_actions_logprob`的绝对值过大时，`torch.exp()`将产生inf或0
**建议**: 添加数值裁剪，如`torch.clamp(ratio, 1e-6, 1e6)`

### M2. TwoTSRunner的next_seq_embed使用临时简化方案
**文件**: `runners/two_ts_runner.py` (行232)
**问题**: slow buffer存储的next_seq_embed直接使用当前值，而非下一个interval的嵌入
**影响**: slow layer学习到错误的时序转换关系

### M3. 文件句柄未使用上下文管理器
**文件**: `runners/off_policy_base_runner.py` (行92-94)
**问题**: 如果后续初始化代码抛出异常，文件句柄不会被关闭
**建议**: 使用`with`语句或在`close()`方法中确保关闭

### M4. Off-policy runner缺少完整的训练状态保存
**文件**: `runners/off_policy_base_runner.py` (行699-708)
**问题**: 只保存模型参数，不保存训练状态，无法真正实现断点续训
**建议**: 参考on_policy_base_runner的training_state保存逻辑

### M5. 裸except捕获过于宽泛
**文件**: `envs/env_wrappers.py` (行518-523)
```python
except:
    pass
```
**问题**: 静默吞掉所有异常，难以调试
**建议**: 使用具体异常类型如`except AttributeError as e`

### M6. 观测空间字典与列表混用不一致
**文件**: `envs/stackelberg/stackelberg_powerzoo_env.py` (行139-151, 360-374)
**问题**: 有些地方假设是dict使用`.values()`，有些地方假设是list用索引

### M7. 电池无功功率计算可能除零
**文件**: `envs/smartgrid/circuit_system/circuit.py` (行458-462)
**问题**: 当`batt.pf = 0`时会发生除零错误
**建议**: 添加条件检查`if batt.pf < 1.0 and batt.pf > 0`

---

## 🟢 Minor问题（代码质量）

### m1. 生产代码中过多print语句
**文件**: 多个runner文件
**建议**: 使用logging模块，支持日志级别控制

### m2. 注释语言不一致
**文件**: 全项目
**问题**: 中英文注释混用
**建议**: 统一使用中文或英文

### m3. 未使用的导入
- `envs/env_wrappers.py`: `import torch` 未使用
- `algorithms/actors/hatrpo.py`: 部分导入可移至局部

### m4. 缺少`__all__`导出声明
**文件**: 所有环境模块
**影响**: 不清楚哪些是公开API

### m5. 魔法数字硬编码
**文件**: `algorithms/actors/sn_mappo.py` (行181、360、376)
```python
equilibrium_reg = 0.01 * policy_stability
cooperation_factor = 0.8
```
**建议**: 移至配置参数

### m6. 重复的optimizer.zero_grad()调用
**文件**: `algorithms/actors/dan_happo.py` (行207、212)
**问题**: 连续两次调用`zero_grad()`，第一次多余

### m7. 布尔索引比较冗余
**文件**: `runners/on_policy_base_runner.py`
```python
rnn_states[dones_env == True] = np.zeros(...)
```
**建议**: 直接使用`rnn_states[dones_env]`

---

## 📝 建议优先级

| 优先级 | 问题 | 原因 |
|--------|------|------|
| P2 | M4 Off-policy断点续训 | 长训练需要 |
| P2 | M5 裸except | 调试困难 |
| P2 | M1 数值溢出 | 训练稳定性 |
| P3 | M2-M7 | 代码健壮性 |
| P3 | m1-m7 | 代码质量 |

---

*此文档由代码审查自动生成 - 2026-01-25*
