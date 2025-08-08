# PowerZoo LLM 配置系统说明

## 概述

PowerZoo LLM 环境的配置系统已经完成重构，实现了统一、简洁的配置管理。

## 配置来源优先级

配置参数按以下优先级加载（高优先级覆盖低优先级）：

1. **env_args 参数**（最高优先级）
   - 从 envs_tools.py 传入的直接参数
   - 支持所有环境参数的覆盖

2. **PV场景配置**
   - `powerzoo_llm_pv_plans/` 目录下的 YAML 文件
   - 三种场景：aggressive, conservative, optimized

3. **环境基础配置**
   - `sys_cfgs/environments_info.json` 中的环境配置
   - 包含默认的奖励权重、动作空间等

4. **系统信息**
   - `sys_cfgs/system_info.json` 中的系统参数
   - 包含节点可视化、布局等参数

## 使用方式

### 1. 基础使用

```python
# 在训练脚本中
env_args = {
    'env_name': '34Bus_pv',
    'max_episode_steps': 360,
    'pv_control': True
}
```

### 2. 使用PV场景

```python
# 使用激进PV方案
env_args = {
    'env_name': '34Bus_pv',
    'pv_scenario': 'aggressive'  # 或 'conservative', 'optimized'
}
```

### 3. 自定义参数覆盖

```python
env_args = {
    'env_name': '34Bus_pv',
    'pv_scenario': 'aggressive',
    'max_episode_steps': 500,  # 覆盖默认值
    'pv_act_num': 21,          # 离散控制
    'llm_enhanced': True       # 启用LLM增强
}
```

### 4. 指定DSS文件路径

```python
env_args = {
    'dss_file': './node_systems/34Bus_PV_Aggressive/ieee34Mod1_duty.dss'
}
# 系统会自动解析路径，提取 system_name 和 dss_file
```

## 配置文件结构

### environments_info.json
- 所有环境的基础配置
- 包含各种Bus配置和PV方案
- 奖励权重默认值

### system_info.json
- 系统可视化参数
- 节点布局信息
- 系统描述

### powerzoo_llm_pv_plans/
- aggressive.yaml: 激进方案（1080kW, 61%渗透率）
- conservative.yaml: 保守方案（720kW, 40.7%渗透率）
- optimized.yaml: 优化方案（900kW, 50.8%渗透率）

## 核心模块

### config_loader.py
- 统一的配置加载器
- 整合所有配置来源
- 处理参数优先级和覆盖

### env_register.py
- 简化的配置获取函数
- 使用 config_loader 加载配置
- 向后兼容性保证

### envs_tools.py
- 最小化的修改
- 只传递 env_args 给 powerzoo_llm
- 不影响其他环境

## 主要改进

1. **配置统一**：所有配置通过一个加载器管理
2. **代码简洁**：移除了大量硬编码和重复代码
3. **易于扩展**：添加新配置只需修改配置文件
4. **向后兼容**：保持了与现有系统的兼容性

## 注意事项

- envs_tools.py 的接口保持不变，不影响其他环境
- 所有硬编码配置已移除，统一使用配置文件
- 配置加载器只在 powerzoo_llm 内部使用，不影响系统其他部分