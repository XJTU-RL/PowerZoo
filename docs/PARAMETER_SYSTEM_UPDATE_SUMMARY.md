# PowerZoo 参数系统更新总结

## 更新概述

参数系统已经重新组织，实现了从配置文件统一管理所有可配置参数的目标。

## 主要改进

### 1. 参数配置文件结构化

#### 1.1 YAML配置文件结构
- **主配置文件**: `configs/exp_cfgs/happo_powerzoo_pv.yaml`
- 新增`environment_specific`部分，包含：
  - 设备控制配置（电容器、调压器、电池、光伏系统）
  - 奖励权重配置（所有权重参数可调）
  - 约束配置（电压限制、惩罚系数）

#### 1.2 系统信息JSON文件
- **文件位置**: `configs/sys_cfgs/system_info.json`
- 包含所有电力系统的静态信息
- 将静态系统信息与动态配置参数分离
- 更容易维护和扩展

### 2. 代码更新

#### 2.1 env_register.py
- 新增`get_info_from_config()`函数，从配置文件读取参数
- 新增`load_system_info()`函数，从JSON文件加载系统信息
- `make_base_env()`函数增加`config_dict`参数
- 保持向后兼容，当没有配置文件时使用_ENV_INFO

#### 2.2 envs_tools.py  
- 更新`make_train_env()`和`make_eval_env()`
- 传递完整配置字典到`make_base_env()`
- 通过`env_args['full_config']`传递配置

#### 2.3 train.py
- 修改配置加载逻辑
- 将完整配置字典添加到`env_args['full_config']`
- 确保环境可以访问所有配置参数

### 3. 参数流程

```
配置文件加载流程：
1. train.py 加载 YAML 配置文件
2. 提取 env_args 并添加 full_config
3. envs_tools.py 传递 config_dict 到 make_base_env()
4. env_register.py 从配置文件读取参数
5. 环境使用配置文件中的参数初始化
```

### 4. 参数优先级

1. **配置文件参数** (最高优先级)
   - `environment_specific`中的参数
   - `env_args`中的参数
   - `train`中的参数

2. **默认值** (最低优先级)
   - _ENV_INFO中的硬编码值
   - 代码中的默认值

## 使用方法

### 1. 修改奖励权重

在`configs/exp_cfgs/happo_powerzoo_pv.yaml`中修改：

```yaml
environment_specific:
  reward_weights:
    power_loss: 20.0       # 修改功率损耗权重
    capacitor: 0.05        # 修改电容器权重
    regulator: 0.05        # 修改调压器权重
    battery_soc: 0.1       # 修改电池SOC权重
    battery_discharge: 0.2 # 修改电池放电权重
    pv_control: 0.1        # 修改光伏控制权重
```

### 2. 修改设备配置

```yaml
environment_specific:
  devices:
    pv_systems:
      control_enabled: true
      action_space: "continuous"  # 或 "discrete"
      action_num: 21  # 离散控制时的动作数
```

### 3. 修改约束配置

```yaml
environment_specific:
  constraints:
    voltage_min: 0.97      # 修改最小电压限制
    voltage_max: 1.03      # 修改最大电压限制
    voltage_penalty_scale: 2.0  # 修改惩罚系数
```

### 4. 添加新的电力系统

在`configs/system_info.json`中添加：

```json
{
  "system_info": {
    "NewSystem": {
      "source_bus": "sourcebus",
      "node_size": 500,
      "shift": 50,
      "show_node_labels": true,
      "description": "New power system description"
    }
  }
}
```

## 向后兼容性

系统保持完全向后兼容：
- 如果没有提供配置文件，将使用_ENV_INFO中的默认值
- 旧的训练脚本无需修改即可继续工作
- 可以逐步迁移到新的配置系统

## 优势

1. **统一管理**: 所有参数在一个配置文件中管理
2. **清晰分离**: 静态系统信息与动态配置参数分离
3. **易于调试**: 参数来源明确，易于追踪
4. **灵活配置**: 可以轻松调整任何参数而无需修改代码
5. **版本控制**: 配置文件可以版本控制，便于实验管理

## 注意事项

1. 确保`episode_length`和`num_steps`保持一致
2. 修改奖励权重时注意保持合理的比例关系
3. 系统信息JSON文件主要存储静态信息，不应频繁修改
4. 配置文件支持YAML和JSON格式，推荐使用YAML

## 文件列表

更新的文件：
- `/configs/exp_cfgs/happo_powerzoo_pv.yaml` - 添加environment_specific部分
- `/configs/system_info.json` - 新建，存储系统静态信息
- `/envs/power_envs/powerzoo_llm/env_register.py` - 支持从配置文件读取参数
- `/utils/envs_tools.py` - 传递配置字典到环境
- `/examples/scripts/train.py` - 添加完整配置传递
- `/docs/PARAMETER_CONFIG_TEMPLATE.md` - 参数配置模板文档
- `/docs/PARAMETER_SYSTEM_UPDATE_SUMMARY.md` - 本文档