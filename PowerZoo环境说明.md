# PowerZoo环境说明

## 环境概述

PowerZoo是基于OpenDSS的电力系统强化学习环境，采用CRBP架构（电容器Capacitor、调压器Regulator、电池Battery、光伏PV），支持多智能体协同控制和混合动作空间。

## 动作空间

### 架构设计
- **混合动作空间**：支持离散动作和连续动作的组合
- **设备类型**：电容器(C)、调压器(R)、电池(B)、光伏(P)
- **实现类**：`ActionSpace`类封装所有动作空间逻辑

### 具体动作

#### 1. 电容器动作（离散）
- **动作数量**：每个电容器2个动作
- **动作含义**：0=断开，1=投入
- **控制目标**：无功功率补偿

#### 2. 调压器动作（离散）
- **动作数量**：`reg_act_num`个档位（默认33）
- **动作含义**：调压器分接头位置
- **控制目标**：电压调节

#### 3. 电池动作（离散/连续可配置）
- **离散模式**：`bat_act_num`个功率等级
- **连续模式**：`bat_act_num = float('inf')`，动作范围[-1,1]
- **控制目标**：有功功率充放电

#### 4. 光伏动作（离散/连续可配置）
- **启用条件**：`pv_control_enabled = True`
- **连续模式**：每个PV系统2维动作
  - 维度1：有功功率比例 [-1,1] → [0,1]×Pmpp
  - 维度2：功率因数 [-1,1]
- **离散模式**：`pv_act_num`个功率等级

### 动作空间构建
```python
# 离散动作
discrete_actions = [2]*cap_num + [reg_act_num]*reg_num + [bat_act_num]*bat_num + [pv_act_num]*pv_num

# 连续动作维度
continuous_shape = bat_num + pv_num*2  # (电池连续时) + (PV连续时：有功+功率因数)

# 最终空间
if continuous_shape > 0:
    space = Tuple((MultiDiscrete(discrete_actions), Box(-1, 1, (continuous_shape,))))
else:
    space = MultiDiscrete(discrete_actions)
```

## 状态空间

### 观测组件

#### 1. 节点电压 (`bus_voltages`)
- **数据类型**：字典，键为节点名称
- **数值范围**：[0.8, 1.2] p.u.
- **物理含义**：各节点的电压幅值

#### 2. 电容器状态 (`cap_statuses`)
- **数据类型**：离散数组
- **数值范围**：{0, 1}
- **物理含义**：电容器投切状态

#### 3. 调压器状态 (`reg_statuses`)
- **数据类型**：离散数组
- **数值范围**：[0, reg_act_num-1]
- **物理含义**：调压器分接头位置

#### 4. 电池状态 (`bat_statuses`)
- **数据类型**：字典，每个电池2维
- **维度1**：SOC (State of Charge) [0, 1]
- **维度2**：功率输出 [-1, 1]
- **物理含义**：电池荷电状态和功率状态

#### 5. 光伏状态 (`pv_statuses`，可选）
- **启用条件**：`pv_control_enabled = True`
- **数据类型**：字典，每个PV系统2维
- **维度1**：有功功率输出比例 [0, 1]
- **维度2**：功率因数 [-1, 1]

#### 6. 负荷曲线 (`load_profile_t`，可选）
- **启用条件**：`observe_load = True`
- **数据类型**：数组
- **数值范围**：[0, 1]
- **物理含义**：当前时刻的负荷水平

### 观测空间配置
```python
# 扁平化观测（wrap_observation=True）
observation_space = Box(low=combined_low, high=combined_high)

# 字典观测（wrap_observation=False）
observation_space = Dict({
    'bus_voltages': Box(0.8, 1.2, shape=(nnode,)),
    'cap_statuses': MultiDiscrete([2]*cap_num),
    'reg_statuses': MultiDiscrete([reg_act_num]*reg_num),
    'bat_statuses': Dict({bat: Box([0,-1], [1,1]) for bat in bat_names}),
    'pv_statuses': Dict({pv: Box([0,-1], [1,1]) for pv in pv_names})  # 可选
})
```

## 奖励函数

### 设计理念
- **多目标优化**：平衡系统效率、电压质量和控制成本
- **约束感知**：强化电压约束和系统稳定性
- **可配置权重**：支持不同应用场景的权重调整

### 奖励组件

#### 1. 功率损耗奖励 (`powerloss_reward`)
```python
reward = -current_loss_ratio * power_w
```
- **目标**：最小化系统功率损耗
- **权重**：`power_w`（默认10.0）

#### 2. 电压奖励 (`voltage_reward`)
```python
# 标准约束：0.95 ≤ V ≤ 1.05 p.u.
violation = sum(max(0, V-1.05) + max(0, 0.95-V) for V in all_voltages)
reward = -violation
```
- **目标**：维持电压在安全范围内
- **约束**：IEEE标准电压范围

#### 3. 控制奖励 (`ctrl_reward`)
```python
reward = -(cap_w*cap_changes + reg_w*reg_changes + 
          soc_w*soc_errors + dis_w*discharge_errors + 
          pv_w*pv_changes)
```
- **目标**：减少控制动作频繁变化
- **权重配置**：
  - `cap_w`：电容器动作权重
  - `reg_w`：调压器动作权重
  - `soc_w`：电池SOC偏差权重（终端时刻）
  - `dis_w`：电池放电动作权重
  - `pv_w`：光伏控制动作权重

#### 4. 约束感知奖励（可选）
- **增强电压奖励**：渐进式惩罚机制
- **功率平衡奖励**：系统功率平衡优化
- **PV优化奖励**：光伏发电效率优化

### 综合奖励
```python
total_reward = ctrl_reward + voltage_reward + powerloss_reward * 0.1 + 
               power_balance_reward + pv_optimization_reward
```

## 环境配置

### 主要参数
- `max_episode_steps`：仿真时长（小时）
- `reg_act_num`：调压器动作数量（默认33）
- `bat_act_num`：电池动作数量（默认33或inf）
- `pv_control`：是否启用PV控制（默认False）
- `pv_act_num`：PV动作数量（默认inf连续控制）
- `scale`：负荷缩放因子

### 预定义环境
- `13Bus`：IEEE 13节点系统
- `34Bus`：IEEE 34节点系统
- `34Bus_pv`：带PV控制的34节点系统
- `123Bus`：IEEE 123节点系统
- `8500Node`：大规模8500节点系统

### 变体说明
- `_cbat`：连续电池控制
- `_soc`：考虑SOC约束
- `_pv`：启用PV控制
- `_discrete`：离散PV控制
- `_s{scale}`：负荷缩放版本

## 使用示例

### PowerZoo环境

```python
from envs.powerzoo import PowerZooEnv

# 创建基础环境
config = {
	"dss_file": "path/to/ieee13Nodeckt.dss",
	"num_agents": 3,
	"episode_length": 96
}
env = PowerZooEnv(**config)

# 环境交互
obs = env.reset()
if isinstance(env.action_space, list):
	actions = [space.sample() for space in env.action_space]
else:
	actions = env.action_space.sample()
obs, reward, done, info = env.step(actions)
```

### PowerZoo_LLM环境

```python
from envs.smartgrid.base_env import PowerZooEnv

# 创建完整环境
config = {
	"dss_folder_path": "path/to/34Bus_PV_Aggressive",
	"dss_file": "ieee34Mod1_duty.dss",
	"num_agents": 10,
	"episode_length": 96,
	"reward_type": "powerzoo"
}
env = PowerZooEnv(**config)

# 环境交互
obs = env.reset()
if isinstance(env.action_space, list):
	actions = [space.sample() for space in env.action_space]
elif isinstance(env.action_space, dict):
	actions = {key: space.sample() for key, space in env.action_space.items()}
else:
	actions = env.action_space.sample()
obs, reward, done, info = env.step(actions)
```

### 单智能体环境（使用gymnasium）

```python
from envs.smartgrid.single_agent import SingleAgentPowerZooEnv

# 创建单智能体环境
config = {
	"dss_folder_path": "path/to/34Bus_PV_Aggressive",
	"max_episode_steps": 96
}
env = SingleAgentPowerZooEnv(**config)

# 使用Stable-Baselines3训练
from stable_baselines3 import PPO

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```