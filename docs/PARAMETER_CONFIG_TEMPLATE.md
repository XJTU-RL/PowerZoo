# PowerZoo 参数配置模板

这是PowerZoo项目的完整参数配置模板，包含所有可配置的参数及其说明。

## 配置文件结构

PowerZoo使用YAML格式的配置文件来管理动态参数，使用JSON格式的配置文件来管理静态系统信息。

### 配置文件类型

1. **YAML配置文件** (`configs/exp_cfgs/`): 实验参数、算法参数、训练参数
2. **JSON系统信息** (`configs/sys_cfgs/system_info.json`): 电力系统静态信息
3. **JSON环境信息** (`configs/sys_cfgs/environments_info.json`): 预定义环境配置

```yaml
# 完整的参数配置模板
# 文件位置: configs/exp_cfgs/your_experiment.yaml

# ========================================
# 1. 算法配置 (Algorithm Configuration)
# ========================================
algo_name: "happo"  # 算法名称

# 随机种子配置
seed:
  seed_specify: true  # 是否使用指定的种子
  seed: 12345        # 随机种子值

# 计算设备配置
device:
  cuda: true              # 是否使用CUDA
  cuda_deterministic: true # 是否设置CUDA确定性
  torch_threads: 4        # PyTorch线程数

# ========================================
# 2. 训练配置 (Training Configuration)
# ========================================
train:
  n_rollout_threads: 2     # 并行环境数量
  num_env_steps: 2000000   # 总训练步数
  episode_length: 360      # 每个episode的步数（统一参数）
  log_interval: 1080       # 日志记录间隔（步数）
  eval_interval: 720       # 评估间隔（步数）
  save_interval: 5         # 模型保存间隔（episode数）
  use_valuenorm: true      # 是否使用值函数归一化
  use_linear_lr_decay: false # 是否使用线性学习率衰减
  use_proper_time_limits: true # 是否考虑episode结束时截断
  model_dir: null          # 预训练模型目录（null表示从头训练）

# ========================================
# 3. 评估配置 (Evaluation Configuration)
# ========================================
eval:
  use_eval: true           # 是否启用评估
  n_eval_rollout_threads: 2 # 评估并行环境数量
  eval_episodes: 10        # 每次评估使用的episode数量

# ========================================
# 4. 网络模型配置 (Model Configuration)
# ========================================
model:
  # 网络架构
  hidden_sizes: [128, 128]  # MLP隐藏层大小
  activation_func: "relu"    # 激活函数: sigmoid, tanh, relu, leaky_relu, selu
  use_feature_normalization: true # 是否使用特征归一化
  initialization_method: "orthogonal_" # 参数初始化方法
  gain: 0.2                 # 输出层增益
  
  # 循环网络配置
  use_naive_recurrent_policy: false # 是否使用简单循环网络
  use_recurrent_policy: true # 是否使用循环网络（带记忆）
  recurrent_n: 1            # 循环层数量
  data_chunk_length: 60     # 数据分块长度
  
  # 优化器配置
  lr: 0.0001               # Actor学习率
  critic_lr: 0.0003        # Critic学习率
  opti_eps: 0.00001        # Adam优化器的epsilon
  weight_decay: 0.0        # 权重衰减
  std_x_coef: 1            # 标准差系数x
  std_y_coef: 0.5          # 标准差系数y

# ========================================
# 5. PPO算法参数 (Algorithm Parameters)
# ========================================
algo:
  ppo_epoch: 5             # Actor更新轮数
  critic_epoch: 5          # Critic更新轮数
  use_clipped_value_loss: true # 是否使用裁剪值损失
  clip_param: 0.25         # PPO裁剪参数
  actor_num_mini_batch: 4  # Actor mini-batch数量
  critic_num_mini_batch: 4 # Critic mini-batch数量
  entropy_coef: 0.08       # 熵系数（探索）
  value_loss_coef: 1.0     # 值损失系数
  use_max_grad_norm: true  # 是否裁剪梯度
  max_grad_norm: 3.0       # 最大梯度范数
  use_gae: true            # 是否使用GAE
  gamma: 0.99              # 折扣因子
  gae_lambda: 0.95         # GAE的lambda
  use_huber_loss: true     # 是否使用Huber损失
  use_policy_active_masks: true # 是否使用策略激活掩码
  huber_delta: 10.0        # Huber损失的delta
  action_aggregation: "prod" # 动作聚合方法: prod, mean
  share_param: false       # 是否共享参数
  fixed_order: false       # 是否使用固定优化顺序
  sentivity_order: false   # 是否使用敏感度优化顺序

# ========================================
# 6. 环境基础配置 (Environment Args)
# ========================================
env_args:
  env_name: "34Bus_pv"     # 环境名称
  seed: 123456             # 环境随机种子
  num_steps: 360           # 最大步数（与episode_length统一）
  num_workers: null        # 并行进程数
  use_plot: false          # 是否使用绘图
  do_testing: false        # 是否进行测试
  mode: "single"           # 运行模式: single, parallel, episodic, dss
  useS: false              # 是否使用S矩阵
  big2small: false         # 是否从大到小排序（SHOM算法）
  use_render: false        # 是否渲染
  record_node: false       # 是否记录节点信息
  dss_act: false           # 是否使用DSS自动控制
  
  # LLM增强功能
  llm_enhanced: true       # 是否启用LLM增强
  use_natural_language_obs: false # 是否使用自然语言观察
  use_action_explanation: false   # 是否使用动作解释
  
  # 系统日志配置
  enable_system_logging: true     # 是否启用系统日志
  system_log_dir: "./logs/system_params" # 系统日志目录
  log_buffer_size: 5000    # 日志缓冲区大小
  log_save_interval: 50    # 日志保存间隔（步数）
  enable_realtime_log: true # 是否启用实时日志

# ========================================
# 7. 环境特定配置 (Environment Specific)
# ========================================
environment_specific:
  # 系统配置
  system_name: "34Bus_PV"   # 电力系统名称
  dss_file: "ieee34Mod1_duty.dss" # OpenDSS文件名
  
  # 设备控制配置
  devices:
    # 电容器配置
    capacitors:
      num: 2               # 电容器数量
      action_space: "discrete" # 动作空间类型
      
    # 调压器配置
    regulators:
      num: 2               # 调压器数量
      action_space: "discrete"
      action_num: 33       # 离散动作数量
      
    # 电池配置
    batteries:
      num: 2               # 电池数量
      action_space: "discrete"
      action_num: 33       # 离散动作数量
      
    # 光伏系统配置
    pv_systems:
      num: 3               # 光伏系统数量
      control_enabled: true # 是否启用控制
      action_space: "continuous" # 动作空间类型: continuous, discrete
      action_num: 21       # 离散动作数量（仅离散控制时使用）
      
  # 奖励权重配置（可调节）
  reward_weights:
    power_loss: 10.0       # 功率损耗权重
    capacitor: 0.0303      # 电容器切换权重
    regulator: 0.0303      # 调压器调节权重
    battery_soc: 0.0       # 电池SOC权重
    battery_discharge: 0.303 # 电池放电权重
    pv_control: 0.0606     # 光伏控制权重
    
  # 约束配置
  constraints:
    voltage_min: 0.95      # 最小电压限制 (p.u.)
    voltage_max: 1.05      # 最大电压限制 (p.u.)
    voltage_penalty_scale: 1.0 # 电压违约惩罚缩放
    constraint_aware: true  # 是否启用约束感知训练

# ========================================
# 8. 电力系统训练配置 (Power System Training)
# ========================================
power_system:
  # 约束惩罚
  voltage_violation_penalty: 50.0  # 电压违约惩罚
  power_loss_weight: 5.0           # 功率损耗权重
  control_cost_weight: 0.05        # 控制成本权重
  
  # 课程学习
  curriculum_learning: true        # 是否启用课程学习
  phases:
    - name: "exploration_phase"
      steps: 800000
      voltage_penalty_scale: 0.3
      exploration_bonus: 0.03
      
    - name: "optimization_phase"
      steps: 800000
      voltage_penalty_scale: 0.7
      exploration_bonus: 0.015
      
    - name: "refinement_phase"
      steps: 400000
      voltage_penalty_scale: 1.0
      exploration_bonus: 0.005
  
  # 收敛标准
  convergence:
    voltage_compliance_threshold: 0.95 # 电压合规阈值
    power_loss_improvement: 0.15      # 功率损耗改进目标
    policy_stability_window: 100000   # 策略稳定性窗口
    reward_std_threshold: 0.1         # 奖励标准差阈值

# ========================================
# 9. 日志和监控 (Logging and Monitoring)
# ========================================
logger:
  log_dir: "./results"     # 日志目录
  log_interval: 720        # 日志间隔
  use_tensorboard: true    # 是否使用TensorBoard
  use_wandb: false         # 是否使用Weights & Biases
  save_gifs: false         # 是否保存GIF动画
  
  # 电力系统特定指标
  track_voltage_violations: true # 跟踪电压违约
  track_power_losses: true      # 跟踪功率损耗
  track_control_costs: true     # 跟踪控制成本
  track_pv_utilization: true    # 跟踪光伏利用率

# ========================================
# 10. 模型保存和恢复 (Save and Resume)
# ========================================
save:
  save_dir: "./results/happo_powerzoo_pv" # 保存目录
  experiment_name: "happo_34bus_pv_experiment" # 实验名称

resume:
  use_resume: false        # 是否恢复训练
  resume_path: ""          # 恢复路径

# ========================================
# 11. 调试和可视化 (Debug and Visualization)
# ========================================
debug:
  use_render: false        # 是否渲染
  render_episodes: false   # 是否渲染episodes
  ifi: 0.1                # 界面间隔

render:
  use_render: false        # 是否使用渲染
  render_episodes: 10      # 渲染的episode数量
```

## 参数优先级

参数的优先级从高到低：

1. **命令行参数** - 最高优先级，覆盖所有其他配置
2. **实验配置文件** - `configs/exp_cfgs/`中的文件
3. **环境默认配置** - `configs/envs_cfgs/`中的文件  
4. **算法默认配置** - `configs/algos_cfgs/`中的文件
5. **代码默认值** - 最低优先级，仅作为最后的备选

## 使用建议

1. **创建新实验**：复制此模板并修改所需参数
2. **参数命名统一**：使用`episode_length`而不是`num_steps`或`max_episode_steps`
3. **奖励权重调节**：通过`environment_specific.reward_weights`调整各项奖励权重
4. **约束配置**：通过`environment_specific.constraints`配置电压约束
5. **设备控制**：通过`environment_specific.devices`配置各类设备的控制参数

## 常见配置示例

### 示例1: 离散PV控制
```yaml
environment_specific:
  devices:
    pv_systems:
      control_enabled: true
      action_space: "discrete"
      action_num: 21  # 21个离散等级
```

### 示例2: 连续PV控制
```yaml
environment_specific:
  devices:
    pv_systems:
      control_enabled: true
      action_space: "continuous"
```

### 示例3: 调整奖励权重
```yaml
environment_specific:
  reward_weights:
    power_loss: 20.0      # 增加功率损耗权重
    pv_control: 0.1       # 增加PV控制权重
    battery_discharge: 0.2 # 降低电池放电权重
```

### 示例4: 严格电压约束
```yaml
environment_specific:
  constraints:
    voltage_min: 0.97     # 更严格的最小电压
    voltage_max: 1.03     # 更严格的最大电压
    voltage_penalty_scale: 2.0 # 加倍惩罚
```

## 注意事项

1. `episode_length`和`num_steps`应保持一致
2. `data_chunk_length`必须能被`episode_length`整除
3. 光伏控制需要设置`pv_control: true`并配置相应的动作空间
4. 奖励权重的总和不需要为1，系统会自动处理
5. 并行环境数量(`n_rollout_threads`)应根据系统资源调整