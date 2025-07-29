# DAN-HAPPO算法详细文档

## 算法概述

DAN-HAPPO（Dynamic Agent Network - Heterogeneous Agent Proximal Policy Optimization）是一个结合了动态智能体网络（DAN）和异构智能体近端策略优化（HAPPO）的多智能体强化学习算法，专门设计用于配电系统恢复（DSR）问题。

### 核心特性
- **动态智能体网络**：处理智能体数量动态变化和复杂交互
- **异构智能体支持**：支持不同类型智能体的协同决策
- **注意力机制**：有效聚合邻居智能体信息
- **配电系统优化**：专门针对电力系统约束和恢复目标设计

## 核心架构组件

### 1. DAN模块（Dynamic Agent Network）

**文件位置**: `models/base/dan.py`

#### 双编码器架构

```python
# 环境信息编码器
self.env_encoder = nn.Sequential(
    nn.Linear(env_obs_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU()
)

# 智能体交互信息编码器
self.agent_encoder = nn.Sequential(
    nn.Linear(neighbor_obs_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU()
)
```

#### 多头注意力机制

```python
# 多头注意力聚合邻居智能体特征
self.attention = nn.MultiheadAttention(
    embed_dim=hidden_dim,
    num_heads=num_heads,
    dropout=dropout,
    batch_first=True
)
```

#### 特征融合层

```python
# 融合环境特征和智能体特征
self.fusion = nn.Sequential(
    nn.Linear(hidden_dim * 2, hidden_dim),
    nn.LayerNorm(hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(hidden_dim, hidden_dim)
)
```

### 2. HAPPO基础算法

**文件位置**: `algorithms/actors/dan_happo.py`

#### 关键特性
- **异构智能体支持**：不同类型的智能体可以有不同的策略
- **截断代理目标函数**：使用PPO的截断机制保证训练稳定性
- **中心化价值函数**：使用全局信息进行价值估计

#### 核心参数
```python
self.clip_param = 0.2           # PPO截断参数
self.ppo_epoch = 15             # PPO训练轮数
self.entropy_coef = 0.01        # 熵系数
self.dan_hidden_dim = 128       # DAN隐藏层维度
self.dan_num_heads = 4          # 注意力头数
self.env_obs_ratio = 0.6        # 环境观测比例
self.max_neighbors = 5          # 最大邻居数量
```

## 算法工作流程

### 1. 观测处理流程

```python
def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, 
                available_actions=None, neighbor_obs=None, agent_mask=None, deterministic=False):
    if self.use_dan and neighbor_obs is not None:
        # 步骤1: 分离环境观测（占总观测的60%）
        env_obs = obs[..., :self.env_obs_dim]
        
        # 步骤2: 使用DAN编码观测
        with torch.no_grad():
            encoded_obs, attention_weights = self.dan(env_obs, neighbor_obs, agent_mask)
        
        # 步骤3: 使用编码后的观测获取动作
        actions, action_log_probs, rnn_states_actor = self.actor(
            encoded_obs, rnn_states_actor, masks, available_actions, deterministic
        )
    
    # 步骤4: 获取价值估计
    values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
    
    return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
```

### 2. DAN前向传播

```python
def forward(self, env_obs, neighbor_obs, agent_mask=None):
    """
    Args:
        env_obs: 环境观测 [batch_size, env_obs_dim]
        neighbor_obs: 邻居观测 [batch_size, num_neighbors, neighbor_obs_dim]
        agent_mask: 智能体掩码 [batch_size, num_neighbors]
    Returns:
        output_features: 编码特征 [batch_size, hidden_dim]
        attention_weights: 注意力权重 [batch_size, num_heads, 1, num_neighbors]
    """
    
    # 步骤1: 编码环境信息
    env_features = self.env_encoder(env_obs)  # [batch_size, hidden_dim]
    
    # 步骤2: 编码邻居智能体信息
    neighbor_features = self.agent_encoder(neighbor_obs.view(-1, neighbor_obs_dim))
    neighbor_features = neighbor_features.view(batch_size, num_neighbors, hidden_dim)
    
    # 步骤3: 注意力机制聚合邻居特征
    if self.use_attention and num_neighbors > 0:
        query = env_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        key = value = neighbor_features    # [batch_size, num_neighbors, hidden_dim]
        
        # 处理注意力掩码
        key_padding_mask = None
        if agent_mask is not None:
            key_padding_mask = (agent_mask == 0)  # True for invalid, False for valid
            
        attended_features, attention_weights = self.attention(
            query, key, value, key_padding_mask=key_padding_mask
        )
        agent_features = attended_features.squeeze(1)  # [batch_size, hidden_dim]
    else:
        # 使用平均池化
        if agent_mask is not None and num_neighbors > 0:
            masked_features = neighbor_features * agent_mask.unsqueeze(-1)
            valid_count = agent_mask.sum(dim=1, keepdim=True).clamp(min=1)
            agent_features = masked_features.sum(dim=1) / valid_count
        else:
            agent_features = neighbor_features.mean(dim=1) if num_neighbors > 0 else torch.zeros_like(env_features)
    
    # 步骤4: 特征融合
    fused_features = torch.cat([env_features, agent_features], dim=-1)
    encoded_features = self.fusion(fused_features)
    
    # 步骤5: 输出投影
    output_features = self.output_proj(encoded_features)
    
    return output_features, attention_weights
```

### 3. 训练更新流程

```python
def update(self, sample):
    """
    HAPPO训练更新流程
    """
    # 步骤1: 解析训练样本
    (obs_batch, rnn_states_batch, actions_batch, masks_batch, 
     active_masks_batch, old_action_log_probs_batch, adv_targ, 
     available_actions_batch, factor_batch) = sample[:9]
    
    # 提取DAN相关数据
    if len(sample) > 10:
        neighbor_obs_batch = sample[10]
        agent_mask_batch = sample[11]
    else:
        neighbor_obs_batch = None
        agent_mask_batch = None
    
    # 步骤2: 使用DAN评估动作
    values, action_log_probs, dist_entropy = self.evaluate_actions(
        obs_batch, obs_batch, rnn_states_batch, rnn_states_batch,
        actions_batch, masks_batch, available_actions_batch,
        active_masks_batch, neighbor_obs_batch, agent_mask_batch
    )
    
    # 步骤3: 计算重要性采样权重
    imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
    
    # 步骤4: HAPPO截断代理目标函数
    surr1 = imp_weights * adv_targ
    surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
    
    if self.use_policy_active_masks:
        policy_action_loss = (-torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True) 
                             * active_masks_batch).sum() / active_masks_batch.sum()
    else:
        policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
    
    policy_loss = policy_action_loss
    
    # 步骤5: 计算总损失并反向传播
    total_loss = policy_loss - dist_entropy * self.entropy_coef
    
    # 清零梯度
    self.actor_optimizer.zero_grad()
    if self.use_dan:
        self.dan_optimizer.zero_grad()
    
    # 反向传播
    total_loss.backward()
    
    # 梯度裁剪
    if self.use_max_grad_norm:
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        if self.use_dan:
            dan_grad_norm = nn.utils.clip_grad_norm_(self.dan.parameters(), self.max_grad_norm)
    
    # 更新参数
    self.actor_optimizer.step()
    if self.use_dan:
        self.dan_optimizer.step()
    
    return policy_loss, dist_entropy, actor_grad_norm, imp_weights
```

## 配电系统恢复（DSR）应用

### 1. 环境特性

**文件位置**: `envs/dsr/dsr_env.py`

#### 智能体配置
- **95个异构智能体**：
  - 1个开关控制器（Switch Controller）
  - 9个PV智能体（Photovoltaic Agents）
  - 85个负载智能体（Load Agents）

#### Episode设置
- **15步episode**：代表配电系统恢复的时间段
- **目标**：在配电系统故障后最大化供电恢复

#### 系统约束
- **电压限制**：维持电压在安全范围内
- **线路容量**：不超过线路传输容量
- **负载优先级**：优先恢复重要负载

### 2. 观测空间设计

```python
# 观测空间分割
self.env_obs_dim = int(obs_dim * self.env_obs_ratio)      # 环境观测（60%）
self.neighbor_obs_dim = obs_dim                          # 邻居观测（完整维度）

# 环境观测内容：
# - 节点电压
# - 功率流
# - 开关状态
# - 负载状态

# 邻居观测内容：
# - 邻居智能体的完整状态信息
# - 邻居的动作历史
# - 邻居的控制设备状态
```

### 3. 增强特性

#### 渐进式过载惩罚
```python
# 配置参数
args.use_progressive_overload_penalty = True
args.overload_penalty_levels = [1.1, 1.3, 1.5, 2.0]      # 过载比例阈值
args.overload_penalty_weights = [2.0, 5.0, 10.0, 25.0]   # 对应惩罚权重
args.severe_overload_threshold = 1.5                      # 严重过载阈值（150%）
```

#### 增强动作掩码
```python
args.use_enhanced_action_mask = True
# 功能：
# - 防止导致电压越限的动作
# - 防止超过线路容量的动作
# - 防止违反系统拓扑约束的动作
```

#### 提前终止机制
```python
args.terminate_on_severe_overload = True
args.max_consecutive_overloads = 5
# 在严重过载或连续过载时提前终止episode
```

## 数据缓冲区

### DANSharedReplayBuffer

**文件位置**: `utils/dan_buffer.py`

#### 扩展功能
```python
class DANSharedReplayBuffer(OnPolicyActorBuffer):
    def __init__(self, args, obs_space, act_space):
        super().__init__(args, obs_space, act_space)
        
        # DAN特定的缓冲区
        if self.use_neighbor_obs:
            # 邻居观测缓冲区
            self.neighbor_obs = np.zeros(
                (self.episode_length + 1, self.n_rollout_threads, 
                 self.max_neighbors, *self.obs_shape), dtype=np.float32
            )
            
            # 智能体掩码缓冲区
            self.agent_masks = np.ones(
                (self.episode_length + 1, self.n_rollout_threads, 
                 self.max_neighbors), dtype=np.float32
            )
            
            # 注意力权重缓冲区（用于分析）
            self.attention_weights = np.zeros(
                (self.episode_length, self.n_rollout_threads, 
                 self.max_neighbors), dtype=np.float32
            )
```

## 训练配置

### 主要超参数

**文件位置**: `configs/dan_happo_config.py`

```python
# === DAN架构参数 ===
args.use_dan = True                    # 启用DAN架构
args.dan_hidden_dim = 128              # DAN隐藏层维度
args.dan_num_heads = 4                 # 注意力头数
args.dan_use_attention = True          # 使用注意力机制
args.dan_dropout = 0.1                 # Dropout率
args.dan_layer_norm = True             # 使用层归一化

# === DAN观测处理参数 ===
args.env_obs_ratio = 0.6               # 环境观测比例
args.use_neighbor_obs = True           # 使用邻居观测
args.max_neighbors = 5                 # 最大邻居数量

# === DAN训练参数 ===
args.dan_lr = args.lr                  # DAN学习率（与主学习率相同）
args.dan_weight_decay = 1e-5           # DAN权重衰减
args.dan_grad_norm_max_norm = 10.0     # DAN梯度裁剪

# === HAPPO参数 ===
args.algorithm_name = "dan_happo"
args.use_centralized_V = True          # 使用中心化价值函数
args.use_obs_instead_of_state = False  # 使用全局状态而非观测拼接
args.use_popart = True                 # 使用PopArt归一化
args.use_valuenorm = False             # 不使用值归一化
args.use_feature_normalization = True  # 使用特征归一化
args.use_orthogonal = True             # 使用正交初始化

# === 训练参数 ===
args.num_mini_batch = 1                # 小批次数量
args.ppo_epoch = 15                    # PPO训练轮数
args.use_clipped_value_loss = True     # 使用截断价值损失
args.clip_param = 0.2                  # PPO截断参数
args.entropy_coef = 0.01               # 熵系数
args.value_loss_coef = 1               # 价值损失系数
args.use_max_grad_norm = True          # 使用梯度裁剪
args.max_grad_norm = 10.0              # 最大梯度范数

# === 网络架构 ===
args.hidden_size = 128                 # 隐藏层大小（与dan_hidden_dim一致）
args.layer_N = 2                       # 网络层数
args.use_ReLU = True                   # 使用ReLU激活
args.use_common_layer = True           # 使用公共层

# === 实验参数 ===
args.experiment_name = "dan_happo_dsr"
args.seed = 1
args.n_training_threads = 1
args.n_rollout_threads = 8
args.num_env_steps = 2e6               # 总训练步数
args.episode_length = 200              # Episode长度

# === 环境特定参数 ===
args.env_name = "DSR"
args.scenario_name = "dsr_optimized"
```

### 奖励函数设计

```python
# === 改进的奖励函数参数 ===
# 增加过载惩罚权重
args.reward_overload = 5.0             # 从1.0增加到5.0
args.reward_severe_overload = 20.0     # 新增严重过载惩罚
args.severe_overload_threshold = 1.5   # 严重过载阈值（150%）

# 渐进式过载惩罚
args.use_progressive_overload_penalty = True
args.overload_penalty_levels = [1.1, 1.3, 1.5, 2.0]      # 过载比例阈值
args.overload_penalty_weights = [2.0, 5.0, 10.0, 25.0]   # 对应惩罚权重

# 终止条件
args.terminate_on_severe_overload = True
args.max_consecutive_overloads = 5     # 最大连续过载步数
```

## 关键创新点

### 1. 动态智能体网络（DAN）
- **处理动态性**：能够处理智能体数量的动态变化
- **注意力聚合**：通过多头注意力机制有效聚合邻居信息
- **信息分离**：将环境信息和智能体交互信息分别处理

### 2. 异构智能体支持
- **类型多样性**：支持开关控制器、PV智能体、负载智能体等不同类型
- **策略差异化**：不同类型智能体可以有不同的策略参数
- **观测适应性**：支持不同观测和动作空间

### 3. 配电系统专门优化
- **约束处理**：考虑电力系统的物理约束
- **奖励设计**：针对配电系统恢复目标的专门奖励函数
- **安全机制**：渐进式惩罚和提前终止机制

### 4. 训练稳定性
- **梯度裁剪**：防止梯度爆炸
- **层归一化**：提高训练稳定性
- **正交初始化**：改善收敛性能

## 运行方式

### 训练命令
```bash
# 基本训练
python train_dan_happo.py

# 带参数训练
python train_dan_happo.py --experiment_name "my_experiment" --seed 42 --n_rollout_threads 16

# 使用配置文件
python train_dan_happo.py --config configs/dan_happo_config.py
```

### 主要文件结构
```
├── train_dan_happo.py              # 主训练脚本
├── algorithms/actors/dan_happo.py  # DAN-HAPPO算法实现
├── models/base/dan.py              # DAN网络架构
├── configs/dan_happo_config.py     # 配置文件
├── runners/shared/dsr_dan_runner.py # DSR专用运行器
├── utils/dan_buffer.py             # DAN数据缓冲区
└── envs/dsr/dsr_env.py             # DSR环境
```

## 性能特点

### 优势
1. **处理复杂交互**：DAN模块能够有效处理多智能体间的复杂交互
2. **异构智能体支持**：支持不同类型智能体的协同决策
3. **训练稳定性**：HAPPO提供稳定的训练过程
4. **领域适应性**：专门针对配电系统恢复问题优化

### 适用场景
1. **配电系统恢复**：主要应用场景
2. **多智能体协调**：需要复杂协调的多智能体任务
3. **异构系统**：包含不同类型智能体的系统
4. **约束优化**：有强约束条件的优化问题

## 扩展性

### 环境扩展
- 可以适配其他电力系统拓扑（13Bus、34Bus、123Bus、8500Node等）
- 可以扩展到其他需要多智能体协调的领域

### 算法扩展
- 可以与其他多智能体算法结合（如MADDPG、QMIX等）
- 可以添加更多的注意力机制变体
- 可以集成更复杂的图神经网络结构

---

*本文档基于DAN-HAPPO算法的代码实现，详细描述了算法的架构、工作流程和应用特点。*