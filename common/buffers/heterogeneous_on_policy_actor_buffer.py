# -*- coding: utf-8 -*-
"""
@File      : heterogeneous_on_policy_actor_buffer.py
@Time      : 2025-07-31 15:50
@Author    : Claude & Xiaodong Zheng
@Description: 单智能体异构动作空间兼容的On-Policy Actor Buffer
- 专门设计用于处理单个智能体的各种动作空间类型：
  * 离散动作空间 (Discrete)
  * 连续动作空间 (Box)
  * 多离散动作空间 (MultiDiscrete)
  * 混合动作空间 (Tuple of Discrete/MultiDiscrete + Box)
- 解决HAPPO算法在处理异构智能体时的动作存储形状不一致问题
- 完全兼容原有OnPolicyActorBuffer接口，可直接替换使用
"""

import torch
import numpy as np
from gym.spaces import Discrete, Box, Tuple as SpaceTuple, MultiDiscrete
from utils.trans_tools import _flatten, _sa_cast
from utils.envs_tools import get_shape_from_obs_space


class HeterogeneousOnPolicyActorBuffer:
    """单智能体异构动作空间兼容的On-Policy Actor Buffer"""
    
    # 明确的类型标识，用于运行时识别异构buffer
    _buffer_class_type = 'heterogeneous'

    def __init__(self, args, obs_space, act_space):
        """初始化单智能体异构动作空间Buffer
        
        Args:
            args: (dict) 参数配置
            obs_space: (gym.Space or list) 观测空间
            act_space: (gym.Space) 单个智能体的动作空间（支持Discrete或Box）
        """
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.hidden_sizes = args["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = args["recurrent_n"]

        # 观测空间处理
        obs_shape = get_shape_from_obs_space(obs_space)
        if isinstance(obs_shape[-1], list):
            obs_shape = obs_shape[:1]

        # 观测缓冲区
        self.obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
            dtype=np.float32,
        )

        # RNN状态缓冲区
        self.rnn_states = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # 解析动作空间并初始化动作缓冲区
        self._parse_action_space(act_space)
        self._init_action_buffers()
        
        # 其他缓冲区
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        ) # 记录每个episode中每个thread的有效动作
        self.active_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        ) # 记录每个episode中每个thread的有效动作
        
        self.factor = None
        self.step = 0

    def _parse_action_space(self, act_space):
        """解析单个智能体的动作空间结构"""
        self.action_space = act_space
        
        if isinstance(act_space, Discrete):
            self.action_shape = (1,)  # 离散动作形状
            self.action_type = 'discrete'
            self.action_dim = 1
        elif isinstance(act_space, Box):
            # 处理Box空间的形状，有些Box空间可能是标量(shape=())或一维(shape=(1,))
            if len(act_space.shape) == 0:
                self.action_shape = (1,)
                self.action_dim = 1
            else:
                self.action_shape = act_space.shape
                self.action_dim = act_space.shape[0]
            self.action_type = 'continuous'
        elif isinstance(act_space, (SpaceTuple, tuple)):
            # 混合动作空间：通常是 (MultiDiscrete, Box) 的组合
            self.action_type = 'mixed'
            self._parse_mixed_action_space(act_space)
        elif isinstance(act_space, MultiDiscrete):
            self.action_shape = (len(act_space.nvec),)  # MultiDiscrete动作形状
            self.action_type = 'multi_discrete'
            self.action_dim = len(act_space.nvec)
        else:
            raise NotImplementedError(f"不支持的动作空间类型: {type(act_space)}")
    
    def _parse_mixed_action_space(self, act_space):
        """解析混合动作空间结构"""
        self.mixed_spaces = []
        self.mixed_action_shapes = []
        total_dim = 0
        
        for i, space in enumerate(act_space):
            if isinstance(space, (Discrete, MultiDiscrete)):
                if isinstance(space, Discrete):
                    shape = (1,)
                    dim = 1
                else:  # MultiDiscrete
                    shape = (len(space.nvec),)
                    dim = len(space.nvec)
                self.mixed_spaces.append(('discrete', space, shape))
            elif isinstance(space, Box):
                shape = space.shape
                dim = space.shape[0] if len(space.shape) > 0 else 1
                self.mixed_spaces.append(('continuous', space, shape))
            else:
                raise NotImplementedError(f"混合动作空间中不支持的子空间类型: {type(space)}")
            
            self.mixed_action_shapes.append(shape)
            total_dim += dim
        
        # 混合动作空间的总维度
        self.action_shape = (total_dim,)
        self.action_dim = total_dim
    
    def _init_action_buffers(self):
        """初始化动作缓冲区"""
        if self.action_type == 'mixed':
            # 混合动作空间：为每个子空间创建独立的缓冲区
            self.actions = {} # 记录每个episode中每个thread的动作
            self.available_actions = {} # 记录每个episode中每个thread的可用动作
            
            for i, (space_type, space, shape) in enumerate(self.mixed_spaces):
                # 动作缓冲区
                self.actions[i] = np.zeros(
                    (self.episode_length, self.n_rollout_threads, *shape),
                    dtype=np.float32
                )
                
                # 可用动作缓冲区（仅离散类型需要）
                if space_type == 'discrete':
                    if isinstance(space, Discrete):
                        av_shape = (self.episode_length + 1, self.n_rollout_threads, space.n)
                    else:  # MultiDiscrete
                        av_shape = (self.episode_length + 1, self.n_rollout_threads, sum(space.nvec))
                    self.available_actions[i] = np.ones(av_shape, dtype=np.float32)
                else:
                    self.available_actions[i] = None
        else:
            # 非混合动作空间：保持原有逻辑
            self.actions = np.zeros(
                (self.episode_length, self.n_rollout_threads, *self.action_shape),
                dtype=np.float32
            )
            
            # 可用动作缓冲区
            if self.action_type == 'discrete':
                self.available_actions = np.ones(
                    (self.episode_length + 1, self.n_rollout_threads, self.action_space.n),
                    dtype=np.float32,
                )
            elif self.action_type == 'multi_discrete':
                self.available_actions = np.ones(
                    (self.episode_length + 1, self.n_rollout_threads, sum(self.action_space.nvec)),
                    dtype=np.float32,
                )
            else:
                # 连续动作空间不需要available_actions
                self.available_actions = None

    def insert(
        self,
        obs,
        rnn_states,
        actions,
        action_log_probs,
        masks,
        active_masks=None,
        available_actions=None,
    ):
        """插入数据到Buffer中
        
        Args:
            obs: 观测数据 (n_rollout_threads, obs_dim)
            rnn_states: RNN状态 (n_rollout_threads, recurrent_n, hidden_size)
            actions: 动作数据 (n_rollout_threads, action_dim)
            action_log_probs: 动作对数概率 (n_rollout_threads, 1)
            masks: 掩码 (n_rollout_threads, 1)
            active_masks: 激活掩码 (n_rollout_threads, 1)
            available_actions: 可用动作 (n_rollout_threads, action_n)
        """
        # 插入观测和状态数据
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.masks[self.step + 1] = masks.copy()
        
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()

        # 插入动作数据
        self._insert_actions(actions)
        
        # 处理可用动作
        if available_actions is not None:
            if self.action_type == 'mixed':
                # 混合动作空间：available_actions应该是dict或list
                if isinstance(available_actions, dict):
                    for i, av_act in available_actions.items():
                        if self.available_actions[i] is not None:
                            self.available_actions[i][self.step + 1] = av_act.copy()
                elif isinstance(available_actions, (list, tuple)):
                    for i, av_act in enumerate(available_actions):
                        if i < len(self.available_actions) and self.available_actions[i] is not None:
                            self.available_actions[i][self.step + 1] = av_act.copy()
            elif self.available_actions is not None:
                # 非混合动作空间
                self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def _insert_actions(self, actions):
        """插入动作数据，处理不同的动作类型"""
        if self.action_type == 'mixed':
            # 混合动作空间：actions应该是tuple或list
            if isinstance(actions, np.ndarray) and len(self.mixed_spaces) == 1:
                # 单个子空间的情况
                actions = [actions]
            elif not isinstance(actions, (tuple, list)):
                raise ValueError(f"混合动作空间期望tuple或list类型的动作，但收到{type(actions)}")
            
            # 分别处理每个子空间的动作
            for i, (space_type, space, shape) in enumerate(self.mixed_spaces):
                sub_action = actions[i] if i < len(actions) else actions[0]  # 容错处理
                
                if not isinstance(sub_action, np.ndarray):
                    sub_action = np.array(sub_action)
                
                if space_type == 'discrete':
                    if sub_action.ndim == 1:
                        sub_action = sub_action.reshape(-1, *shape)
                    self.actions[i][self.step] = sub_action.astype(np.int32)
                else:  # continuous
                    if sub_action.shape[-1] != shape[0]:
                        raise ValueError(f"子空间{i}动作维度不匹配: 期望{shape[0]}, 实际{sub_action.shape[-1]}")
                    self.actions[i][self.step] = sub_action.astype(np.float32)
        else:
            # 非混合动作空间：保持原有逻辑
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)
            
            # 确保动作维度正确
            if self.action_type in ['discrete', 'multi_discrete']:
                # 离散动作：确保形状为 (n_threads, action_dim)
                if actions.ndim == 1:
                    actions = actions.reshape(-1, *self.action_shape)
                self.actions[self.step] = actions.astype(np.int32)
            else:
                # 连续动作：确保形状匹配
                # 处理不同形状的情况
                if actions.ndim == 1:
                    # 如果是一维数组，根据action_shape进行重塑
                    if self.action_shape == (1,):
                        actions = actions.reshape(-1, 1)
                    else:
                        # 假设第一维是batch size
                        batch_size = len(actions) // self.action_shape[0]
                        if batch_size * self.action_shape[0] == len(actions):
                            actions = actions.reshape(batch_size, self.action_shape[0])
                        else:
                            # 尝试作为单个样本处理
                            actions = actions.reshape(1, -1)
                elif actions.ndim == 2:
                    # 检查最后一维是否匹配
                    if actions.shape[-1] != self.action_shape[0]:
                        # 尝试自动修复常见情况
                        if actions.shape[-1] == 1 and self.action_shape[0] == 1:
                            pass  # 形状兼容
                        elif actions.shape[0] == self.action_shape[0] and actions.shape[1] < self.n_rollout_threads:
                            # 处理形状转置的情况：如果是 (action_dim, batch_size) 而不是 (batch_size, action_dim)
                            # 这种情况可能发生在PV动作被错误地组织为2x1而不是1x2时
                            actions = actions.T  # 转置为正确的形状
                        elif self.n_rollout_threads == 1 and actions.shape[0] == self.action_shape[0]:
                            # 单线程情况下，如果形状是 (action_dim, 1)，转换为 (1, action_dim)
                            actions = actions.reshape(1, -1)
                        else:
                            raise ValueError(f"动作维度不匹配: 期望形状 (batch_size, {self.action_shape[0]}), 实际形状 {actions.shape}")
                
                # 最终形状验证
                if actions.ndim == 2 and actions.shape[0] != self.n_rollout_threads:
                    # 如果批次大小不匹配，尝试调整
                    if actions.shape[0] == 1 and self.n_rollout_threads > 1:
                        # 广播单个动作到所有线程
                        actions = np.broadcast_to(actions, (self.n_rollout_threads, actions.shape[1]))
                    elif actions.shape[0] > self.n_rollout_threads:
                        # 截取前n_rollout_threads个
                        actions = actions[:self.n_rollout_threads]
                
                self.actions[self.step] = actions.astype(np.float32)

    def after_update(self):
        """更新后的数据复制"""
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        
        if self.action_type == 'mixed':
            # 混合动作空间：复制每个子空间的available_actions
            for i in self.available_actions:
                if self.available_actions[i] is not None:
                    self.available_actions[i][0] = self.available_actions[i][-1].copy()
        elif self.available_actions is not None:
            # 非混合动作空间
            self.available_actions[0] = self.available_actions[-1].copy()

    def update_factor(self, factor):
        """更新因子（保持接口兼容性）"""
        self.factor = factor.copy()

    def feed_forward_generator_actor(
        self, advantages, actor_num_mini_batch=None, mini_batch_size=None
    ):
        """为Actor生成训练数据（MLP网络）"""
        episode_length, n_rollout_threads = self.obs.shape[0:2]
        batch_size = n_rollout_threads * (episode_length - 1)

        if mini_batch_size is None:
            assert batch_size >= actor_num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({}).".format(
                    n_rollout_threads, episode_length, batch_size, actor_num_mini_batch
                )
            )
            mini_batch_size = batch_size // actor_num_mini_batch

        # 创建随机采样索引
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]

        # 处理数据
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, 1)
        
        # 处理动作数据
        if self.action_type == 'mixed':
            # 混合动作空间：将各子空间的动作拼接成一个向量
            actions_list = []
            for i in self.actions:
                sub_actions = self.actions[i].reshape(-1, *self.actions[i].shape[2:])
                actions_list.append(sub_actions)
            actions = np.concatenate(actions_list, axis=-1)
            
            # 处理可用动作
            if any(av is not None for av in self.available_actions.values()):
                available_actions_list = []
                for i in self.available_actions:
                    if self.available_actions[i] is not None:
                        av = self.available_actions[i][:-1].reshape(-1, self.available_actions[i].shape[-1])
                        available_actions_list.append(av)
                # 对于混合动作空间，可用动作通常只适用于离散部分
                # 这里返回第一个非None的available_actions（通常是离散动作部分）
                available_actions = available_actions_list[0] if available_actions_list else None
            else:
                available_actions = None
        else:
            # 非混合动作空间
            actions = self.actions.reshape(-1, *self.actions.shape[2:])
            if self.available_actions is not None:
                available_actions = self.available_actions[:-1].reshape(
                    -1, self.available_actions.shape[-1]
                )
            else:
                available_actions = None
            
        if self.factor is not None:
            factor = self.factor.reshape(-1, 1)
        
        advantages = advantages.reshape(-1, 1)

        # 生成mini-batch
        for indices in sampler:
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            actions_batch = actions[indices]
            
            if available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
                
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            if self.factor is None:
                yield (obs_batch, rnn_states_batch, actions_batch, 
                      masks_batch, active_masks_batch, old_action_log_probs_batch, 
                      adv_targ, available_actions_batch)
            else:
                factor_batch = factor[indices]
                yield (obs_batch, rnn_states_batch, actions_batch, 
                      masks_batch, active_masks_batch, old_action_log_probs_batch, 
                      adv_targ, available_actions_batch, factor_batch)

    def naive_recurrent_generator_actor(self, advantages, actor_num_mini_batch):
        """为Actor生成循环网络训练数据（简单版本）"""
        n_rollout_threads = advantages.shape[1]
        assert n_rollout_threads >= actor_num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, actor_num_mini_batch)
        )
        
        num_envs_per_batch = n_rollout_threads // actor_num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()

        T, N = self.episode_length, num_envs_per_batch

        for batch_id in range(actor_num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[start_id : start_id + num_envs_per_batch]
            
            obs_batch = _flatten(T, N, self.obs[:-1, ids])
            masks_batch = _flatten(T, N, self.masks[:-1, ids])
            active_masks_batch = _flatten(T, N, self.active_masks[:-1, ids])
            old_action_log_probs_batch = _flatten(T, N, self.action_log_probs[:, ids])
            adv_targ = _flatten(T, N, advantages[:, ids])
            
            # 处理动作数据
            if self.action_type == 'mixed':
                # 混合动作空间：拼接所有子动作
                actions_list = []
                for i in self.actions:
                    sub_actions = _flatten(T, N, self.actions[i][:, ids])
                    actions_list.append(sub_actions)
                actions_batch = np.concatenate(actions_list, axis=-1)
                
                # 处理可用动作（通常只有离散部分有）
                available_actions_batch = None
                for i in self.available_actions:
                    if self.available_actions[i] is not None:
                        available_actions_batch = _flatten(T, N, self.available_actions[i][:-1, ids])
                        break  # 使用第一个非None的available_actions
            else:
                # 非混合动作空间
                actions_batch = _flatten(T, N, self.actions[:, ids])
                if self.available_actions is not None:
                    available_actions_batch = _flatten(T, N, self.available_actions[:-1, ids])
                else:
                    available_actions_batch = None
                
            if self.factor is not None:
                factor_batch = _flatten(T, N, self.factor[:, ids])
            else:
                factor_batch = None
                
            rnn_states_batch = self.rnn_states[0, ids]
            
            if self.factor is not None:
                yield (obs_batch, rnn_states_batch, actions_batch, 
                      masks_batch, active_masks_batch, old_action_log_probs_batch, 
                      adv_targ, available_actions_batch, factor_batch)
            else:
                yield (obs_batch, rnn_states_batch, actions_batch, 
                      masks_batch, active_masks_batch, old_action_log_probs_batch, 
                      adv_targ, available_actions_batch)

    def recurrent_generator_actor(self, advantages, actor_num_mini_batch, data_chunk_length):
        """为Actor生成循环网络训练数据（分块版本）"""
        episode_length, n_rollout_threads = self.obs.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // actor_num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO要求进程数量n_rollout_threads ({}) * 回合长度 episode_length({}) "
            "必须大于或等于数据块长度 data_chunk_length长度({}).".format(
                n_rollout_threads, episode_length, data_chunk_length
            )
        )
        assert data_chunks >= 2, "need larger batch size"

        # 生成随机排列
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]

        # 重塑数据
        obs = _sa_cast(self.obs[:-1])
        rnn_states = (
            self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        )
        action_log_probs = _sa_cast(self.action_log_probs)
        advantages = _sa_cast(advantages)
        masks = _sa_cast(self.masks[:-1])
        active_masks = _sa_cast(self.active_masks[:-1])
        
        # 处理动作和可用动作
        if self.action_type == 'mixed':
            # 混合动作空间需要特殊处理
            actions_dict = {}
            available_actions_dict = {}
            for i in self.actions:
                actions_dict[i] = _sa_cast(self.actions[i])
                if self.available_actions[i] is not None:
                    available_actions_dict[i] = _sa_cast(self.available_actions[i][:-1])
            # 后续在生成batch时进行拼接
        else:
            # 非混合动作空间
            actions = _sa_cast(self.actions)
            if self.available_actions is not None:
                available_actions = _sa_cast(self.available_actions[:-1])
            else:
                available_actions = None
            
        if self.factor is not None:
            factor = _sa_cast(self.factor)

        # 生成mini-batch
        for indices in sampler:
            obs_batch = []
            rnn_states_batch = []
            actions_batch = []
            available_actions_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []

            for index in indices:
                ind = index * data_chunk_length
                obs_batch.append(obs[ind : ind + data_chunk_length])
                
                if self.action_type == 'mixed':
                    # 混合动作空间：收集所有子动作
                    sub_actions = []
                    for i in actions_dict:
                        sub_actions.append(actions_dict[i][ind : ind + data_chunk_length])
                    # 拼接动作
                    concat_actions = np.concatenate(sub_actions, axis=-1)
                    actions_batch.append(concat_actions)
                    
                    # 处理可用动作
                    for i in available_actions_dict:
                        if i in available_actions_dict:
                            available_actions_batch.append(available_actions_dict[i][ind : ind + data_chunk_length])
                            break  # 只使用第一个available_actions
                else:
                    # 非混合动作空间
                    actions_batch.append(actions[ind : ind + data_chunk_length])
                    if available_actions is not None:
                        available_actions_batch.append(available_actions[ind : ind + data_chunk_length])
                    
                masks_batch.append(masks[ind : ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind : ind + data_chunk_length])
                adv_targ.append(advantages[ind : ind + data_chunk_length])
                rnn_states_batch.append(rnn_states[ind])
                
                if self.factor is not None:
                    factor_batch.append(factor[ind : ind + data_chunk_length])

            L, N = data_chunk_length, mini_batch_size
            
            # Stack数据
            obs_batch = np.stack(obs_batch, axis=1)
            actions_batch = np.stack(actions_batch, axis=1)
            
            if available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
                
            if self.factor is not None:
                factor_batch = np.stack(factor_batch, axis=1)
                
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])

            # Flatten数据
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            
            if available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
                
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
                
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
            
            if self.factor is not None:
                yield (obs_batch, rnn_states_batch, actions_batch, 
                      masks_batch, active_masks_batch, old_action_log_probs_batch, 
                      adv_targ, available_actions_batch, factor_batch)
            else:
                yield (obs_batch, rnn_states_batch, actions_batch, 
                      masks_batch, active_masks_batch, old_action_log_probs_batch, 
                      adv_targ, available_actions_batch)
    
    def debug_available_actions_info(self):
        """调试方法：打印available_actions的维度信息"""
        print("=== 单智能体异构Buffer Available Actions 调试信息 ===")
        print(f"episode_length: {self.episode_length}")
        print(f"n_rollout_threads: {self.n_rollout_threads}")
        print(f"current step: {self.step}")
        print(f"action_type: {self.action_type}")
        print(f"action_shape: {self.action_shape}")
        
        if self.available_actions is not None:
            print(f"available_actions shape: {self.available_actions.shape}")
            print(f"action_space size: {self.action_space.n}")
        else:
            print("available_actions: None (连续动作空间)")
        print("=" * 50)