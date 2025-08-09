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
        )
        self.active_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )
        
        self.factor = None
        self.step = 0

    def _parse_action_space(self, act_space):
        """解析单个智能体的动作空间结构"""
        self.action_space = act_space
        
        if isinstance(act_space, Discrete):
            self.action_shape = (1,)
            self.action_type = 'discrete'
            self.action_dim = 1
        elif isinstance(act_space, Box):
            # 处理Box空间的形状
            self.action_shape = (1,) if len(act_space.shape) == 0 else act_space.shape
            self.action_dim = self.action_shape[0]
            self.action_type = 'continuous'
        elif isinstance(act_space, (SpaceTuple, tuple)):
            # 混合动作空间
            self.action_type = 'mixed'
            self._parse_mixed_action_space(act_space)
        elif isinstance(act_space, MultiDiscrete):
            self.action_shape = (len(act_space.nvec),)
            self.action_type = 'multi_discrete'
            self.action_dim = len(act_space.nvec)
        else:
            raise NotImplementedError(f"不支持的动作空间类型: {type(act_space)}")
    
    def _parse_mixed_action_space(self, act_space):
        """解析混合动作空间结构"""
        self.mixed_spaces = []
        self.mixed_action_shapes = []
        total_dim = 0
        
        for space in act_space:
            if isinstance(space, Discrete):
                shape = (1,)
                dim = 1
                space_type = 'discrete'
            elif isinstance(space, MultiDiscrete):
                shape = (len(space.nvec),)
                dim = len(space.nvec)
                space_type = 'discrete'
            elif isinstance(space, Box):
                shape = space.shape
                dim = space.shape[0] if len(space.shape) > 0 else 1
                space_type = 'continuous'
            else:
                raise NotImplementedError(f"混合动作空间中不支持的子空间类型: {type(space)}")
            
            self.mixed_spaces.append((space_type, space, shape))
            self.mixed_action_shapes.append(shape)
            total_dim += dim
        
        # 混合动作空间的总维度
        self.action_shape = (total_dim,)
        self.action_dim = total_dim
    
    def _init_action_buffers(self):
        """初始化动作缓冲区"""
        if self.action_type == 'mixed':
            # 混合动作空间：为每个子空间创建独立的缓冲区
            self.actions = {}
            self.available_actions = {}
            
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
            # 非混合动作空间
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
        """插入数据到Buffer中"""
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.masks[self.step + 1] = masks.copy()
        
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()

        self._insert_actions(actions)
        self._insert_available_actions(available_actions)
        
        self.step = (self.step + 1) % self.episode_length

    def _insert_actions(self, actions):
        """插入动作数据，处理不同的动作类型"""
        if self.action_type == 'mixed':
            self._insert_mixed_actions(actions)
        else:
            self._insert_simple_actions(actions)

    def _insert_mixed_actions(self, actions):
        """处理混合动作空间的动作插入"""
        # 确保actions是list或tuple
        if isinstance(actions, np.ndarray) and len(self.mixed_spaces) == 1:
            actions = [actions]
        elif not isinstance(actions, (tuple, list)):
            raise ValueError(f"混合动作空间期望tuple或list类型的动作，但收到{type(actions)}")
        
        # 分别处理每个子空间的动作
        for i, (space_type, _, shape) in enumerate(self.mixed_spaces):
            sub_action = actions[i] if i < len(actions) else actions[0]
            
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

    def _insert_simple_actions(self, actions):
        """处理非混合动作空间的动作插入"""
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        
        # 确保动作维度正确
        if self.action_type in ['discrete', 'multi_discrete']:
            actions = self._reshape_discrete_actions(actions)
            # 验证形状匹配
            if actions.shape[1:] != self.action_shape:
                raise ValueError(
                    f"离散动作形状不匹配: 期望 {self.action_shape}, "
                    f"实际 {actions.shape[1:]}, 完整形状 {actions.shape}"
                )
            self.actions[self.step] = actions.astype(np.int32)
        else:
            actions = self._reshape_continuous_actions(actions)
            # 验证形状匹配
            if actions.shape[1:] != self.action_shape:
                raise ValueError(
                    f"连续动作形状不匹配: 期望 {self.action_shape}, "
                    f"实际 {actions.shape[1:]}, 完整形状 {actions.shape}"
                )
            self.actions[self.step] = actions.astype(np.float32)

    def _reshape_discrete_actions(self, actions):
        """重塑离散动作"""
        if actions.ndim == 1:
            return actions.reshape(-1, *self.action_shape)
        elif actions.ndim == 2:
            # 如果是2维，检查是否需要截取
            if actions.shape[-1] > self.action_shape[0]:
                # 只取需要的维度
                actions = actions[:, :self.action_shape[0]]
            elif actions.shape[-1] < self.action_shape[0]:
                # 维度不足，报错
                raise ValueError(
                    f"离散动作维度不足: 期望至少 {self.action_shape[0]} 维, "
                    f"实际只有 {actions.shape[-1]} 维"
                )
        return actions

    def _reshape_continuous_actions(self, actions):
        """重塑连续动作"""
        if actions.ndim == 1:
            if self.action_shape == (1,):
                return actions.reshape(-1, 1)
            else:
                # 尝试自动推断形状
                batch_size = len(actions) // self.action_shape[0]
                if batch_size * self.action_shape[0] == len(actions):
                    return actions.reshape(batch_size, self.action_shape[0])
                return actions.reshape(1, -1)
        
        if actions.ndim == 2:
            # 如果动作维度超过需要，截取
            if actions.shape[-1] > self.action_shape[0]:
                actions = actions[:, :self.action_shape[0]]
            elif actions.shape[-1] < self.action_shape[0]:
                # 维度不足，尝试修复
                actions = self._fix_action_shape_mismatch(actions)
            
            # 处理批次大小不匹配
            if actions.shape[0] != self.n_rollout_threads:
                actions = self._fix_batch_size_mismatch(actions)
        
        return actions

    def _fix_action_shape_mismatch(self, actions):
        """修复动作形状不匹配的问题"""
        if actions.shape[-1] == 1 and self.action_shape[0] == 1:
            return actions
        
        # 处理转置情况
        if (actions.shape[0] == self.action_shape[0] and 
            actions.shape[1] < self.n_rollout_threads):
            return actions.T
        
        # 单线程特殊情况
        if (self.n_rollout_threads == 1 and 
            actions.shape[0] == self.action_shape[0]):
            return actions.reshape(1, -1)
        
        raise ValueError(
            f"动作维度不匹配: 期望形状 (batch_size, {self.action_shape[0]}), "
            f"实际形状 {actions.shape}"
        )

    def _fix_batch_size_mismatch(self, actions):
        """修复批次大小不匹配的问题"""
        if actions.shape[0] == 1 and self.n_rollout_threads > 1:
            # 广播单个动作到所有线程
            return np.broadcast_to(actions, (self.n_rollout_threads, actions.shape[1]))
        elif actions.shape[0] > self.n_rollout_threads:
            # 截取前n_rollout_threads个
            return actions[:self.n_rollout_threads]
        return actions

    def _insert_available_actions(self, available_actions):
        """插入可用动作数据"""
        if available_actions is None:
            return
        
        if self.action_type == 'mixed':
            # 混合动作空间
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

    def _prepare_batch_data(self):
        """准备批处理数据的通用方法"""
        episode_length, n_rollout_threads = self.obs.shape[0:2]
        batch_size = n_rollout_threads * (episode_length - 1)
        
        # 基础数据处理
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, 1)
        
        # 处理动作数据
        if self.action_type == 'mixed':
            actions = self._prepare_mixed_actions()
            available_actions = self._prepare_mixed_available_actions()
        else:
            actions = self.actions.reshape(-1, *self.actions.shape[2:])
            available_actions = self._prepare_simple_available_actions()
        
        factor = self.factor.reshape(-1, 1) if self.factor is not None else None
        
        return {
            'obs': obs,
            'rnn_states': rnn_states,
            'actions': actions,
            'masks': masks,
            'active_masks': active_masks,
            'action_log_probs': action_log_probs,
            'available_actions': available_actions,
            'factor': factor,
            'batch_size': batch_size
        }

    def _prepare_mixed_actions(self):
        """准备混合动作空间的动作数据"""
        actions_list = []
        for i in self.actions:
            sub_actions = self.actions[i].reshape(-1, *self.actions[i].shape[2:])
            actions_list.append(sub_actions)
        return np.concatenate(actions_list, axis=-1)

    def _prepare_mixed_available_actions(self):
        """准备混合动作空间的可用动作数据"""
        if any(av is not None for av in self.available_actions.values()):
            for i in self.available_actions:
                if self.available_actions[i] is not None:
                    return self.available_actions[i][:-1].reshape(
                        -1, self.available_actions[i].shape[-1]
                    )
        return None

    def _prepare_simple_available_actions(self):
        """准备非混合动作空间的可用动作数据"""
        if self.available_actions is not None:
            return self.available_actions[:-1].reshape(
                -1, self.available_actions.shape[-1]
            )
        return None

    def feed_forward_generator_actor(
        self, advantages, actor_num_mini_batch=None, mini_batch_size=None
    ):
        """为Actor生成训练数据（MLP网络）"""
        data = self._prepare_batch_data()
        batch_size = data['batch_size']
        
        if mini_batch_size is None:
            assert batch_size >= actor_num_mini_batch, (
                f"PPO requires the number of processes ({self.n_rollout_threads}) "
                f"* number of steps ({self.episode_length - 1}) = {batch_size} "
                f"to be greater than or equal to the number of PPO mini batches ({actor_num_mini_batch})."
            )
            mini_batch_size = batch_size // actor_num_mini_batch

        # 创建随机采样索引
        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]

        advantages = advantages.reshape(-1, 1)

        # 生成mini-batch
        for indices in sampler:
            batch_data = (
                data['obs'][indices],
                data['rnn_states'][indices],
                data['actions'][indices],
                data['masks'][indices],
                data['active_masks'][indices],
                data['action_log_probs'][indices],
                advantages[indices] if advantages is not None else None,
                data['available_actions'][indices] if data['available_actions'] is not None else None,
            )
            
            if data['factor'] is not None:
                batch_data += (data['factor'][indices],)
            
            yield batch_data

    def naive_recurrent_generator_actor(self, advantages, actor_num_mini_batch):
        """为Actor生成循环网络训练数据（简单版本）"""
        n_rollout_threads = advantages.shape[1]
        assert n_rollout_threads >= actor_num_mini_batch, (
            f"PPO requires the number of processes ({n_rollout_threads}) "
            f"to be greater than or equal to the number of "
            f"PPO mini batches ({actor_num_mini_batch})."
        )
        
        num_envs_per_batch = n_rollout_threads // actor_num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()

        T, N = self.episode_length, num_envs_per_batch

        for batch_id in range(actor_num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[start_id : start_id + num_envs_per_batch]
            
            batch_data = self._prepare_recurrent_batch(T, N, ids, advantages)
            yield batch_data

    def _prepare_recurrent_batch(self, T, N, ids, advantages):
        """准备循环网络的批处理数据"""
        obs_batch = _flatten(T, N, self.obs[:-1, ids])
        masks_batch = _flatten(T, N, self.masks[:-1, ids])
        active_masks_batch = _flatten(T, N, self.active_masks[:-1, ids])
        old_action_log_probs_batch = _flatten(T, N, self.action_log_probs[:, ids])
        adv_targ = _flatten(T, N, advantages[:, ids])
        
        # 处理动作数据
        if self.action_type == 'mixed':
            actions_list = []
            for i in self.actions:
                sub_actions = _flatten(T, N, self.actions[i][:, ids])
                actions_list.append(sub_actions)
            actions_batch = np.concatenate(actions_list, axis=-1)
            
            available_actions_batch = None
            for i in self.available_actions:
                if self.available_actions[i] is not None:
                    available_actions_batch = _flatten(T, N, self.available_actions[i][:-1, ids])
                    break
        else:
            actions_batch = _flatten(T, N, self.actions[:, ids])
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, self.available_actions[:-1, ids])
            else:
                available_actions_batch = None
        
        factor_batch = _flatten(T, N, self.factor[:, ids]) if self.factor is not None else None
        rnn_states_batch = self.rnn_states[0, ids]
        
        batch_data = (
            obs_batch, rnn_states_batch, actions_batch,
            masks_batch, active_masks_batch, old_action_log_probs_batch,
            adv_targ, available_actions_batch
        )
        
        if factor_batch is not None:
            batch_data += (factor_batch,)
        
        return batch_data

    def recurrent_generator_actor(self, advantages, actor_num_mini_batch, data_chunk_length):
        """为Actor生成循环网络训练数据（分块版本）"""
        episode_length, n_rollout_threads = self.obs.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // actor_num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            f"PPO要求进程数量n_rollout_threads ({n_rollout_threads}) * 回合长度 episode_length({episode_length}) "
            f"必须大于或等于数据块长度 data_chunk_length长度({data_chunk_length})."
        )
        assert data_chunks >= 2, "need larger batch size"

        # 生成随机排列
        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]

        # 准备数据
        prepared_data = self._prepare_chunked_data()
        
        # 准备advantages
        if advantages is not None:
            # advantages 的形状应该是 (episode_length, n_rollout_threads, 1)
            prepared_data['advantages'] = _sa_cast(advantages)

        # 生成mini-batch
        for indices in sampler:
            batch_data = self._collect_chunk_batch(
                indices, data_chunk_length, mini_batch_size, prepared_data
            )
            yield batch_data

    def _prepare_chunked_data(self):
        """准备分块数据"""
        obs = _sa_cast(self.obs[:-1])
        rnn_states = (
            self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        )
        action_log_probs = _sa_cast(self.action_log_probs)
        masks = _sa_cast(self.masks[:-1])
        active_masks = _sa_cast(self.active_masks[:-1])
        
        if self.action_type == 'mixed':
            actions_dict = {}
            available_actions_dict = {}
            for i in self.actions:
                actions_dict[i] = _sa_cast(self.actions[i])
                if self.available_actions[i] is not None:
                    available_actions_dict[i] = _sa_cast(self.available_actions[i][:-1])
            return {
                'obs': obs,
                'rnn_states': rnn_states,
                'action_log_probs': action_log_probs,
                'masks': masks,
                'active_masks': active_masks,
                'actions_dict': actions_dict,
                'available_actions_dict': available_actions_dict,
                'factor': _sa_cast(self.factor) if self.factor is not None else None
            }
        else:
            return {
                'obs': obs,
                'rnn_states': rnn_states,
                'action_log_probs': action_log_probs,
                'masks': masks,
                'active_masks': active_masks,
                'actions': _sa_cast(self.actions),
                'available_actions': _sa_cast(self.available_actions[:-1]) if self.available_actions is not None else None,
                'factor': _sa_cast(self.factor) if self.factor is not None else None
            }

    def _collect_chunk_batch(self, indices, data_chunk_length, mini_batch_size, prepared_data):
        """收集分块批处理数据"""
        L, N = data_chunk_length, mini_batch_size
        
        obs_batch = []
        rnn_states_batch = []
        actions_batch = []
        available_actions_batch = []
        masks_batch = []
        active_masks_batch = []
        old_action_log_probs_batch = []
        factor_batch = []
        adv_targ_batch = []

        for index in indices:
            ind = index * data_chunk_length
            obs_batch.append(prepared_data['obs'][ind : ind + data_chunk_length])
            
            if self.action_type == 'mixed':
                # 混合动作空间：收集所有子动作
                sub_actions = []
                for i in prepared_data['actions_dict']:
                    sub_actions.append(prepared_data['actions_dict'][i][ind : ind + data_chunk_length])
                concat_actions = np.concatenate(sub_actions, axis=-1)
                actions_batch.append(concat_actions)
                
                # 处理可用动作
                for i in prepared_data['available_actions_dict']:
                    if i in prepared_data['available_actions_dict']:
                        available_actions_batch.append(
                            prepared_data['available_actions_dict'][i][ind : ind + data_chunk_length]
                        )
                        break
            else:
                actions_batch.append(prepared_data['actions'][ind : ind + data_chunk_length])
                if prepared_data['available_actions'] is not None:
                    available_actions_batch.append(
                        prepared_data['available_actions'][ind : ind + data_chunk_length]
                    )
            
            masks_batch.append(prepared_data['masks'][ind : ind + data_chunk_length])
            active_masks_batch.append(prepared_data['active_masks'][ind : ind + data_chunk_length])
            old_action_log_probs_batch.append(
                prepared_data['action_log_probs'][ind : ind + data_chunk_length]
            )
            rnn_states_batch.append(prepared_data['rnn_states'][ind])
            
            if prepared_data['factor'] is not None:
                factor_batch.append(prepared_data['factor'][ind : ind + data_chunk_length])
            
            # 处理advantages
            if 'advantages' in prepared_data and prepared_data['advantages'] is not None:
                adv_targ_batch.append(prepared_data['advantages'][ind : ind + data_chunk_length])

        # Stack数据
        obs_batch = _flatten(L, N, np.stack(obs_batch, axis=1))
        actions_batch = _flatten(L, N, np.stack(actions_batch, axis=1))
        masks_batch = _flatten(L, N, np.stack(masks_batch, axis=1))
        active_masks_batch = _flatten(L, N, np.stack(active_masks_batch, axis=1))
        old_action_log_probs_batch = _flatten(L, N, np.stack(old_action_log_probs_batch, axis=1))
        rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])
        
        if available_actions_batch:
            available_actions_batch = _flatten(L, N, np.stack(available_actions_batch, axis=1))
        else:
            available_actions_batch = None
        
        if prepared_data['factor'] is not None:
            factor_batch = _flatten(L, N, np.stack(factor_batch, axis=1))
        else:
            factor_batch = None
        
        # 处理advantages
        if adv_targ_batch:
            # advantages 和其他数据一样需要 stack 和 flatten
            adv_targ = _flatten(L, N, np.stack(adv_targ_batch, axis=1))
        else:
            adv_targ = None
        
        batch_data = (
            obs_batch, rnn_states_batch, actions_batch,
            masks_batch, active_masks_batch, old_action_log_probs_batch,
            adv_targ, available_actions_batch
        )
        
        if factor_batch is not None:
            batch_data += (factor_batch,)
        
        return batch_data