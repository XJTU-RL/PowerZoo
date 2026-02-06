# -*- coding: utf-8 -*-
"""
@File      : shared_heterogeneous_buffer.py
@Time      : 2025-08-05
@Author    : Zheng
@Description: 支持HAPPO算法的共享异构动作空间Buffer
- 支持多智能体共享critic的数据存储
- 兼容各种动作空间类型：离散、连续、多离散、混合
- 提供共享的value和advantage计算
- 完全兼容HAPPO算法的训练需求
"""

import torch
import numpy as np
try:
    from gymnasium.spaces import Discrete, Box, Tuple as SpaceTuple, MultiDiscrete
except ImportError:
    from gym.spaces import Discrete, Box, Tuple as SpaceTuple, MultiDiscrete
from utils.trans_tools import _flatten, _sa_cast
from utils.envs_tools import get_shape_from_obs_space, get_shape_from_act_space


class SharedHeterogeneousBuffer:
    """HAPPO算法的共享异构动作空间Buffer"""

    def __init__(self, args, share_obs_space, obs_spaces, act_spaces):
        """初始化共享异构动作空间Buffer
        
        Args:
            args: (dict) 参数配置
            share_obs_space: (gym.Space) 共享观测空间（用于centralized critic）
            obs_spaces: (list of gym.Space) 各智能体的观测空间
            act_spaces: (list of gym.Space) 各智能体的动作空间
        """
        self.episode_length = args["episode_length"]
        self.n_rollout_threads = args["n_rollout_threads"]
        self.hidden_sizes = args["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = args["recurrent_n"]
        self.gamma = args.get("gamma", 0.99)
        self.gae_lambda = args.get("gae_lambda", 0.95)
        self.use_gae = args.get("use_gae", True)
        self.use_proper_time_limits = args.get("use_proper_time_limits", False)

        self.num_agents = len(obs_spaces)

        # 共享观测空间处理（用于centralized value function）
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        # 共享观测缓冲区
        self.share_obs = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, *share_obs_shape),
            dtype=np.float32,
        )

        # 共享的value和returns缓冲区
        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1),
            dtype=np.float32,
        )

        self.returns = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1),
            dtype=np.float32,
        )

        # 奖励缓冲区
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents, 1),
            dtype=np.float32,
        )

        # Critic的RNN状态缓冲区
        self.rnn_states_critic = np.zeros(
            (
                self.episode_length + 1,
                self.n_rollout_threads,
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # 为每个智能体创建独立的actor缓冲区
        self.actor_buffers = []
        for agent_id in range(self.num_agents):
            obs_shape = get_shape_from_obs_space(obs_spaces[agent_id])
            if isinstance(obs_shape[-1], list):
                obs_shape = obs_shape[:1]

            actor_buffer = {
                'obs': np.zeros(
                    (self.episode_length + 1, self.n_rollout_threads, *obs_shape),
                    dtype=np.float32,
                ),
                'rnn_states': np.zeros(
                    (
                        self.episode_length + 1,
                        self.n_rollout_threads,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                ),
                'action_log_probs': np.zeros(
                    (self.episode_length, self.n_rollout_threads, 1),
                    dtype=np.float32,
                ),
                'masks': np.ones(
                    (self.episode_length + 1, self.n_rollout_threads, 1),
                    dtype=np.float32,
                ),
                'active_masks': np.ones(
                    (self.episode_length + 1, self.n_rollout_threads, 1),
                    dtype=np.float32,
                ),
                'bad_masks': np.ones(
                    (self.episode_length + 1, self.n_rollout_threads, 1),
                    dtype=np.float32,
                ),
            }

            # 解析并初始化动作空间相关的缓冲区
            action_info = self._parse_action_space(act_spaces[agent_id])
            actor_buffer.update(action_info)
            actor_buffer.update(self._init_action_buffers(
                act_spaces[agent_id],
                action_info['action_type'],
                action_info['action_shape'],
                action_info.get('mixed_spaces', None)
            ))

            self.actor_buffers.append(actor_buffer)

        self.step = 0

    def _parse_action_space(self, act_space):
        """解析单个智能体的动作空间结构"""
        info = {'action_space': act_space}

        if isinstance(act_space, Discrete):
            info['action_shape'] = (1,)
            info['action_type'] = 'discrete'
            info['action_dim'] = 1
        elif isinstance(act_space, Box):
            if len(act_space.shape) == 0:
                info['action_shape'] = (1,)
                info['action_dim'] = 1
            else:
                info['action_shape'] = act_space.shape
                info['action_dim'] = act_space.shape[0]
            info['action_type'] = 'continuous'
        elif isinstance(act_space, (SpaceTuple, tuple)):
            info['action_type'] = 'mixed'
            mixed_spaces, action_shape, action_dim = self._parse_mixed_action_space(act_space)
            info['mixed_spaces'] = mixed_spaces
            info['action_shape'] = action_shape
            info['action_dim'] = action_dim
        elif isinstance(act_space, MultiDiscrete):
            info['action_shape'] = (len(act_space.nvec),)
            info['action_type'] = 'multi_discrete'
            info['action_dim'] = len(act_space.nvec)
        else:
            raise NotImplementedError(f"不支持的动作空间类型: {type(act_space)}")

        return info

    def _parse_mixed_action_space(self, act_space):
        """解析混合动作空间结构"""
        mixed_spaces = []
        mixed_action_shapes = []
        total_dim = 0

        for i, space in enumerate(act_space):
            if isinstance(space, (Discrete, MultiDiscrete)):
                if isinstance(space, Discrete):
                    shape = (1,)
                    dim = 1
                else:  # MultiDiscrete
                    shape = (len(space.nvec),)
                    dim = len(space.nvec)
                mixed_spaces.append(('discrete', space, shape))
            elif isinstance(space, Box):
                shape = space.shape
                dim = space.shape[0] if len(space.shape) > 0 else 1
                mixed_spaces.append(('continuous', space, shape))
            else:
                raise NotImplementedError(f"混合动作空间中不支持的子空间类型: {type(space)}")

            mixed_action_shapes.append(shape)
            total_dim += dim

        action_shape = (total_dim,)
        return mixed_spaces, action_shape, total_dim

    def _init_action_buffers(self, act_space, action_type, action_shape, mixed_spaces=None):
        """初始化动作缓冲区"""
        buffers = {}

        if action_type == 'mixed':
            # 混合动作空间：为每个子空间创建独立的缓冲区
            buffers['actions'] = {}
            buffers['available_actions'] = {}

            for i, (space_type, space, shape) in enumerate(mixed_spaces):
                # 动作缓冲区
                buffers['actions'][i] = np.zeros(
                    (self.episode_length, self.n_rollout_threads, *shape),
                    dtype=np.float32
                )

                # 可用动作缓冲区（仅离散类型需要）
                if space_type == 'discrete':
                    if isinstance(space, Discrete):
                        av_shape = (self.episode_length + 1, self.n_rollout_threads, space.n)
                    else:  # MultiDiscrete
                        av_shape = (self.episode_length + 1, self.n_rollout_threads, sum(space.nvec))
                    buffers['available_actions'][i] = np.ones(av_shape, dtype=np.float32)
                else:
                    buffers['available_actions'][i] = None
        else:
            # 非混合动作空间
            buffers['actions'] = np.zeros(
                (self.episode_length, self.n_rollout_threads, *action_shape),
                dtype=np.float32
            )

            # 可用动作缓冲区
            if action_type == 'discrete':
                buffers['available_actions'] = np.ones(
                    (self.episode_length + 1, self.n_rollout_threads, act_space.n),
                    dtype=np.float32,
                )
            elif action_type == 'multi_discrete':
                buffers['available_actions'] = np.ones(
                    (self.episode_length + 1, self.n_rollout_threads, sum(act_space.nvec)),
                    dtype=np.float32,
                )
            else:
                buffers['available_actions'] = None

        return buffers

    def insert(
        self,
        share_obs,
        obs,
        rnn_states_actor,
        rnn_states_critic,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        bad_masks=None,
        active_masks=None,
        available_actions=None,
    ):
        """插入数据到Buffer中
        
        Args:
            share_obs: 共享观测 (n_rollout_threads, n_agents, share_obs_dim)
            obs: 各智能体观测 list of (n_rollout_threads, obs_dim)
            rnn_states_actor: Actor RNN状态 list of (n_rollout_threads, recurrent_n, hidden_size)
            rnn_states_critic: Critic RNN状态 (n_rollout_threads, n_agents, recurrent_n, hidden_size)
            actions: 动作 list of (n_rollout_threads, action_dim)
            action_log_probs: 动作对数概率 list of (n_rollout_threads, 1)
            value_preds: 价值预测 (n_rollout_threads, n_agents, 1)
            rewards: 奖励 (n_rollout_threads, n_agents, 1)
            masks: 掩码 (n_rollout_threads, n_agents, 1)
            bad_masks: 坏掩码 (n_rollout_threads, n_agents, 1)
            active_masks: 激活掩码 list of (n_rollout_threads, 1)
            available_actions: 可用动作 list of (n_rollout_threads, action_n)
        """
        # 插入共享数据
        self.share_obs[self.step + 1] = share_obs.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()

        # 插入各智能体的独立数据
        for agent_id in range(self.num_agents):
            buffer = self.actor_buffers[agent_id]

            buffer['obs'][self.step + 1] = obs[agent_id].copy()
            buffer['rnn_states'][self.step + 1] = rnn_states_actor[agent_id].copy()
            buffer['action_log_probs'][self.step] = action_log_probs[agent_id].copy()
            buffer['masks'][self.step + 1] = masks[:, agent_id].copy()

            if bad_masks is not None:
                buffer['bad_masks'][self.step + 1] = bad_masks[:, agent_id].copy()

            if active_masks is not None:
                buffer['active_masks'][self.step + 1] = active_masks[agent_id].copy()

            # 插入动作数据
            self._insert_actions(agent_id, actions[agent_id])

            # 处理可用动作
            if available_actions is not None and available_actions[agent_id] is not None:
                self._insert_available_actions(agent_id, available_actions[agent_id])

        self.step = (self.step + 1) % self.episode_length

    def _insert_actions(self, agent_id, actions):
        """插入动作数据"""
        buffer = self.actor_buffers[agent_id]
        action_type = buffer['action_type']

        if action_type == 'mixed':
            # 混合动作空间处理
            if isinstance(actions, np.ndarray) and len(buffer['mixed_spaces']) == 1:
                actions = [actions]
            elif not isinstance(actions, (tuple, list)):
                raise ValueError(f"混合动作空间期望tuple或list类型的动作")

            for i, (space_type, space, shape) in enumerate(buffer['mixed_spaces']):
                sub_action = actions[i] if i < len(actions) else actions[0]

                if not isinstance(sub_action, np.ndarray):
                    sub_action = np.array(sub_action)

                if space_type == 'discrete':
                    if sub_action.ndim == 1:
                        sub_action = sub_action.reshape(-1, *shape)
                    buffer['actions'][i][self.step] = sub_action.astype(np.int32)
                else:  # continuous
                    buffer['actions'][i][self.step] = sub_action.astype(np.float32)
        else:
            # 非混合动作空间
            if not isinstance(actions, np.ndarray):
                actions = np.array(actions)

            if buffer['action_type'] in ['discrete', 'multi_discrete']:
                if actions.ndim == 1:
                    actions = actions.reshape(-1, *buffer['action_shape'])
                buffer['actions'][self.step] = actions.astype(np.int32)
            else:
                # 连续动作处理
                if actions.ndim == 1:
                    if buffer['action_shape'] == (1,):
                        actions = actions.reshape(-1, 1)
                    else:
                        batch_size = len(actions) // buffer['action_shape'][0]
                        if batch_size * buffer['action_shape'][0] == len(actions):
                            actions = actions.reshape(batch_size, buffer['action_shape'][0])
                        else:
                            actions = actions.reshape(1, -1)

                buffer['actions'][self.step] = actions.astype(np.float32)

    def _insert_available_actions(self, agent_id, available_actions):
        """插入可用动作数据"""
        buffer = self.actor_buffers[agent_id]

        if buffer['action_type'] == 'mixed':
            if isinstance(available_actions, dict):
                for i, av_act in available_actions.items():
                    if buffer['available_actions'][i] is not None:
                        buffer['available_actions'][i][self.step + 1] = av_act.copy()
            elif isinstance(available_actions, (list, tuple)):
                for i, av_act in enumerate(available_actions):
                    if i < len(buffer['available_actions']) and buffer['available_actions'][i] is not None:
                        buffer['available_actions'][i][self.step + 1] = av_act.copy()
        elif buffer['available_actions'] is not None:
            buffer['available_actions'][self.step + 1] = available_actions.copy()

    def after_update(self):
        """更新后的数据复制"""
        self.share_obs[0] = self.share_obs[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.value_preds[0] = self.value_preds[-1].copy()

        for agent_id in range(self.num_agents):
            buffer = self.actor_buffers[agent_id]
            buffer['obs'][0] = buffer['obs'][-1].copy()
            buffer['rnn_states'][0] = buffer['rnn_states'][-1].copy()
            buffer['masks'][0] = buffer['masks'][-1].copy()
            buffer['active_masks'][0] = buffer['active_masks'][-1].copy()
            buffer['bad_masks'][0] = buffer['bad_masks'][-1].copy()

            if buffer['action_type'] == 'mixed':
                for i in buffer['available_actions']:
                    if buffer['available_actions'][i] is not None:
                        buffer['available_actions'][i][0] = buffer['available_actions'][i][-1].copy()
            elif buffer['available_actions'] is not None:
                buffer['available_actions'][0] = buffer['available_actions'][-1].copy()

    def compute_returns(self, next_value):
        """计算returns和advantages
        
        Args:
            next_value: 下一步的价值预测 (n_rollout_threads, n_agents, 1)
        """
        self.value_preds[-1] = next_value
        gae = 0

        for step in reversed(range(self.rewards.shape[0])):
            if self.use_proper_time_limits:
                # 使用bad_masks来正确处理episode边界
                bad_masks = np.stack([self.actor_buffers[i]['bad_masks'][step + 1]
                                    for i in range(self.num_agents)], axis=1)
                delta = (self.rewards[step] +
                        self.gamma * self.value_preds[step + 1] * bad_masks -
                        self.value_preds[step])
                masks = np.stack([self.actor_buffers[i]['masks'][step + 1]
                                for i in range(self.num_agents)], axis=1)
                gae = delta + self.gamma * self.gae_lambda * bad_masks * masks * gae
            else:
                masks = np.stack([self.actor_buffers[i]['masks'][step + 1]
                                for i in range(self.num_agents)], axis=1)
                delta = (self.rewards[step] +
                        self.gamma * self.value_preds[step + 1] * masks -
                        self.value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * masks * gae

            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(
        self, 
        advantages, 
        num_mini_batch=None, 
        mini_batch_size=None,
        agent_id=None
    ):
        """为指定智能体生成前馈网络训练数据
        
        Args:
            advantages: 优势值
            num_mini_batch: mini-batch数量
            mini_batch_size: mini-batch大小
            agent_id: 智能体ID，如果为None则返回所有智能体的数据
        """
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        # 准备共享数据
        share_obs = self.share_obs[:-1].reshape(-1, self.num_agents, *self.share_obs.shape[3:])
        value_preds = self.value_preds[:-1].reshape(-1, self.num_agents, 1)
        returns = self.returns[:-1].reshape(-1, self.num_agents, 1)
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, self.num_agents, *self.rnn_states_critic.shape[3:]
        )

        # 准备各智能体数据
        agent_data = []
        for aid in range(self.num_agents):
            buffer = self.actor_buffers[aid]

            obs = buffer['obs'][:-1].reshape(-1, *buffer['obs'].shape[2:])
            rnn_states = buffer['rnn_states'][:-1].reshape(-1, *buffer['rnn_states'].shape[2:])
            masks = buffer['masks'][:-1].reshape(-1, 1)
            active_masks = buffer['active_masks'][:-1].reshape(-1, 1)
            action_log_probs = buffer['action_log_probs'].reshape(-1, 1)

            # 处理动作数据
            if buffer['action_type'] == 'mixed':
                actions_list = []
                for i in buffer['actions']:
                    sub_actions = buffer['actions'][i].reshape(-1, *buffer['actions'][i].shape[2:])
                    actions_list.append(sub_actions)
                actions = np.concatenate(actions_list, axis=-1)

                # 处理可用动作
                available_actions = None
                if any(av is not None for av in buffer['available_actions'].values()):
                    for i in buffer['available_actions']:
                        if buffer['available_actions'][i] is not None:
                            available_actions = buffer['available_actions'][i][:-1].reshape(
                                -1, buffer['available_actions'][i].shape[-1]
                            )
                            break
            else:
                actions = buffer['actions'].reshape(-1, *buffer['actions'].shape[2:])
                if buffer['available_actions'] is not None:
                    available_actions = buffer['available_actions'][:-1].reshape(
                        -1, buffer['available_actions'].shape[-1]
                    )
                else:
                    available_actions = None

            agent_data.append({
                'obs': obs,
                'rnn_states': rnn_states,
                'actions': actions,
                'masks': masks,
                'active_masks': active_masks,
                'action_log_probs': action_log_probs,
                'available_actions': available_actions,
            })

        advantages = advantages.reshape(-1, self.num_agents, 1)

        # 生成mini-batch
        for indices in sampler:
            # 共享数据
            share_obs_batch = share_obs[indices]
            value_preds_batch = value_preds[indices]
            returns_batch = returns[indices]
            advantages_batch = advantages[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]

            if agent_id is not None:
                # 返回特定智能体的数据
                data = agent_data[agent_id]
                yield (
                    share_obs_batch[:, agent_id],
                    data['obs'][indices],
                    data['rnn_states'][indices],
                    data['actions'][indices],
                    value_preds_batch[:, agent_id],
                    returns_batch[:, agent_id],
                    data['masks'][indices],
                    data['active_masks'][indices],
                    data['action_log_probs'][indices],
                    advantages_batch[:, agent_id],
                    data['available_actions'][indices] if data['available_actions'] is not None else None,
                )
            else:
                # 返回所有智能体的数据
                all_obs_batch = []
                all_actions_batch = []
                all_masks_batch = []
                all_active_masks_batch = []
                all_action_log_probs_batch = []
                all_available_actions_batch = []

                for aid in range(self.num_agents):
                    data = agent_data[aid]
                    all_obs_batch.append(data['obs'][indices])
                    all_actions_batch.append(data['actions'][indices])
                    all_masks_batch.append(data['masks'][indices])
                    all_active_masks_batch.append(data['active_masks'][indices])
                    all_action_log_probs_batch.append(data['action_log_probs'][indices])
                    all_available_actions_batch.append(
                        data['available_actions'][indices] if data['available_actions'] is not None else None
                    )

                yield (
                    share_obs_batch,
                    all_obs_batch,
                    rnn_states_critic_batch,
                    all_actions_batch,
                    value_preds_batch,
                    returns_batch,
                    all_masks_batch,
                    all_active_masks_batch,
                    all_action_log_probs_batch,
                    advantages_batch,
                    all_available_actions_batch,
                )

    def recurrent_generator(
        self,
        advantages,
        num_mini_batch,
        data_chunk_length,
        agent_id=None
    ):
        """为指定智能体生成循环网络训练数据"""
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length
        assert data_chunks >= 2

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        # 准备数据（转置并重塑）
        share_obs = self.share_obs[:-1].transpose(1, 0, 2, 3, 4).reshape(
            -1, self.num_agents, *self.share_obs.shape[3:]
        )
        value_preds = self.value_preds[:-1].transpose(1, 0, 2, 3).reshape(
            -1, self.num_agents, 1
        )
        returns = self.returns[:-1].transpose(1, 0, 2, 3).reshape(
            -1, self.num_agents, 1
        )
        advantages = advantages.transpose(1, 0, 2, 3).reshape(
            -1, self.num_agents, 1
        )
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 0, 2, 3, 4).reshape(
            -1, self.num_agents, *self.rnn_states_critic.shape[3:]
        )

        # 准备各智能体数据
        agent_data = []
        for aid in range(self.num_agents):
            buffer = self.actor_buffers[aid]

            # 转置并重塑数据
            obs = _sa_cast(buffer['obs'][:-1])
            rnn_states = buffer['rnn_states'][:-1].transpose(1, 0, 2, 3).reshape(
                -1, *buffer['rnn_states'].shape[2:]
            )
            masks = _sa_cast(buffer['masks'][:-1])
            active_masks = _sa_cast(buffer['active_masks'][:-1])
            action_log_probs = _sa_cast(buffer['action_log_probs'])

            # 处理动作
            if buffer['action_type'] == 'mixed':
                actions_dict = {}
                available_actions_dict = {}
                for i in buffer['actions']:
                    actions_dict[i] = _sa_cast(buffer['actions'][i])
                    if buffer['available_actions'][i] is not None:
                        available_actions_dict[i] = _sa_cast(buffer['available_actions'][i][:-1])

                agent_data.append({
                    'obs': obs,
                    'rnn_states': rnn_states,
                    'actions_dict': actions_dict,
                    'masks': masks,
                    'active_masks': active_masks,
                    'action_log_probs': action_log_probs,
                    'available_actions_dict': available_actions_dict,
                    'action_type': 'mixed',
                })
            else:
                actions = _sa_cast(buffer['actions'])
                available_actions = (_sa_cast(buffer['available_actions'][:-1])
                                    if buffer['available_actions'] is not None else None)

                agent_data.append({
                    'obs': obs,
                    'rnn_states': rnn_states,
                    'actions': actions,
                    'masks': masks,
                    'active_masks': active_masks,
                    'action_log_probs': action_log_probs,
                    'available_actions': available_actions,
                    'action_type': buffer['action_type'],
                })

        # 生成mini-batch
        for indices in sampler:
            share_obs_batch = []
            value_preds_batch = []
            returns_batch = []
            advantages_batch = []
            rnn_states_critic_batch = []

            agent_batches = [[] for _ in range(self.num_agents)]

            for index in indices:
                ind = index * data_chunk_length

                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                returns_batch.append(returns[ind : ind + data_chunk_length])
                advantages_batch.append(advantages[ind : ind + data_chunk_length])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

                for aid in range(self.num_agents):
                    data = agent_data[aid]
                    batch = {
                        'obs': data['obs'][ind : ind + data_chunk_length],
                        'masks': data['masks'][ind : ind + data_chunk_length],
                        'active_masks': data['active_masks'][ind : ind + data_chunk_length],
                        'action_log_probs': data['action_log_probs'][ind : ind + data_chunk_length],
                        'rnn_states': data['rnn_states'][ind],
                    }

                    if data['action_type'] == 'mixed':
                        # 处理混合动作
                        sub_actions = []
                        for i in data['actions_dict']:
                            sub_actions.append(data['actions_dict'][i][ind : ind + data_chunk_length])
                        batch['actions'] = np.concatenate(sub_actions, axis=-1)

                        # 处理可用动作
                        batch['available_actions'] = None
                        for i in data['available_actions_dict']:
                            if i in data['available_actions_dict']:
                                batch['available_actions'] = data['available_actions_dict'][i][ind : ind + data_chunk_length]
                                break
                    else:
                        batch['actions'] = data['actions'][ind : ind + data_chunk_length]
                        if data['available_actions'] is not None:
                            batch['available_actions'] = data['available_actions'][ind : ind + data_chunk_length]
                        else:
                            batch['available_actions'] = None

                    agent_batches[aid].append(batch)

            L, N = data_chunk_length, mini_batch_size

            # Stack并flatten共享数据
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            returns_batch = np.stack(returns_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, self.num_agents, *self.rnn_states_critic.shape[3:]
            )

            share_obs_batch = _flatten(L, N, share_obs_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            returns_batch = _flatten(L, N, returns_batch)
            advantages_batch = _flatten(L, N, advantages_batch)

            if agent_id is not None:
                # 返回特定智能体的数据
                agent_batch_data = []
                for batch_dict in agent_batches[agent_id]:
                    agent_batch_data.append(batch_dict)

                # Stack并flatten智能体数据
                obs_batch = _flatten(L, N, np.stack([b['obs'] for b in agent_batch_data], axis=1))
                actions_batch = _flatten(L, N, np.stack([b['actions'] for b in agent_batch_data], axis=1))
                masks_batch = _flatten(L, N, np.stack([b['masks'] for b in agent_batch_data], axis=1))
                active_masks_batch = _flatten(L, N, np.stack([b['active_masks'] for b in agent_batch_data], axis=1))
                old_action_log_probs_batch = _flatten(L, N, np.stack([b['action_log_probs'] for b in agent_batch_data], axis=1))
                rnn_states_batch = np.stack([b['rnn_states'] for b in agent_batch_data]).reshape(
                    N, *agent_data[agent_id]['rnn_states'].shape[1:]
                )

                if agent_batch_data[0]['available_actions'] is not None:
                    available_actions_batch = _flatten(L, N, np.stack([b['available_actions'] for b in agent_batch_data], axis=1))
                else:
                    available_actions_batch = None

                yield (
                    share_obs_batch[:, agent_id],
                    obs_batch,
                    rnn_states_batch,
                    rnn_states_critic_batch[:, agent_id],
                    actions_batch,
                    value_preds_batch[:, agent_id],
                    returns_batch[:, agent_id],
                    masks_batch,
                    active_masks_batch,
                    old_action_log_probs_batch,
                    advantages_batch[:, agent_id],
                    available_actions_batch,
                )
            else:
                # 返回所有智能体的数据
                all_obs_batch = []
                all_actions_batch = []
                all_masks_batch = []
                all_active_masks_batch = []
                all_action_log_probs_batch = []
                all_available_actions_batch = []
                all_rnn_states_batch = []

                for aid in range(self.num_agents):
                    agent_batch_data = agent_batches[aid]

                    obs = _flatten(L, N, np.stack([b['obs'] for b in agent_batch_data], axis=1))
                    actions = _flatten(L, N, np.stack([b['actions'] for b in agent_batch_data], axis=1))
                    masks = _flatten(L, N, np.stack([b['masks'] for b in agent_batch_data], axis=1))
                    active_masks = _flatten(L, N, np.stack([b['active_masks'] for b in agent_batch_data], axis=1))
                    action_log_probs = _flatten(L, N, np.stack([b['action_log_probs'] for b in agent_batch_data], axis=1))
                    rnn_states = np.stack([b['rnn_states'] for b in agent_batch_data]).reshape(
                        N, *agent_data[aid]['rnn_states'].shape[1:]
                    )

                    if agent_batch_data[0]['available_actions'] is not None:
                        available_actions = _flatten(L, N, np.stack([b['available_actions'] for b in agent_batch_data], axis=1))
                    else:
                        available_actions = None

                    all_obs_batch.append(obs)
                    all_actions_batch.append(actions)
                    all_masks_batch.append(masks)
                    all_active_masks_batch.append(active_masks)
                    all_action_log_probs_batch.append(action_log_probs)
                    all_available_actions_batch.append(available_actions)
                    all_rnn_states_batch.append(rnn_states)

                yield (
                    share_obs_batch,
                    all_obs_batch,
                    all_rnn_states_batch,
                    rnn_states_critic_batch,
                    all_actions_batch,
                    value_preds_batch,
                    returns_batch,
                    all_masks_batch,
                    all_active_masks_batch,
                    all_action_log_probs_batch,
                    advantages_batch,
                    all_available_actions_batch,
                )
