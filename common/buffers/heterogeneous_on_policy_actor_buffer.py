# -*- coding: utf-8 -*-
"""
@File      : heterogeneous_on_policy_actor_buffer.py
@Time      : 2025-07-31 15:50
@Author    : Claude & Xiaodong Zheng
@Description: 异构动作空间兼容的On-Policy Actor Buffer
- 专门设计用于处理混合动作空间（离散+连续）的多智能体强化学习
- 解决HAPPO算法在处理异构智能体时的动作存储形状不一致问题
- 完全兼容原有OnPolicyActorBuffer接口，可直接替换使用
"""

import torch
import numpy as np
from gym.spaces import Discrete, Box
from utils.trans_tools import _flatten, _sa_cast
from utils.envs_tools import get_shape_from_obs_space


class HeterogeneousOnPolicyActorBuffer:
    """异构动作空间兼容的On-Policy Actor Buffer"""
    
    # 明确的类型标识，用于运行时识别异构buffer
    _buffer_class_type = 'heterogeneous'

    def __init__(self, args, obs_space, act_space):
        """初始化异构动作空间Buffer
        
        Args:
            args: (dict) 参数配置
            obs_space: (gym.Space or list) 观测空间
            act_space: (gym.Space or List[gym.Space]) 动作空间，支持异构
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

        # 解析异构动作空间
        self._parse_heterogeneous_action_space(act_space)
        
        # 初始化异构动作缓冲区
        self._init_heterogeneous_action_buffers()
        
        # 其他缓冲区（与原版相同）
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )
        self.active_masks = np.ones(
            (self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32
        )

        self.step = 0

    def _parse_heterogeneous_action_space(self, act_space):
        """解析异构动作空间结构"""
        if isinstance(act_space, list):
            # 异构动作空间：List[Union[Discrete, Box]]
            self.action_spaces = act_space
            self.is_heterogeneous = True
            self.num_agents = len(act_space)
        else:
            # 同构动作空间，转换为列表形式处理
            self.action_spaces = [act_space]
            self.is_heterogeneous = False
            self.num_agents = 1
        
        # 分析各智能体动作空间属性
        self.action_shapes = {}
        self.action_types = {}
        self.max_action_dim = 0
        
        for agent_id, space in enumerate(self.action_spaces):
            if isinstance(space, Discrete):
                self.action_shapes[agent_id] = (1,)  # 离散动作形状
                self.action_types[agent_id] = 'discrete'
                self.max_action_dim = max(self.max_action_dim, 1)
            elif isinstance(space, Box):
                self.action_shapes[agent_id] = space.shape
                self.action_types[agent_id] = 'continuous'
                self.max_action_dim = max(self.max_action_dim, space.shape[0])
            else:
                raise NotImplementedError(f"不支持的动作空间类型: {type(space)}")
    
    def _init_heterogeneous_action_buffers(self):
        """初始化异构动作缓冲区"""
        self.actions = {}
        self.available_actions = {}
        
        for agent_id in range(self.num_agents):
            action_shape = self.action_shapes[agent_id]
            
            # 为每个智能体创建独立的动作缓冲区
            self.actions[agent_id] = np.zeros(
                (self.episode_length, self.n_rollout_threads, *action_shape),
                dtype=np.float32
            )
            
            # 可用动作缓冲区
            if self.action_types[agent_id] == 'discrete':
                act_space = self.action_spaces[agent_id]
                self.available_actions[agent_id] = np.ones(
                    (self.episode_length + 1, self.n_rollout_threads, act_space.n),
                    dtype=np.float32,
                )
            else:
                # 连续动作空间不需要available_actions
                self.available_actions[agent_id] = None

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
        """插入数据到异构Buffer中
        
        Args:
            obs: 观测数据
            rnn_states: RNN状态
            actions: 动作数据 (n_rollout_threads, num_agents, action_dim)
            action_log_probs: 动作对数概率
            masks: 掩码
            active_masks: 激活掩码
            available_actions: 可用动作
        """
        # 插入观测和状态数据（与原版相同）
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.masks[self.step + 1] = masks.copy()
        
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()

        # 智能体级异构动作插入
        self._insert_heterogeneous_actions(actions)
        
        # 处理可用动作
        if available_actions is not None:
            self._insert_available_actions(available_actions)

        self.step = (self.step + 1) % self.episode_length

    def _insert_heterogeneous_actions(self, actions):
        """插入异构动作数据"""
        if not isinstance(actions, np.ndarray):
            # 如果actions不是numpy数组，尝试转换
            actions = np.array(actions)
        
        # 检查动作数据结构
        if actions.ndim == 2:  # (n_rollout_threads, num_agents)
            # 标准格式：每个环境每个智能体一个动作
            for agent_id in range(self.num_agents):
                agent_actions = actions[:, agent_id]
                
                # 根据动作类型进行适配
                if self.action_types[agent_id] == 'discrete':
                    # 离散动作：确保为整数并reshape为(n_threads, 1)
                    if agent_actions.ndim == 1:
                        agent_actions = agent_actions.reshape(-1, 1)
                    self.actions[agent_id][self.step] = agent_actions.astype(np.int32)
                else:
                    # 连续动作：保持原始形状
                    expected_shape = self.action_shapes[agent_id]
                    if agent_actions.shape[-len(expected_shape):] != expected_shape: 
                        # 形状不匹配时的处理
                        agent_actions = self._reshape_continuous_action(
                            agent_actions, expected_shape
                        )
                    self.actions[agent_id][self.step] = agent_actions.astype(np.float32)
        
        elif actions.ndim == 3:  # (n_rollout_threads, num_agents, action_dim)
            # 三维格式：处理不同动作维度
            for agent_id in range(self.num_agents):
                agent_actions = actions[:, agent_id, :]
                
                if self.action_types[agent_id] == 'discrete':
                    # 离散动作：取第一个元素作为动作值
                    if agent_actions.shape[-1] > 1:
                        agent_actions = agent_actions[:, 0:1]
                    self.actions[agent_id][self.step] = agent_actions.astype(np.int32)
                else:
                    # 连续动作：直接使用
                    expected_shape = self.action_shapes[agent_id]
                    if agent_actions.shape[-1] != expected_shape[0]:
                        agent_actions = agent_actions[:, :expected_shape[0]]
                    self.actions[agent_id][self.step] = agent_actions.astype(np.float32)
        
        else:
            raise ValueError(f"不支持的动作数据结构: {actions.shape}")

    def _reshape_continuous_action(self, action, expected_shape):
        """重塑连续动作到期望形状"""
        if action.ndim == 1:
            # 一维动作，扩展到期望形状
            if len(expected_shape) == 1:
                return action[:expected_shape[0]].reshape(-1, expected_shape[0])
            else:
                return action.reshape(-1, *expected_shape)
        else:
            # 多维动作，截断或填充
            return action[:, :expected_shape[0]]

    def _insert_available_actions(self, available_actions):
        """插入可用动作数据，增强维度验证和错误处理"""
        if isinstance(available_actions, list):
            for agent_id in range(self.num_agents):
                if (self.available_actions[agent_id] is not None and 
                    agent_id < len(available_actions)):
                    agent_avail = available_actions[agent_id]
                    if isinstance(agent_avail, list):
                        agent_avail = np.array(agent_avail)
                    
                    # 维度验证：确保数据赋值不越界
                    target_array = self.available_actions[agent_id]
                    target_step = self.step + 1
                    
                    if target_step >= target_array.shape[0]:
                        raise IndexError(
                            f"尝试访问 available_actions[{agent_id}][{target_step}]，"
                            f"但数组第一维大小只有 {target_array.shape[0]}。"
                            f"当前 step={self.step}, episode_length={self.episode_length}"
                        )
                    
                    # 形状验证：确保数据形状匹配
                    expected_shape = target_array.shape[1:]  # 除第一维外的形状
                    if agent_avail.shape != expected_shape:
                        raise ValueError(
                            f"智能体 {agent_id} 的 available_actions 形状不匹配。"
                            f"期望形状: {expected_shape}, 实际形状: {agent_avail.shape}"
                        )
                    
                    self.available_actions[agent_id][target_step] = agent_avail.copy()

    def after_update(self):
        """更新后的数据复制"""
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        
        for agent_id in range(self.num_agents):
            if self.available_actions[agent_id] is not None:
                self.available_actions[agent_id][0] = self.available_actions[agent_id][-1].copy()

    def get_agent_actions(self, agent_id):
        """获取指定智能体的动作数据"""
        return self.actions[agent_id]

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

        # 处理观测、状态、掩码数据
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, 1)

        # 处理异构动作数据
        actions_dict = {}
        available_actions_dict = {}
        
        for agent_id in range(self.num_agents):
            # 动作数据
            agent_actions = self.actions[agent_id].reshape(-1, *self.actions[agent_id].shape[2:])
            actions_dict[agent_id] = agent_actions
            
            # 可用动作数据
            if self.available_actions[agent_id] is not None:
                agent_avail = self.available_actions[agent_id][:-1].reshape(
                    -1, *self.available_actions[agent_id].shape[2:]
                )
                available_actions_dict[agent_id] = agent_avail
            else:
                available_actions_dict[agent_id] = None

        # 生成mini-batch
        for indices in sampler:
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            actions_batch = {}
            
            for agent_id in range(self.num_agents):
                actions_batch[agent_id] = actions_dict[agent_id][indices]
            
            action_log_probs_batch = action_log_probs[indices]
            
            available_actions_batch = {}
            for agent_id in range(self.num_agents):
                if available_actions_dict[agent_id] is not None:
                    available_actions_batch[agent_id] = available_actions_dict[agent_id][indices]
                else:
                    available_actions_batch[agent_id] = None
            
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            advantages_batch = advantages[indices]

            yield obs_batch, rnn_states_batch, actions_batch, action_log_probs_batch, \
                  available_actions_batch, masks_batch, active_masks_batch, advantages_batch

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

        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            idxes = perm[start_ind : start_ind + num_envs_per_batch]
            
            obs_batch = self.obs[:-1, idxes]
            rnn_states_batch = self.rnn_states[:-1, idxes]
            
            # 处理异构动作
            actions_batch = {}
            available_actions_batch = {}
            
            for agent_id in range(self.num_agents):
                actions_batch[agent_id] = self.actions[agent_id][:, idxes]
                
                if self.available_actions[agent_id] is not None:
                    available_actions_batch[agent_id] = self.available_actions[agent_id][:-1, idxes]
                else:
                    available_actions_batch[agent_id] = None
            
            action_log_probs_batch = self.action_log_probs[:, idxes]
            advantages_batch = advantages[:, idxes]
            masks_batch = self.masks[:-1, idxes]
            active_masks_batch = self.active_masks[:-1, idxes]

            yield obs_batch, rnn_states_batch, actions_batch, action_log_probs_batch, \
                  available_actions_batch, masks_batch, active_masks_batch, advantages_batch

    def recurrent_generator_actor(self, advantages, actor_num_mini_batch, data_chunk_length):
        """为Actor生成循环网络训练数据（分块版本）"""
        episode_length, n_rollout_threads = self.obs.shape[0:2] # 获取episode_length和n_rollout_threads
        batch_size = n_rollout_threads * episode_length # 计算出batch_size， batch_size是由线程数*episode_length 得到的
        data_chunks = batch_size // data_chunk_length #计算出data_chuncks 数量
        mini_batch_size = data_chunks // actor_num_mini_batch  

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO要求进程数量n_rollout_threads ({}) *  回合长度 episode_length({}) "
            "必须大于或等于数据块长度 data_chunk_length长度({}).".format(
                n_rollout_threads, episode_length, data_chunk_length
            )
        )

        # 生成随机排列
        rand = torch.randperm(data_chunks).numpy()
        # 根据mini_batch_size将随机索引分成actor_num_mini_batch组
        # 每组大小为mini_batch_size，用于后续批量训练
        # 例如: rand=[1,5,3,2,0,4], mini_batch_size=2, actor_num_mini_batch=3
        # 则sampler=[[1,5], [3,2], [0,4]]
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(actor_num_mini_batch)
        ]

        # 重塑数据
        # _sa_cast 将形状为 (episode_length, n_rollout_threads, *dim) 的数据
        # （其中 episode_length 是时间步长， n_rollout_threads 是并行收集数据的线程数，
        # `*dim` 是数据本身的维度，例如观测或RNN状态的维度）转换为 (n_rollout_threads * episode_length, *dim)。
        ''' 
            _sa_cast 函数在 recurrent_generator_actor 方法中的作用，是将 同一个智能体在多个rollout线程中收集到的数据 
            进行重塑，以便于高效地进行批处理。
            具体来说，它将形状为 (episode_length, n_rollout_threads, *dim) 的数据（其中 episode_length 是时间步长， n_rollout_threads 是并行收集数据的线程数， *dim 是数据本身的维度，例如观测或RNN状态的维度）转换为 (n_rollout_threads * episode_length, *dim) 。
            这个转换的目的是：
            1.批处理效率 ：将所有线程在所有时间步的数据展平，形成一个大的批次，
              这样可以一次性输入到神经网络中进行计算，提高GPU利用率和训练效率。
            2. RNN输入格式 ：虽然RNN在逻辑上处理序列数据，但在实际的PyTorch等深度学习框架中，
            通常需要将序列数据展平为 (batch_size, feature_dim) 的形式进行批处理。
            _sa_cast 正是完成了这一展平操作，使得每个智能体的RNN状态或观测数据能够以正确的格式送入其对应的RNN模型。
        '''
        obs = _sa_cast(self.obs[:-1]) 
        rnn_states = (
            self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        )
        # 处理异构动作数据
        actions_dict = {}
        available_actions_dict = {}
        
        for agent_id in range(self.num_agents):
            # 同样的将不同进程的动作收集并展平
            actions_dict[agent_id] = _sa_cast(self.actions[agent_id])
            
            if self.available_actions[agent_id] is not None:
                available_actions_dict[agent_id] = _sa_cast(self.available_actions[agent_id][:-1])
            else:
                available_actions_dict[agent_id] = None

        action_log_probs = _sa_cast(self.action_log_probs)
        advantages = _sa_cast(advantages)
        masks = _sa_cast(self.masks[:-1])
        active_masks = _sa_cast(self.active_masks[:-1])

        for indices in sampler:
            # 初始化用于存储当前mini-batch数据的列表
            obs_batch = []
            rnn_states_batch = []
            # 为每个智能体初始化动作批次字典，用于存储异构动作数据
            actions_batch_dict = {agent_id: [] for agent_id in range(self.num_agents)}
            # 为每个智能体初始化可用动作批次字典
            available_actions_batch_dict = {agent_id: [] for agent_id in range(self.num_agents)}
            action_log_probs_batch = []
            advantages_batch = []
            masks_batch = []
            active_masks_batch = []

            # 遍历当前mini-batch的索引
            for index in indices:
                # 计算当前数据块的起始索引
                ind = index * data_chunk_length
                # 从预处理好的数据中切片获取观测数据，并添加到当前批次列表
                obs_batch.append(obs[ind : ind + data_chunk_length])
                # 从预处理好的数据中切片获取RNN状态数据，并添加到当前批次列表
                rnn_states_batch.append(rnn_states[ind])
                
                # 遍历所有智能体，处理其动作和可用动作数据
                for agent_id in range(self.num_agents):
                    # 获取当前智能体的动作数据块，并添加到其对应的批次列表
                    actions_batch_dict[agent_id].append(
                        actions_dict[agent_id][ind : ind + data_chunk_length]
                    )
                    
                    # 如果存在可用动作，则获取并添加到批次列表
                    if available_actions_dict[agent_id] is not None:
                        available_actions_batch_dict[agent_id].append(
                            available_actions_dict[agent_id][ind : ind + data_chunk_length]
                        )
                    # 否则，添加None到批次列表
                    else:
                        available_actions_dict[agent_id].append(None)

                # 获取动作对数概率数据块，并添加到批次列表
                action_log_probs_batch.append(action_log_probs[ind : ind + data_chunk_length])
                # 获取优势函数数据块，并添加到批次列表
                advantages_batch.append(advantages[ind : ind + data_chunk_length])
                # 获取掩码数据块（用于处理episode结束等情况），并添加到批次列表
                masks_batch.append(masks[ind : ind + data_chunk_length])
                # 获取活跃掩码数据块（用于指示哪些智能体在当前时间步是活跃的），并添加到批次列表
                active_masks_batch.append(active_masks[ind : ind + data_chunk_length])

            # 转换为numpy数组 (data_chunk_length, mini_batch_size, *dim)
            L, N = data_chunk_length, mini_batch_size
            actions_batch = {}
            available_actions_batch = {}
            
            obs_batch = np.stack(obs_batch, axis=1)
            for agent_id in range(self.num_agents):
                actions_batch[agent_id] = np.stack(actions_batch_dict[agent_id], axis=1)
                if actions_batch_dict[agent_id][0] is not None:
                    available_actions_batch[agent_id] = np.stack(
                        available_actions_batch_dict[agent_id], axis=1
                    )
                else:
                    available_actions_batch[agent_id] = None
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])
            action_log_probs_batch = np.stack(action_log_probs_batch, axis=1)
            advantages_batch = np.stack(advantages_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)

            
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
            
            yield obs_batch, rnn_states_batch, actions_batch,masks_batch, active_masks_batch, action_log_probs_batch,  advantages_batch, available_actions_batch,


    def update_factor(self, factor):
        """更新因子（保持接口兼容性）"""
        self.factor = factor.copy()
    
    def debug_available_actions_info(self):
        """调试方法：打印available_actions的维度信息"""
        print("=== 异构Buffer Available Actions 调试信息 ===")
        print(f"num_agents: {self.num_agents}")
        print(f"episode_length: {self.episode_length}")
        print(f"n_rollout_threads: {self.n_rollout_threads}")
        print(f"current step: {self.step}")
        
        for agent_id in range(self.num_agents):
            if self.available_actions[agent_id] is not None:
                shape = self.available_actions[agent_id].shape
                print(f"Agent {agent_id}: available_actions shape = {shape}")
                print(f"  - 动作类型: {self.action_types[agent_id]}")
                if hasattr(self, 'action_spaces'):
                    space = self.action_spaces[agent_id]
                    if hasattr(space, 'n'):
                        print(f"  - 动作空间大小: {space.n}")
            else:
                print(f"Agent {agent_id}: available_actions = None (连续动作空间)")
        print("=" * 50)