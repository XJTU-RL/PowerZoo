"""
@File    : dan_happo.py
@Time    : 2024/05/24
@Author  : Xiaodong Zheng
@Description: DAN-HAPPO algorithm implementation combining Dynamic Agent Network with HAPPO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from algorithms.actors.on_policy_base import OnPolicyBase
from models.base.dan import DAN
from utils.util import check
from utils.valuenorm import ValueNorm

class DAN_HAPPO(OnPolicyBase):
    """DAN-HAPPO algorithm combining Dynamic Agent Network with HAPPO
    
    This implementation integrates the Dynamic Agent Network (DAN) architecture
    with the HAPPO algorithm to handle dynamic agent numbers and improve
    coordination in multi-agent environments.
    """
    
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        # Store DAN parameters before calling parent init
        self.use_dan = getattr(args, 'use_dan', True)
        self.dan_hidden_dim = getattr(args, 'dan_hidden_dim', 128)
        self.dan_num_heads = getattr(args, 'dan_num_heads', 4)
        self.dan_use_attention = getattr(args, 'dan_use_attention', True)
        self.dan_dropout = getattr(args, 'dan_dropout', 0.1)
        self.dan_layer_norm = getattr(args, 'dan_layer_norm', True)
        
        # DAN observation splitting parameters
        self.env_obs_ratio = getattr(args, 'env_obs_ratio', 0.6)  # Ratio of environmental obs
        self.use_neighbor_obs = getattr(args, 'use_neighbor_obs', True)
        
        # DAN training parameters
        self.dan_lr = getattr(args, 'dan_lr', args.lr)
        self.dan_weight_decay = getattr(args, 'dan_weight_decay', 1e-5)
        self.dan_grad_norm_max_norm = getattr(args, 'dan_grad_norm_max_norm', 10.0)
        
        super(DAN_HAPPO, self).__init__(args, obs_space, cent_obs_space, act_space, device)
        
        # Initialize DAN model if enabled
        if self.use_dan:
            self._init_dan_model(obs_space)
            
    def _init_dan_model(self, obs_space):
        """Initialize DAN model
        Args:
            obs_space: Observation space
        """
        # Determine observation dimensions
        if hasattr(obs_space, 'shape'):
            total_obs_dim = obs_space.shape[0]
        else:
            total_obs_dim = obs_space.n
            
        # Split observation into environmental and agent components
        self.env_obs_dim = int(total_obs_dim * self.env_obs_ratio)
        self.agent_obs_dim = total_obs_dim - self.env_obs_dim
        
        # Initialize DAN model
        self.dan_model = DAN(
            env_obs_dim=self.env_obs_dim,
            agent_obs_dim=self.agent_obs_dim,
            hidden_dim=self.dan_hidden_dim,
            num_heads=self.dan_num_heads,
            use_attention=self.dan_use_attention,
            dropout=self.dan_dropout,
            layer_norm=self.dan_layer_norm
        ).to(self.device)
        
        # Initialize DAN optimizer
        self.dan_optimizer = torch.optim.Adam(
            self.dan_model.parameters(),
            lr=self.dan_lr,
            weight_decay=self.dan_weight_decay
        )
        
        # Update encoded observation dimension
        self.dan_encoded_dim = self.dan_hidden_dim
        
        print(f"DAN Model initialized:")
        print(f"  - Total obs dim: {total_obs_dim}")
        print(f"  - Env obs dim: {self.env_obs_dim}")
        print(f"  - Agent obs dim: {self.agent_obs_dim}")
        print(f"  - Encoded dim: {self.dan_encoded_dim}")
        print(f"  - Use attention: {self.dan_use_attention}")
        print(f"  - Num heads: {self.dan_num_heads}")
        
    def split_observations(self, obs):
        """Split observations into environmental and agent components
        Args:
            obs: Original observations [batch_size, obs_dim]
        Returns:
            env_obs: Environmental observations [batch_size, env_obs_dim]
            agent_obs: Agent observations [batch_size, agent_obs_dim]
        """
        if not self.use_dan:
            return obs, None
            
        env_obs = obs[..., :self.env_obs_dim]
        agent_obs = obs[..., self.env_obs_dim:]
        
        return env_obs, agent_obs
        
    def encode_observations(self, obs, neighbor_obs=None, agent_mask=None):
        """Encode observations using DAN
        Args:
            obs: Original observations [batch_size, obs_dim]
            neighbor_obs: Neighboring agent observations [batch_size, num_neighbors, agent_obs_dim]
            agent_mask: Mask for valid neighboring agents [batch_size, num_neighbors]
        Returns:
            encoded_obs: DAN encoded observations [batch_size, dan_encoded_dim]
            attention_weights: Attention weights if using attention
        """
        if not self.use_dan:
            return obs, None
            
        # Split observations
        env_obs, agent_obs = self.split_observations(obs)
        
        # Prepare neighbor observations
        if neighbor_obs is None and self.use_neighbor_obs:
            # Use current agent observation as neighbor observation
            neighbor_obs = agent_obs.unsqueeze(1)  # [batch_size, 1, agent_obs_dim]
            if agent_mask is None:
                agent_mask = torch.ones(obs.shape[0], 1, device=obs.device)
        elif neighbor_obs is None:
            # Create dummy neighbor observations
            neighbor_obs = torch.zeros(obs.shape[0], 1, self.agent_obs_dim, device=obs.device)
            agent_mask = torch.zeros(obs.shape[0], 1, device=obs.device)
            
        # Encode using DAN
        encoded_obs, attention_weights = self.dan_model(env_obs, neighbor_obs, agent_mask)
        
        return encoded_obs, attention_weights
        
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, 
                   available_actions=None, deterministic=False, neighbor_obs=None, agent_mask=None):
        """Get actions using DAN encoded observations
        Args:
            cent_obs: Centralized observations
            obs: Decentralized observations
            rnn_states_actor: RNN states for actor
            rnn_states_critic: RNN states for critic
            masks: Masks
            available_actions: Available actions
            deterministic: Whether to use deterministic actions
            neighbor_obs: Neighboring agent observations
            agent_mask: Agent mask
        Returns:
            values: Value estimates
            actions: Actions
            action_log_probs: Action log probabilities
            rnn_states_actor: Updated RNN states for actor
            rnn_states_critic: Updated RNN states for critic
        """
        # Encode observations using DAN
        if self.use_dan:
            encoded_obs, _ = self.encode_observations(obs, neighbor_obs, agent_mask)
        else:
            encoded_obs = obs
            
        # Call parent method with encoded observations
        return super().get_actions(
            cent_obs, encoded_obs, rnn_states_actor, rnn_states_critic, masks,
            available_actions, deterministic
        )
        
    def get_values(self, cent_obs, obs, rnn_states_critic, masks, neighbor_obs=None, agent_mask=None):
        """Get values using DAN encoded observations
        Args:
            cent_obs: Centralized observations
            obs: Decentralized observations
            rnn_states_critic: RNN states for critic
            masks: Masks
            neighbor_obs: Neighboring agent observations
            agent_mask: Agent mask
        Returns:
            values: Value estimates
        """
        # Encode observations using DAN
        if self.use_dan:
            encoded_obs, _ = self.encode_observations(obs, neighbor_obs, agent_mask)
        else:
            encoded_obs = obs
            
        # Call parent method with encoded observations
        return super().get_values(cent_obs, encoded_obs, rnn_states_critic, masks)
        
    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                        available_actions=None, active_masks=None, neighbor_obs=None, agent_mask=None):
        """Evaluate actions using DAN encoded observations
        Args:
            cent_obs: Centralized observations
            obs: Decentralized observations
            rnn_states_actor: RNN states for actor
            rnn_states_critic: RNN states for critic
            action: Actions to evaluate
            masks: Masks
            available_actions: Available actions
            active_masks: Active masks
            neighbor_obs: Neighboring agent observations
            agent_mask: Agent mask
        Returns:
            values: Value estimates
            action_log_probs: Action log probabilities
            dist_entropy: Action distribution entropy
            rnn_states_actor: Updated RNN states for actor
            rnn_states_critic: Updated RNN states for critic
        """
        # Encode observations using DAN
        if self.use_dan:
            encoded_obs, _ = self.encode_observations(obs, neighbor_obs, agent_mask)
        else:
            encoded_obs = obs
            
        # Call parent method with encoded observations
        return super().evaluate_actions(
            cent_obs, encoded_obs, rnn_states_actor, rnn_states_critic, action, masks,
            available_actions, active_masks
        )
        
    def update(self, sample):
        """Update the DAN-HAPPO algorithm
        Args:
            sample: Training sample containing observations, actions, rewards, etc.
        Returns:
            train_info: Training information dictionary
        """
        # Extract DAN specific data from sample
        neighbor_obs = sample.get('neighbor_obs', None)
        agent_mask = sample.get('agent_mask', None)
        
        # Encode observations using DAN
        if self.use_dan:
            obs = check(sample['obs']).to(**self.tpdv)
            encoded_obs, attention_weights = self.encode_observations(obs, neighbor_obs, agent_mask)
            sample['obs'] = encoded_obs.cpu().numpy()
            
        # Call parent HAPPO update method
        train_info = super().update(sample)
        
        # Add DAN specific training information
        if self.use_dan:
            train_info['dan_grad_norm'] = 0.0  # Placeholder
            if attention_weights is not None:
                train_info['dan_attention_entropy'] = self._compute_attention_entropy(attention_weights)
                
        return train_info
        
    def _compute_attention_entropy(self, attention_weights):
        """Compute attention entropy for analysis
        Args:
            attention_weights: Attention weights [batch_size, num_heads, 1, num_neighbors]
        Returns:
            entropy: Average attention entropy
        """
        if attention_weights is None:
            return 0.0
            
        # Compute entropy across the neighbor dimension
        attention_probs = F.softmax(attention_weights, dim=-1)
        log_probs = F.log_softmax(attention_weights, dim=-1)
        entropy = -(attention_probs * log_probs).sum(dim=-1).mean()
        
        return entropy.item()
        
    def train(self, buffer, update_actor=True):
        """Train the DAN-HAPPO algorithm
        Args:
            buffer: Experience buffer
            update_actor: Whether to update the actor network
        Returns:
            train_infos: List of training information dictionaries
        """
        # Prepare data with DAN encoding if needed
        if self.use_dan:
            # TODO: Implement buffer preprocessing for DAN
            # This would involve encoding all observations in the buffer
            pass
            
        # Call parent HAPPO train method
        train_infos = super().train(buffer, update_actor)
        
        # Update DAN model parameters
        if self.use_dan and update_actor:
            self._update_dan_parameters()
            
        return train_infos
        
    def _update_dan_parameters(self):
        """Update DAN model parameters"""
        # Apply gradient clipping
        if self.dan_grad_norm_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.dan_model.parameters(), 
                self.dan_grad_norm_max_norm
            )
            
        # Update DAN optimizer
        self.dan_optimizer.step()
        self.dan_optimizer.zero_grad()
        
    def prep_training(self):
        """Prepare for training"""
        super().prep_training()
        if self.use_dan:
            self.dan_model.train()
            
    def prep_rollout(self):
        """Prepare for rollout"""
        super().prep_rollout()
        if self.use_dan:
            self.dan_model.eval()
            
    def save(self, save_dir):
        """Save model parameters
        Args:
            save_dir: Directory to save models
        """
        super().save(save_dir)
        if self.use_dan:
            torch.save(self.dan_model.state_dict(), str(save_dir) + "/dan_model.pt")
            torch.save(self.dan_optimizer.state_dict(), str(save_dir) + "/dan_optimizer.pt")
            
    def restore(self, model_dir):
        """Restore model parameters
        Args:
            model_dir: Directory containing saved models
        """
        super().restore(model_dir)
        if self.use_dan:
            dan_model_path = str(model_dir) + "/dan_model.pt"
            dan_optimizer_path = str(model_dir) + "/dan_optimizer.pt"
            
            if torch.cuda.is_available():
                self.dan_model.load_state_dict(torch.load(dan_model_path))
                self.dan_optimizer.load_state_dict(torch.load(dan_optimizer_path))
            else:
                self.dan_model.load_state_dict(torch.load(dan_model_path, map_location=torch.device('cpu')))
                self.dan_optimizer.load_state_dict(torch.load(dan_optimizer_path, map_location=torch.device('cpu')))
"""
@File    : dan_happo.py
@Time    : 2024/05/24
@Author  : Xiaodong Zheng
@Description: DAN-HAPPO algorithm implementation combining Dynamic Agent Network with HAPPO.
"""

import numpy as np
import torch
import torch.nn as nn
from utils.envs_tools import check
from utils.models_tools import get_grad_norm
from algorithms.actors.on_policy_base import OnPolicyBase
from models.base.dan import DAN

class DAN_HAPPO(OnPolicyBase):
    """DAN-HAPPO算法：结合动态智能体网络和HAPPO算法"""
    
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        """初始化 DAN-HAPPO 算法。
        Args:
            args: (dict) arguments.
            obs_space: (gym.spaces or list) observation space.
            act_space: (gym.spaces) action space.
            device: (torch.device) device to use for tensor operations.
        """
        super(DAN_HAPPO, self).__init__(args, obs_space, act_space, device)
        
        # HAPPO参数
        self.clip_param = args["clip_param"]
        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.entropy_coef = args["entropy_coef"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]
        
        # DAN参数
        self.use_dan = getattr(args, 'use_dan', True)
        self.dan_hidden_size = getattr(args, 'dan_hidden_size', 128)
        self.dan_attention_heads = getattr(args, 'dan_attention_heads', 4)
        self.dan_use_attention = getattr(args, 'dan_use_attention', True)
        self.dan_max_agents = getattr(args, 'dan_max_agents', 20)
        
        if self.use_dan:
            # 初始化DAN模块
            self.dan = DAN(obs_space, args).to(device)
            # 重新构建策略网络以使用DAN编码的观测
            self._rebuild_policy_with_dan()
            
    def _rebuild_policy_with_dan(self):
        """重新构建策略网络以使用DAN编码的观测"""
        # NOTE: 这里需要根据具体的策略网络结构进行调整
        # TODO: 实现策略网络的重构，使其能够接收DAN编码的观测
        pass
        
    def encode_observations(self, obs, agent_mask=None, neighbor_obs=None):
        """使用DAN编码观测
        Args:
            obs: 原始观测
            agent_mask: 智能体掩码
            neighbor_obs: 邻居智能体观测
        Returns:
            encoded_obs: 编码后的观测
        """
        if self.use_dan:
            return self.dan(obs, agent_mask, neighbor_obs)
        else:
            return obs
            
    def update(self, sample):
        """更新actor网络。
        Args:
            sample: (Tuple) 包含用于更新网络的数据批次。
        Returns:
            policy_loss: (torch.Tensor) actor(policy) 损失值。
            dist_entropy: (torch.Tensor) 动作熵。
            actor_grad_norm: (torch.Tensor) actor更新的梯度范数。
            imp_weights: (torch.Tensor) 重要性采样权重。
        """
        (
            obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            active_masks_batch,
            old_action_log_probs_batch,
            adv_targ,
            available_actions_batch,
            factor_batch,
        ) = sample
        
        # 如果有额外的DAN相关数据，从sample中提取
        agent_mask_batch = None
        neighbor_obs_batch = None
        if len(sample) > 9:
            agent_mask_batch = sample[9] if len(sample) > 9 else None
            neighbor_obs_batch = sample[10] if len(sample) > 10 else None

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        
        # 使用DAN编码观测
        if self.use_dan:
            encoded_obs_batch = self.encode_observations(
                obs_batch, agent_mask_batch, neighbor_obs_batch
            )
        else:
            encoded_obs_batch = obs_batch

        # 重塑以在一次前向传递中对所有步骤进行评估
        action_log_probs, dist_entropy, _ = self.evaluate_actions(
            encoded_obs_batch,
            rnn_states_batch,
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
        )

        # actor update
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs_batch),
            dim=-1,
            keepdim=True,
        )
        
        # HAPPO的截断代理目标函数
        surr1 = imp_weights * adv_targ
        surr2 = (
            torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True)
                * active_masks_batch
            ).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(
                factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True
            ).mean()

        policy_loss = policy_action_loss

        self.actor_optimizer.zero_grad()

        if self.use_dan:
            # 同时更新DAN参数
            dan_loss = policy_loss  # DAN的损失与策略损失相同
            total_loss = policy_loss + dan_loss
        else:
            total_loss = policy_loss

        (total_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            # 计算所有参数的梯度范数
            all_params = list(self.actor.parameters())
            if self.use_dan:
                all_params.extend(list(self.dan.parameters()))
            actor_grad_norm = nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
        else:
            all_params = list(self.actor.parameters())
            if self.use_dan:
                all_params.extend(list(self.dan.parameters()))
            actor_grad_norm = get_grad_norm(all_params)

        self.actor_optimizer.step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights
        
    def train(self, actor_buffer, advantages, num_agents_list, actor_train_infos):
        """训练actor网络。
        Args:
            actor_buffer: (OnPolicyActorBuffer) actor缓冲区。
            advantages: (np.ndarray) 优势值。
            num_agents_list: (list) 每个环境中的智能体数量列表。
            actor_train_infos: (list) actor训练信息列表。
        Returns:
            actor_train_infos: (list) 更新后的actor训练信息列表。
        """
        if self.use_recurrent_policy:
            data_generator = actor_buffer.recurrent_generator_actor(
                advantages, self.actor_num_mini_batch
            )
        elif self.use_naive_recurrent:
            data_generator = actor_buffer.naive_recurrent_generator_actor(
                advantages, self.actor_num_mini_batch
            )
        else:
            data_generator = actor_buffer.feed_forward_generator_actor(
                advantages, self.actor_num_mini_batch
            )

        for _ in range(self.ppo_epoch):
            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update(
                    sample
                )

                actor_train_infos.append(
                    {
                        "policy_loss": policy_loss.item(),
                        "dist_entropy": dist_entropy.item(),
                        "actor_grad_norm": actor_grad_norm.item(),
                        "ratio": imp_weights.mean().item(),
                    }
                )

        return actor_train_infos
        
    def prep_training(self):
        """准备训练模式"""
        super().prep_training()
        if self.use_dan:
            self.dan.train()
            
    def prep_rollout(self):
        """准备推理模式"""
        super().prep_rollout()
        if self.use_dan:
            self.dan.eval()