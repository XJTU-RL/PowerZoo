#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from utils.envs_tools import get_shape_from_obs_space

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
        self.dan_hidden_dim = getattr(args, 'dan_hidden_dim', 128)
        self.dan_num_heads = getattr(args, 'dan_num_heads', 4)
        self.dan_use_attention = getattr(args, 'dan_use_attention', True)
        self.dan_dropout = getattr(args, 'dan_dropout', 0.1)
        self.use_layer_norm = getattr(args, 'use_layer_norm', True)
        self.env_obs_ratio = getattr(args, 'env_obs_ratio', 0.7)
        self.max_neighbors = getattr(args, 'max_neighbors', 5)
        
        if self.use_dan:
            # 获取观测空间维度
            obs_shape = get_shape_from_obs_space(obs_space)
            if isinstance(obs_shape[0], list):
                obs_dim = obs_shape[0][0]
            else:
                obs_dim = obs_shape[0]
            
            # 根据论文，将观测分为环境观测和邻居观测
            # env_obs_dim是环境观测的维度，neighbor_obs_dim是邻居观测的完整维度
            self.env_obs_dim = int(obs_dim * self.env_obs_ratio)
            self.neighbor_obs_dim = obs_dim  # 邻居观测包含完整的观测信息
            
            # 初始化DAN模块
            self.dan = DAN(
                env_obs_dim=self.env_obs_dim,
                neighbor_obs_dim=self.neighbor_obs_dim,
                hidden_dim=self.dan_hidden_dim,
                num_heads=self.dan_num_heads,
                use_attention=self.dan_use_attention,
                dropout=self.dan_dropout,
                layer_norm=self.use_layer_norm
            ).to(device)
            
            # 为DAN创建优化器
            self.dan_optimizer = torch.optim.Adam(
                self.dan.parameters(),
                lr=getattr(args, 'dan_lr', args["lr"]),
                weight_decay=getattr(args, 'dan_weight_decay', 1e-5)
            )
            
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                   neighbor_obs=None, agent_mask=None, deterministic=False):
        """Get actions for all agents with DAN encoding."""
        if self.use_dan and neighbor_obs is not None:
            # 分离环境观测和其他观测
            env_obs = obs[..., :self.env_obs_dim]
            
            # 使用DAN编码观测
            with torch.no_grad():
                encoded_obs, _ = self.dan(env_obs, neighbor_obs, agent_mask)
            
            # 使用编码后的观测获取动作
            actions, action_log_probs, rnn_states_actor = self.actor(
                encoded_obs, rnn_states_actor, masks, available_actions, deterministic
            )
        else:
            # 标准HAPPO行为
            actions, action_log_probs, rnn_states_actor = self.actor(
                obs, rnn_states_actor, masks, available_actions, deterministic
            )
        
        # 获取价值估计
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        
    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, actions, masks,
                        available_actions=None, active_masks=None, neighbor_obs=None, agent_mask=None):
        """Evaluate actions with DAN encoding."""
        if self.use_dan and neighbor_obs is not None:
            # 分离环境观测
            env_obs = obs[..., :self.env_obs_dim]
            
            # 使用DAN编码观测
            encoded_obs, attention_weights = self.dan(env_obs, neighbor_obs, agent_mask)
            
            # 评估动作
            action_log_probs, dist_entropy = self.actor.evaluate_actions(
                encoded_obs, rnn_states_actor, actions, masks, available_actions, active_masks
            )
        else:
            # 标准HAPPO行为
            action_log_probs, dist_entropy = self.actor.evaluate_actions(
                obs, rnn_states_actor, actions, masks, available_actions, active_masks
            )
        
        # 获取价值估计
        values = self.critic.get_values(cent_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy
            
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
        if len(sample) > 10:
            neighbor_obs_batch = sample[10]
            agent_mask_batch = sample[11]
        else:
            neighbor_obs_batch = None
            agent_mask_batch = None

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        
        # 使用DAN评估动作
        values, action_log_probs, dist_entropy = self.evaluate_actions(
            obs_batch,  # 这里传入完整观测，evaluate_actions内部会处理
            obs_batch,  # cent_obs
            rnn_states_batch,
            rnn_states_batch,  # rnn_states_critic
            actions_batch,
            masks_batch,
            available_actions_batch,
            active_masks_batch,
            neighbor_obs_batch,
            agent_mask_batch
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

        # 计算总损失
        total_loss = policy_loss - dist_entropy * self.entropy_coef
        
        self.actor_optimizer.zero_grad()
        if self.use_dan:
            self.dan_optimizer.zero_grad()
        
        total_loss.backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            if self.use_dan:
                dan_grad_norm = nn.utils.clip_grad_norm_(self.dan.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actor.parameters())
            if self.use_dan:
                dan_grad_norm = get_grad_norm(self.dan.parameters())

        self.actor_optimizer.step()
        if self.use_dan:
            self.dan_optimizer.step()

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
            
    def save(self, save_dir):
        """保存模型
        Args:
            save_dir: (str) 保存目录
        """
        super().save(save_dir)
        if self.use_dan:
            torch.save(self.dan.state_dict(), str(save_dir) + "/dan.pt")
            
    def restore(self, model_dir):
        """恢复模型
        Args:
            model_dir: (str) 模型目录
        """
        super().restore(model_dir)
        if self.use_dan:
            dan_model_path = str(model_dir) + "/dan.pt"
            if self.device == torch.device("cpu"):
                self.dan.load_state_dict(torch.load(dan_model_path, map_location=torch.device('cpu')))
            else:
                self.dan.load_state_dict(torch.load(dan_model_path))