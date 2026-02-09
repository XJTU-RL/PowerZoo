# -*- coding: utf-8 -*-
"""
模型推理引擎

加载训练好的 actor 模型 (StochasticPolicy)，执行确定性或随机推理。
支持 MLP 和 RNN 两种策略架构。负责管理 RNN 隐状态的初始化和更新。
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# StochasticPolicy 构建所需的默认参数
_DEFAULT_POLICY_ARGS = {
	"hidden_sizes": [128, 128],
	"gain": 0.01,
	"initialization_method": "orthogonal_",
	"use_policy_active_masks": True,
	"use_naive_recurrent_policy": False,
	"use_recurrent_policy": False,
	"recurrent_n": 1,
	"use_feature_normalization": True,
	"use_ReLU": True,
	"activation_func": "relu",
	"stacked_frames": 1,
	"layer_N": 2,
	"std_x_coef": 1.0,
	"std_y_coef": 0.5,
}


class InferenceEngine:
	"""模型推理引擎

	加载训练好的 actor 模型，执行推理生成动作。
	支持 MLP 和 RNN 两种策略。每个 episode 开始前应调用
	reset_rnn_states() 重置隐状态。

	参数:
		checkpoint_dir: 检查点目录路径
		n_agents: 智能体数量
		obs_dims: 每个智能体的观测维度列表
		action_dims: 每个智能体的动作维度列表
		hidden_size: 隐藏层大小
		device: 推理设备 ("cpu" 或 "cuda")
	"""

	def __init__(
		self,
		checkpoint_dir: str,
		n_agents: int,
		obs_dims: List[int],
		action_dims: List[int],
		hidden_size: int = 128,
		device: str = "cpu",
	):
		self.checkpoint_dir = checkpoint_dir
		self.n_agents = n_agents
		self.obs_dims = obs_dims
		self.action_dims = action_dims
		self.hidden_size = hidden_size
		self.device = torch.device(device)

		# 模型列表，load_actors() 后填充
		self._actors: List[Optional[Any]] = [None] * n_agents
		self._loaded = False

		# RNN 隐状态 (每个 agent 独立维护)
		self._rnn_states: List[np.ndarray] = []
		self._use_rnn = False

		# 统一 masks (MLP 模式下恒为 1)
		self._masks = np.ones((1, 1), dtype=np.float32)

		# 动作维度上限 (用于输出零填充对齐)
		self._max_action_dim = max(action_dims) if action_dims else 0

		logger.info(
			f"InferenceEngine 初始化: n_agents={n_agents}, "
			f"obs_dims={obs_dims}, action_dims={action_dims}, "
			f"hidden_size={hidden_size}, device={device}"
		)

	def load_actors(self, policy_args: Optional[Dict[str, Any]] = None) -> None:
		"""加载所有 actor 模型的 state_dict

		为每个智能体构建与训练时相同架构的 StochasticPolicy，
		然后加载对应的 state_dict。

		参数:
			policy_args: 覆盖默认策略参数的字典。
				若为 None 则使用 _DEFAULT_POLICY_ARGS。

		异常:
			FileNotFoundError: actor 文件不存在
			RuntimeError: state_dict 加载失败 (架构不匹配)
		"""
		# NOTE: 延迟导入避免循环依赖和顶层模块加载开销
		from gymnasium.spaces import Box
		from models.policy_models.stochastic_policy import StochasticPolicy

		args = dict(_DEFAULT_POLICY_ARGS)
		if policy_args:
			args.update(policy_args)

		# 使用传入的 hidden_size 覆盖
		args["hidden_sizes"] = [self.hidden_size, self.hidden_size]

		# 检测是否使用 RNN
		self._use_rnn = (
			args.get("use_naive_recurrent_policy", False)
			or args.get("use_recurrent_policy", False)
		)

		for agent_id in range(self.n_agents):
			pt_path = os.path.join(
				self.checkpoint_dir, f"actor_agent{agent_id}.pt"
			)
			if not os.path.isfile(pt_path):
				raise FileNotFoundError(
					f"Actor 模型文件不存在: {pt_path}"
				)

			obs_dim = self.obs_dims[agent_id]
			act_dim = self.action_dims[agent_id]

			obs_space = Box(
				low=-np.inf, high=np.inf,
				shape=(obs_dim,), dtype=np.float32,
			)
			action_space = Box(
				low=-1.0, high=1.0,
				shape=(act_dim,), dtype=np.float32,
			)

			actor = StochasticPolicy(args, obs_space, action_space, self.device)

			state_dict = torch.load(pt_path, map_location=self.device, weights_only=False)
			actor.load_state_dict(state_dict)
			actor.eval()

			self._actors[agent_id] = actor
			logger.debug(
				f"Agent {agent_id}: 加载 {pt_path} "
				f"(obs={obs_dim}, act={act_dim})"
			)

		self._loaded = True
		self.reset_rnn_states()
		logger.info(
			f"已加载 {self.n_agents} 个 actor 模型 "
			f"(RNN={'on' if self._use_rnn else 'off'})"
		)

	def infer(
		self,
		obs_list: List[np.ndarray],
		deterministic: bool = True,
	) -> np.ndarray:
		"""执行推理

		将每个智能体的观测输入对应 actor，生成动作。
		非 RNN 模式下 rnn_states 和 masks 使用固定值。

		参数:
			obs_list: 每个智能体的观测 [np.ndarray(obs_dim,)]
			deterministic: 是否确定性推理 (True=取 mode, False=采样)

		返回:
			actions: np.ndarray(n_agents, max_action_dim)
				各 agent 的动作，短维度用零填充

		异常:
			RuntimeError: 模型未加载
		"""
		if not self._loaded:
			raise RuntimeError("模型未加载，请先调用 load_actors()")

		actions = np.zeros(
			(self.n_agents, self._max_action_dim), dtype=np.float32
		)

		with torch.no_grad():
			for agent_id in range(self.n_agents):
				actor = self._actors[agent_id]
				obs = obs_list[agent_id]

				# 增加 batch 维度: (obs_dim,) -> (1, obs_dim)
				obs_input = np.expand_dims(obs, axis=0).astype(np.float32)
				rnn_state = self._rnn_states[agent_id]
				masks = self._masks

				act_tensor, _, new_rnn = actor.forward(
					obs_input, rnn_state, masks,
					available_actions=None,
					deterministic=deterministic,
				)

				# 更新 RNN 隐状态
				if self._use_rnn:
					self._rnn_states[agent_id] = new_rnn.cpu().numpy()

				# 提取动作并填充到统一维度
				act_np = act_tensor.cpu().numpy().flatten()
				act_dim = self.action_dims[agent_id]
				actions[agent_id, :act_dim] = act_np[:act_dim]

		return actions

	def reset_rnn_states(self) -> None:
		"""重置 RNN 隐状态

		每个 episode 开始时调用，将所有 agent 的隐状态
		初始化为零向量。MLP 模式下同样初始化 (forward 忽略)。
		"""
		recurrent_n = _DEFAULT_POLICY_ARGS.get("recurrent_n", 1)
		self._rnn_states = [
			np.zeros((1, recurrent_n, self.hidden_size), dtype=np.float32)
			for _ in range(self.n_agents)
		]

	def get_model_info(self) -> Dict[str, Any]:
		"""获取模型信息

		返回各 agent 的参数量、架构摘要和推理配置。

		返回:
			模型信息字典，包含:
			- n_agents: 智能体数量
			- device: 推理设备
			- use_rnn: 是否使用 RNN
			- hidden_size: 隐藏层大小
			- checkpoint_dir: 检查点路径
			- agents: 各 agent 的详细信息列表
				- agent_id: 智能体索引
				- obs_dim: 观测维度
				- action_dim: 动作维度
				- total_params: 总参数量
				- trainable_params: 可训练参数量
		"""
		info: Dict[str, Any] = {
			"n_agents": self.n_agents,
			"device": str(self.device),
			"use_rnn": self._use_rnn,
			"hidden_size": self.hidden_size,
			"checkpoint_dir": self.checkpoint_dir,
			"loaded": self._loaded,
			"agents": [],
		}

		if not self._loaded:
			return info

		for agent_id in range(self.n_agents):
			actor = self._actors[agent_id]
			total_params = sum(p.numel() for p in actor.parameters())
			trainable_params = sum(
				p.numel() for p in actor.parameters() if p.requires_grad
			)

			info["agents"].append({
				"agent_id": agent_id,
				"obs_dim": self.obs_dims[agent_id],
				"action_dim": self.action_dims[agent_id],
				"total_params": total_params,
				"trainable_params": trainable_params,
			})

		return info
