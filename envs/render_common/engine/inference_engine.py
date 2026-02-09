"""
Inference Engine (Common)
模型推理引擎

加载训练好的 actor 模型 (StochasticPolicy)，执行确定性或随机推理。
支持 MLP 和 RNN 两种策略架构。从 District Dispatch 提取，完全可复用。
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

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
	支持 MLP 和 RNN 两种策略。

	Args:
		checkpoint_dir: 检查点目录路径
		n_agents: 智能体数量
		obs_dims: 每个智能体的观测维度列表
		action_dims: 每个智能体的动作维度列表
		hidden_size: 隐藏层大小
		device: 推理设备
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

		self._actors: List[Optional[Any]] = [None] * n_agents
		self._loaded = False
		self._rnn_states: List[np.ndarray] = []
		self._use_rnn = False
		self._masks = np.ones((1, 1), dtype=np.float32)
		self._max_action_dim = max(action_dims) if action_dims else 0

		logger.info(
			f"InferenceEngine init: n_agents={n_agents}, "
			f"obs_dims={obs_dims}, action_dims={action_dims}, "
			f"hidden_size={hidden_size}, device={device}"
		)

	def load_actors(self, policy_args: Optional[Dict[str, Any]] = None) -> None:
		"""加载所有 actor 模型的 state_dict

		Args:
			policy_args: 覆盖默认策略参数的字典
		"""
		from gymnasium.spaces import Box
		from models.policy_models.stochastic_policy import StochasticPolicy

		args = dict(_DEFAULT_POLICY_ARGS)
		if policy_args:
			args.update(policy_args)

		args["hidden_sizes"] = [self.hidden_size, self.hidden_size]

		self._use_rnn = (
			args.get("use_naive_recurrent_policy", False)
			or args.get("use_recurrent_policy", False)
		)

		for agent_id in range(self.n_agents):
			pt_path = os.path.join(
				self.checkpoint_dir, f"actor_agent{agent_id}.pt"
			)
			if not os.path.isfile(pt_path):
				raise FileNotFoundError(f"Actor model not found: {pt_path}")

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
				f"Agent {agent_id}: loaded {pt_path} (obs={obs_dim}, act={act_dim})"
			)

		self._loaded = True
		self.reset_rnn_states()
		logger.info(
			f"Loaded {self.n_agents} actor models "
			f"(RNN={'on' if self._use_rnn else 'off'})"
		)

	def infer(
		self,
		obs_list: List[np.ndarray],
		deterministic: bool = True,
		available_actions: Optional[List[Optional[np.ndarray]]] = None,
	) -> np.ndarray:
		"""执行推理

		Args:
			obs_list: 每个智能体的观测
			deterministic: 是否确定性推理
			available_actions: 每个智能体的可用动作掩码 (DSR 环境需要)

		Returns:
			actions: np.ndarray(n_agents, max_action_dim)
		"""
		if not self._loaded:
			raise RuntimeError("Models not loaded, call load_actors() first")

		actions = np.zeros(
			(self.n_agents, self._max_action_dim), dtype=np.float32
		)

		with torch.no_grad():
			for agent_id in range(self.n_agents):
				actor = self._actors[agent_id]
				obs = obs_list[agent_id]

				obs_input = np.expand_dims(obs, axis=0).astype(np.float32)
				rnn_state = self._rnn_states[agent_id]
				masks = self._masks

				avail = None
				if available_actions is not None and agent_id < len(available_actions):
					avail = available_actions[agent_id]

				act_tensor, _, new_rnn = actor.forward(
					obs_input, rnn_state, masks,
					available_actions=avail,
					deterministic=deterministic,
				)

				if self._use_rnn:
					self._rnn_states[agent_id] = new_rnn.cpu().numpy()

				act_np = act_tensor.cpu().numpy().flatten()
				act_dim = self.action_dims[agent_id]
				actions[agent_id, :act_dim] = act_np[:act_dim]

		return actions

	def reset_rnn_states(self) -> None:
		"""重置 RNN 隐状态"""
		recurrent_n = _DEFAULT_POLICY_ARGS.get("recurrent_n", 1)
		self._rnn_states = [
			np.zeros((1, recurrent_n, self.hidden_size), dtype=np.float32)
			for _ in range(self.n_agents)
		]

	def get_model_info(self) -> Dict[str, Any]:
		"""获取模型信息

		Returns:
			模型信息字典
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
