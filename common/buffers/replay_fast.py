# common/buffers/replay_fast.py
import torch
import numpy as np
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class FastReplayBuffer:
	"""Replay buffer for fast layer (minute-level transitions)."""
	
	def __init__(self, capacity: int, obs_dim: int, action_dim: int):
		self.capacity = capacity
		self.obs_dim = obs_dim
		self.action_dim = action_dim
		
		# Pre-allocate memory
		self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
		self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
		self.rewards = np.zeros(capacity, dtype=np.float32)
		self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
		self.dones = np.zeros(capacity, dtype=np.float32)
		
		self.ptr = 0
		self.size = 0
		
		logger.info(f"Fast replay buffer initialized: capacity={capacity}, obs_dim={obs_dim}, action_dim={action_dim}")
	
	def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
			next_obs: np.ndarray, done: bool):
		"""Add a transition to the buffer."""
		self.obs[self.ptr] = obs
		self.actions[self.ptr] = action
		self.rewards[self.ptr] = reward
		self.next_obs[self.ptr] = next_obs
		self.dones[self.ptr] = float(done)
		
		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)
	
	def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
		"""Sample a batch of transitions."""
		if self.size < batch_size:
			raise ValueError(f"Buffer size {self.size} < batch_size {batch_size}")
		
		indices = np.random.choice(self.size, batch_size, replace=False)
		
		batch = {
			"obs": torch.FloatTensor(self.obs[indices]),
			"actions": torch.FloatTensor(self.actions[indices]),
			"rewards": torch.FloatTensor(self.rewards[indices]).unsqueeze(-1),
			"next_obs": torch.FloatTensor(self.next_obs[indices]),
			"dones": torch.FloatTensor(self.dones[indices]).unsqueeze(-1)
		}
		
		return batch
	
	def __len__(self) -> int:
		return self.size
	
	def clear(self):
		"""Clear the buffer."""
		self.ptr = 0
		self.size = 0
		logger.info("Fast replay buffer cleared")