# common/buffers/replay_slow.py
import torch
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SlowReplayBuffer:
	"""Replay buffer for slow layer (interval-level transitions with sequence embedding)."""
	
	def __init__(self, capacity: int, state_dim: int, action_dims: List[int], 
				 seq_embed_dim: int, T: int):
		self.capacity = capacity
		self.state_dim = state_dim
		self.action_dims = action_dims
		self.n_actions = len(action_dims)
		self.seq_embed_dim = seq_embed_dim
		self.T = T
		
		# Pre-allocate memory
		self.states = np.zeros((capacity, state_dim), dtype=np.float32)
		self.actions = [np.zeros(capacity, dtype=np.int32) for _ in range(self.n_actions)]
		self.seq_embeds = np.zeros((capacity, seq_embed_dim), dtype=np.float32)
		self.rewards = np.zeros(capacity, dtype=np.float32)
		self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
		self.next_seq_embeds = np.zeros((capacity, seq_embed_dim), dtype=np.float32)
		self.dones = np.zeros(capacity, dtype=np.float32)
		
		# Optional: store raw sequences for online encoding
		self.store_raw_seq = False
		if self.store_raw_seq:
			# Assuming seq_input_dim will be set later
			self.raw_sequences = None
		
		self.ptr = 0
		self.size = 0
		
		logger.info(f"Slow replay buffer initialized: capacity={capacity}, state_dim={state_dim}, "
					f"action_dims={action_dims}, seq_embed_dim={seq_embed_dim}")
	
	def add(self, state: np.ndarray, actions: List[int], seq_embed: np.ndarray,
			reward: float, next_state: np.ndarray, next_seq_embed: np.ndarray, done: bool):
		"""Add an interval transition to the buffer."""
		self.states[self.ptr] = state
		for i, action in enumerate(actions):
			self.actions[i][self.ptr] = action
		self.seq_embeds[self.ptr] = seq_embed
		self.rewards[self.ptr] = reward
		self.next_states[self.ptr] = next_state
		self.next_seq_embeds[self.ptr] = next_seq_embed
		self.dones[self.ptr] = float(done)
		
		self.ptr = (self.ptr + 1) % self.capacity
		self.size = min(self.size + 1, self.capacity)
	
	def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
		"""Sample a batch of interval transitions."""
		if self.size < batch_size:
			raise ValueError(f"Buffer size {self.size} < batch_size {batch_size}")
		
		indices = np.random.choice(self.size, batch_size, replace=False)
		
		batch = {
			"states": torch.FloatTensor(self.states[indices]),
			"seq_embeds": torch.FloatTensor(self.seq_embeds[indices]),
			"rewards": torch.FloatTensor(self.rewards[indices]).unsqueeze(-1),
			"next_states": torch.FloatTensor(self.next_states[indices]),
			"next_seq_embeds": torch.FloatTensor(self.next_seq_embeds[indices]),
			"dones": torch.FloatTensor(self.dones[indices]).unsqueeze(-1)
		}
		
		# Add individual action tensors
		for i in range(self.n_actions):
			batch[f"action_{i}"] = torch.LongTensor(self.actions[i][indices])
		
		return batch
	
	def __len__(self) -> int:
		return self.size
	
	def clear(self):
		"""Clear the buffer."""
		self.ptr = 0
		self.size = 0
		logger.info("Slow replay buffer cleared")