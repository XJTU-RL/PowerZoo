# algorithms/twots_vvc/coordinator.py
"""Two-timescale VVC coordinator algorithm."""
import torch
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from algorithms.twots_vvc.fast_ddpg import TwoTSFastDDPG
from algorithms.twots_vvc.slow_sacd import TwoTSSlowSACD
from algorithms.twots_vvc.reward_processor import TwoTSRewardProcessor
from algorithms.twots_vvc.obs_processor import TwoTSObsProcessor

logger = logging.getLogger(__name__)


class TwoTSVVC:
	"""Two-timescale VVC coordinator algorithm.

	This class coordinates the two-timescale VVC algorithm,
	managing fast and slow agents, and processing observations/rewards.
	All algorithm-specific logic is contained here, with no
	environment-specific code.
	"""

	def __init__(self, cfg: Dict[str, Any]):
		self.cfg = cfg
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		# Initialize algorithm components
		self.reward_processor = TwoTSRewardProcessor(cfg)
		self.obs_processor = TwoTSObsProcessor(cfg)

		# Environment manager will be injected by runner
		self.env_manager = None

		# Initialize after environment info is available
		self.fast_agent = None
		self.slow_agent = None
		self.initialized = False

		# Training state
		self.current_interval = 0
		self.pretrain_intervals = cfg.get("train", {}).get("pretrain_fast_intervals", 200)
		self.joint_intervals = cfg.get("train", {}).get("joint_intervals", 1000)

		# Update frequency control
		self.fast_update_count = 0
		self.slow_update_count = 0
		self.fast_critic_update_freq = cfg.get("train", {}).get("fast_critic_update_freq", 1)
		self.fast_actor_update_freq = cfg.get("train", {}).get("fast_actor_update_freq", 2)
		self.slow_critic_update_freq = cfg.get("train", {}).get("slow_critic_update_freq", 1)
		self.slow_actor_update_freq = cfg.get("train", {}).get("slow_actor_update_freq", 2)

		logger.info("TwoTSVVC coordinator initialized with decoupled architecture")

	def set_env_manager(self, env_manager):
		"""Inject environment manager.

		Args:
			env_manager: TwoTSEnvManager instance
		"""
		self.env_manager = env_manager
		logger.info("Environment manager injected into coordinator")

	def init_agents(self, fast_obs_dim: int, fast_action_dim: int,
					slow_obs_dim: int, slow_action_dims: List[int],
					seq_input_dim: int):
		"""Initialize fast and slow agents with environment dimensions."""
		if self.initialized:
			return

		# Initialize fast agent (DDPG)
		self.fast_agent = TwoTSFastDDPG(fast_obs_dim, fast_action_dim, self.cfg)

		# Initialize slow agent (SAC-D)
		self.slow_agent = TwoTSSlowSACD(slow_obs_dim, slow_action_dims, seq_input_dim, self.cfg)

		self.initialized = True
		logger.info(f"Agents initialized - Fast: {fast_obs_dim}→{fast_action_dim}, Slow: {slow_obs_dim}→{slow_action_dims}")
	
	def act_fast(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
		"""Get fast layer action."""
		if not self.initialized:
			raise RuntimeError("Agents not initialized. Call init_agents first.")
		return self.fast_agent.act(obs, explore=explore)
	
	def act_slow(self, state: np.ndarray, deterministic: bool = False) -> List[int]:
		"""Get slow layer actions."""
		if not self.initialized:
			raise RuntimeError("Agents not initialized. Call init_agents first.")
		
		# During pretraining, use random actions for slow layer
		if self.current_interval < self.pretrain_intervals and not deterministic:
			return [np.random.randint(dim) for dim in self.slow_agent.action_dims]
		
		return self.slow_agent.act(state, deterministic=deterministic)
	
	def encode_sequence(self, seq: np.ndarray) -> np.ndarray:
		"""Encode fast trajectory sequence for slow layer."""
		if not self.initialized:
			raise RuntimeError("Agents not initialized. Call init_agents first.")
		return self.slow_agent.encode_sequence(seq)
	
	def update_fast(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
		"""Update fast layer with frequency control."""
		if not self.initialized:
			raise RuntimeError("Agents not initialized. Call init_agents first.")
		
		self.fast_update_count += 1
		critic_update = (self.fast_update_count % self.fast_critic_update_freq) == 0
		actor_update = (self.fast_update_count % self.fast_actor_update_freq) == 0
		
		return self.fast_agent.update(batch, critic_update=critic_update, actor_update=actor_update)
	
	def update_slow(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
		"""Update slow layer with frequency control."""
		if not self.initialized:
			raise RuntimeError("Agents not initialized. Call init_agents first.")
		
		# Only update during joint training phase
		if self.current_interval < self.pretrain_intervals:
			return {}
		
		self.slow_update_count += 1
		critic_update = (self.slow_update_count % self.slow_critic_update_freq) == 0
		actor_update = (self.slow_update_count % self.slow_actor_update_freq) == 0
		
		return self.slow_agent.update(batch, critic_update=critic_update, actor_update=actor_update)
	
	def increment_interval(self):
		"""Increment training interval counter."""
		self.current_interval += 1
	
	def is_pretraining(self) -> bool:
		"""Check if in pretraining phase."""
		return self.current_interval < self.pretrain_intervals
	
	def prep_training(self):
		"""Set agents to training mode."""
		if self.initialized:
			self.fast_agent.prep_training()
			self.slow_agent.prep_training()
	
	def prep_rollout(self):
		"""Set agents to evaluation mode."""
		if self.initialized:
			self.fast_agent.prep_rollout()
			self.slow_agent.prep_rollout()
	
	def save(self, path: str):
		"""Save both agents."""
		if not self.initialized:
			logger.warning("Cannot save uninitialized agents")
			return
		
		import os
		base_path = path.replace(".pt", "")
		self.fast_agent.save(f"{base_path}_fast.pt")
		self.slow_agent.save(f"{base_path}_slow.pt")
		
		# Save training state
		torch.save({
			"current_interval": self.current_interval,
			"cfg": self.cfg
		}, f"{base_path}_state.pt")
		
		logger.info(f"TwoTSVVC models saved to {base_path}_*.pt")
	
	def load(self, path: str):
		"""Load both agents."""
		if not self.initialized:
			logger.warning("Cannot load to uninitialized agents")
			return
		
		base_path = path.replace(".pt", "")
		self.fast_agent.load(f"{base_path}_fast.pt")
		self.slow_agent.load(f"{base_path}_slow.pt")
		
		# Load training state
		import os
		state_path = f"{base_path}_state.pt"
		if os.path.exists(state_path):
			state = torch.load(state_path, map_location=self.device)
			self.current_interval = state["current_interval"]
		
		logger.info(f"TwoTSVVC models loaded from {base_path}_*.pt")