# algorithms/twots_vvc/fast_ddpg.py
"""Fast layer DDPG for continuous PV/BESS control."""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OrnsteinUhlenbeckNoise:
	"""Ornstein-Uhlenbeck process for exploration noise with decay."""
	
	def __init__(self, size: int, theta: float = 0.15, sigma: float = 0.2, 
				 sigma_min: float = 0.05, sigma_decay: float = 0.99995, dt: float = 1e-2):
		self.size = size
		self.theta = theta
		self.sigma = sigma
		self.sigma_min = sigma_min
		self.sigma_decay = sigma_decay
		self.dt = dt
		self.reset()
	
	def reset(self):
		self.state = np.zeros(self.size)
	
	def decay(self):
		"""Apply decay to noise level."""
		self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
	
	def __call__(self) -> np.ndarray:
		dx = self.theta * (-self.state) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
		self.state += dx
		return self.state


class FastActor(nn.Module):
	"""Fast layer actor network for continuous control."""
	
	def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] = [256, 128]):
		super().__init__()
		layers = []
		prev_dim = obs_dim
		for dim in hidden_dims:
			layers.extend([
				nn.Linear(prev_dim, dim),
				nn.ReLU(),
				nn.LayerNorm(dim)
			])
			prev_dim = dim
		layers.append(nn.Linear(prev_dim, action_dim))
		layers.append(nn.Tanh())  # Output in [-1, 1]
		self.net = nn.Sequential(*layers)
	
	def forward(self, obs: torch.Tensor) -> torch.Tensor:
		return self.net(obs)


class FastCritic(nn.Module):
	"""Fast layer critic network (Q-function)."""
	
	def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int] = [256, 128]):
		super().__init__()
		layers = []
		prev_dim = obs_dim + action_dim
		for dim in hidden_dims:
			layers.extend([
				nn.Linear(prev_dim, dim),
				nn.ReLU(),
				nn.LayerNorm(dim)
			])
			prev_dim = dim
		layers.append(nn.Linear(prev_dim, 1))
		self.net = nn.Sequential(*layers)
	
	def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
		x = torch.cat([obs, action], dim=-1)
		return self.net(x)


class TwoTSFastDDPG:
	"""Fast layer DDPG for continuous PV/BESS control."""
	
	def __init__(self, obs_dim: int, action_dim: int, cfg: Dict[str, Any]):
		# Input validation
		assert obs_dim > 0, f"obs_dim must be positive, got {obs_dim}"
		assert action_dim > 0, f"action_dim must be positive, got {action_dim}"
		
		self.obs_dim = obs_dim
		self.action_dim = action_dim
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Extract DDPG config with validation
		ddpg_cfg = cfg.get("fast", {}).get("ddpg", {})
		self.lr = ddpg_cfg.get("lr", 1e-3)
		self.tau = ddpg_cfg.get("tau", 0.005)
		self.gamma = ddpg_cfg.get("gamma", 0.99)
		
		# Validate hyperparameters
		assert 0 < self.tau <= 1, f"tau must be in (0, 1], got {self.tau}"
		assert 0 < self.gamma <= 1, f"gamma must be in (0, 1], got {self.gamma}"
		assert self.lr > 0, f"learning rate must be positive, got {self.lr}"
		
		# Networks
		self.actor = FastActor(obs_dim, action_dim).to(self.device)
		self.actor_target = FastActor(obs_dim, action_dim).to(self.device)
		self.critic = FastCritic(obs_dim, action_dim).to(self.device)
		self.critic_target = FastCritic(obs_dim, action_dim).to(self.device)
		
		# Initialize target networks
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.critic_target.load_state_dict(self.critic.state_dict())
		
		# Optimizers
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)
		
		# Exploration noise with decay
		noise_cfg = ddpg_cfg.get("noise", {})
		self.noise = OrnsteinUhlenbeckNoise(
			action_dim,
			theta=noise_cfg.get("theta", 0.15),
			sigma=noise_cfg.get("sigma", 0.2),
			sigma_min=noise_cfg.get("sigma_min", 0.05),
			sigma_decay=noise_cfg.get("sigma_decay", 0.99995)
		)
		
		self.training = True
		self.update_count = 0
	
	def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
		"""Select action given observation."""
		with torch.no_grad():
			obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
			action = self.actor(obs_t).cpu().numpy()[0]
		
		if explore and self.training:
			noise = self.noise()
			action = np.clip(action + noise, -1.0, 1.0)
		
		return action
	
	def update(self, batch: Dict[str, torch.Tensor], critic_update: bool = True, 
			   actor_update: bool = True) -> Dict[str, float]:
		"""Update actor and critic networks with frequency control."""
		obs = batch["obs"].to(self.device)
		actions = batch["actions"].to(self.device)
		rewards = batch["rewards"].to(self.device)
		next_obs = batch["next_obs"].to(self.device)
		dones = batch["dones"].to(self.device)
		
		metrics = {}
		
		# Update critic if scheduled
		if critic_update:
			with torch.no_grad():
				next_actions = self.actor_target(next_obs)
				target_q = self.critic_target(next_obs, next_actions)
				target_value = rewards + self.gamma * target_q * (1 - dones)
			
			current_q = self.critic(obs, actions)
			critic_loss = nn.MSELoss()(current_q, target_value)
			
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
			self.critic_optimizer.step()
			
			metrics["fast_critic_loss"] = critic_loss.item()
			metrics["fast_q_value"] = current_q.mean().item()
		
		# Update actor if scheduled
		if actor_update:
			actor_actions = self.actor(obs)
			actor_loss = -self.critic(obs, actor_actions).mean()
			
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
			self.actor_optimizer.step()
			
			metrics["fast_actor_loss"] = actor_loss.item()
		
		# Soft update target networks
		self._soft_update()
		
		# Decay exploration noise
		self.noise.decay()
		
		self.update_count += 1
		metrics["fast_noise_sigma"] = self.noise.sigma
		
		return metrics
	
	def _soft_update(self):
		"""Soft update target networks."""
		for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
	
	def prep_training(self):
		"""Set networks to training mode."""
		self.actor.train()
		self.critic.train()
		self.training = True
		self.noise.reset()
	
	def prep_rollout(self):
		"""Set networks to evaluation mode."""
		self.actor.eval()
		self.critic.eval()
		self.training = False
	
	def save(self, path: str):
		"""Save model checkpoint."""
		torch.save({
			"actor": self.actor.state_dict(),
			"actor_target": self.actor_target.state_dict(),
			"critic": self.critic.state_dict(),
			"critic_target": self.critic_target.state_dict(),
			"actor_optimizer": self.actor_optimizer.state_dict(),
			"critic_optimizer": self.critic_optimizer.state_dict(),
			"update_count": self.update_count
		}, path)
		logger.info(f"Fast DDPG model saved to {path}")
	
	def load(self, path: str):
		"""Load model checkpoint."""
		checkpoint = torch.load(path, map_location=self.device)
		self.actor.load_state_dict(checkpoint["actor"])
		self.actor_target.load_state_dict(checkpoint["actor_target"])
		self.critic.load_state_dict(checkpoint["critic"])
		self.critic_target.load_state_dict(checkpoint["critic_target"])
		self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
		self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
		self.update_count = checkpoint["update_count"]
		logger.info(f"Fast DDPG model loaded from {path}")