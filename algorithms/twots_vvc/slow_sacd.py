# algorithms/twots_vvc/slow_sacd.py
"""Slow layer multi-head discrete SAC with CTDE."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SlowActor(nn.Module):
	"""Multi-head discrete actor for slow layer devices."""
	
	def __init__(self, obs_dim: int, action_dims: List[int], hidden_dims: List[int] = [256, 128]):
		super().__init__()
		self.action_dims = action_dims
		self.n_heads = len(action_dims)
		
		# Shared layers
		layers = []
		prev_dim = obs_dim
		for dim in hidden_dims:
			layers.extend([
				nn.Linear(prev_dim, dim),
				nn.ReLU(),
				nn.LayerNorm(dim)
			])
			prev_dim = dim
		self.shared = nn.Sequential(*layers)
		
		# Multi-head output layers
		self.heads = nn.ModuleList([
			nn.Linear(prev_dim, action_dim) for action_dim in action_dims
		])
	
	def forward(self, obs: torch.Tensor) -> List[torch.Tensor]:
		"""Forward pass returning logits for each head."""
		shared_features = self.shared(obs)
		logits_list = [head(shared_features) for head in self.heads]
		return logits_list
	
	def get_action_probs(self, obs: torch.Tensor) -> List[torch.Tensor]:
		"""Get action probabilities for each head."""
		logits_list = self.forward(obs)
		probs_list = [F.softmax(logits, dim=-1) for logits in logits_list]
		return probs_list
	
	def sample_action(self, obs: torch.Tensor) -> Tuple[List[int], torch.Tensor]:
		"""Sample actions and compute log probabilities."""
		probs_list = self.get_action_probs(obs)
		actions = []
		log_probs = []
		
		for probs in probs_list:
			dist = Categorical(probs)
			action = dist.sample()
			log_prob = dist.log_prob(action)
			actions.append(action.item() if obs.dim() == 1 else action.cpu().numpy())
			log_probs.append(log_prob)
		
		total_log_prob = torch.stack(log_probs).sum(dim=0)
		return actions, total_log_prob
	
	def sample_action_batch(self, obs: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
		"""Sample actions for batch and compute log probabilities."""
		probs_list = self.get_action_probs(obs)
		actions = []
		log_probs = []
		
		for probs in probs_list:
			dist = Categorical(probs)
			action = dist.sample()
			log_prob = dist.log_prob(action)
			actions.append(action)
			log_probs.append(log_prob)
		
		total_log_prob = torch.stack(log_probs).sum(dim=0)
		return actions, total_log_prob
	
	def evaluate_actions(self, obs: torch.Tensor, actions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Evaluate given actions."""
		probs_list = self.get_action_probs(obs)
		log_probs = []
		entropies = []
		
		for probs, action in zip(probs_list, actions):
			dist = Categorical(probs)
			log_prob = dist.log_prob(action)
			entropy = dist.entropy()
			log_probs.append(log_prob)
			entropies.append(entropy)
		
		total_log_prob = torch.stack(log_probs).sum(dim=0)
		total_entropy = torch.stack(entropies).sum(dim=0)
		return total_log_prob, total_entropy


class SlowCritic(nn.Module):
	"""CTDE critic for slow layer (reads S_τ, φ_seq, A_onehot)."""
	
	def __init__(self, state_dim: int, seq_embed_dim: int, action_dims: List[int], 
				 hidden_dims: List[int] = [256, 256, 128]):
		super().__init__()
		self.action_dims = action_dims
		total_action_dim = sum(action_dims)
		input_dim = state_dim + seq_embed_dim + total_action_dim
		
		# Q1 network
		q1_layers = []
		prev_dim = input_dim
		for dim in hidden_dims:
			q1_layers.extend([
				nn.Linear(prev_dim, dim),
				nn.ReLU(),
				nn.LayerNorm(dim)
			])
			prev_dim = dim
		q1_layers.append(nn.Linear(prev_dim, 1))
		self.q1 = nn.Sequential(*q1_layers)
		
		# Q2 network (twin Q-learning)
		q2_layers = []
		prev_dim = input_dim
		for dim in hidden_dims:
			q2_layers.extend([
				nn.Linear(prev_dim, dim),
				nn.ReLU(),
				nn.LayerNorm(dim)
			])
			prev_dim = dim
		q2_layers.append(nn.Linear(prev_dim, 1))
		self.q2 = nn.Sequential(*q2_layers)
	
	def forward(self, state: torch.Tensor, seq_embed: torch.Tensor, 
				actions: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Forward pass with action one-hot encoding."""
		# Convert actions to one-hot
		action_one_hots = []
		for action, action_dim in zip(actions, self.action_dims):
			one_hot = F.one_hot(action.long(), num_classes=action_dim).float()
			action_one_hots.append(one_hot)
		action_concat = torch.cat(action_one_hots, dim=-1)
		
		# Concatenate all inputs
		x = torch.cat([state, seq_embed, action_concat], dim=-1)
		
		q1_value = self.q1(x)
		q2_value = self.q2(x)
		return q1_value, q2_value


class SequenceEncoder(nn.Module):
	"""LSTM encoder for fast trajectory embedding."""
	
	def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, 
				 output_dim: int = 128):
		super().__init__()
		self.hidden_dim = hidden_dim
		self.num_layers = num_layers
		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
		self.output_projection = nn.Linear(hidden_dim, output_dim)
	
	def forward(self, seq: torch.Tensor, h0: Optional[torch.Tensor] = None, 
				c0: Optional[torch.Tensor] = None) -> torch.Tensor:
		"""Encode sequence to fixed-size embedding with explicit state initialization."""
		# seq shape: (batch, T, input_dim)
		batch_size = seq.size(0)
		
		# Initialize hidden states if not provided (ensure device compatibility)
		if h0 is None:
			h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, 
							 dtype=seq.dtype).to(seq.device)
		if c0 is None:
			c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim,
							 dtype=seq.dtype).to(seq.device)
		
		lstm_out, (h_n, c_n) = self.lstm(seq, (h0, c0))
		# Use last hidden state
		final_hidden = h_n[-1]  # (batch, hidden_dim)
		seq_embed = self.output_projection(final_hidden)
		return seq_embed


class TwoTSSlowSACD:
	"""Slow layer multi-head discrete SAC with CTDE."""
	
	def __init__(self, state_dim: int, action_dims: List[int], seq_input_dim: int, cfg: Dict[str, Any]):
		# Input validation
		assert state_dim > 0, f"state_dim must be positive, got {state_dim}"
		assert seq_input_dim > 0, f"seq_input_dim must be positive, got {seq_input_dim}"
		assert len(action_dims) > 0, "action_dims must not be empty"
		assert all(d > 0 for d in action_dims), f"All action dimensions must be positive, got {action_dims}"
		
		self.state_dim = state_dim
		self.action_dims = action_dims
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# Extract config with validation
		slow_cfg = cfg.get("slow", {})
		sacd_cfg = slow_cfg.get("sacd", {})
		self.lr = sacd_cfg.get("lr", 3e-4)
		self.tau = sacd_cfg.get("tau", 0.005)
		self.gamma = sacd_cfg.get("gamma", 0.99)
		self.alpha = sacd_cfg.get("alpha", 0.2)  # Entropy coefficient
		
		# Validate hyperparameters
		assert 0 < self.tau <= 1, f"tau must be in (0, 1], got {self.tau}"
		assert 0 < self.gamma <= 1, f"gamma must be in (0, 1], got {self.gamma}"
		assert self.lr > 0, f"learning rate must be positive, got {self.lr}"
		assert self.alpha >= 0, f"alpha must be non-negative, got {self.alpha}"
		
		seq_embed_dim = slow_cfg.get("seq_embed_dim", 128)
		lstm_hidden = slow_cfg.get("lstm_hidden", 64)
		lstm_layers = slow_cfg.get("lstm_layers", 2)
		
		# Exploration parameters
		self.epsilon = sacd_cfg.get("epsilon_start", 0.3)
		self.epsilon_min = sacd_cfg.get("epsilon_min", 0.01)
		self.epsilon_decay = sacd_cfg.get("epsilon_decay", 0.9999)
		
		# Networks
		self.actor = SlowActor(state_dim, action_dims).to(self.device)
		self.critic = SlowCritic(state_dim, seq_embed_dim, action_dims).to(self.device)
		self.critic_target = SlowCritic(state_dim, seq_embed_dim, action_dims).to(self.device)
		self.seq_encoder = SequenceEncoder(seq_input_dim, lstm_hidden, lstm_layers, seq_embed_dim).to(self.device)
		
		# Initialize target
		self.critic_target.load_state_dict(self.critic.state_dict())
		
		# Optimizers
		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
		self.critic_optimizer = optim.Adam(
			list(self.critic.parameters()) + list(self.seq_encoder.parameters()), 
			lr=self.lr
		)
		
		# Automatic entropy tuning
		self.target_entropy = -sum(np.log(dim) for dim in action_dims)
		self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
		self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.lr)
		
		self.training = True
		self.update_count = 0
	
	def act(self, state: np.ndarray, deterministic: bool = False) -> List[int]:
		"""Select actions for slow devices with epsilon-greedy exploration. Always returns List[int]."""
		# Epsilon-greedy exploration (only during training, not in deterministic mode)
		if not deterministic and self.training and np.random.random() < self.epsilon:
			# Random actions
			actions = [np.random.randint(dim) for dim in self.action_dims]
		else:
			with torch.no_grad():
				state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
				
				if deterministic:
					probs_list = self.actor.get_action_probs(state_t)
					actions = [int(probs.argmax().item()) for probs in probs_list]
				else:
					actions, _ = self.actor.sample_action(state_t)
					# Ensure consistent return format: List[int]
					if isinstance(actions[0], (np.ndarray, np.integer)):
						actions = [int(a) for a in actions]
					elif not isinstance(actions[0], int):
						actions = [int(a) for a in actions]
		
		return actions
	
	def encode_sequence(self, seq: np.ndarray) -> np.ndarray:
		"""Encode fast trajectory sequence."""
		with torch.no_grad():
			# Ensure tensor is on correct device
			seq_t = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
			seq_embed = self.seq_encoder(seq_t).cpu().numpy()[0]
		return seq_embed
	
	def update(self, batch: Dict[str, torch.Tensor], critic_update: bool = True,
			   actor_update: bool = True) -> Dict[str, float]:
		"""Update actor and critic networks with frequency control."""
		states = batch["states"].to(self.device)
		actions = [batch[f"action_{i}"].to(self.device) for i in range(len(self.action_dims))]
		seq_embeds = batch["seq_embeds"].to(self.device)
		rewards = batch["rewards"].to(self.device)
		next_states = batch["next_states"].to(self.device)
		next_seq_embeds = batch["next_seq_embeds"].to(self.device)
		dones = batch["dones"].to(self.device)
		
		metrics = {}
		
		# Update critic if scheduled
		if critic_update:
			with torch.no_grad():
				# Sample next actions
				next_probs_list = self.actor.get_action_probs(next_states)
				next_actions = []
				next_log_probs = []
				
				for probs in next_probs_list:
					dist = Categorical(probs)
					action = dist.sample()
					log_prob = dist.log_prob(action)
					next_actions.append(action)
					next_log_probs.append(log_prob)
				
				next_log_prob = torch.stack(next_log_probs).sum(dim=0)
				
				# Compute target Q-value
				target_q1, target_q2 = self.critic_target(next_states, next_seq_embeds, next_actions)
				target_q = torch.min(target_q1, target_q2)
				target_value = rewards + self.gamma * (target_q - self.alpha * next_log_prob) * (1 - dones)
			
			# Current Q-values
			current_q1, current_q2 = self.critic(states, seq_embeds, actions)
			
			# Critic loss (ensure dimensions match)
			critic_loss = F.mse_loss(current_q1.squeeze(-1), target_value.squeeze(-1)) + \
						  F.mse_loss(current_q2.squeeze(-1), target_value.squeeze(-1))
			
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()) + list(self.seq_encoder.parameters()), 1.0)
			self.critic_optimizer.step()
			
			metrics["slow_critic_loss"] = critic_loss.item()
			metrics["slow_q_value"] = current_q1.mean().item()
		
		# Update actor if scheduled
		if actor_update:
			# Sample new actions from current policy
			sampled_actions, sampled_log_probs = self.actor.sample_action_batch(states)
			
			# Compute Q-values for newly sampled actions
			q1_new, q2_new = self.critic(states, seq_embeds, sampled_actions)
			min_q_new = torch.min(q1_new, q2_new)
			
			# Actor loss with correctly sampled actions
			actor_loss = (self.alpha * sampled_log_probs - min_q_new).mean()
			
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
			self.actor_optimizer.step()
			
			metrics["slow_actor_loss"] = actor_loss.item()
			metrics["slow_log_prob"] = sampled_log_probs.mean().item()
			
			# Update alpha (entropy coefficient) using sampled log probs
			alpha_loss = -(self.log_alpha * (sampled_log_probs + self.target_entropy).detach()).mean()
			
			self.alpha_optimizer.zero_grad()
			alpha_loss.backward()
			self.alpha_optimizer.step()
		
		self.alpha = self.log_alpha.exp().item()
		metrics["slow_alpha"] = self.alpha
		
		# Decay epsilon
		self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
		metrics["slow_epsilon"] = self.epsilon
		
		# Soft update target network
		self._soft_update()
		
		self.update_count += 1
		
		return metrics
	
	def _soft_update(self):
		"""Soft update target network."""
		for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
	
	def prep_training(self):
		"""Set networks to training mode."""
		self.actor.train()
		self.critic.train()
		self.seq_encoder.train()
		self.training = True
	
	def prep_rollout(self):
		"""Set networks to evaluation mode."""
		self.actor.eval()
		self.critic.eval()
		self.seq_encoder.eval()
		self.training = False
	
	def save(self, path: str):
		"""Save model checkpoint."""
		torch.save({
			"actor": self.actor.state_dict(),
			"critic": self.critic.state_dict(),
			"critic_target": self.critic_target.state_dict(),
			"seq_encoder": self.seq_encoder.state_dict(),
			"actor_optimizer": self.actor_optimizer.state_dict(),
			"critic_optimizer": self.critic_optimizer.state_dict(),
			"alpha_optimizer": self.alpha_optimizer.state_dict(),
			"log_alpha": self.log_alpha,
			"update_count": self.update_count
		}, path)
		logger.info(f"Slow SAC-D model saved to {path}")
	
	def load(self, path: str):
		"""Load model checkpoint."""
		checkpoint = torch.load(path, map_location=self.device)
		self.actor.load_state_dict(checkpoint["actor"])
		self.critic.load_state_dict(checkpoint["critic"])
		self.critic_target.load_state_dict(checkpoint["critic_target"])
		self.seq_encoder.load_state_dict(checkpoint["seq_encoder"])
		self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
		self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
		self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
		self.log_alpha = checkpoint["log_alpha"]
		self.update_count = checkpoint["update_count"]
		logger.info(f"Slow SAC-D model loaded from {path}")