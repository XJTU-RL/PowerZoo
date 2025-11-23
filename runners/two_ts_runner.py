# runners/two_ts_runner.py
import torch
import numpy as np
import os
from typing import Dict, Any, Optional
import logging
from tensorboardX import SummaryWriter
from common.buffers.replay_fast import FastReplayBuffer
from common.buffers.replay_slow import SlowReplayBuffer
from algorithms.twots_vvc import TwoTSVVC
from algorithms.twots_vvc.env_manager import TwoTSEnvManager

logger = logging.getLogger(__name__)


class TwoTSRunner:
	"""Runner for two-timescale VVC training.

	This runner coordinates the training process using the
	decoupled architecture where algorithm logic is separate
	from environment interaction.
	"""

	def __init__(self, env, cfg: Dict[str, Any], exp_name: str = "2ts_vvc"):
		self.cfg = cfg
		self.exp_name = exp_name

		# Use standard environment with algorithm-side manager
		self.env = env
		self.env_manager = TwoTSEnvManager(env, cfg)

		# Get dimensions from environment manager
		self.fast_obs_dim = len(self.env_manager.fast_ids) * 4 + 5  # Device features + system features
		self.fast_action_dim = self.env_manager.fast_action_dim
		self.slow_obs_dim = self.env_manager.T * 2 + len(self.env_manager.slow_ids) + 4  # Forecasts + states + time
		self.slow_action_dims = self.env_manager.slow_action_dims
		self.seq_input_dim = self.fast_obs_dim + self.fast_action_dim + 1  # obs + action + reward
		self.T = self.env_manager.T
		
		# Initialize algorithm and inject environment manager
		self.algo = TwoTSVVC(cfg)
		self.algo.set_env_manager(self.env_manager)
		self.algo.init_agents(
			self.fast_obs_dim,
			self.fast_action_dim,
			self.slow_obs_dim,
			self.slow_action_dims,
			self.seq_input_dim
		)
		
		# Initialize replay buffers
		fast_cfg = cfg.get("fast", {})
		slow_cfg = cfg.get("slow", {})
		
		self.fast_buffer = FastReplayBuffer(
			capacity=fast_cfg.get("buffer", 200000),
			obs_dim=self.fast_obs_dim,
			action_dim=self.fast_action_dim
		)
		
		self.slow_buffer = SlowReplayBuffer(
			capacity=slow_cfg.get("buffer", 50000),
			state_dim=self.slow_obs_dim,
			action_dims=self.slow_action_dims,
			seq_embed_dim=slow_cfg.get("seq_embed_dim", 128),
			T=self.T
		)
		
		# Training parameters
		self.fast_batch_size = fast_cfg.get("batch", 256)
		self.fast_warmup = fast_cfg.get("warmup", 5000)
		self.fast_updates_per_interval = fast_cfg.get("updates_per_interval", 50)
		
		self.slow_batch_size = slow_cfg.get("batch", 128)
		self.slow_warmup = slow_cfg.get("warmup", 200)
		self.slow_updates_per_interval = slow_cfg.get("updates_per_interval", 10)
		
		train_cfg = cfg.get("train", {})
		self.pretrain_intervals = train_cfg.get("pretrain_fast_intervals", 200)
		self.joint_intervals = train_cfg.get("joint_intervals", 1000)
		self.total_intervals = self.pretrain_intervals + self.joint_intervals
		
		# Runtime parameters
		runtime_cfg = cfg.get("runtime", {})
		self.seed = runtime_cfg.get("seed", 1)
		self.save_every = runtime_cfg.get("save_every_ep", 5)
		
		# Logging
		self.log_dir = f"results/logs/{exp_name}"
		self.model_dir = f"results/models/{exp_name}"
		os.makedirs(self.log_dir, exist_ok=True)
		os.makedirs(self.model_dir, exist_ok=True)
		
		self.writer = SummaryWriter(self.log_dir)
		self.interval_count = 0
		
		# Set random seed
		self._set_seed(self.seed)
		
		logger.info(f"TwoTSRunner initialized: fast_dim={self.fast_obs_dim}→{self.fast_action_dim}, "
					f"slow_dim={self.slow_obs_dim}→{self.slow_action_dims}")
	
	def train(self):
		"""Main training loop."""
		logger.info(f"Starting training for {self.total_intervals} intervals")
		
		for interval in range(self.total_intervals):
			self.interval_count = interval
			self.algo.increment_interval()
			
			# Determine training phase
			is_pretraining = interval < self.pretrain_intervals
			phase = "pretrain" if is_pretraining else "joint"
			
			# Run one interval
			metrics = self._run_interval(is_pretraining)
			
			# Update fast layer
			if len(self.fast_buffer) >= self.fast_warmup:
				fast_metrics = self._update_fast()
				metrics.update(fast_metrics)
			
			# Update slow layer (only during joint training)
			if not is_pretraining and len(self.slow_buffer) >= self.slow_warmup:
				slow_metrics = self._update_slow()
				metrics.update(slow_metrics)
			
			# Logging
			self._log_metrics(metrics, phase)
			
			# Save model
			if (interval + 1) % self.save_every == 0:
				self._save_model(interval + 1)
			
			# Progress report
			if (interval + 1) % 10 == 0:
				logger.info(f"[{phase}] Interval {interval+1}/{self.total_intervals}: "
						   f"R_τ={metrics.get('interval_reward', 0):.2f}, "
						   f"fast_buffer={len(self.fast_buffer)}, "
						   f"slow_buffer={len(self.slow_buffer)}")
		
		logger.info("Training completed")
		self._save_model(self.total_intervals)
	
	def _run_interval(self, is_pretraining: bool) -> Dict[str, float]:
		"""Run one slow interval using decoupled architecture."""
		# Reset interval through environment manager
		reset_info = self.env_manager.reset_interval(seed=self.seed + self.interval_count)

		# Build observations using algorithm's processor
		device_groups = {
			"slow": self.env_manager.slow_ids,
			"fast": self.env_manager.fast_ids
		}
		slow_obs = self.algo.obs_processor.build_slow_obs(
			reset_info["obs"], reset_info["share_obs"], device_groups
		)
		fast_obs = self.algo.obs_processor.build_fast_obs(
			reset_info["obs"], reset_info["share_obs"], device_groups
		)

		# Select slow actions
		if is_pretraining:
			# Random actions during pretraining
			slow_actions = [np.random.randint(dim) for dim in self.slow_action_dims]
		else:
			# Policy actions during joint training
			slow_actions = self.algo.act_slow(slow_obs, deterministic=False)

		# Apply slow actions through environment manager
		switch_cost = self.env_manager.apply_slow_actions(slow_actions)

		# Store initial state for slow buffer
		initial_slow_obs = slow_obs.copy()
		initial_slow_actions = slow_actions.copy()

		# Run T fast steps
		interval_metrics = {
			"line_loss": [],
			"volt_violation": [],
			"convergence_rate": []
		}
		obs_sequence = []
		action_sequence = []
		reward_sequence = []

		for t in range(self.T):
			# Get fast action
			fast_action = self.algo.act_fast(fast_obs, explore=True)

			# Step environment through manager
			step_info, interval_done = self.env_manager.step_fast(fast_action)

			# Process reward using algorithm's processor
			delta_q = self.algo.reward_processor._compute_step_delta_q({"obs_dict": step_info["obs_dict"]})
			minute_reward = self.algo.reward_processor.compute_minute_reward(step_info["info"], delta_q)

			# Build next observation
			next_fast_obs = self.algo.obs_processor.build_fast_obs(
				step_info["obs"], step_info["share_obs"], device_groups
			)

			# Store in fast buffer
			self.fast_buffer.add(fast_obs, fast_action, minute_reward, next_fast_obs, interval_done)

			# Track for sequence
			obs_sequence.append(fast_obs)
			action_sequence.append(fast_action)
			reward_sequence.append(minute_reward)

			# Track metrics
			interval_metrics["line_loss"].append(step_info["info"].get("line_loss", 0))
			interval_metrics["volt_violation"].append(step_info["info"].get("volt_violation_cost", 0))
			interval_metrics["convergence_rate"].append(float(step_info["info"].get("convergence", True)))

			# Update observation
			fast_obs = next_fast_obs

			if interval_done:
				break

		# Process interval summary
		summary = self.env_manager.get_interval_summary()
		
		# Process sequence for embedding
		sequence = self.algo.obs_processor.process_sequence(
			obs_sequence, action_sequence, reward_sequence
		)
		seq_embed = self.algo.encode_sequence(sequence)

		# For next interval's seq_embed (placeholder, would be from next interval)
		next_seq_embed = seq_embed  # Simplified for now

		# Build next slow observation for buffer
		current_state = self.env_manager.get_current_state()
		next_slow_obs = self.algo.obs_processor.build_slow_obs(
			current_state["obs"], current_state["share_obs"], device_groups
		)

		# Compute interval reward using algorithm's processor
		reward_info = self.algo.reward_processor.compute_interval_reward(
			self.env_manager.interval_data, switch_cost
		)

		# Store in slow buffer
		self.slow_buffer.add(
			initial_slow_obs,
			initial_slow_actions,
			seq_embed,
			reward_info["interval_reward"],
			next_slow_obs,
			next_seq_embed,
			False  # Episode not done
		)
		
		# Aggregate metrics from algorithm's reward processor
		metrics = {
			"interval_reward": reward_info["interval_reward"],
			"switch_cost": reward_info["switch_cost"],
			"avg_minute_reward": reward_info["minute_rewards_sum"] / max(self.T, 1),
			"avg_line_loss": reward_info["avg_line_loss"],
			"avg_volt_violation": reward_info["avg_volt_violation"],
			"convergence_rate": reward_info["convergence_rate"],
			"rms_delta_q": reward_info["rms_delta_q"]
		}

		return metrics
	
	def _update_fast(self) -> Dict[str, float]:
		"""Update fast layer."""
		total_metrics = {}
		
		for _ in range(self.fast_updates_per_interval):
			batch = self.fast_buffer.sample(self.fast_batch_size)
			metrics = self.algo.update_fast(batch)
			
			# Accumulate metrics
			for k, v in metrics.items():
				if k not in total_metrics:
					total_metrics[k] = []
				total_metrics[k].append(v)
		
		# Average metrics
		avg_metrics = {k: np.mean(v) for k, v in total_metrics.items()}
		return avg_metrics
	
	def _update_slow(self) -> Dict[str, float]:
		"""Update slow layer."""
		total_metrics = {}
		
		for _ in range(self.slow_updates_per_interval):
			batch = self.slow_buffer.sample(self.slow_batch_size)
			metrics = self.algo.update_slow(batch)
			
			# Accumulate metrics
			for k, v in metrics.items():
				if k not in total_metrics:
					total_metrics[k] = []
				total_metrics[k].append(v)
		
		# Average metrics
		avg_metrics = {k: np.mean(v) for k, v in total_metrics.items()}
		return avg_metrics
	
	def _log_metrics(self, metrics: Dict[str, float], phase: str):
		"""Log metrics to TensorBoard."""
		for key, value in metrics.items():
			self.writer.add_scalar(f"{phase}/{key}", value, self.interval_count)
	
	def _save_model(self, interval: int):
		"""Save model checkpoint."""
		path = os.path.join(self.model_dir, f"model_ep{interval}.pt")
		self.algo.save(path)
		logger.info(f"Model saved at interval {interval}")
	
	def _set_seed(self, seed: int):
		"""Set random seeds."""
		np.random.seed(seed)
		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed(seed)