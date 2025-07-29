# -*- coding: utf-8 -*-
"""
Simple example demonstrating Stackelberg game environment usage.

This example shows how to:
1. Create a Stackelberg environment
2. Run a simple episode with random actions
3. Display basic metrics
"""

import numpy as np
import logging
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StackelbergExample')


def run_simple_episode():
	"""Run a simple episode with random actions."""
	try:
		# Import environment factory
		from envs.stackelberg.stackelberg_game.env_factory import make_stackelberg_env
		
		# Create environment
		logger.info("Creating Stackelberg 13Bus environment...")
		env = make_stackelberg_env('stackelberg_13bus', use_async_wrapper=False)
		
		# Reset environment
		observations = env.reset()
		logger.info(f"Environment reset. Number of agents: {len(observations)}")
		
		# Episode variables
		episode_rewards = {agent_id: 0.0 for agent_id in observations}
		done = False
		step = 0
		
		# Run episode
		logger.info("Running episode with random actions...")
		while not done:
			# Create random actions for all agents
			actions = {}
			for agent_id in observations:
				action_space = env.action_spaces[agent_id]
				actions[agent_id] = action_space.sample()
			
			# Execute step
			observations, rewards, done, infos = env.step(actions)
			
			# Accumulate rewards
			for agent_id, reward in rewards.items():
				episode_rewards[agent_id] += reward
			
			# Log progress
			if step % 5 == 0:
				uc_reward = rewards[0]
				avg_consumer_reward = np.mean([r for aid, r in rewards.items() if aid > 0])
				logger.info(f"Step {step}: UC reward={uc_reward:.3f}, Avg consumer reward={avg_consumer_reward:.3f}")
			
			step += 1
		
		# Episode summary
		logger.info("\nEpisode completed!")
		logger.info(f"Total steps: {step}")
		logger.info(f"UC total reward: {episode_rewards[0]:.2f}")
		consumer_rewards = [r for aid, r in episode_rewards.items() if aid > 0]
		logger.info(f"Average consumer total reward: {np.mean(consumer_rewards):.2f}")
		
		# Calculate Nash gap
		nash_gap = abs(episode_rewards[0] - np.mean(consumer_rewards))
		logger.info(f"Final Nash gap: {nash_gap:.2f}")
		
		return True
		
	except Exception as e:
		logger.error(f"Example failed: {e}")
		import traceback
		traceback.print_exc()
		return False


def run_async_episode():
	"""Run episode with async wrapper (leader-follower structure)."""
	try:
		from envs.stackelberg.stackelberg_game.env_factory import make_stackelberg_env
		
		logger.info("\n" + "="*50)
		logger.info("Running async episode (leader-follower)...")
		
		# Create environment with async wrapper
		env = make_stackelberg_env('stackelberg_13bus', use_async_wrapper=True)
		
		# Reset
		observations = env.reset()
		done = False
		step = 0
		
		while not done and step < 20:  # Limit steps for example
			# Get current phase
			phase_info = env.get_phase_info()
			
			if phase_info['current_phase'] == 'uc_decision':
				# UC makes decision
				uc_action = env.action_spaces[0].sample()
				actions = {0: uc_action}
				logger.info(f"Step {step}: UC decision phase")
			else:
				# Consumers respond
				actions = {}
				for cid in env.consumer_agent_ids:
					actions[cid] = env.action_spaces[cid].sample()
				logger.info(f"Step {step}: Consumer response phase")
			
			# Execute step
			observations, rewards, done, infos = env.step(actions)
			step += 1
		
		logger.info("Async episode completed!")
		return True
		
	except Exception as e:
		logger.error(f"Async example failed: {e}")
		import traceback
		traceback.print_exc()
		return False


def display_environment_info():
	"""Display environment configuration information."""
	try:
		from envs.stackelberg.stackelberg_game.env_factory import load_stackelberg_config
		
		logger.info("\n" + "="*50)
		logger.info("Environment Configuration Info")
		
		# Load configuration
		config = load_stackelberg_config('stackelberg_13bus')
		
		# Display key settings
		logger.info(f"\nSystem: {config['system_name']}")
		logger.info(f"Max episode steps: {config['max_episode_steps']}")
		logger.info(f"Number of consumer agents: {config['n_consumer_agents']}")
		
		logger.info("\nUC Action Space:")
		for action, bounds in config['uc_action_space'].items():
			logger.info(f"  - {action}: [{bounds['low']}, {bounds['high']}]")
		
		logger.info("\nConsumer Action Space:")
		for action, bounds in config['consumer_action_space'].items():
			logger.info(f"  - {action}: [{bounds['low']}, {bounds['high']}]")
		
		logger.info("\nReward Weights (UC):")
		for component, weight in config['reward_weights']['uc'].items():
			logger.info(f"  - {component}: {weight}")
		
		return True
		
	except Exception as e:
		logger.error(f"Config display failed: {e}")
		return False


def main():
	"""Run all examples."""
	logger.info("STACKELBERG GAME ENVIRONMENT EXAMPLE")
	logger.info("="*50)
	
	# Display configuration
	display_environment_info()
	
	# Run simple episode
	logger.info("\n" + "="*50)
	if run_simple_episode():
		logger.info("✓ Simple episode example completed successfully")
	else:
		logger.info("✗ Simple episode example failed")
	
	# Run async episode
	if run_async_episode():
		logger.info("✓ Async episode example completed successfully")
	else:
		logger.info("✗ Async episode example failed")
	
	logger.info("\n" + "="*50)
	logger.info("Example completed!")


if __name__ == "__main__":
	main()