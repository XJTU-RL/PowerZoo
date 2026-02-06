# -*- coding: utf-8 -*-
"""
Integration tests for Stackelberg game environment.

This module provides comprehensive tests for the Stackelberg-Nash game
environment implementation with PowerZoo-compatible interface.
"""

import numpy as np
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('StackelbergTest')


def test_basic_environment():
	"""Test basic environment creation and reset."""
	logger.info("=" * 50)
	logger.info("Testing basic environment functionality...")

	try:
		from envs.stackelberg.stackelberg_game.env_factory import make_stackelberg_env

		# Create environment
		env = make_stackelberg_env('stackelberg_13bus', use_async_wrapper=False)
		logger.info("OK Environment created successfully")

		# Test reset - returns (obs_dict)
		observations = env.reset()
		logger.info(f"OK Environment reset successful")
		logger.info(f"  - Number of agents: {len(observations)}")
		logger.info(f"  - UC observation shape: {observations[0].shape}")
		if 1 in observations:
			logger.info(f"  - Consumer observation shape: {observations[1].shape}")

		# Test agent properties
		logger.info(f"  - UC agent ID: {env.uc_agent_id}")
		logger.info(f"  - Consumer agent IDs: {env.consumer_agent_ids}")
		logger.info(f"  - Agent types: {env.agent_types}")

		return True

	except Exception as e:
		logger.error(f"FAIL Basic environment test failed: {e}")
		import traceback
		traceback.print_exc()
		return False


def test_action_spaces():
	"""Test action space configuration."""
	logger.info("=" * 50)
	logger.info("Testing action spaces...")
	
	try:
		from envs.stackelberg.stackelberg_game.env_factory import make_stackelberg_env
		
		env = make_stackelberg_env('stackelberg_13bus', use_async_wrapper=False)
		env.reset()
		
		# Test UC action space
		uc_action_space = env.action_spaces[0]
		logger.info(f"✓ UC action space: {uc_action_space}")
		logger.info(f"  - Shape: {uc_action_space.shape}")
		logger.info(f"  - Low: {uc_action_space.low}")
		logger.info(f"  - High: {uc_action_space.high}")
		
		# Test consumer action spaces
		for cid in env.consumer_agent_ids[:3]:  # Test first 3 consumers
			consumer_action_space = env.action_spaces[cid]
			logger.info(f"✓ Consumer {cid} action space: {consumer_action_space.shape}")
		
		# Test sample actions
		uc_action = uc_action_space.sample()
		logger.info(f"✓ UC sample action: {uc_action}")
		
		return True
		
	except Exception as e:
		logger.error(f"✗ Action space test failed: {e}")
		return False


def test_step_execution():
	"""Test environment step execution."""
	logger.info("=" * 50)
	logger.info("Testing step execution...")
	
	try:
		from envs.stackelberg.stackelberg_game.env_factory import make_stackelberg_env
		
		env = make_stackelberg_env('stackelberg_13bus', use_async_wrapper=False)
		observations = env.reset()
		
		# Create actions for all agents
		actions = {}
		for agent_id in env.action_spaces:
			actions[agent_id] = env.action_spaces[agent_id].sample()
		
		# Execute step
		logger.info("Executing environment step...")
		next_obs, rewards, done, infos = env.step(actions)
		
		logger.info("✓ Step executed successfully")
		logger.info(f"  - Rewards: UC={rewards[0]:.3f}, Avg Consumer={np.mean([r for aid, r in rewards.items() if aid > 0]):.3f}")
		logger.info(f"  - Done: {done}")
		logger.info(f"  - UC info keys: {list(infos[0].keys())}")
		
		# Check reward structure
		assert 0 in rewards, "UC reward missing"
		assert len(rewards) == len(env.action_spaces), "Reward count mismatch"
		
		return True
		
	except Exception as e:
		logger.error(f"✗ Step execution test failed: {e}")
		return False


def test_async_wrapper():
	"""Test asynchronous wrapper functionality."""
	logger.info("=" * 50)
	logger.info("Testing async wrapper...")
	
	try:
		from envs.stackelberg.stackelberg_game.env_factory import make_stackelberg_env
		
		env = make_stackelberg_env('stackelberg_13bus', use_async_wrapper=True)
		observations = env.reset()
		
		logger.info("✓ Async environment created")
		
		# Phase 1: UC decision
		logger.info("Phase 1: UC decision...")
		phase_info = env.get_phase_info()
		logger.info(f"  - Current phase: {phase_info['current_phase']}")
		
		uc_action = env.action_spaces[0].sample()
		actions = {0: uc_action}
		
		obs1, rewards1, done1, infos1 = env.step(actions)
		logger.info(f"✓ UC phase executed")
		logger.info(f"  - UC reward: {rewards1[0]:.3f}")
		
		# Phase 2: Consumer response
		logger.info("Phase 2: Consumer response...")
		phase_info = env.get_phase_info()
		logger.info(f"  - Current phase: {phase_info['current_phase']}")
		
		consumer_actions = {}
		for cid in env.consumer_agent_ids:
			consumer_actions[cid] = env.action_spaces[cid].sample()
		
		obs2, rewards2, done2, infos2 = env.step(consumer_actions)
		logger.info(f"✓ Consumer phase executed")
		logger.info(f"  - Avg consumer reward: {np.mean([r for aid, r in rewards2.items() if aid > 0]):.3f}")
		
		# Check phase cycling
		phase_info = env.get_phase_info()
		if not done2:
			assert phase_info['current_phase'] == 'uc_decision', "Phase should cycle back to UC"
		
		return True
		
	except Exception as e:
		logger.error(f"✗ Async wrapper test failed: {e}")
		return False


def test_episode_completion():
	"""Test full episode execution."""
	logger.info("=" * 50)
	logger.info("Testing full episode...")
	
	try:
		from envs.stackelberg.stackelberg_game.env_factory import make_stackelberg_env
		
		env = make_stackelberg_env('stackelberg_13bus', use_async_wrapper=False)
		observations = env.reset()
		
		episode_rewards = {agent_id: 0.0 for agent_id in env.action_spaces}
		step_count = 0
		done = False
		
		logger.info("Running episode...")
		
		while not done and step_count < 100:  # Safety limit
			# Create actions
			actions = {}
			for agent_id in env.action_spaces:
				actions[agent_id] = env.action_spaces[agent_id].sample()
			
			# Step
			observations, rewards, done, infos = env.step(actions)
			
			# Accumulate rewards
			for agent_id, reward in rewards.items():
				episode_rewards[agent_id] += reward
			
			step_count += 1
			
			if step_count % 5 == 0:
				logger.info(f"  Step {step_count}: UC reward={rewards[0]:.3f}")
		
		logger.info(f"✓ Episode completed in {step_count} steps")
		logger.info(f"  - Total UC reward: {episode_rewards[0]:.2f}")
		logger.info(f"  - Avg total consumer reward: {np.mean([r for aid, r in episode_rewards.items() if aid > 0]):.2f}")
		
		return True
		
	except Exception as e:
		logger.error(f"✗ Episode completion test failed: {e}")
		return False


def test_monitoring():
	"""Test monitoring functionality."""
	logger.info("=" * 50)
	logger.info("Testing monitoring...")
	
	try:
		from envs.stackelberg.stackelberg_game.env_factory import make_stackelberg_env
		from envs.stackelberg.stackelberg_game.stackelberg_monitor import StackelbergMonitor
		
		# Create environment with monitoring
		config = {
			'monitoring_config': {
				'enable': True,
				'log_dir': 'logs/stackelberg_test',
				'save_interval': 10,
				'plot_interval': 20
			}
		}
		
		env = make_stackelberg_env('stackelberg_13bus', config=config, use_async_wrapper=False)
		monitor = StackelbergMonitor(
			log_dir='logs/stackelberg_test',
			experiment_name='integration_test'
		)
		
		observations = env.reset()
		
		# Run a few steps with monitoring
		for step in range(5):
			actions = {}
			for agent_id in env.action_spaces:
				actions[agent_id] = env.action_spaces[agent_id].sample()
			
			observations, rewards, done, infos = env.step(actions)
			
			# Log to monitor
			monitor.log_step(step, observations, actions, rewards, infos)
		
		# Get summary
		summary = monitor.get_summary_statistics()
		logger.info("✓ Monitoring functional")
		logger.info(f"  - Total steps logged: {summary['total_steps']}")
		logger.info(f"  - Metrics tracked: {len(summary['current_metrics'])}")
		
		# Clean up
		monitor.close()
		
		return True
		
	except Exception as e:
		logger.error(f"✗ Monitoring test failed: {e}")
		return False


def test_different_systems():
	"""Test different power system configurations."""
	logger.info("=" * 50)
	logger.info("Testing different power systems...")
	
	try:
		from envs.stackelberg.stackelberg_game.env_factory import (
			make_stackelberg_13bus,
			make_stackelberg_34bus,
			make_stackelberg_123bus
		)
		
		systems = [
			('13bus', make_stackelberg_13bus),
			('34bus', make_stackelberg_34bus),
			('123bus', make_stackelberg_123bus)
		]
		
		for system_name, factory_fn in systems:
			logger.info(f"\nTesting {system_name} system...")
			
			env = factory_fn(use_async_wrapper=False)
			observations = env.reset()
			
			# One step test
			actions = {}
			for agent_id in env.action_spaces:
				actions[agent_id] = env.action_spaces[agent_id].sample()
			
			observations, rewards, done, infos = env.step(actions)
			
			logger.info(f"✓ {system_name} system functional")
			logger.info(f"  - Agents: {len(env.action_spaces)}")
			logger.info(f"  - Observation dims: UC={observations[0].shape}, Consumer={observations[1].shape}")
		
		return True
		
	except Exception as e:
		logger.error(f"✗ Different systems test failed: {e}")
		return False


def test_nash_gap_tracking():
	"""Test Nash gap tracking between UC and consumers."""
	logger.info("=" * 50)
	logger.info("Testing Nash gap tracking...")
	
	try:
		from envs.stackelberg.stackelberg_game.env_factory import make_stackelberg_env
		
		env = make_stackelberg_env('stackelberg_13bus', use_async_wrapper=False)
		observations = env.reset()
		
		nash_gaps = []
		
		# Run several steps
		for step in range(10):
			actions = {}
			for agent_id in env.action_spaces:
				actions[agent_id] = env.action_spaces[agent_id].sample()
			
			observations, rewards, done, infos = env.step(actions)
			
			# Calculate Nash gap
			uc_reward = rewards[0]
			consumer_rewards = [r for aid, r in rewards.items() if aid > 0]
			avg_consumer_reward = np.mean(consumer_rewards)
			nash_gap = abs(uc_reward - avg_consumer_reward)
			nash_gaps.append(nash_gap)
			
			if step % 3 == 0:
				logger.info(f"  Step {step}: Nash gap = {nash_gap:.4f}")
		
		logger.info(f"✓ Nash gap tracking functional")
		logger.info(f"  - Average Nash gap: {np.mean(nash_gaps):.4f}")
		logger.info(f"  - Min Nash gap: {np.min(nash_gaps):.4f}")
		logger.info(f"  - Max Nash gap: {np.max(nash_gaps):.4f}")
		
		return True
		
	except Exception as e:
		logger.error(f"✗ Nash gap tracking test failed: {e}")
		return False


def test_powerzoo_compatibility():
	"""Test PowerZoo interface compatibility."""
	logger.info("=" * 50)
	logger.info("Testing PowerZoo interface compatibility...")

	try:
		from envs.stackelberg.stackelberg_vvc_env import StackelbergVVCEnv, make_stackelberg_env

		# Create environment using factory
		args = {
			'env_name': 'stackelberg_13bus',
			'seed': 42,
		}
		env = StackelbergVVCEnv(args)

		# Test required attributes
		assert hasattr(env, 'n_agents'), "Missing n_agents"
		assert hasattr(env, 'share_observation_space'), "Missing share_observation_space"
		assert hasattr(env, 'observation_space'), "Missing observation_space"
		assert hasattr(env, 'action_space'), "Missing action_space"
		logger.info("OK Required attributes present")

		# Test share_observation_space
		assert len(env.share_observation_space) == env.n_agents, "share_observation_space length mismatch"
		logger.info(f"OK share_observation_space: {len(env.share_observation_space)} spaces")

		# Test reset returns (obs, share_obs, avail_actions)
		result = env.reset()
		assert len(result) == 3, f"reset() should return 3 items, got {len(result)}"
		local_obs, share_obs, avail_actions = result
		logger.info(f"OK reset() returns 3 items: obs, share_obs, avail_actions")

		assert len(local_obs) == env.n_agents, "local_obs length mismatch"
		assert len(share_obs) == env.n_agents, "share_obs length mismatch"
		logger.info("OK reset() output shapes correct")

		# Test get_avail_actions
		avail = env.get_avail_actions()
		assert len(avail) == env.n_agents, "avail_actions length mismatch"
		logger.info(f"OK get_avail_actions() returns {len(avail)} items")

		# Test step returns (obs, share_obs, rewards, dones, infos, avail_actions)
		actions = [env.action_space[i].sample() for i in range(env.n_agents)]
		result = env.step(actions)
		assert len(result) == 6, f"step() should return 6 items, got {len(result)}"
		local_obs, share_obs, rewards, dones, infos, avail_actions = result

		assert len(local_obs) == env.n_agents, "step local_obs length mismatch"
		assert len(rewards) == env.n_agents, "step rewards length mismatch"
		assert len(dones) == env.n_agents, "step dones length mismatch"
		logger.info("OK step() returns 6 items with correct shapes")

		# Test reward format [[r]]
		for r in rewards:
			assert isinstance(r, list), f"Reward should be list, got {type(r)}"
			assert len(r) == 1, f"Reward should have length 1, got {len(r)}"
		logger.info("OK Reward format [[r]] correct")

		env.close()
		return True

	except Exception as e:
		logger.error(f"FAIL PowerZoo compatibility test failed: {e}")
		import traceback
		traceback.print_exc()
		return False


def run_all_tests():
	"""Run all integration tests."""
	logger.info("\n" + "=" * 70)
	logger.info("STACKELBERG GAME ENVIRONMENT INTEGRATION TESTS")
	logger.info("=" * 70 + "\n")

	tests = [
		("Basic Environment", test_basic_environment),
		("Action Spaces", test_action_spaces),
		("Step Execution", test_step_execution),
		("Async Wrapper", test_async_wrapper),
		("Episode Completion", test_episode_completion),
		("PowerZoo Compatibility", test_powerzoo_compatibility),
		("Monitoring", test_monitoring),
		("Nash Gap Tracking", test_nash_gap_tracking)
	]

	results = []

	for test_name, test_func in tests:
		start_time = time.time()
		try:
			success = test_func()
			elapsed = time.time() - start_time
			results.append((test_name, success, elapsed))
		except Exception as e:
			logger.error(f"Test {test_name} crashed: {e}")
			import traceback
			traceback.print_exc()
			results.append((test_name, False, 0))

		logger.info("")  # Blank line between tests

	# Summary
	logger.info("\n" + "=" * 70)
	logger.info("TEST SUMMARY")
	logger.info("=" * 70)

	total_tests = len(results)
	passed_tests = sum(1 for _, success, _ in results if success)

	for test_name, success, elapsed in results:
		status = "OK PASSED" if success else "FAIL"
		logger.info(f"{test_name:.<40} {status} ({elapsed:.2f}s)")

	logger.info("=" * 70)
	logger.info(f"Total: {passed_tests}/{total_tests} tests passed")

	if passed_tests == total_tests:
		logger.info("OK ALL TESTS PASSED!")
	else:
		logger.info("FAIL Some tests failed.")

	return passed_tests == total_tests


if __name__ == "__main__":
	success = run_all_tests()
	exit(0 if success else 1)