#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test imports for all PowerZoo modules.

This script systematically tests imports for all Python modules in the project,
focusing on powerzoo and powerzoo_llm environments.
"""

import sys
import os
import importlib
import traceback
from pathlib import Path
from typing import List, Tuple, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def test_import(module_name: str) -> Tuple[bool, str]:
	"""
	Test importing a module.

	Args:
		module_name: Fully qualified module name

	Returns:
		Tuple of (success, error_message)
	"""
	try:
		importlib.import_module(module_name)
		return True, ""
	except Exception as e:
		return False, f"{type(e).__name__}: {str(e)}"


def get_module_name_from_path(file_path: Path, root: Path) -> str:
	"""Convert file path to module name."""
	relative = file_path.relative_to(root)
	parts = list(relative.parts)
	# Remove .py extension
	if parts[-1].endswith(".py"):
		parts[-1] = parts[-1][:-3]
	# Handle __init__.py
	if parts[-1] == "__init__":
		parts = parts[:-1]
	return ".".join(parts)


def find_python_files(directory: Path, exclude_patterns: List[str] = None) -> List[Path]:
	"""Find all Python files in a directory."""
	if exclude_patterns is None:
		exclude_patterns = ["__pycache__", ".git", "venv", ".venv", "build", "dist", "*.egg-info"]

	python_files = []
	for file_path in directory.rglob("*.py"):
		# Check exclusions
		skip = False
		for pattern in exclude_patterns:
			if pattern in str(file_path):
				skip = True
				break
		if not skip:
			python_files.append(file_path)

	return sorted(python_files)


def test_core_imports() -> Dict[str, Tuple[bool, str]]:
	"""Test core module imports."""
	core_modules = [
		# Top-level modules
		"envs",
		"algorithms",
		"runners",
		"models",
		"utils",
		"common",

		# PowerZoo environment
		"envs.powerzoo",
		"envs.powerzoo.powerzoo_env",
		"envs.powerzoo.powerzoo_logger",
		"envs.powerzoo.powerzoo.env",
		"envs.powerzoo.powerzoo.circuit",
		"envs.powerzoo.powerzoo.loadprofile",
		"envs.powerzoo.powerzoo.env_register",

		# PowerZoo_LLM environment
		"envs.powerzoo_llm",
		"envs.powerzoo_llm.base_env",
		"envs.powerzoo_llm.base_env.powerzoo_env",
		"envs.powerzoo_llm.base_env.env",
		"envs.powerzoo_llm.base_env.env_register",
		"envs.powerzoo_llm.circuit_system",
		"envs.powerzoo_llm.circuit_system.circuit",
		"envs.powerzoo_llm.circuit_system.components",
		"envs.powerzoo_llm.data_process",
		"envs.powerzoo_llm.data_process.loadprofile",
		"envs.powerzoo_llm.data_process.loadprofile_core",
		"envs.powerzoo_llm.rewards",
		"envs.powerzoo_llm.rewards.powerzoo_reward",
		"envs.powerzoo_llm.rewards.lagrangian",
		"envs.powerzoo_llm.logging",
		"envs.powerzoo_llm.logging.base_logger",
		"envs.powerzoo_llm.single_agent",
		"envs.powerzoo_llm.single_agent.single_agent_env",

		# Algorithms
		"algorithms.algo_registry",

		# Runners
		"runners.on_policy_runner_base",
		"runners.off_policy_runner_base",

		# Utils
		"utils.config_utils",
	]

	results = {}
	for module in core_modules:
		success, error = test_import(module)
		results[module] = (success, error)

	return results


def test_all_imports(directory: Path) -> Dict[str, Tuple[bool, str]]:
	"""Test imports for all Python files in a directory."""
	files = find_python_files(directory)
	results = {}

	for file_path in files:
		try:
			module_name = get_module_name_from_path(file_path, PROJECT_ROOT)
			if module_name:
				success, error = test_import(module_name)
				results[module_name] = (success, error)
		except Exception as e:
			results[str(file_path)] = (False, str(e))

	return results


def print_results(results: Dict[str, Tuple[bool, str]], title: str):
	"""Print test results."""
	print(f"\n{BLUE}{'='*60}{RESET}")
	print(f"{BLUE}{title}{RESET}")
	print(f"{BLUE}{'='*60}{RESET}\n")

	success_count = 0
	failure_count = 0

	# Group by category
	categories = {}
	for module, (success, error) in results.items():
		parts = module.split(".")
		category = parts[0] if len(parts) > 0 else "other"
		if category not in categories:
			categories[category] = []
		categories[category].append((module, success, error))

	for category, items in sorted(categories.items()):
		print(f"\n{YELLOW}[{category}]{RESET}")
		for module, success, error in sorted(items):
			if success:
				print(f"  {GREEN}✓{RESET} {module}")
				success_count += 1
			else:
				print(f"  {RED}✗{RESET} {module}")
				print(f"    {RED}→ {error}{RESET}")
				failure_count += 1

	print(f"\n{BLUE}{'='*60}{RESET}")
	print(f"Total: {success_count + failure_count} modules")
	print(f"{GREEN}Passed: {success_count}{RESET}")
	print(f"{RED}Failed: {failure_count}{RESET}")
	print(f"{BLUE}{'='*60}{RESET}")

	return success_count, failure_count


def main():
	"""Main entry point."""
	print(f"\n{BLUE}PowerZoo Import Test{RESET}")
	print(f"{BLUE}Project Root: {PROJECT_ROOT}{RESET}")

	# Test core imports
	print(f"\n{YELLOW}Testing core imports...{RESET}")
	core_results = test_core_imports()
	core_success, core_failure = print_results(core_results, "Core Module Imports")

	# Test powerzoo directory
	print(f"\n{YELLOW}Testing envs/powerzoo imports...{RESET}")
	powerzoo_dir = PROJECT_ROOT / "envs" / "powerzoo"
	if powerzoo_dir.exists():
		powerzoo_results = test_all_imports(powerzoo_dir)
		pz_success, pz_failure = print_results(powerzoo_results, "PowerZoo Environment Imports")
	else:
		print(f"{RED}Directory not found: {powerzoo_dir}{RESET}")
		pz_success, pz_failure = 0, 0

	# Test powerzoo_llm directory
	print(f"\n{YELLOW}Testing envs/powerzoo_llm imports...{RESET}")
	powerzoo_llm_dir = PROJECT_ROOT / "envs" / "powerzoo_llm"
	if powerzoo_llm_dir.exists():
		powerzoo_llm_results = test_all_imports(powerzoo_llm_dir)
		pzl_success, pzl_failure = print_results(powerzoo_llm_results, "PowerZoo_LLM Environment Imports")
	else:
		print(f"{RED}Directory not found: {powerzoo_llm_dir}{RESET}")
		pzl_success, pzl_failure = 0, 0

	# Summary
	total_success = core_success + pz_success + pzl_success
	total_failure = core_failure + pz_failure + pzl_failure

	print(f"\n{BLUE}{'='*60}{RESET}")
	print(f"{BLUE}FINAL SUMMARY{RESET}")
	print(f"{BLUE}{'='*60}{RESET}")
	print(f"Total modules tested: {total_success + total_failure}")
	print(f"{GREEN}Total passed: {total_success}{RESET}")
	print(f"{RED}Total failed: {total_failure}{RESET}")

	if total_failure == 0:
		print(f"\n{GREEN}All imports successful!{RESET}")
		return 0
	else:
		print(f"\n{RED}Some imports failed. Please check the errors above.{RESET}")
		return 1


if __name__ == "__main__":
	sys.exit(main())
