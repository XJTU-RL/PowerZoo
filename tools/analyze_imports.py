#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Static analysis of imports for all PowerZoo modules.

This script analyzes import statements without executing them,
checking for potential issues in the import structure.
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent

# Known third-party packages
THIRD_PARTY_PACKAGES = {
	"numpy", "np", "scipy", "pandas", "pd", "torch", "gym", "gymnasium",
	"matplotlib", "plt", "seaborn", "sns", "h5py", "yaml", "pyyaml",
	"tensorboard", "tensorboardX", "setproctitle", "networkx", "nx",
	"imageio", "plotly", "sklearn", "stable_baselines3", "sb3",
	"dss", "opendssdirect", "absl", "logging", "os", "sys", "typing",
	"pathlib", "json", "copy", "time", "datetime", "collections",
	"functools", "itertools", "warnings", "traceback", "abc", "enum",
	"dataclasses", "re", "math", "random", "pickle", "tempfile",
	"shutil", "glob", "subprocess", "multiprocessing", "threading",
	"queue", "socket", "struct", "argparse", "configparser", "csv",
	"io", "contextlib", "operator", "inspect", "importlib", "types"
}

# Project modules
PROJECT_MODULES = {
	"envs", "algorithms", "runners", "models", "utils", "common", "configs"
}


class ImportAnalyzer(ast.NodeVisitor):
	"""AST visitor to extract import statements."""

	def __init__(self, file_path: Path):
		self.file_path = file_path
		self.imports: List[Dict] = []
		self.from_imports: List[Dict] = []

	def visit_Import(self, node):
		for alias in node.names:
			self.imports.append({
				"module": alias.name,
				"alias": alias.asname,
				"line": node.lineno
			})
		self.generic_visit(node)

	def visit_ImportFrom(self, node):
		module = node.module or ""
		for alias in node.names:
			self.from_imports.append({
				"module": module,
				"name": alias.name,
				"alias": alias.asname,
				"line": node.lineno,
				"level": node.level  # 0 = absolute, >0 = relative
			})
		self.generic_visit(node)


def analyze_file(file_path: Path) -> Tuple[List[Dict], List[Dict], List[str]]:
	"""Analyze imports in a single file."""
	try:
		with open(file_path, "r", encoding="utf-8") as f:
			source = f.read()

		tree = ast.parse(source, filename=str(file_path))
		analyzer = ImportAnalyzer(file_path)
		analyzer.visit(tree)

		issues = []

		# Check for relative imports
		for imp in analyzer.from_imports:
			if imp["level"] > 0:
				issues.append(
					f"Line {imp['line']}: Relative import 'from {'.'*imp['level']}{imp['module']} import {imp['name']}'"
				)

		return analyzer.imports, analyzer.from_imports, issues

	except SyntaxError as e:
		return [], [], [f"SyntaxError: {e}"]
	except Exception as e:
		return [], [], [f"Error: {e}"]


def check_internal_import(module: str, all_modules: Set[str]) -> Tuple[bool, str]:
	"""Check if an internal import is valid."""
	# Get the base module
	parts = module.split(".")
	base = parts[0]

	if base not in PROJECT_MODULES:
		return True, ""  # Not a project module, skip

	# Check if the full module path exists
	if module in all_modules:
		return True, ""

	# Check if it's a package import (might be valid)
	for existing in all_modules:
		if existing.startswith(module + "."):
			return True, ""

	return False, f"Module '{module}' not found in project"


def find_all_modules(directory: Path) -> Set[str]:
	"""Find all Python modules in the project."""
	modules = set()

	for py_file in directory.rglob("*.py"):
		if "__pycache__" in str(py_file):
			continue

		relative = py_file.relative_to(PROJECT_ROOT)
		parts = list(relative.parts)

		# Remove .py extension
		if parts[-1].endswith(".py"):
			parts[-1] = parts[-1][:-3]

		# Handle __init__.py
		if parts[-1] == "__init__":
			parts = parts[:-1]

		if parts:
			module_name = ".".join(parts)
			modules.add(module_name)

			# Also add all parent packages
			for i in range(1, len(parts)):
				parent = ".".join(parts[:i])
				modules.add(parent)

	return modules


def analyze_project():
	"""Analyze all imports in the project."""
	print("=" * 60)
	print("PowerZoo Import Static Analysis")
	print("=" * 60)
	print(f"Project root: {PROJECT_ROOT}\n")

	# Find all modules
	all_modules = find_all_modules(PROJECT_ROOT)
	print(f"Found {len(all_modules)} Python modules\n")

	# Directories to analyze
	dirs_to_analyze = [
		PROJECT_ROOT / "envs" / "vvc",
		PROJECT_ROOT / "envs" / "smartgrid",
		PROJECT_ROOT / "algorithms",
		PROJECT_ROOT / "runners",
		PROJECT_ROOT / "utils",
	]

	total_files = 0
	total_imports = 0
	total_from_imports = 0
	all_issues = []
	third_party_deps = set()
	internal_deps = defaultdict(set)

	for directory in dirs_to_analyze:
		if not directory.exists():
			print(f"[SKIP] Directory not found: {directory}")
			continue

		print(f"\n[Analyzing] {directory.relative_to(PROJECT_ROOT)}/")
		print("-" * 40)

		for py_file in sorted(directory.rglob("*.py")):
			if "__pycache__" in str(py_file):
				continue

			total_files += 1
			rel_path = py_file.relative_to(PROJECT_ROOT)
			imports, from_imports, issues = analyze_file(py_file)

			total_imports += len(imports)
			total_from_imports += len(from_imports)

			# Track dependencies
			for imp in imports:
				base = imp["module"].split(".")[0]
				if base in THIRD_PARTY_PACKAGES or base not in PROJECT_MODULES:
					third_party_deps.add(imp["module"].split(".")[0])
				else:
					internal_deps[str(rel_path)].add(imp["module"])

			for imp in from_imports:
				if imp["level"] == 0:  # Only absolute imports
					base = imp["module"].split(".")[0] if imp["module"] else ""
					if base in THIRD_PARTY_PACKAGES or (base and base not in PROJECT_MODULES):
						third_party_deps.add(base)
					elif base in PROJECT_MODULES:
						internal_deps[str(rel_path)].add(imp["module"])

			# Check for issues
			if issues:
				for issue in issues:
					all_issues.append(f"{rel_path}: {issue}")

			# Check internal imports
			for imp in from_imports:
				if imp["level"] == 0 and imp["module"]:
					base = imp["module"].split(".")[0]
					if base in PROJECT_MODULES:
						valid, msg = check_internal_import(imp["module"], all_modules)
						if not valid:
							all_issues.append(f"{rel_path}:{imp['line']}: {msg}")

	# Summary
	print("\n" + "=" * 60)
	print("ANALYSIS SUMMARY")
	print("=" * 60)

	print(f"\nFiles analyzed: {total_files}")
	print(f"Import statements: {total_imports}")
	print(f"From-import statements: {total_from_imports}")

	print(f"\n[Third-party dependencies detected]")
	for dep in sorted(third_party_deps):
		status = "OK" if dep in THIRD_PARTY_PACKAGES or dep.startswith("_") else "?"
		print(f"  - {dep} [{status}]")

	if all_issues:
		print(f"\n[Issues Found: {len(all_issues)}]")
		for issue in sorted(set(all_issues)):
			print(f"  ! {issue}")
	else:
		print("\n[No issues found]")

	# Check requirements.txt coverage
	print("\n" + "=" * 60)
	print("REQUIREMENTS.TXT COVERAGE CHECK")
	print("=" * 60)

	req_file = PROJECT_ROOT / "requirements.txt"
	if req_file.exists():
		with open(req_file) as f:
			req_content = f.read().lower()

		# Map package names to requirements.txt names
		package_mapping = {
			"numpy": "numpy", "np": "numpy",
			"pandas": "pandas", "pd": "pandas",
			"scipy": "scipy",
			"torch": "torch",
			"gym": "gym",
			"gymnasium": "gymnasium",
			"matplotlib": "matplotlib", "plt": "matplotlib",
			"seaborn": "seaborn", "sns": "seaborn",
			"yaml": "pyyaml",
			"h5py": "h5py",
			"networkx": "networkx", "nx": "networkx",
			"imageio": "imageio",
			"plotly": "plotly",
			"sklearn": "scikit-learn",
			"stable_baselines3": "stable-baselines3",
			"dss": "dss-python",
			"opendssdirect": "dss-python",
			"absl": "absl-py",
			"tensorboard": "tensorboard",
			"tensorboardX": "tensorboardx",
		}

		missing = []
		covered = []
		for dep in third_party_deps:
			req_name = package_mapping.get(dep, dep).lower()
			if req_name in req_content or dep.lower() in req_content:
				covered.append(dep)
			elif dep not in THIRD_PARTY_PACKAGES:
				# Skip standard library
				pass
			else:
				missing.append(dep)

		print(f"\nCovered in requirements.txt: {len(covered)}")
		for dep in sorted(covered):
			print(f"  OK {dep}")

		if missing:
			print(f"\nMissing from requirements.txt: {len(missing)}")
			for dep in sorted(missing):
				print(f"  !! {dep}")
		else:
			print("\nAll third-party dependencies are covered!")

	return len(all_issues) == 0


if __name__ == "__main__":
	success = analyze_project()
	sys.exit(0 if success else 1)
