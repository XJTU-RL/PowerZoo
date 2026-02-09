"""Training subprocess lifecycle management.

Provides a singleton ``process_manager`` that can start, stop, and monitor
training subprocesses.  Each task is launched via
``python examples/multi_agent/scripts/train.py`` with the generated config,
and its stdout/stderr are redirected to a per-task log file.
"""

import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Project root: two levels up from training_ui/core/
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Directory for task log files
LOG_DIR = PROJECT_ROOT / "training_ui" / ".cache" / "logs"


@dataclass
class TaskInfo:
	"""Runtime metadata for a single training task."""

	task_id: str
	algo: str
	env: str
	exp_name: str
	config_path: str
	status: str  # "running" | "completed" | "failed" | "stopped"
	pid: int | None = None
	start_time: str = ""
	end_time: str = ""
	log_file: str = ""


class ProcessManager:
	"""Manage training subprocesses.

	Holds references to running ``Popen`` objects and their metadata.
	Call ``list_all()`` or ``status()`` to get refreshed status info --
	the manager polls process state lazily on each query.
	"""

	def __init__(self) -> None:
		self._tasks: dict[str, TaskInfo] = {}
		self._processes: dict[str, subprocess.Popen] = {}

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def start(
		self,
		algo: str,
		env: str,
		exp_name: str,
		config_path: str,
	) -> str:
		"""Launch a new training subprocess.

		Command template::

			python examples/multi_agent/scripts/train.py \\
				--algo {algo} --env {env} --exp_name {exp_name} \\
				--load_config {config_path}

		Args:
			algo: Algorithm name.
			env: Environment name.
			exp_name: Experiment name.
			config_path: Absolute path to the generated YAML config file.

		Returns:
			task_id (first 8 hex chars of a uuid4).

		Raises:
			RuntimeError: If the subprocess fails to start.
		"""
		task_id = uuid.uuid4().hex[:8]

		# Ensure log directory exists
		LOG_DIR.mkdir(parents=True, exist_ok=True)
		log_file = LOG_DIR / f"task_{task_id}.log"

		train_script = PROJECT_ROOT / "examples" / "multi_agent" / "scripts" / "train.py"
		cmd = [
			sys.executable,
			str(train_script),
			"--algo", algo,
			"--env", env,
			"--exp_name", exp_name,
			"--load_config", config_path,
		]

		# Build environment with PYTHONPATH pointing at project root
		env_vars = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}

		try:
			log_fh = open(log_file, "w", encoding="utf-8")
			proc = subprocess.Popen(
				cmd,
				stdout=log_fh,
				stderr=subprocess.STDOUT,
				cwd=str(PROJECT_ROOT),
				env=env_vars,
				start_new_session=True,  # isolate from parent signals
			)
		except Exception as exc:
			raise RuntimeError(f"Failed to start training process: {exc}") from exc

		now = datetime.now().isoformat(timespec="seconds")
		task = TaskInfo(
			task_id=task_id,
			algo=algo,
			env=env,
			exp_name=exp_name,
			config_path=config_path,
			status="running",
			pid=proc.pid,
			start_time=now,
			log_file=str(log_file),
		)

		self._tasks[task_id] = task
		self._processes[task_id] = proc
		return task_id

	def stop(self, task_id: str) -> bool:
		"""Stop a running task gracefully.

		Sends SIGTERM first, waits up to 5 seconds, then sends SIGKILL
		if the process is still alive.

		Args:
			task_id: Task identifier.

		Returns:
			True if the task was stopped (or was already finished),
			False if task_id is unknown.
		"""
		if task_id not in self._tasks:
			return False

		proc = self._processes.get(task_id)
		if proc is None or proc.poll() is not None:
			# Already finished
			self._refresh_status(task_id)
			return True

		# Graceful termination
		try:
			os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
		except (ProcessLookupError, PermissionError):
			pass

		# Wait up to 5 seconds
		deadline = time.monotonic() + 5.0
		while time.monotonic() < deadline:
			if proc.poll() is not None:
				break
			time.sleep(0.2)

		# Force kill if still alive
		if proc.poll() is None:
			try:
				os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
				proc.wait(timeout=3)
			except (ProcessLookupError, PermissionError, subprocess.TimeoutExpired):
				pass

		task = self._tasks[task_id]
		task.status = "stopped"
		task.end_time = datetime.now().isoformat(timespec="seconds")
		return True

	def status(self, task_id: str) -> TaskInfo | None:
		"""Get current status of a task (refreshes from process state).

		Args:
			task_id: Task identifier.

		Returns:
			TaskInfo with up-to-date status, or None if unknown.
		"""
		if task_id not in self._tasks:
			return None
		self._refresh_status(task_id)
		return self._tasks[task_id]

	def list_all(self) -> list[TaskInfo]:
		"""List all tasks with refreshed statuses.

		Returns:
			List of TaskInfo, most-recently started first.
		"""
		for tid in list(self._tasks):
			self._refresh_status(tid)
		return sorted(
			self._tasks.values(),
			key=lambda t: t.start_time,
			reverse=True,
		)

	def log_tail(self, task_id: str, lines: int = 100) -> str:
		"""Read the last *lines* lines from a task's log file.

		Args:
			task_id: Task identifier.
			lines: Number of trailing lines to return.

		Returns:
			Log content string, or an empty string if unavailable.
		"""
		task = self._tasks.get(task_id)
		if task is None or not task.log_file:
			return ""

		log_path = Path(task.log_file)
		if not log_path.exists():
			return ""

		try:
			with open(log_path, "r", encoding="utf-8", errors="replace") as f:
				all_lines = f.readlines()
			return "".join(all_lines[-lines:])
		except OSError:
			return ""

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _refresh_status(self, task_id: str) -> None:
		"""Poll the subprocess and update task status accordingly."""
		task = self._tasks.get(task_id)
		if task is None:
			return

		# Only refresh if currently marked as running
		if task.status != "running":
			return

		proc = self._processes.get(task_id)
		if proc is None:
			task.status = "failed"
			return

		retcode = proc.poll()
		if retcode is None:
			# Still running
			return

		task.end_time = datetime.now().isoformat(timespec="seconds")
		task.status = "completed" if retcode == 0 else "failed"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

process_manager = ProcessManager()
