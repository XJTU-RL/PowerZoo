"""
Training Log Parser (Common)
训练结果解析工具

扫描训练输出目录，解析 progress.txt、training.log 等文件，
为 Training Progress 标签页提供数据支撑。
从 District Dispatch 提取，完全可复用。
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


# -- 训练目录结构约定 --
# results/<algo>_<env>_YYYYMMDD_HHMMSS/
#   ├── progress.txt       # 制表符分隔的训练指标
#   ├── training.log       # 控制台输出日志
#   └── logs/              # TensorBoard 事件文件目录

_PROGRESS_REWARD_COLS = [
	"average_episode_rewards",
	"average_step_rewards",
	"eval_average_episode_rewards",
]

_CONFIG_KEYS = [
	"algorithm",
	"algo",
	"env_name",
	"env",
	"seed",
	"num_agents",
	"n_agents",
	"num_env_steps",
	"episode_length",
	"lr",
	"learning_rate",
	"gamma",
	"batch_size",
]

_RUN_DIR_PATTERN = re.compile(
	r"^(?P<algorithm>.+?)_(?P<env>.+?)_(?P<date>\d{8})_(?P<time>\d{6})$"
)


def scan_training_results(results_dir: str) -> List[Dict[str, Any]]:
	"""扫描训练结果目录，发现所有可用的训练运行记录。

	Args:
		results_dir: 训练结果根目录路径

	Returns:
		训练运行信息列表，按时间倒序排列
	"""
	if not os.path.isdir(results_dir):
		return []

	runs: List[Dict[str, Any]] = []

	for entry in sorted(os.listdir(results_dir)):
		run_path = os.path.join(results_dir, entry)
		if not os.path.isdir(run_path):
			continue

		run_info: Dict[str, Any] = {
			"name": entry,
			"path": os.path.abspath(run_path),
			"algorithm": "unknown",
			"env": "unknown",
			"timestamp": "",
			"datetime": None,
			"has_progress": False,
			"has_log": False,
			"has_tensorboard": False,
			"episodes": None,
		}

		match = _RUN_DIR_PATTERN.match(entry)
		if match:
			run_info["algorithm"] = match.group("algorithm")
			run_info["env"] = match.group("env")
			date_str = match.group("date")
			time_str = match.group("time")
			run_info["timestamp"] = f"{date_str}_{time_str}"
			try:
				run_info["datetime"] = datetime.strptime(
					f"{date_str}{time_str}", "%Y%m%d%H%M%S"
				)
			except ValueError:
				pass

		progress_path = os.path.join(run_path, "progress.txt")
		log_path = os.path.join(run_path, "training.log")
		logs_dir = os.path.join(run_path, "logs")

		run_info["has_progress"] = os.path.isfile(progress_path)
		run_info["has_log"] = os.path.isfile(log_path)
		run_info["has_tensorboard"] = os.path.isdir(logs_dir)

		if run_info["has_progress"]:
			try:
				df = parse_progress_file(progress_path)
				if not df.empty:
					run_info["episodes"] = len(df)
			except Exception:
				pass

		runs.append(run_info)

	runs.sort(key=lambda r: r["timestamp"], reverse=True)
	return runs


def parse_progress_file(progress_path: str) -> pd.DataFrame:
	"""解析 progress.txt 为 DataFrame。

	Args:
		progress_path: progress.txt 文件路径

	Returns:
		训练指标 DataFrame
	"""
	if not os.path.isfile(progress_path):
		return pd.DataFrame()

	try:
		df = pd.read_csv(progress_path, sep="\t")
	except Exception:
		try:
			df = pd.read_csv(progress_path, sep=r"\s+", engine="python")
		except Exception:
			return pd.DataFrame()

	if df.empty:
		return df

	df.columns = [col.strip() for col in df.columns]

	for col in df.columns:
		try:
			df[col] = pd.to_numeric(df[col], errors="coerce")
		except Exception:
			pass

	return df


def parse_training_log(log_path: str) -> Dict[str, Any]:
	"""解析 training.log 提取配置信息和最终指标。

	Args:
		log_path: training.log 文件路径

	Returns:
		解析出的配置和指标字典
	"""
	result: Dict[str, Any] = {
		"config": {},
		"final_reward": None,
		"total_episodes": None,
		"total_steps": None,
		"training_time": None,
	}

	if not os.path.isfile(log_path):
		return result

	try:
		with open(log_path, "r", encoding="utf-8", errors="replace") as f:
			content = f.read()
	except Exception:
		return result

	for key in _CONFIG_KEYS:
		pattern = rf"(?:^|\s){key}\s*[:=]\s*(.+?)(?:\s*$|\s*,)"
		match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
		if match:
			value = match.group(1).strip().strip("'\"")
			try:
				value = int(value)
			except ValueError:
				try:
					value = float(value)
				except ValueError:
					pass
			result["config"][key] = value

	reward_patterns = [
		r"average[_ ]episode[_ ]rewards?\s*[:=]\s*([-\d.eE+]+)",
		r"episode[_ ]reward\s*[:=]\s*([-\d.eE+]+)",
		r"reward\s*[:=]\s*([-\d.eE+]+)",
	]
	last_reward = None
	for pat in reward_patterns:
		matches = re.findall(pat, content, re.IGNORECASE)
		if matches:
			try:
				last_reward = float(matches[-1])
			except ValueError:
				pass
			break
	result["final_reward"] = last_reward

	episode_match = re.findall(
		r"episode\s*[:=]\s*(\d+)", content, re.IGNORECASE
	)
	if episode_match:
		try:
			result["total_episodes"] = int(episode_match[-1])
		except ValueError:
			pass

	steps_match = re.findall(
		r"total[_ ](?:num[_ ])?steps?\s*[:=]\s*(\d+)", content, re.IGNORECASE
	)
	if steps_match:
		try:
			result["total_steps"] = int(steps_match[-1])
		except ValueError:
			pass

	time_match = re.search(
		r"(?:training|total)[_ ]time\s*[:=]\s*(.+?)$",
		content,
		re.MULTILINE | re.IGNORECASE,
	)
	if time_match:
		result["training_time"] = time_match.group(1).strip()

	return result


def get_reward_curves(results_path: str) -> pd.DataFrame:
	"""从训练结果目录获取 episode-reward 曲线数据。

	Args:
		results_path: 训练运行目录路径

	Returns:
		包含奖励列的 DataFrame
	"""
	progress_path = os.path.join(results_path, "progress.txt")
	df = parse_progress_file(progress_path)

	if df.empty:
		return df

	index_candidates = ["episode", "total_num_steps", "timestep"]
	reward_candidates = _PROGRESS_REWARD_COLS

	keep_cols: List[str] = []

	for col_name in index_candidates:
		matching = [c for c in df.columns if col_name in c.lower()]
		keep_cols.extend(matching)

	for col_name in reward_candidates:
		matching = [c for c in df.columns if col_name in c.lower()]
		keep_cols.extend(matching)

	seen = set()
	unique_cols: List[str] = []
	for c in keep_cols:
		if c not in seen and c in df.columns:
			seen.add(c)
			unique_cols.append(c)

	if not unique_cols:
		return df

	return df[unique_cols].copy()


def get_episode_metrics(results_path: str) -> Dict[str, List[float]]:
	"""从 progress.txt 提取每个 episode 的详细指标。

	Args:
		results_path: 训练运行目录路径

	Returns:
		{指标名: [值列表]}
	"""
	progress_path = os.path.join(results_path, "progress.txt")
	df = parse_progress_file(progress_path)

	if df.empty:
		return {}

	metrics: Dict[str, List[float]] = {}

	for col in df.columns:
		if df[col].dtype in ("float64", "float32", "int64", "int32"):
			values = df[col].dropna().tolist()
			if values:
				metrics[col] = values

	return metrics
