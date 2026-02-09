# -*- coding: utf-8 -*-
"""
@File      : training_log_parser.py
@Description: 训练结果解析工具。
			  扫描训练输出目录，解析 progress.txt、training.log 等文件，
			  为 Training Progress 标签页提供数据支撑。
"""

import glob
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

# progress.txt 常见列名
_PROGRESS_REWARD_COLS = [
	"average_episode_rewards",
	"average_step_rewards",
	"eval_average_episode_rewards",
]

# training.log 中需要提取的配置键
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

# 目录名解析正则：<algo>_<env>_YYYYMMDD_HHMMSS
_RUN_DIR_PATTERN = re.compile(
	r"^(?P<algorithm>.+?)_(?P<env>.+?)_(?P<date>\d{8})_(?P<time>\d{6})$"
)


def scan_training_results(results_dir: str) -> List[Dict[str, Any]]:
	"""扫描训练结果目录，发现所有可用的训练运行记录。

	每个训练运行对应 results_dir 下的一个子目录，
	目录名格式为 <algorithm>_<env>_YYYYMMDD_HHMMSS。

	Args:
		results_dir: 训练结果根目录路径

	Returns:
		List[Dict[str, Any]]: 训练运行信息列表，每项包含:
			- name (str): 运行目录名
			- path (str): 运行目录绝对路径
			- algorithm (str): 算法名称
			- env (str): 环境名称
			- timestamp (str): 时间戳字符串 YYYYMMDD_HHMMSS
			- datetime (Optional[datetime]): 解析后的 datetime 对象
			- has_progress (bool): 是否存在 progress.txt
			- has_log (bool): 是否存在 training.log
			- has_tensorboard (bool): 是否存在 logs/ 目录
			- episodes (Optional[int]): 已训练 episode 数（从 progress.txt 推断）
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

		# 从目录名解析元信息
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

		# 检测可用文件
		progress_path = os.path.join(run_path, "progress.txt")
		log_path = os.path.join(run_path, "training.log")
		logs_dir = os.path.join(run_path, "logs")

		run_info["has_progress"] = os.path.isfile(progress_path)
		run_info["has_log"] = os.path.isfile(log_path)
		run_info["has_tensorboard"] = os.path.isdir(logs_dir)

		# 从 progress.txt 推断 episode 数
		if run_info["has_progress"]:
			try:
				df = parse_progress_file(progress_path)
				if not df.empty:
					run_info["episodes"] = len(df)
			except Exception:
				pass

		runs.append(run_info)

	# 按时间倒序排列，最新的排前面
	runs.sort(key=lambda r: r["timestamp"], reverse=True)
	return runs


def parse_progress_file(progress_path: str) -> pd.DataFrame:
	"""解析 progress.txt 为 DataFrame。

	progress.txt 是制表符分隔的文本文件，首行为列名。
	典型列：episode, total_num_steps, average_episode_rewards,
	average_step_rewards, value_loss, policy_loss 等。

	Args:
		progress_path: progress.txt 文件路径

	Returns:
		pd.DataFrame: 训练指标数据。文件不存在或格式错误时返回空 DataFrame。
	"""
	if not os.path.isfile(progress_path):
		return pd.DataFrame()

	try:
		df = pd.read_csv(progress_path, sep="\t")
	except Exception:
		# 尝试空格分隔
		try:
			df = pd.read_csv(progress_path, sep=r"\s+", engine="python")
		except Exception:
			return pd.DataFrame()

	if df.empty:
		return df

	# 清理列名中的前后空白
	df.columns = [col.strip() for col in df.columns]

	# 尝试将数值列转换为 float
	for col in df.columns:
		try:
			df[col] = pd.to_numeric(df[col], errors="coerce")
		except Exception:
			pass

	return df


def parse_training_log(log_path: str) -> Dict[str, Any]:
	"""解析 training.log 提取配置信息和最终指标。

	从日志文本中正则匹配关键配置项和训练终止时的最终指标。

	Args:
		log_path: training.log 文件路径

	Returns:
		Dict[str, Any]: 解析出的配置和指标，包含:
			- config: Dict 配置项
			- final_reward: Optional[float] 最终 episode 奖励
			- total_episodes: Optional[int] 总 episode 数
			- total_steps: Optional[int] 总步数
			- training_time: Optional[str] 训练耗时
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

	# 提取配置项 (匹配 "key: value" 或 "key = value" 模式)
	for key in _CONFIG_KEYS:
		pattern = rf"(?:^|\s){key}\s*[:=]\s*(.+?)(?:\s*$|\s*,)"
		match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
		if match:
			value = match.group(1).strip().strip("'\"")
			# 尝试转换为数值
			try:
				value = int(value)
			except ValueError:
				try:
					value = float(value)
				except ValueError:
					pass
			result["config"][key] = value

	# 提取最终奖励 (匹配日志中的奖励输出行)
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

	# 提取总 episode 数
	episode_match = re.findall(
		r"episode\s*[:=]\s*(\d+)", content, re.IGNORECASE
	)
	if episode_match:
		try:
			result["total_episodes"] = int(episode_match[-1])
		except ValueError:
			pass

	# 提取总步数
	steps_match = re.findall(
		r"total[_ ](?:num[_ ])?steps?\s*[:=]\s*(\d+)", content, re.IGNORECASE
	)
	if steps_match:
		try:
			result["total_steps"] = int(steps_match[-1])
		except ValueError:
			pass

	# 提取训练耗时
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

	便捷函数，自动定位 progress.txt 并提取奖励列。

	Args:
		results_path: 训练运行目录路径（包含 progress.txt 的目录）

	Returns:
		pd.DataFrame: 至少包含一个索引列和奖励列的 DataFrame。
			列名可能包含 episode/total_num_steps 作为 x 轴，
			average_episode_rewards 等作为 y 轴。
			文件不存在时返回空 DataFrame。
	"""
	progress_path = os.path.join(results_path, "progress.txt")
	df = parse_progress_file(progress_path)

	if df.empty:
		return df

	# 筛选奖励相关列 + 索引列
	index_candidates = ["episode", "total_num_steps", "timestep"]
	reward_candidates = _PROGRESS_REWARD_COLS

	keep_cols: List[str] = []

	# 选择索引列
	for col_name in index_candidates:
		matching = [c for c in df.columns if col_name in c.lower()]
		keep_cols.extend(matching)

	# 选择奖励列
	for col_name in reward_candidates:
		matching = [c for c in df.columns if col_name in c.lower()]
		keep_cols.extend(matching)

	# 去重并保持顺序
	seen = set()
	unique_cols: List[str] = []
	for c in keep_cols:
		if c not in seen and c in df.columns:
			seen.add(c)
			unique_cols.append(c)

	if not unique_cols:
		# 没有匹配到任何已知列名，返回全部数据
		return df

	return df[unique_cols].copy()


def get_episode_metrics(results_path: str) -> Dict[str, List[float]]:
	"""从 progress.txt 提取每个 episode 的详细指标。

	Args:
		results_path: 训练运行目录路径

	Returns:
		Dict[str, List[float]]: {指标名: [值列表]}，
			常见指标包括 reward, voltage_violation, policy_loss, value_loss 等。
			文件不存在或无数据时返回空字典。
	"""
	progress_path = os.path.join(results_path, "progress.txt")
	df = parse_progress_file(progress_path)

	if df.empty:
		return {}

	metrics: Dict[str, List[float]] = {}

	for col in df.columns:
		# 只保留数值列
		if df[col].dtype in ("float64", "float32", "int64", "int32"):
			values = df[col].dropna().tolist()
			if values:
				metrics[col] = values

	return metrics
