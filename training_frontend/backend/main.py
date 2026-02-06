"""
PowerZoo Training Frontend - FastAPI Backend

提供训练任务配置和管理的REST API服务。
"""
import asyncio
import os
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = FastAPI(
	title="PowerZoo Training API",
	description="训练任务配置和管理API",
	version="1.0.0"
)

# CORS配置
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)


# ============ 数据模型 ============

class AlgorithmInfo(BaseModel):
	"""算法信息"""
	name: str
	display_name: str
	type: str  # on_policy / off_policy / two_timescale
	category: str  # ha / ma / qmix / single
	runner: str
	config_file: str
	description: str


class EnvironmentInfo(BaseModel):
	"""环境信息"""
	name: str
	display_name: str
	type: str
	config_file: str
	description: str
	default_agents: int


class TrainingConfig(BaseModel):
	"""训练配置"""
	algo: str
	env: str
	exp_name: str

	# 种子配置
	seed_specify: bool = True
	seed: int = 12345

	# 设备配置
	cuda: bool = True
	cuda_deterministic: bool = True
	torch_threads: int = 4

	# 训练配置
	n_rollout_threads: int = 4
	num_env_steps: int = 2000000
	episode_length: int = 360
	log_interval: int = 1
	eval_interval: int = 2
	save_interval: int = 5
	use_valuenorm: bool = True
	use_linear_lr_decay: bool = False

	# 评估配置
	use_eval: bool = True
	n_eval_rollout_threads: int = 2
	eval_episodes: int = 10

	# 模型配置
	hidden_sizes: list[int] = Field(default_factory=lambda: [128, 128])
	activation_func: str = "relu"
	use_feature_normalization: bool = True
	use_recurrent_policy: bool = True
	recurrent_n: int = 1
	data_chunk_length: int = 60

	# 算法配置（通用）
	lr: float = 5e-4
	critic_lr: float = 5e-4
	gamma: float = 0.99
	gae_lambda: float = 0.95
	entropy_coef: float = 0.08
	max_grad_norm: float = 10.0

	# PPO/HAPPO特定
	ppo_epoch: int = 5
	clip_param: float = 0.25
	actor_num_mini_batch: int = 1
	critic_num_mini_batch: int = 1

	# Off-policy特定
	buffer_size: int = 100000
	batch_size: int = 256
	polyak: float = 0.005

	# 环境特定配置
	env_args: dict[str, Any] = Field(default_factory=dict)


class TrainingTask(BaseModel):
	"""训练任务"""
	task_id: str
	config: TrainingConfig
	status: str  # pending / running / completed / failed / stopped
	created_at: str
	started_at: str | None = None
	ended_at: str | None = None
	pid: int | None = None
	log_file: str | None = None


class PresetConfig(BaseModel):
	"""预设配置"""
	name: str
	description: str
	config: TrainingConfig


# ============ 全局状态 ============

# 存储运行中的任务
running_tasks: dict[str, TrainingTask] = {}
task_processes: dict[str, subprocess.Popen] = {}


# ============ 算法和环境定义 ============

ALGORITHMS: list[AlgorithmInfo] = [
	# HA系列 On-Policy
	AlgorithmInfo(
		name="happo", display_name="HAPPO", type="on_policy", category="ha",
		runner="OnPolicyHARunner", config_file="happo.yaml",
		description="Heterogeneous-Agent Proximal Policy Optimization"
	),
	AlgorithmInfo(
		name="hatrpo", display_name="HATRPO", type="on_policy", category="ha",
		runner="OnPolicyHARunner", config_file="hatrpo.yaml",
		description="Heterogeneous-Agent Trust Region Policy Optimization"
	),
	AlgorithmInfo(
		name="haa2c", display_name="HAA2C", type="on_policy", category="ha",
		runner="OnPolicyHARunner", config_file="haa2c.yaml",
		description="Heterogeneous-Agent Advantage Actor-Critic"
	),
	AlgorithmInfo(
		name="shom", display_name="SHOM", type="on_policy", category="ha",
		runner="OnPolicyHARunner", config_file="shom.yaml",
		description="Shared Hierarchical On-policy MARL"
	),
	AlgorithmInfo(
		name="sn_mappo", display_name="SN-MAPPO", type="on_policy", category="ha",
		runner="OnPolicyHARunner", config_file="sn_mappo.yaml",
		description="Stackelberg Network MAPPO"
	),
	AlgorithmInfo(
		name="dan_happo", display_name="DAN-HAPPO", type="on_policy", category="ha",
		runner="OnPolicyHARunner", config_file="dan_happo.yaml",
		description="Dynamic Attention Network HAPPO"
	),
	# MA系列 On-Policy
	AlgorithmInfo(
		name="mappo", display_name="MAPPO", type="on_policy", category="ma",
		runner="OnPolicyMARunner", config_file="mappo.yaml",
		description="Multi-Agent Proximal Policy Optimization"
	),
	# HA系列 Off-Policy
	AlgorithmInfo(
		name="haddpg", display_name="HADDPG", type="off_policy", category="ha",
		runner="OffPolicyHARunner", config_file="haddpg.yaml",
		description="Heterogeneous-Agent Deep Deterministic Policy Gradient"
	),
	AlgorithmInfo(
		name="hatd3", display_name="HATD3", type="off_policy", category="ha",
		runner="OffPolicyHARunner", config_file="hatd3.yaml",
		description="Heterogeneous-Agent Twin Delayed DDPG"
	),
	AlgorithmInfo(
		name="hasac", display_name="HASAC", type="off_policy", category="ha",
		runner="OffPolicyHARunner", config_file="hasac.yaml",
		description="Heterogeneous-Agent Soft Actor-Critic"
	),
	AlgorithmInfo(
		name="had3qn", display_name="HAD3QN", type="off_policy", category="ha",
		runner="OffPolicyHARunner", config_file="had3qn.yaml",
		description="Heterogeneous-Agent Dueling Double DQN"
	),
	# MA系列 Off-Policy
	AlgorithmInfo(
		name="maddpg", display_name="MADDPG", type="off_policy", category="ma",
		runner="OffPolicyMARunner", config_file="maddpg.yaml",
		description="Multi-Agent Deep Deterministic Policy Gradient"
	),
	AlgorithmInfo(
		name="matd3", display_name="MATD3", type="off_policy", category="ma",
		runner="OffPolicyMARunner", config_file="matd3.yaml",
		description="Multi-Agent Twin Delayed DDPG"
	),
	# QMIX
	AlgorithmInfo(
		name="qmix", display_name="QMIX", type="off_policy", category="qmix",
		runner="QMIXRunner", config_file="qmix.yaml",
		description="QMIX Value Decomposition"
	),
	# 两时间尺度
	AlgorithmInfo(
		name="2ts_vvc", display_name="2TS-VVC", type="two_timescale", category="special",
		runner="TwoTSRunner", config_file="2ts_vvc.yaml",
		description="Two-Timescale Volt-VAR Control"
	),
]

SINGLE_AGENT_ALGORITHMS: list[AlgorithmInfo] = [
	AlgorithmInfo(
		name="ppo", display_name="PPO", type="on_policy", category="single",
		runner="SingleAgentRunner", config_file="ppo.yaml",
		description="Proximal Policy Optimization"
	),
	AlgorithmInfo(
		name="a2c", display_name="A2C", type="on_policy", category="single",
		runner="SingleAgentRunner", config_file="a2c.yaml",
		description="Advantage Actor-Critic"
	),
	AlgorithmInfo(
		name="ddpg", display_name="DDPG", type="off_policy", category="single",
		runner="SingleAgentRunner", config_file="ddpg.yaml",
		description="Deep Deterministic Policy Gradient"
	),
	AlgorithmInfo(
		name="td3", display_name="TD3", type="off_policy", category="single",
		runner="SingleAgentRunner", config_file="td3.yaml",
		description="Twin Delayed DDPG"
	),
	AlgorithmInfo(
		name="sac", display_name="SAC", type="off_policy", category="single",
		runner="SingleAgentRunner", config_file="sac.yaml",
		description="Soft Actor-Critic"
	),
	AlgorithmInfo(
		name="dqn", display_name="DQN", type="off_policy", category="single",
		runner="SingleAgentRunner", config_file="dqn.yaml",
		description="Deep Q-Network"
	),
]

ENVIRONMENTS: list[EnvironmentInfo] = [
	EnvironmentInfo(
		name="vvc", display_name="VVC (Volt-VAR Control)", type="vvc",
		config_file="vvc.yaml",
		description="电压无功控制环境，基于OpenDSS仿真",
		default_agents=6
	),
	EnvironmentInfo(
		name="smartgrid", display_name="SmartGrid", type="microgrid",
		config_file="smartgrid.yaml",
		description="智能微网环境，模块化电网仿真",
		default_agents=12
	),
	EnvironmentInfo(
		name="stackelberg_13bus", display_name="Stackelberg 13-Bus", type="game",
		config_file="stackelberg_13bus.yaml",
		description="13节点Stackelberg博弈环境",
		default_agents=9
	),
	EnvironmentInfo(
		name="stackelberg_34bus", display_name="Stackelberg 34-Bus", type="game",
		config_file="stackelberg_34bus.yaml",
		description="34节点Stackelberg博弈环境",
		default_agents=11
	),
	EnvironmentInfo(
		name="stackelberg_123bus", display_name="Stackelberg 123-Bus", type="game",
		config_file="stackelberg_123bus.yaml",
		description="123节点大规模Stackelberg博弈环境",
		default_agents=123
	),
	EnvironmentInfo(
		name="dsr", display_name="DSR", type="restoration",
		config_file="dsr.yaml",
		description="配电网服务恢复环境",
		default_agents=10
	),
	EnvironmentInfo(
		name="dsr_13bus", display_name="DSR 13-Bus", type="restoration",
		config_file="dsr_13bus.yaml",
		description="13节点配电网恢复环境",
		default_agents=8
	),
	EnvironmentInfo(
		name="dsr_8500node", display_name="DSR 8500-Node", type="restoration",
		config_file="dsr_8500node.yaml",
		description="8500节点大规模配电网恢复环境",
		default_agents=50
	),
	EnvironmentInfo(
		name="gym", display_name="OpenAI Gym", type="standard",
		config_file="gym.yaml",
		description="标准OpenAI Gym环境",
		default_agents=1
	),
]

PRESETS: list[PresetConfig] = [
	PresetConfig(
		name="quick_test",
		description="快速测试 - 用于验证配置和代码",
		config=TrainingConfig(
			algo="happo",
			env="smartgrid",
			exp_name="quick_test",
			n_rollout_threads=2,
			num_env_steps=100000,
			episode_length=24,
			hidden_sizes=[64, 64],
			use_recurrent_policy=False,
		)
	),
	PresetConfig(
		name="standard_training",
		description="标准训练 - 电力系统推荐配置",
		config=TrainingConfig(
			algo="happo",
			env="smartgrid",
			exp_name="standard_training",
			n_rollout_threads=4,
			num_env_steps=2000000,
			episode_length=360,
			hidden_sizes=[128, 128],
			use_recurrent_policy=True,
			entropy_coef=0.08,
		)
	),
	PresetConfig(
		name="high_performance",
		description="高性能训练 - 更大网络和更多环境",
		config=TrainingConfig(
			algo="mappo",
			env="smartgrid",
			exp_name="high_performance",
			n_rollout_threads=8,
			num_env_steps=5000000,
			episode_length=360,
			hidden_sizes=[256, 256],
			use_recurrent_policy=True,
		)
	),
	PresetConfig(
		name="stackelberg_game",
		description="Stackelberg博弈 - 领导者-跟随者博弈",
		config=TrainingConfig(
			algo="sn_mappo",
			env="stackelberg_34bus",
			exp_name="stackelberg_game",
			n_rollout_threads=4,
			num_env_steps=1000000,
			episode_length=24,
		)
	),
	PresetConfig(
		name="dsr_restoration",
		description="配电网恢复 - 故障恢复调度",
		config=TrainingConfig(
			algo="happo",
			env="dsr",
			exp_name="dsr_restoration",
			n_rollout_threads=3,
			num_env_steps=10000000,
			episode_length=96,
		)
	),
	PresetConfig(
		name="off_policy_training",
		description="Off-Policy训练 - 实时控制场景",
		config=TrainingConfig(
			algo="hasac",
			env="vvc",
			exp_name="off_policy_training",
			n_rollout_threads=4,
			num_env_steps=1000000,
			episode_length=360,
			buffer_size=100000,
			batch_size=256,
		)
	),
]


# ============ API 路由 ============

@app.get("/")
async def root():
	"""API根路径"""
	return {
		"name": "PowerZoo Training API",
		"version": "1.0.0",
		"status": "running"
	}


@app.get("/api/algorithms")
async def get_algorithms():
	"""获取所有可用算法"""
	return {
		"multi_agent": [algo.model_dump() for algo in ALGORITHMS],
		"single_agent": [algo.model_dump() for algo in SINGLE_AGENT_ALGORITHMS]
	}


@app.get("/api/algorithms/{algo_name}")
async def get_algorithm(algo_name: str):
	"""获取特定算法信息"""
	for algo in ALGORITHMS + SINGLE_AGENT_ALGORITHMS:
		if algo.name == algo_name:
			return algo.model_dump()
	raise HTTPException(status_code=404, detail=f"算法 {algo_name} 不存在")


@app.get("/api/environments")
async def get_environments():
	"""获取所有可用环境"""
	return [env.model_dump() for env in ENVIRONMENTS]


@app.get("/api/environments/{env_name}")
async def get_environment(env_name: str):
	"""获取特定环境信息"""
	for env in ENVIRONMENTS:
		if env.name == env_name:
			return env.model_dump()
	raise HTTPException(status_code=404, detail=f"环境 {env_name} 不存在")


@app.get("/api/presets")
async def get_presets():
	"""获取所有预设配置"""
	return [preset.model_dump() for preset in PRESETS]


@app.get("/api/presets/{preset_name}")
async def get_preset(preset_name: str):
	"""获取特定预设配置"""
	for preset in PRESETS:
		if preset.name == preset_name:
			return preset.model_dump()
	raise HTTPException(status_code=404, detail=f"预设 {preset_name} 不存在")


@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
	"""启动训练任务"""
	task_id = str(uuid.uuid4())[:8]

	# 创建任务记录
	task = TrainingTask(
		task_id=task_id,
		config=config,
		status="pending",
		created_at=datetime.now().isoformat(),
	)

	# 生成配置文件
	config_dir = PROJECT_ROOT / "training_frontend" / "configs"
	config_dir.mkdir(exist_ok=True)
	config_file = config_dir / f"task_{task_id}.yaml"

	# 构建完整配置
	full_config = _build_full_config(config)

	import yaml
	with open(config_file, "w") as f:
		yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True)

	# 创建日志目录
	log_dir = PROJECT_ROOT / "training_frontend" / "logs"
	log_dir.mkdir(exist_ok=True)
	log_file = log_dir / f"task_{task_id}.log"
	task.log_file = str(log_file)

	# 启动训练进程
	train_script = PROJECT_ROOT / "examples" / "multi_agent" / "scripts" / "train.py"

	cmd = [
		sys.executable,
		str(train_script),
		"--algo", config.algo,
		"--env", config.env,
		"--exp_name", config.exp_name,
		"--load_config", str(config_file),
	]

	try:
		with open(log_file, "w") as log_f:
			process = subprocess.Popen(
				cmd,
				stdout=log_f,
				stderr=subprocess.STDOUT,
				cwd=str(PROJECT_ROOT),
				env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)}
			)

		task.status = "running"
		task.started_at = datetime.now().isoformat()
		task.pid = process.pid

		running_tasks[task_id] = task
		task_processes[task_id] = process

		return {"task_id": task_id, "status": "started", "pid": process.pid}

	except Exception as e:
		task.status = "failed"
		task.ended_at = datetime.now().isoformat()
		raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/tasks")
async def get_tasks():
	"""获取所有训练任务"""
	# 更新任务状态
	for task_id, process in list(task_processes.items()):
		if process.poll() is not None:
			task = running_tasks.get(task_id)
			if task:
				task.status = "completed" if process.returncode == 0 else "failed"
				task.ended_at = datetime.now().isoformat()

	return [task.model_dump() for task in running_tasks.values()]


@app.get("/api/training/tasks/{task_id}")
async def get_task(task_id: str):
	"""获取特定任务状态"""
	if task_id not in running_tasks:
		raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

	# 更新状态
	if task_id in task_processes:
		process = task_processes[task_id]
		if process.poll() is not None:
			task = running_tasks[task_id]
			task.status = "completed" if process.returncode == 0 else "failed"
			task.ended_at = datetime.now().isoformat()

	return running_tasks[task_id].model_dump()


@app.get("/api/training/tasks/{task_id}/logs")
async def get_task_logs(task_id: str, lines: int = 100):
	"""获取任务日志"""
	if task_id not in running_tasks:
		raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

	task = running_tasks[task_id]
	if not task.log_file or not Path(task.log_file).exists():
		return {"logs": ""}

	try:
		with open(task.log_file, "r") as f:
			all_lines = f.readlines()
			return {"logs": "".join(all_lines[-lines:])}
	except Exception as e:
		return {"logs": f"Error reading logs: {e}"}


@app.post("/api/training/tasks/{task_id}/stop")
async def stop_task(task_id: str):
	"""停止训练任务"""
	if task_id not in running_tasks:
		raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

	if task_id not in task_processes:
		raise HTTPException(status_code=400, detail="任务未在运行")

	process = task_processes[task_id]
	if process.poll() is None:
		process.terminate()
		await asyncio.sleep(1)
		if process.poll() is None:
			process.kill()

	task = running_tasks[task_id]
	task.status = "stopped"
	task.ended_at = datetime.now().isoformat()

	return {"task_id": task_id, "status": "stopped"}


@app.delete("/api/training/tasks/{task_id}")
async def delete_task(task_id: str):
	"""删除任务记录"""
	if task_id not in running_tasks:
		raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")

	# 如果任务还在运行，先停止
	if task_id in task_processes:
		process = task_processes[task_id]
		if process.poll() is None:
			process.terminate()
			await asyncio.sleep(1)
			if process.poll() is None:
				process.kill()
		del task_processes[task_id]

	# 删除配置文件和日志
	task = running_tasks[task_id]
	config_file = PROJECT_ROOT / "training_frontend" / "configs" / f"task_{task_id}.yaml"
	if config_file.exists():
		config_file.unlink()
	if task.log_file and Path(task.log_file).exists():
		Path(task.log_file).unlink()

	del running_tasks[task_id]

	return {"task_id": task_id, "status": "deleted"}


@app.post("/api/config/validate")
async def validate_config(config: TrainingConfig):
	"""验证配置"""
	errors = []
	warnings = []

	# 检查算法是否存在
	algo_names = [a.name for a in ALGORITHMS + SINGLE_AGENT_ALGORITHMS]
	if config.algo not in algo_names:
		errors.append(f"算法 '{config.algo}' 不存在")

	# 检查环境是否存在
	env_names = [e.name for e in ENVIRONMENTS]
	if config.env not in env_names:
		errors.append(f"环境 '{config.env}' 不存在")

	# 参数范围检查
	if config.n_rollout_threads < 1 or config.n_rollout_threads > 32:
		errors.append("n_rollout_threads 应在 1-32 之间")

	if config.num_env_steps < 10000:
		warnings.append("num_env_steps 过小，可能无法充分训练")

	if config.episode_length < 1:
		errors.append("episode_length 必须大于 0")

	if config.use_recurrent_policy and config.episode_length % config.data_chunk_length != 0:
		warnings.append("episode_length 应为 data_chunk_length 的整数倍")

	if config.lr < 1e-7 or config.lr > 1e-1:
		warnings.append("lr 建议在 1e-6 到 1e-2 之间")

	if config.entropy_coef < 0 or config.entropy_coef > 1:
		errors.append("entropy_coef 应在 0-1 之间")

	return {
		"valid": len(errors) == 0,
		"errors": errors,
		"warnings": warnings
	}


@app.get("/api/config/default/{algo}/{env}")
async def get_default_config(algo: str, env: str):
	"""获取算法和环境的默认配置"""
	# 基础默认配置
	config = TrainingConfig(
		algo=algo,
		env=env,
		exp_name=f"{algo}_{env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
	)

	# 根据环境调整默认值
	env_info = next((e for e in ENVIRONMENTS if e.name == env), None)
	if env_info:
		if env_info.type == "game":
			config.episode_length = 24
		elif env_info.type == "restoration":
			config.episode_length = 96
		else:
			config.episode_length = 360

	# 根据算法类型调整默认值
	algo_info = next((a for a in ALGORITHMS + SINGLE_AGENT_ALGORITHMS if a.name == algo), None)
	if algo_info:
		if algo_info.type == "off_policy":
			config.buffer_size = 100000
			config.batch_size = 256

	return config.model_dump()


def _build_full_config(config: TrainingConfig) -> dict:
	"""构建完整的配置字典"""
	return {
		"algo_name": config.algo,
		"env_name": config.env,
		"exp_name": config.exp_name,

		"seed": {
			"seed_specify": config.seed_specify,
			"seed": config.seed,
		},

		"device": {
			"cuda": config.cuda,
			"cuda_deterministic": config.cuda_deterministic,
			"torch_threads": config.torch_threads,
		},

		"train": {
			"n_rollout_threads": config.n_rollout_threads,
			"num_env_steps": config.num_env_steps,
			"episode_length": config.episode_length,
			"log_interval": config.log_interval,
			"eval_interval": config.eval_interval,
			"save_interval": config.save_interval,
			"use_valuenorm": config.use_valuenorm,
			"use_linear_lr_decay": config.use_linear_lr_decay,
			"model_dir": None,
		},

		"eval": {
			"use_eval": config.use_eval,
			"n_eval_rollout_threads": config.n_eval_rollout_threads,
			"eval_episodes": config.eval_episodes,
		},

		"model": {
			"hidden_sizes": config.hidden_sizes,
			"activation_func": config.activation_func,
			"use_feature_normalization": config.use_feature_normalization,
			"use_recurrent_policy": config.use_recurrent_policy,
			"recurrent_n": config.recurrent_n,
			"data_chunk_length": config.data_chunk_length,
		},

		"algo": {
			"lr": config.lr,
			"critic_lr": config.critic_lr,
			"gamma": config.gamma,
			"gae_lambda": config.gae_lambda,
			"entropy_coef": config.entropy_coef,
			"max_grad_norm": config.max_grad_norm,
			"ppo_epoch": config.ppo_epoch,
			"clip_param": config.clip_param,
			"actor_num_mini_batch": config.actor_num_mini_batch,
			"critic_num_mini_batch": config.critic_num_mini_batch,
			"buffer_size": config.buffer_size,
			"batch_size": config.batch_size,
			"polyak": config.polyak,
		},

		"env_args": config.env_args,
	}


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8000)
