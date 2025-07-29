#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAPPO训练启动脚本 - PowerZoo CRBP环境
约束感知多智能体强化学习训练脚本

@Author: Xiaodong Zheng
@Description: 使用HAPPO算法训练PowerZoo CRBP环境中的光伏系统控制
"""

import os
import sys
import argparse
import datetime
import subprocess
import json
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

def setup_experiment_directory(exp_name: str) -> Path:
	"""设置实验目录结构"""
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	exp_dir = project_root / "results" / "happo_powerzoo_pv" / f"{exp_name}_{timestamp}"
	
	# 创建必要的目录
	exp_dir.mkdir(parents=True, exist_ok=True)
	(exp_dir / "models").mkdir(exist_ok=True)
	(exp_dir / "logs").mkdir(exist_ok=True)
	(exp_dir / "configs").mkdir(exist_ok=True)
	(exp_dir / "plots").mkdir(exist_ok=True)
	
	return exp_dir

def check_environment():
	"""检查训练环境是否正确配置"""
	print("🔍 检查训练环境...")
	
	# 检查必要文件
	required_files = [
		project_root / "configs" / "algos_cfgs" / "happo_powerzoo_pv.yaml",
		project_root / "envs" / "power_envs" / "powerzoo_llm" / "env.py",
		project_root / "examples" / "train.py"
	]
	
	for file_path in required_files:
		if not file_path.exists():
			raise FileNotFoundError(f"关键文件不存在: {file_path}")
		print(f"✅ {file_path.name}")
	
	# 检查DSS文件
	dss_file = project_root / "envs" / "power_envs" / "powerzoo_llm" / "node_systems_with_pv" / "34Bus" / "ieee34Mod1_duty.dss"
	if not dss_file.exists():
		print(f"⚠️  DSS文件不存在: {dss_file}")
	else:
		print(f"✅ {dss_file.name}")
	
	print("✅ 环境检查完成\n")

def validate_config(config_path: Path):
	"""验证配置文件有效性"""
	print("🔧 验证配置文件...")
	
	try:
		import yaml
		with open(config_path, 'r', encoding='utf-8') as f:
			config = yaml.safe_load(f)
		
		# 检查关键配置项
		required_keys = ['algo_name', 'env_name', 'experiment']
		for key in required_keys:
			if key not in config:
				raise ValueError(f"配置文件缺少必要键: {key}")
		
		# 验证环境配置
		env_args = config.get('env_args', {})
		if env_args.get('env_name') != '34Bus_pv':
			print("⚠️  建议使用 34Bus_pv 环境进行PV控制训练")
		
		# 验证训练步数
		num_steps = config['experiment'].get('num_env_steps', 0)
		if num_steps < 1000000:
			print(f"⚠️  训练步数较少 ({num_steps})，可能影响收敛")
		
		print(f"✅ 配置验证完成，总训练步数: {num_steps:,}")
		
	except Exception as e:
		raise ValueError(f"配置文件验证失败: {e}")

def create_training_command(config_path: Path, exp_name: str, additional_args: dict = None) -> list:
	"""创建训练命令"""
	cmd = [
		sys.executable,
		str(project_root / "examples" / "train.py"),
		"--algo", "happo",
		"--env", "powerzoo_llm", 
		"--exp_name", exp_name,
		"--load_config", str(config_path)
	]
	
	# 添加额外参数
	if additional_args:
		for key, value in additional_args.items():
			cmd.extend([f"--{key}", str(value)])
	
	return cmd

def monitor_training_progress(log_file: Path):
	"""监控训练进度（简化版）"""
	print(f"📊 训练日志位置: {log_file}")
	print("💡 可使用以下命令监控训练进度:")
	print(f"   tail -f {log_file}")
	print("   或使用TensorBoard查看详细指标")

def main():
	parser = argparse.ArgumentParser(description="HAPPO PowerZoo PV训练脚本")
	parser.add_argument("--exp_name", type=str, default="happo_34bus_pv_default", 
	                   help="实验名称")
	parser.add_argument("--config", type=str, 
	                   default="configs/algos_cfgs/happo_powerzoo_pv.yaml",
	                   help="配置文件路径")
	parser.add_argument("--resume", type=str, default="", 
	                   help="恢复训练的检查点路径")
	parser.add_argument("--dry_run", action="store_true", 
	                   help="只显示训练命令，不实际执行")
	parser.add_argument("--gpu", type=int, default=0, 
	                   help="使用的GPU ID")
	parser.add_argument("--threads", type=int, default=32, 
	                   help="并行线程数")
	
	args = parser.parse_args()
	
	print("🚀 HAPPO PowerZoo PV 训练启动器")
	print("=" * 50)
	
	# 检查环境
	try:
		check_environment()
	except Exception as e:
		print(f"❌ 环境检查失败: {e}")
		return 1
	
	# 设置配置文件路径
	config_path = project_root / args.config
	if not config_path.exists():
		print(f"❌ 配置文件不存在: {config_path}")
		return 1
	
	# 验证配置
	try:
		validate_config(config_path)
	except Exception as e:
		print(f"❌ 配置验证失败: {e}")
		return 1
	
	# 设置实验目录
	exp_dir = setup_experiment_directory(args.exp_name)
	print(f"📁 实验目录: {exp_dir}")
	
	# 准备训练参数
	additional_args = {
		"n_rollout_threads": args.threads,
		"cuda": True,
		"cuda_deterministic": True
	}
	
	if args.resume:
		additional_args["use_resume"] = True
		additional_args["resume_path"] = args.resume
	
	# 创建训练命令
	train_cmd = create_training_command(config_path, args.exp_name, additional_args)
	
	print("\n🎯 训练配置:")
	print(f"   算法: HAPPO")
	print(f"   环境: PowerZoo-LLM (34Bus_pv)")
	print(f"   实验名称: {args.exp_name}")
	print(f"   配置文件: {config_path}")
	print(f"   并行线程: {args.threads}")
	print(f"   GPU设备: {args.gpu}")
	
	if args.resume:
		print(f"   恢复训练: {args.resume}")
	
	print(f"\n📝 训练命令:")
	print(" ".join(train_cmd))
	
	if args.dry_run:
		print("\n🔍 试运行模式，不执行实际训练")
		return 0
	
	# 设置环境变量
	env = os.environ.copy()
	env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
	env["PYTHONPATH"] = str(project_root)
	
	# 准备日志文件
	log_file = exp_dir / "logs" / "training.log"
	
	print(f"\n🏃 开始训练...")
	print(f"📊 训练日志: {log_file}")
	print("💡 按 Ctrl+C 停止训练\n")
	
	# 执行训练
	try:
		with open(log_file, 'w', encoding='utf-8') as f:
			# 记录训练开始信息
			start_time = datetime.datetime.now()
			f.write(f"训练开始时间: {start_time}\n")
			f.write(f"训练命令: {' '.join(train_cmd)}\n")
			f.write("=" * 80 + "\n")
			f.flush()
			
			# 启动训练进程
			process = subprocess.Popen(
				train_cmd,
				env=env,
				cwd=str(project_root),
				stdout=subprocess.PIPE,
				stderr=subprocess.STDOUT,
				universal_newlines=True,
				bufsize=1
			)
			
			# 实时输出和记录日志
			for line in process.stdout:
				print(line.rstrip())
				f.write(line)
				f.flush()
			
			# 等待进程结束
			return_code = process.wait()
			
			# 记录训练结束信息
			end_time = datetime.datetime.now()
			duration = end_time - start_time
			f.write("=" * 80 + "\n")
			f.write(f"训练结束时间: {end_time}\n")
			f.write(f"训练持续时间: {duration}\n")
			f.write(f"返回码: {return_code}\n")
			
			if return_code == 0:
				print(f"\n✅ 训练完成!")
				print(f"⏱️  训练耗时: {duration}")
				print(f"📁 结果保存在: {exp_dir}")
			else:
				print(f"\n❌ 训练异常结束，返回码: {return_code}")
				return return_code
				
	except KeyboardInterrupt:
		print(f"\n⏹️  用户中断训练")
		if 'process' in locals():
			process.terminate()
		return 1
	except Exception as e:
		print(f"\n❌ 训练执行失败: {e}")
		return 1
	
	return 0

if __name__ == "__main__":
	sys.exit(main())