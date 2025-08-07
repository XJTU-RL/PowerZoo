#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PowerZoo专用训练脚本
专门针对电力系统强化学习环境优化的训练工具

特点：
- 支持三种PV渗透率方案（保守/优化/激进）
- 内置电力系统性能指标监控
- 自动化训练结果分析
- 简化的参数配置
"""

import argparse
import json
import yaml
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到系统路径
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from utils.configs_tools import get_defaults_yaml_args, update_args
from runners import RUNNER_REGISTRY


class PowerZooTrainer:
	"""PowerZoo专用训练器"""
	
	def __init__(self):
		self.project_root = project_root
		self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.setup_parser()
		
	def setup_parser(self):
		"""设置命令行参数解析器"""
		self.parser = argparse.ArgumentParser(
			description="PowerZoo电力系统强化学习训练工具",
			formatter_class=argparse.ArgumentDefaultsHelpFormatter
		)
		
		# ========== 基础参数 ==========
		self.parser.add_argument(
			"--system",
			type=str,
			default="powerzoo_llm",
			help="电力系统环境配置 (powerzoo_llm, powerzoo, 等)"
		)
		
		self.parser.add_argument(
			"--pv_plan",
			type=str,
			default="optimized",
			choices=["conservative", "optimized", "aggressive"],
			help="PV渗透率方案: conservative(720kW,40.7%), optimized(900kW,50.8%), aggressive(1080kW,61%)"
		)
		
		self.parser.add_argument(
			"--algo",
			type=str,
			default="happo",
			choices=["happo", "mappo", "hatrpo", "maddpg", "matd3", "shom"],
			help="强化学习算法选择"
		)
		
		self.parser.add_argument(
			"--exp_name",
			type=str,
			default="",
			help="实验名称（留空自动生成）"
		)
		
		# ========== 训练参数 ==========
		self.parser.add_argument(
			"--train_steps",
			type=int,
			default=1000000,
			help="总训练步数"
		)
		
		self.parser.add_argument(
			"--episode_length",
			type=int,
			default=360,
			help="每个episode的长度（时间步）"
		)
		
		self.parser.add_argument(
			"--batch_size",
			type=int,
			default=2048,
			help="批次大小"
		)
		
		self.parser.add_argument(
			"--lr",
			type=float,
			default=5e-4,
			help="学习率"
		)
		
		self.parser.add_argument(
			"--gamma",
			type=float,
			default=0.99,
			help="折扣因子"
		)
		
		# ========== 硬件参数 ==========
		self.parser.add_argument(
			"--gpu",
			type=int,
			default=0,
			help="GPU设备ID (-1表示使用CPU)"
		)
		
		self.parser.add_argument(
			"--n_threads",
			type=int,
			default=4,
			help="训练线程数"
		)
		
		self.parser.add_argument(
			"--n_rollout",
			type=int,
			default=8,
			help="并行环境数"
		)
		
		# ========== 电力系统特定参数 ==========
		self.parser.add_argument(
			"--voltage_penalty",
			type=float,
			default=None,
			help="电压越限惩罚权重（留空使用方案默认值）"
		)
		
		self.parser.add_argument(
			"--loss_penalty",
			type=float,
			default=None,
			help="功率损耗惩罚权重（留空使用方案默认值）"
		)
		
		self.parser.add_argument(
			"--pv_reward",
			type=float,
			default=None,
			help="PV利用率奖励权重（留空使用方案默认值）"
		)
		
		self.parser.add_argument(
			"--enable_curtailment",
			action="store_true",
			help="允许弃光（仅激进方案默认启用）"
		)
		
		# ========== 评估参数 ==========
		self.parser.add_argument(
			"--eval_interval",
			type=int,
			default=25,
			help="评估间隔（episodes）"
		)
		
		self.parser.add_argument(
			"--eval_episodes",
			type=int,
			default=5,
			help="每次评估的episode数"
		)
		
		# ========== 日志参数 ==========
		self.parser.add_argument(
			"--save_interval",
			type=int,
			default=5,
			help="模型保存间隔（episodes）"
		)
		
		self.parser.add_argument(
			"--log_interval",
			type=int,
			default=2,
			help="日志记录间隔（episodes）"
		)
		
		self.parser.add_argument(
			"--use_wandb",
			action="store_true",
			help="使用Weights & Biases记录"
		)
		
		self.parser.add_argument(
			"--wandb_project",
			type=str,
			default="PowerZoo_LLM",
			help="WandB项目名"
		)
		
		# ========== 其他参数 ==========
		self.parser.add_argument(
			"--seed",
			type=int,
			default=1,
			help="随机种子"
		)
		
		self.parser.add_argument(
			"--resume",
			type=str,
			default="",
			help="从检查点恢复训练"
		)
		
		self.parser.add_argument(
			"--dry_run",
			action="store_true",
			help="仅显示配置，不执行训练"
		)
		
		self.parser.add_argument(
			"--verbose",
			action="store_true",
			help="显示详细输出"
		)
		
	def get_pv_config(self, plan):
		"""获取PV方案配置"""
		configs = {
			"conservative": {
				"config_file": "configs/envs_cfgs/powerzoo_llm_pv_plans/powerzoo_llm_conservative.yaml",
				"description": "保守方案 (720kW, 40.7%渗透率)",
				"default_weights": {
					"voltage_violation": -1.5,
					"power_loss": -0.6,
					"pv_utilization": 0.2,
					"import_penalty": -0.15
				}
			},
			"optimized": {
				"config_file": "configs/envs_cfgs/powerzoo_llm_pv_plans/powerzoo_llm_optimized.yaml",
				"description": "优化方案 (900kW, 50.8%渗透率)",
				"default_weights": {
					"voltage_violation": -2.0,
					"power_loss": -0.5,
					"pv_utilization": 0.3,
					"import_penalty": -0.1
				}
			},
			"aggressive": {
				"config_file": "configs/envs_cfgs/powerzoo_llm_pv_plans/powerzoo_llm_aggressive.yaml",
				"description": "激进方案 (1080kW, 61%渗透率)",
				"default_weights": {
					"voltage_violation": -3.0,
					"power_loss": -0.4,
					"pv_utilization": 0.4,
					"import_penalty": -0.05,
					"curtailment": -0.2,
					"reverse_flow": -0.5
				}
			}
		}
		return configs[plan]
		
	def setup_directories(self, args):
		"""设置结果目录"""
		exp_name = args.exp_name or f"{args.algo}_pv_{args.pv_plan}_{self.timestamp}"
		result_dir = self.project_root / "results" / exp_name
		result_dir.mkdir(parents=True, exist_ok=True)
		
		# 创建子目录
		(result_dir / "models").mkdir(exist_ok=True)
		(result_dir / "logs").mkdir(exist_ok=True)
		(result_dir / "eval").mkdir(exist_ok=True)
		(result_dir / "plots").mkdir(exist_ok=True)
		
		return result_dir, exp_name
		
	def save_config(self, args, result_dir):
		"""保存训练配置"""
		config = {
			"timestamp": self.timestamp,
			"experiment": {
				"name": args.exp_name,
				"algorithm": args.algo,
				"system": args.system,
				"pv_plan": args.pv_plan,
				"plan_description": self.get_pv_config(args.pv_plan)["description"]
			},
			"training": {
				"total_steps": args.train_steps,
				"episode_length": args.episode_length,
				"batch_size": args.batch_size,
				"learning_rate": args.lr,
				"gamma": args.gamma,
				"seed": args.seed
			},
			"hardware": {
				"gpu": args.gpu,
				"n_threads": args.n_threads,
				"n_rollout": args.n_rollout
			},
			"evaluation": {
				"eval_interval": args.eval_interval,
				"eval_episodes": args.eval_episodes
			},
			"logging": {
				"save_interval": args.save_interval,
				"log_interval": args.log_interval,
				"use_wandb": args.use_wandb
			},
			"paths": {
				"result_dir": str(result_dir),
				"resume_from": args.resume
			}
		}
		
		config_file = result_dir / "training_config.yaml"
		with open(config_file, 'w', encoding='utf-8') as f:
			yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
			
		return config_file
		
	def print_banner(self, args, pv_config, result_dir):
		"""打印训练开始横幅"""
		print("\n" + "="*60)
		print("🚀 PowerZoo 强化学习训练系统")
		print("="*60)
		print(f"📋 配置信息:")
		print(f"   系统环境: {args.system}")
		print(f"   算法: {args.algo.upper()}")
		print(f"   PV方案: {pv_config['description']}")
		print(f"   训练步数: {args.train_steps:,}")
		print(f"   GPU设备: {'CPU' if args.gpu == -1 else f'GPU:{args.gpu}'}")
		print(f"   并行环境: {args.n_rollout}")
		print(f"   结果目录: {result_dir}")
		print("="*60 + "\n")
		
	def run(self):
		"""运行训练"""
		# 解析参数
		args, unparsed_args = self.parser.parse_known_args()
		
		# 获取PV配置
		pv_config = self.get_pv_config(args.pv_plan)
		
		# 设置目录
		result_dir, exp_name = self.setup_directories(args)
		args.exp_name = exp_name
		
		# 保存配置
		config_file = self.save_config(args, result_dir)
		
		# 打印横幅
		self.print_banner(args, pv_config, result_dir)
		
		if args.dry_run:
			print("🔍 模拟运行模式 - 仅显示配置")
			print(f"   配置已保存至: {config_file}")
			print("\n使用 --dry_run=false 执行实际训练")
			return
			
		# 设置环境变量
		if args.gpu >= 0:
			os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
			
		# 加载配置文件
		env_config_path = self.project_root / pv_config["config_file"]
		algo_config_path = self.project_root / "configs" / "algos_cfgs" / f"{args.algo}.yaml"
		
		# 加载算法配置
		with open(algo_config_path, 'r', encoding='utf-8') as f:
			algo_args = yaml.load(f, Loader=yaml.FullLoader)
			
		# 加载环境配置
		with open(env_config_path, 'r', encoding='utf-8') as f:
			env_args = yaml.load(f, Loader=yaml.FullLoader)
			
		# 更新奖励权重（如果用户指定）
		if args.voltage_penalty is not None:
			env_args['reward_weights']['voltage_violation'] = args.voltage_penalty
		if args.loss_penalty is not None:
			env_args['reward_weights']['power_loss'] = args.loss_penalty
		if args.pv_reward is not None:
			env_args['reward_weights']['pv_utilization'] = args.pv_reward
			
		# 处理弃光设置
		if args.enable_curtailment or args.pv_plan == "aggressive":
			if 'special_configs' not in env_args:
				env_args['special_configs'] = {}
			env_args['special_configs']['enable_curtailment'] = True
			
		# 更新训练参数
		main_args = {
			"algo": args.algo,
			"env": args.system,
			"exp_name": exp_name,
			"cuda": args.gpu >= 0,
			"cuda_deterministic": True,
			"seed": args.seed,
			"n_training_threads": args.n_threads,
			"n_rollout_threads": args.n_rollout,
			"num_env_steps": args.train_steps,
			"episode_length": args.episode_length,
			"use_eval": True,
			"eval_interval": args.eval_interval,
			"n_eval_rollout_threads": 2,
			"eval_episodes": args.eval_episodes,
			"save_interval": args.save_interval,
			"log_interval": args.log_interval,
			"use_wandb": args.use_wandb
		}
		
		# 更新算法特定参数
		if 'train' in algo_args:
			algo_args['train']['batch_size'] = args.batch_size
			algo_args['train']['lr'] = args.lr
			algo_args['train']['gamma'] = args.gamma
			
		# 处理未解析的参数
		def process(arg):
			try:
				return eval(arg)
			except:
				return arg
				
		keys = [k[2:] for k in unparsed_args[0::2]]
		values = [process(v) for v in unparsed_args[1::2]]
		unparsed_dict = {k: v for k, v in zip(keys, values)}
		
		# 更新参数
		update_args(unparsed_dict, algo_args, env_args)
		
		# 开始训练
		print("⏳ 开始训练...\n")
		start_time = time.time()
		
		try:
			# 创建并运行训练器
			runner = RUNNER_REGISTRY[args.algo](main_args, algo_args, env_args)
			runner.run()
			runner.close()
			
			# 训练完成
			elapsed_time = time.time() - start_time
			hours = int(elapsed_time // 3600)
			minutes = int((elapsed_time % 3600) // 60)
			seconds = int(elapsed_time % 60)
			
			print("\n" + "="*60)
			print("✅ 训练成功完成!")
			print(f"   总用时: {hours:02d}:{minutes:02d}:{seconds:02d}")
			print(f"   结果保存在: {result_dir}")
			print("="*60)
			
			# 生成训练报告
			self.generate_report(args, result_dir, elapsed_time)
			
		except KeyboardInterrupt:
			print("\n⚠️  训练被用户中断")
			print(f"   部分结果保存在: {result_dir}")
		except Exception as e:
			print(f"\n❌ 训练失败: {e}")
			print(f"   日志保存在: {result_dir / 'logs'}")
			raise
			
	def generate_report(self, args, result_dir, elapsed_time):
		"""生成训练报告"""
		report_file = result_dir / "training_report.md"
		
		hours = int(elapsed_time // 3600)
		minutes = int((elapsed_time % 3600) // 60)
		
		report = f"""# PowerZoo训练报告

## 基本信息
- **完成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **训练用时**: {hours}小时{minutes}分钟
- **实验名称**: {args.exp_name}

## 配置详情
### 系统配置
- **环境名称**: {args.system}

### PV方案
- **方案类型**: {args.pv_plan}
- **描述**: {self.get_pv_config(args.pv_plan)['description']}

### 算法参数
- **算法**: {args.algo.upper()}
- **训练步数**: {args.train_steps:,}
- **批次大小**: {args.batch_size}
- **学习率**: {args.lr}
- **折扣因子**: {args.gamma}

### 硬件配置
- **GPU**: {'CPU' if args.gpu == -1 else f'GPU:{args.gpu}'}
- **训练线程**: {args.n_threads}
- **并行环境**: {args.n_rollout}

## 结果文件
- **模型文件**: {result_dir}/models/
- **日志文件**: {result_dir}/logs/
- **评估结果**: {result_dir}/eval/
- **可视化图表**: {result_dir}/plots/

## 性能指标
请查看评估结果文件夹获取详细的性能指标。

---
*报告生成时间: {datetime.now()}*
"""
		
		with open(report_file, 'w', encoding='utf-8') as f:
			f.write(report)
			
		print(f"\n📄 训练报告已生成: {report_file}")


def main():
	"""主函数"""
	trainer = PowerZooTrainer()
	trainer.run()


if __name__ == "__main__":
	main()