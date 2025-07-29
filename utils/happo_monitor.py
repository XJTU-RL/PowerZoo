#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HAPPO训练监控工具
用于监控PowerZoo PV训练的收敛情况和性能指标

@Author: Xiaodong Zheng
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime
import re

class HAPPOMonitor:
	"""HAPPO训练监控器"""
	
	def __init__(self, log_dir: Path):
		self.log_dir = Path(log_dir)
		self.metrics = {
			'rewards': [],
			'episode_lengths': [],
			'voltage_violations': [],
			'power_losses': [],
			'convergence_metrics': [],
			'training_steps': []
		}
		
	def parse_log_file(self, log_file: Path) -> Dict:
		"""解析训练日志文件"""
		if not log_file.exists():
			print(f"❌ 日志文件不存在: {log_file}")
			return {}
		
		print(f"📖 解析日志文件: {log_file}")
		
		with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
			lines = f.readlines()
		
		# 提取关键指标
		rewards = []
		losses = []
		violations = []
		steps = []
		
		for line in lines:
			# 解析奖励值
			if 'average_episode_rewards' in line:
				match = re.search(r'average_episode_rewards.*?([+-]?\d+\.?\d*)', line)
				if match:
					rewards.append(float(match.group(1)))
			
			# 解析功率损失
			if 'power_loss' in line.lower():
				match = re.search(r'power_loss.*?([+-]?\d+\.?\d*)', line)
				if match:
					losses.append(float(match.group(1)))
			
			# 解析电压违规
			if 'voltage_violation' in line.lower():
				match = re.search(r'voltage_violation.*?([+-]?\d+\.?\d*)', line)
				if match:
					violations.append(float(match.group(1)))
			
			# 解析训练步数
			if 'total_num_steps' in line:
				match = re.search(r'total_num_steps.*?(\d+)', line)
				if match:
					steps.append(int(match.group(1)))
		
		return {
			'rewards': rewards,
			'power_losses': losses, 
			'voltage_violations': violations,
			'training_steps': steps
		}
	
	def analyze_convergence(self, metric_data: List[float], window_size: int = 100) -> Dict:
		"""分析收敛性"""
		if len(metric_data) < window_size:
			return {'converged': False, 'reason': 'insufficient_data'}
		
		# 计算滑动平均
		moving_avg = []
		for i in range(len(metric_data) - window_size + 1):
			avg = np.mean(metric_data[i:i + window_size])
			moving_avg.append(avg)
		
		# 计算方差变化
		if len(moving_avg) >= 2:
			recent_std = np.std(moving_avg[-window_size//2:])
			early_std = np.std(moving_avg[:window_size//2])
			
			# 收敛判断：最近的方差显著小于早期方差
			variance_ratio = recent_std / (early_std + 1e-8)
			
			converged = variance_ratio < 0.1  # 方差减少90%以上
			
			return {
				'converged': converged,
				'variance_ratio': variance_ratio,
				'recent_std': recent_std,
				'trend': 'improving' if moving_avg[-1] > moving_avg[-10] else 'declining'
			}
		
		return {'converged': False, 'reason': 'insufficient_variance_data'}
	
	def generate_convergence_report(self, metrics: Dict) -> str:
		"""生成收敛报告"""
		report = ["=" * 60]
		report.append("🎯 HAPPO PowerZoo PV 训练收敛分析报告")
		report.append("=" * 60)
		report.append(f"📅 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
		report.append("")
		
		# 分析各项指标
		for metric_name, data in metrics.items():
			if not data:
				continue
				
			report.append(f"📊 {metric_name.upper()} 分析:")
			report.append(f"   数据点数: {len(data)}")
			report.append(f"   最新值: {data[-1]:.4f}")
			report.append(f"   平均值: {np.mean(data):.4f}")
			report.append(f"   标准差: {np.std(data):.4f}")
			
			# 收敛分析
			convergence = self.analyze_convergence(data)
			if convergence['converged']:
				report.append("   ✅ 收敛状态: 已收敛")
			else:
				report.append(f"   ⏳ 收敛状态: 未收敛 ({convergence.get('reason', 'unknown')})")
			
			report.append("")
		
		# 整体评估
		report.append("🎯 整体训练评估:")
		
		# 奖励收敛
		if 'rewards' in metrics and len(metrics['rewards']) > 100:
			reward_convergence = self.analyze_convergence(metrics['rewards'])
			if reward_convergence['converged']:
				report.append("   ✅ 奖励函数已收敛")
			else:
				report.append("   ⏳ 奖励函数尚未收敛")
		
		# 约束满足
		if 'voltage_violations' in metrics and metrics['voltage_violations']:
			avg_violations = np.mean(metrics['voltage_violations'][-50:])  # 最近50个数据点
			if avg_violations < 0.05:  # 5%以下违规率
				report.append("   ✅ 电压约束满足良好")
			else:
				report.append(f"   ⚠️  电压违规率较高: {avg_violations:.2%}")
		
		# 性能改善
		if 'power_losses' in metrics and len(metrics['power_losses']) > 50:
			early_loss = np.mean(metrics['power_losses'][:25])
			recent_loss = np.mean(metrics['power_losses'][-25:])
			improvement = (early_loss - recent_loss) / early_loss
			
			if improvement > 0.1:  # 10%以上改善
				report.append(f"   ✅ 功率损失显著改善: {improvement:.1%}")
			else:
				report.append(f"   ⏳ 功率损失改善有限: {improvement:.1%}")
		
		# 训练建议
		report.append("")
		report.append("💡 训练建议:")
		
		if 'rewards' in metrics and len(metrics['rewards']) > 0:
			recent_rewards = metrics['rewards'][-10:]
			if len(recent_rewards) > 5 and np.std(recent_rewards) < 0.01:
				report.append("   • 奖励已趋于稳定，可考虑降低学习率")
			elif len(metrics['rewards']) > 1000:
				report.append("   • 训练步数充足，检查是否需要调整超参数")
		
		report.append("")
		report.append("=" * 60)
		
		return "\n".join(report)
	
	def plot_training_curves(self, metrics: Dict, save_path: Optional[Path] = None):
		"""绘制训练曲线"""
		fig, axes = plt.subplots(2, 2, figsize=(15, 10))
		fig.suptitle('HAPPO PowerZoo PV 训练监控', fontsize=16, fontweight='bold')
		
		# 奖励曲线
		if 'rewards' in metrics and metrics['rewards']:
			axes[0, 0].plot(metrics['rewards'], 'b-', alpha=0.7, linewidth=1)
			if len(metrics['rewards']) > 50:
				# 添加滑动平均
				window = min(50, len(metrics['rewards']) // 10)
				smooth_rewards = pd.Series(metrics['rewards']).rolling(window=window).mean()
				axes[0, 0].plot(smooth_rewards, 'r-', linewidth=2, label=f'滑动平均({window})')
				axes[0, 0].legend()
			axes[0, 0].set_title('平均奖励 (Average Rewards)')
			axes[0, 0].set_xlabel('Episodes')
			axes[0, 0].set_ylabel('Reward')
			axes[0, 0].grid(True, alpha=0.3)
		
		# 功率损失
		if 'power_losses' in metrics and metrics['power_losses']:
			axes[0, 1].plot(metrics['power_losses'], 'g-', alpha=0.7, linewidth=1)
			if len(metrics['power_losses']) > 20:
				window = min(20, len(metrics['power_losses']) // 5)
				smooth_losses = pd.Series(metrics['power_losses']).rolling(window=window).mean()
				axes[0, 1].plot(smooth_losses, 'r-', linewidth=2, label=f'滑动平均({window})')
				axes[0, 1].legend()
			axes[0, 1].set_title('功率损失 (Power Losses)')
			axes[0, 1].set_xlabel('Episodes')
			axes[0, 1].set_ylabel('Power Loss')
			axes[0, 1].grid(True, alpha=0.3)
		
		# 电压违规
		if 'voltage_violations' in metrics and metrics['voltage_violations']:
			axes[1, 0].plot(metrics['voltage_violations'], 'orange', alpha=0.7, linewidth=1)
			if len(metrics['voltage_violations']) > 20:
				window = min(20, len(metrics['voltage_violations']) // 5)
				smooth_violations = pd.Series(metrics['voltage_violations']).rolling(window=window).mean()
				axes[1, 0].plot(smooth_violations, 'r-', linewidth=2, label=f'滑动平均({window})')
				axes[1, 0].legend()
			axes[1, 0].set_title('电压违规率 (Voltage Violations)')
			axes[1, 0].set_xlabel('Episodes')
			axes[1, 0].set_ylabel('Violation Rate')
			axes[1, 0].grid(True, alpha=0.3)
		
		# 训练步数与奖励关系
		if 'training_steps' in metrics and 'rewards' in metrics and metrics['training_steps'] and metrics['rewards']:
			min_len = min(len(metrics['training_steps']), len(metrics['rewards']))
			axes[1, 1].scatter(metrics['training_steps'][:min_len], metrics['rewards'][:min_len], 
			                  alpha=0.6, s=10, c='purple')
			axes[1, 1].set_title('训练步数 vs 奖励 (Steps vs Rewards)')
			axes[1, 1].set_xlabel('Training Steps')
			axes[1, 1].set_ylabel('Reward')
			axes[1, 1].grid(True, alpha=0.3)
		
		plt.tight_layout()
		
		if save_path:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')
			print(f"📊 训练曲线已保存: {save_path}")
		
		plt.show()
	
	def run_analysis(self, log_pattern: str = "*.log"):
		"""运行完整分析"""
		print(f"🔍 搜索日志文件: {self.log_dir / log_pattern}")
		
		# 查找日志文件
		log_files = list(self.log_dir.glob(log_pattern))
		if not log_files:
			print(f"❌ 未找到日志文件: {self.log_dir}")
			return
		
		print(f"📁 找到 {len(log_files)} 个日志文件")
		
		# 解析所有日志文件
		all_metrics = {}
		for log_file in sorted(log_files):
			print(f"📖 解析: {log_file.name}")
			metrics = self.parse_log_file(log_file)
			
			# 合并指标
			for key, values in metrics.items():
				if key not in all_metrics:
					all_metrics[key] = []
				all_metrics[key].extend(values)
		
		if not any(all_metrics.values()):
			print("⚠️  未提取到有效的训练指标")
			return
		
		# 生成报告
		report = self.generate_convergence_report(all_metrics)
		print(report)
		
		# 保存报告
		report_file = self.log_dir / f"convergence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
		with open(report_file, 'w', encoding='utf-8') as f:
			f.write(report)
		print(f"📄 报告已保存: {report_file}")
		
		# 绘制图表
		plot_file = self.log_dir / f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
		self.plot_training_curves(all_metrics, plot_file)
		
		return all_metrics, report

def main():
	parser = argparse.ArgumentParser(description="HAPPO PowerZoo PV训练监控工具")
	parser.add_argument("--log_dir", type=str, required=True,
	                   help="训练日志目录路径")
	parser.add_argument("--pattern", type=str, default="*.log",
	                   help="日志文件匹配模式")
	parser.add_argument("--output", type=str, default=None,
	                   help="输出目录（默认为日志目录）")
	
	args = parser.parse_args()
	
	log_dir = Path(args.log_dir)
	if not log_dir.exists():
		print(f"❌ 日志目录不存在: {log_dir}")
		return 1
	
	print("🎯 HAPPO PowerZoo PV 训练监控分析")
	print("=" * 50)
	
	# 创建监控器
	monitor = HAPPOMonitor(log_dir)
	
	# 运行分析
	try:
		results = monitor.run_analysis(args.pattern)
		if results:
			print("\n✅ 分析完成!")
		else:
			print("\n⚠️  分析未产生有效结果")
			return 1
			
	except Exception as e:
		print(f"\n❌ 分析过程中出现错误: {e}")
		return 1
	
	return 0

if __name__ == "__main__":
	sys.exit(main())