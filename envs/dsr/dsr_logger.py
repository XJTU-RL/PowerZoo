# -*- coding: utf-8 -*-
"""
DSR Environment Logger
配电网恢复环境日志记录器
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    from utils.util import update_linear_schedule
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False


class DSRLogger:
    """DSR环境日志记录器，兼容PowerZoo框架"""
    
    def __init__(self, args, algo_args, env_args, run_dir, writter):
        """
        初始化DSR日志记录器
        
        Args:
            args: 全局参数
            algo_args: 算法参数  
            env_args: 环境参数
            run_dir: 运行目录
            writter: TensorBoard写入器
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.run_dir = run_dir
        self.writter = writter
        
        # DSR特有的日志记录项
        self.dsr_metrics = {
            'restored_load_ratio': [],
            'energized_buses_ratio': [],
            'voltage_violations': [],
            'convergence_rate': [],
            'restoration_steps': [],
            'fault_recovery_time': [],
        }
        
        # 初始化评估指标
        self.eval_metrics = {
            'eval_restored_load_ratio': [],
            'eval_energized_buses_ratio': [],
            'eval_success_rate': [],
            'eval_convergence_rate': [],
        }
        
        # 记录训练过程
        self.episode_count = 0
        self.total_env_steps = 0
        
    def init(self, episodes):
        """初始化日志记录"""
        # 重置DSR指标
        for key in self.dsr_metrics:
            self.dsr_metrics[key] = []
        
        for key in self.eval_metrics:
            self.eval_metrics[key] = []
    
    def per_step(self, data):
        """每步记录"""
        # data包含环境返回的信息
        pass
    
    def episode_log(self, actor_train_infos, critic_train_infos, env_infos):
        """每个episode的日志记录"""
        self.episode_count += 1
        
        # 处理环境信息
        if env_infos is not None and len(env_infos) > 0:
            self._log_env_info(env_infos)
        
        # 处理训练信息
        if actor_train_infos is not None:
            self._log_train_info(actor_train_infos, "actor")
        
        if critic_train_infos is not None:
            self._log_train_info(critic_train_infos, "critic")
    
    def _log_env_info(self, env_infos: List[Dict[str, Any]]):
        """记录环境信息"""
        if not env_infos:
            return
        
        # 取第一个环境的信息（假设所有环境信息类似）
        info = env_infos[0]
        
        # 记录DSR特有指标
        if 'restored_load_ratio' in info:
            self.dsr_metrics['restored_load_ratio'].append(info['restored_load_ratio'])
        
        if 'energized_buses' in info and 'total_buses' in info:
            ratio = info['energized_buses'] / max(info['total_buses'], 1)
            self.dsr_metrics['energized_buses_ratio'].append(ratio)
        
        if 'converged' in info:
            self.dsr_metrics['convergence_rate'].append(float(info['converged']))
        
        if 'current_step' in info:
            self.dsr_metrics['restoration_steps'].append(info['current_step'])
        
        # 计算平均值并记录到TensorBoard
        log_interval = self.env_args.get('log_interval_episodes', 10)
        if len(self.dsr_metrics['restored_load_ratio']) % log_interval == 0:
            self._write_metrics_to_tensorboard()
    
    def _log_train_info(self, train_infos: List[Dict[str, Any]], prefix: str):
        """记录训练信息"""
        if not train_infos:
            return
        
        # 计算平均训练指标
        avg_train_info = {}
        for key in train_infos[0].keys():
            if isinstance(train_infos[0][key], (int, float)):
                values = [info[key] for info in train_infos if key in info]
                if values:
                    avg_train_info[f"{prefix}_{key}"] = np.mean(values)
        
        # 记录到TensorBoard
        for key, value in avg_train_info.items():
            self.writter.add_scalar(f"train/{key}", value, self.episode_count)
    
    def _write_metrics_to_tensorboard(self):
        """将DSR指标写入TensorBoard"""
        recent_window = self.env_args.get('recent_episodes_window', 10)
        for metric_name, values in self.dsr_metrics.items():
            if values:
                avg_value = np.mean(values[-recent_window:])  # 最近窗口的平均值
                self.writter.add_scalar(f"dsr/{metric_name}", avg_value, self.episode_count)
    
    def eval_log(self, eval_episode, eval_env_infos, eval_average_episode_rewards):
        """评估日志记录"""
        # 处理评估环境信息
        if eval_env_infos is not None and len(eval_env_infos) > 0:
            self._log_eval_env_info(eval_env_infos)
        
        # 记录评估奖励
        if eval_average_episode_rewards is not None:
            self.writter.add_scalar("eval/average_episode_rewards", 
                                  np.mean(eval_average_episode_rewards), eval_episode)
            
            if self.args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "eval/average_episode_rewards": np.mean(eval_average_episode_rewards),
                    "eval/episode": eval_episode
                })
    
    def _log_eval_env_info(self, eval_env_infos: List[Dict[str, Any]]):
        """记录评估环境信息"""
        # 收集所有评估信息
        restored_ratios = []
        energized_ratios = []
        success_flags = []
        convergence_flags = []
        
        success_threshold = self.env_args.get('success_threshold', 0.9)
        
        for info in eval_env_infos:
            if 'restored_load_ratio' in info:
                restored_ratios.append(info['restored_load_ratio'])
                success_flags.append(info['restored_load_ratio'] >= success_threshold)  # 配置阈值视为成功
            
            if 'energized_buses' in info and 'total_buses' in info:
                ratio = info['energized_buses'] / max(info['total_buses'], 1)
                energized_ratios.append(ratio)
            
            if 'converged' in info:
                convergence_flags.append(info['converged'])
        
        # 计算评估指标
        if restored_ratios:
            avg_restored_ratio = np.mean(restored_ratios)
            self.eval_metrics['eval_restored_load_ratio'].append(avg_restored_ratio)
            self.writter.add_scalar("eval/restored_load_ratio", avg_restored_ratio, self.episode_count)
        
        if energized_ratios:
            avg_energized_ratio = np.mean(energized_ratios)
            self.eval_metrics['eval_energized_buses_ratio'].append(avg_energized_ratio)
            self.writter.add_scalar("eval/energized_buses_ratio", avg_energized_ratio, self.episode_count)
        
        if success_flags:
            success_rate = np.mean(success_flags)
            self.eval_metrics['eval_success_rate'].append(success_rate)
            self.writter.add_scalar("eval/success_rate", success_rate, self.episode_count)
        
        if convergence_flags:
            convergence_rate = np.mean(convergence_flags)
            self.eval_metrics['eval_convergence_rate'].append(convergence_rate)
            self.writter.add_scalar("eval/convergence_rate", convergence_rate, self.episode_count)
        
        # WandB记录
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb_dict = {}
            if restored_ratios:
                wandb_dict["eval/restored_load_ratio"] = np.mean(restored_ratios)
            if energized_ratios:
                wandb_dict["eval/energized_buses_ratio"] = np.mean(energized_ratios)
            if success_flags:
                wandb_dict["eval/success_rate"] = np.mean(success_flags)
            if convergence_flags:
                wandb_dict["eval/convergence_rate"] = np.mean(convergence_flags)
            
            if wandb_dict:
                wandb_dict["eval/episode"] = self.episode_count
                wandb.log(wandb_dict)
    
    def train_log(self, total_num_steps):
        """训练日志记录"""
        self.total_env_steps = total_num_steps
        
        # 记录训练进度
        self.writter.add_scalar("train/total_env_steps", total_num_steps, self.episode_count)
        
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "train/total_env_steps": total_num_steps,
                "train/episode": self.episode_count
            })
    
    def close(self):
        """关闭日志记录器"""
        # 保存最终的DSR指标统计
        self._save_final_metrics()
        
        # 关闭TensorBoard写入器
        if self.writter:
            self.writter.close()
    
    def _save_final_metrics(self):
        """保存最终指标统计"""
        try:
            import json
            import os
            
            # 计算最终统计
            final_stats = {}
            for metric_name, values in self.dsr_metrics.items():
                if values:
                    final_stats[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'final_value': float(values[-1]) if values else 0.0
                    }
            
            for metric_name, values in self.eval_metrics.items():
                if values:
                    final_stats[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'final_value': float(values[-1]) if values else 0.0
                    }
            
            # 保存到文件
            stats_file = os.path.join(self.run_dir, 'dsr_final_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(final_stats, f, indent=2)
            
            print(f"DSR最终统计已保存到: {stats_file}")
            
        except Exception as e:
            print(f"保存DSR统计时出错: {e}")
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """获取最新的DSR指标"""
        latest = {}
        
        for metric_name, values in self.dsr_metrics.items():
            if values:
                latest[metric_name] = values[-1]
        
        for metric_name, values in self.eval_metrics.items():
            if values:
                latest[metric_name] = values[-1]
        
        return latest