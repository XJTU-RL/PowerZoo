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

try:
    from envs.dsr.dsr_monitor import DSRMonitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False


class DSRLogger:
    """DSR环境日志记录器，兼容PowerZoo框架"""
    
    def __init__(self, args, algo_args, env_args, num_agents, writer, run_dir):
        """
        初始化DSR日志记录器
        
        Args:
            args: 全局参数
            algo_args: 算法参数  
            env_args: 环境参数
            num_agents: 智能体数量
            writer: TensorBoard写入器
            run_dir: 运行目录
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.num_agents = num_agents
        self.run_dir = run_dir
        self.writer = writer
        
        # DSR特有的日志记录项
        self.dsr_metrics = {
            'restored_load_ratio': [],
            'energized_buses_ratio': [],
            'voltage_violations': [],
            'convergence_rate': [],
            'restoration_steps': [],
            'fault_recovery_time': [],
        }
        
        # 系统层面的监控指标
        self.system_metrics = {
            'line_overloads': [],
            'max_line_loading': [],
            'avg_line_loading': [],
            'power_losses_active': [],
            'power_losses_reactive': [],
            'voltage_deviation_max': [],
            'voltage_deviation_avg': [],
            'topology_components': [],
            'isolated_buses': [],
        }
        
        # 智能体行为指标
        self.agent_metrics = {
            'switch_actions': [],
            'switch_success_rate': [],
            'pv_power_adjustments': [],
            'load_restore_attempts': [],
            'load_restore_success_rate': [],
            'action_effectiveness': [],
        }
        
        # 恢复过程指标
        self.restoration_metrics = {
            'priority_1_restoration': [],
            'priority_2_restoration': [],
            'priority_3_restoration': [],
            'restoration_rate': [],
            'total_kw_restored': [],
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
        
        # 初始化监控器（如果启用）
        self.monitor = None
        monitoring_config = env_args.get('monitoring', {})
        if monitoring_config.get('visualization', False) and MONITOR_AVAILABLE:
            self.monitor = DSRMonitor(
                save_dir=run_dir,
                update_interval=monitoring_config.get('visualization_interval', 50)
            )
            print("DSR监控器已启用")
        
    def init(self, episodes):
        """初始化日志记录"""
        # 重置所有指标
        for metrics_dict in [self.dsr_metrics, self.system_metrics, 
                            self.agent_metrics, self.restoration_metrics,
                            self.eval_metrics]:
            for key in metrics_dict:
                metrics_dict[key] = []
    
    def episode_init(self, episode):
        """初始化每个episode的记录器"""
        self.episode = episode
    
    def eval_init(self):
        """初始化评估阶段的记录器"""
        # 重置评估指标
        for key in self.eval_metrics:
            self.eval_metrics[key] = []
        print("DSR评估阶段初始化完成")
    
    def eval_per_step(self, eval_data):
        """评估阶段每步记录"""
        # eval_data包含评估环境返回的信息
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        
        # 记录评估信息
        if eval_infos:
            self._log_env_info(eval_infos)
    
    def eval_thread_done(self, tid):
        """评估线程完成时的回调"""
        # 可以在这里添加线程特定的清理或记录逻辑
        pass
    
    def per_step(self, data):
        """每步记录"""
        # data包含环境返回的信息
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data
        
        # 如果启用了监控器，更新可视化
        if self.monitor and infos:
            # 使用第一个环境的信息
            self.monitor.update(infos[0], self.total_env_steps)
        
        self.total_env_steps += 1
    
    def episode_log(self, actor_train_infos, critic_train_infos, actor_buffer, critic_buffer):
        """每个episode的日志记录"""
        self.episode_count += 1
        
        # 处理训练信息
        if actor_train_infos is not None:
            self._log_train_info(actor_train_infos, "actor")
        
        if critic_train_infos is not None:
            self._log_train_info(critic_train_infos, "critic")
        
        # 可以从buffer中提取环境信息进行记录
        # 这里暂时不处理buffer，保持简单的实现
    
    def _log_env_info(self, env_infos: List[Dict[str, Any]]):
        """记录环境信息"""
        if not env_infos:
            return
        
        # 取第一个环境的信息（假设所有环境信息类似）
        info = env_infos[0]
        
        # 记录基础DSR指标
        if 'restored_load_ratio' in info:
            self.dsr_metrics['restored_load_ratio'].append(info['restored_load_ratio'])
        
        if 'energized_buses' in info and 'total_buses' in info:
            ratio = info['energized_buses'] / max(info['total_buses'], 1)
            self.dsr_metrics['energized_buses_ratio'].append(ratio)
        
        if 'converged' in info:
            self.dsr_metrics['convergence_rate'].append(float(info['converged']))
        
        if 'current_step' in info:
            self.dsr_metrics['restoration_steps'].append(info['current_step'])
        
        # 记录系统状态指标
        if 'system_state' in info:
            sys_state = info['system_state']
            if 'voltage_violations' in sys_state:
                self.dsr_metrics['voltage_violations'].append(sys_state['voltage_violations'])
            
            if 'line_overloads' in sys_state:
                self.system_metrics['line_overloads'].append(sys_state['line_overloads'])
            
            if 'power_losses' in sys_state:
                losses = sys_state['power_losses']
                self.system_metrics['power_losses_active'].append(losses.get('active', 0))
                self.system_metrics['power_losses_reactive'].append(losses.get('reactive', 0))
        
        # 记录线路负载详情
        if 'line_loading' in info:
            loading_values = []
            for line_data in info['line_loading'].values():
                if line_data.get('status') == 'active':
                    loading_values.append(line_data.get('loading', 0))
            
            if loading_values:
                self.system_metrics['max_line_loading'].append(max(loading_values))
                self.system_metrics['avg_line_loading'].append(np.mean(loading_values))
        
        # 记录电压偏差
        if 'voltage_profile' in info:
            deviations = []
            for bus_data in info['voltage_profile'].values():
                if bus_data.get('energized', False):
                    avg_v = bus_data.get('avg', 1.0)
                    deviation = abs(avg_v - 1.0)
                    deviations.append(deviation)
            
            if deviations:
                self.system_metrics['voltage_deviation_max'].append(max(deviations))
                self.system_metrics['voltage_deviation_avg'].append(np.mean(deviations))
        
        # 记录拓扑指标
        if 'topology_metrics' in info:
            topo = info['topology_metrics']
            self.system_metrics['topology_components'].append(topo.get('total_components', 0))
            self.system_metrics['isolated_buses'].append(topo.get('isolated_buses', 0))
        
        # 记录智能体行为
        if 'agent_behavior' in info:
            behavior = info['agent_behavior']
            if 'action_results' in behavior:
                results = behavior['action_results']
                
                # 开关智能体
                if 'switch' in results:
                    switch_data = results['switch']
                    if switch_data['attempted'] > 0:
                        self.agent_metrics['switch_actions'].append(1)
                        success_rate = switch_data['successful'] / switch_data['attempted']
                        self.agent_metrics['switch_success_rate'].append(success_rate)
                
                # PV智能体
                if 'pv' in results:
                    pv_data = results['pv']
                    self.agent_metrics['pv_power_adjustments'].append(
                        abs(pv_data.get('total_power_change', 0))
                    )
                
                # 负荷智能体
                if 'load' in results:
                    load_data = results['load']
                    if load_data['attempted_restore'] > 0:
                        self.agent_metrics['load_restore_attempts'].append(
                            load_data['attempted_restore']
                        )
                        success_rate = load_data['successful_restore'] / load_data['attempted_restore']
                        self.agent_metrics['load_restore_success_rate'].append(success_rate)
                
            if 'action_effectiveness' in behavior:
                self.agent_metrics['action_effectiveness'].append(
                    behavior['action_effectiveness']
                )
        
        # 记录恢复过程指标
        if 'restoration_progress' in info:
            progress = info['restoration_progress']
            
            if 'priority_restoration' in progress:
                priority_data = progress['priority_restoration']
                for priority in [1, 2, 3]:
                    if priority in priority_data:
                        ratio = priority_data[priority].get('restoration_ratio', 0)
                        self.restoration_metrics[f'priority_{priority}_restoration'].append(ratio)
            
            if 'restoration_rate' in progress:
                self.restoration_metrics['restoration_rate'].append(progress['restoration_rate'])
            
            if 'total_load_restored' in progress:
                self.restoration_metrics['total_kw_restored'].append(progress['total_load_restored'])
        
        # 计算平均值并记录到TensorBoard
        log_interval = self.env_args.get('log_interval_episodes', 10)
        if self.episode_count % log_interval == 0:
            self._write_metrics_to_tensorboard()
    
    def _log_train_info(self, train_infos, prefix: str):
        """记录训练信息"""
        if not train_infos:
            return
        
        # 如果是actor_train_infos（列表格式），处理每个agent的信息
        if isinstance(train_infos, list):
            for agent_id, agent_info in enumerate(train_infos):
                if isinstance(agent_info, dict):
                    for key, value in agent_info.items():
                        if isinstance(value, (int, float)):
                            agent_key = f"{prefix}_agent{agent_id}_{key}"
                            self.writer.add_scalar(f"train/{agent_key}", value, self.episode_count)
        # 如果是critic_train_infos（字典格式），直接处理
        elif isinstance(train_infos, dict):
            for key, value in train_infos.items():
                if isinstance(value, (int, float)):
                    critic_key = f"{prefix}_{key}"
                    self.writer.add_scalar(f"train/{critic_key}", value, self.episode_count)
    
    def _write_metrics_to_tensorboard(self):
        """将所有指标写入TensorBoard"""
        recent_window = self.env_args.get('recent_episodes_window', 10)
        
        # 写入DSR基础指标
        for metric_name, values in self.dsr_metrics.items():
            if values:
                avg_value = np.mean(values[-recent_window:])  # 最近窗口的平均值
                self.writer.add_scalar(f"dsr/{metric_name}", avg_value, self.episode_count)
        
        # 写入系统状态指标
        for metric_name, values in self.system_metrics.items():
            if values:
                avg_value = np.mean(values[-recent_window:])
                self.writer.add_scalar(f"system/{metric_name}", avg_value, self.episode_count)
                
                # 对一些关键指标记录最大/最小值
                if 'loading' in metric_name or 'voltage' in metric_name:
                    self.writer.add_scalar(f"system/{metric_name}_max", 
                                          max(values[-recent_window:]), self.episode_count)
                    self.writer.add_scalar(f"system/{metric_name}_min", 
                                          min(values[-recent_window:]), self.episode_count)
        
        # 写入智能体行为指标
        for metric_name, values in self.agent_metrics.items():
            if values:
                avg_value = np.mean(values[-recent_window:])
                self.writer.add_scalar(f"agent/{metric_name}", avg_value, self.episode_count)
        
        # 写入恢复过程指标
        for metric_name, values in self.restoration_metrics.items():
            if values:
                avg_value = np.mean(values[-recent_window:])
                self.writer.add_scalar(f"restoration/{metric_name}", avg_value, self.episode_count)
        
        # 写入综合指标
        if (self.dsr_metrics['restored_load_ratio'] and 
            self.dsr_metrics['convergence_rate'] and
            self.agent_metrics.get('action_effectiveness')):
            
            # 综合性能分数
            restoration_score = np.mean(self.dsr_metrics['restored_load_ratio'][-recent_window:])
            convergence_score = np.mean(self.dsr_metrics['convergence_rate'][-recent_window:])
            effectiveness_score = np.mean(self.agent_metrics['action_effectiveness'][-recent_window:])
            
            overall_score = (restoration_score * 0.5 + 
                           convergence_score * 0.3 + 
                           effectiveness_score * 0.2)
            
            self.writer.add_scalar("overview/overall_performance", overall_score, self.episode_count)
            self.writer.add_scalar("overview/restoration_score", restoration_score, self.episode_count)
            self.writer.add_scalar("overview/convergence_score", convergence_score, self.episode_count)
    
    def eval_log(self, eval_episode, eval_env_infos=None, eval_average_episode_rewards=None):
        """评估日志记录"""
        # 处理评估环境信息
        if eval_env_infos is not None and len(eval_env_infos) > 0:
            self._log_eval_env_info(eval_env_infos)
        
        # 记录评估奖励
        if eval_average_episode_rewards is not None:
            self.writer.add_scalar("eval/average_episode_rewards", 
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
            self.writer.add_scalar("eval/restored_load_ratio", avg_restored_ratio, self.episode_count)
        
        if energized_ratios:
            avg_energized_ratio = np.mean(energized_ratios)
            self.eval_metrics['eval_energized_buses_ratio'].append(avg_energized_ratio)
            self.writer.add_scalar("eval/energized_buses_ratio", avg_energized_ratio, self.episode_count)
        
        if success_flags:
            success_rate = np.mean(success_flags)
            self.eval_metrics['eval_success_rate'].append(success_rate)
            self.writer.add_scalar("eval/success_rate", success_rate, self.episode_count)
        
        if convergence_flags:
            convergence_rate = np.mean(convergence_flags)
            self.eval_metrics['eval_convergence_rate'].append(convergence_rate)
            self.writer.add_scalar("eval/convergence_rate", convergence_rate, self.episode_count)
        
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
        self.writer.add_scalar("train/total_env_steps", total_num_steps, self.episode_count)
        
        if self.args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "train/total_env_steps": total_num_steps,
                "train/episode": self.episode_count
            })
    
    def close(self):
        """关闭日志记录器"""
        # 保存最终的DSR指标统计
        self._save_final_metrics()
        
        # 生成最终报告
        self._generate_final_report()
        
        # 生成最终可视化报告
        if self.monitor:
            self.monitor.generate_final_report()
        
        # 关闭TensorBoard写入器
        if self.writer:
            self.writer.close()
    
    def _save_final_metrics(self):
        """保存最终指标统计"""
        try:
            import json
            import os
            
            # 计算最终统计
            final_stats = {}
            
            # 处理所有指标字典
            all_metrics = {
                'dsr': self.dsr_metrics,
                'system': self.system_metrics,
                'agent': self.agent_metrics,
                'restoration': self.restoration_metrics,
                'eval': self.eval_metrics
            }
            
            for category, metrics_dict in all_metrics.items():
                final_stats[category] = {}
                for metric_name, values in metrics_dict.items():
                    if values:
                        final_stats[category][metric_name] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'min': float(np.min(values)),
                            'max': float(np.max(values)),
                            'final_value': float(values[-1]),
                            'total_samples': len(values)
                        }
            
            # 保存到文件
            stats_file = os.path.join(self.run_dir, 'dsr_final_stats.json')
            with open(stats_file, 'w') as f:
                json.dump(final_stats, f, indent=2)
            
            print(f"DSR最终统计已保存到: {stats_file}")
            
        except Exception as e:
            print(f"保存DSR统计时出错: {e}")
    
    def get_latest_metrics(self) -> Dict[str, float]:
        """获取最新的所有指标"""
        latest = {}
        
        # 收集所有最新指标
        all_metrics = {
            'dsr': self.dsr_metrics,
            'system': self.system_metrics,
            'agent': self.agent_metrics,
            'restoration': self.restoration_metrics,
            'eval': self.eval_metrics
        }
        
        for category, metrics_dict in all_metrics.items():
            for metric_name, values in metrics_dict.items():
                if values:
                    latest[f"{category}/{metric_name}"] = values[-1]
        
        return latest
    
    def _generate_final_report(self):
        """生成最终训练报告"""
        try:
            import os
            
            report_file = os.path.join(self.run_dir, 'dsr_training_report.txt')
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("DSR 训练最终报告\n")
                f.write("="*60 + "\n\n")
                
                # 基础信息
                f.write(f"环境: {self.args.get('env', 'dsr')}\n")
                f.write(f"算法: {self.args.get('algo', 'unknown')}\n")
                f.write(f"实验名称: {self.args.get('exp_name', 'unknown')}\n")
                f.write(f"总训练回合数: {self.episode_count}\n")
                f.write(f"总环境步数: {self.total_env_steps}\n\n")
                
                # 关键性能指标
                f.write("-"*40 + "\n")
                f.write("关键性能指标\n")
                f.write("-"*40 + "\n")
                
                if self.dsr_metrics['restored_load_ratio']:
                    final_restoration = self.dsr_metrics['restored_load_ratio'][-1]
                    avg_restoration = np.mean(self.dsr_metrics['restored_load_ratio'])
                    f.write(f"最终负荷恢复率: {final_restoration:.2%}\n")
                    f.write(f"平均负荷恢复率: {avg_restoration:.2%}\n")
                
                if self.dsr_metrics['convergence_rate']:
                    avg_convergence = np.mean(self.dsr_metrics['convergence_rate'])
                    f.write(f"平均收敛率: {avg_convergence:.2%}\n")
                
                if self.agent_metrics['action_effectiveness']:
                    avg_effectiveness = np.mean(self.agent_metrics['action_effectiveness'])
                    f.write(f"平均动作有效性: {avg_effectiveness:.2%}\n")
                
                # 系统稳定性
                f.write("\n" + "-"*40 + "\n")
                f.write("系统稳定性\n")
                f.write("-"*40 + "\n")
                
                if self.dsr_metrics['voltage_violations']:
                    avg_violations = np.mean(self.dsr_metrics['voltage_violations'])
                    f.write(f"平均电压越限数: {avg_violations:.1f}\n")
                
                if self.system_metrics['line_overloads']:
                    avg_overloads = np.mean(self.system_metrics['line_overloads'])
                    f.write(f"平均线路过载数: {avg_overloads:.1f}\n")
                
                # 智能体协作
                f.write("\n" + "-"*40 + "\n")
                f.write("智能体协作\n")
                f.write("-"*40 + "\n")
                
                if self.agent_metrics['switch_success_rate']:
                    avg_switch_success = np.mean(self.agent_metrics['switch_success_rate'])
                    f.write(f"开关操作成功率: {avg_switch_success:.2%}\n")
                
                if self.agent_metrics['load_restore_success_rate']:
                    avg_load_success = np.mean(self.agent_metrics['load_restore_success_rate'])
                    f.write(f"负荷恢复成功率: {avg_load_success:.2%}\n")
                
                # 优先级恢复
                f.write("\n" + "-"*40 + "\n")
                f.write("优先级恢复\n")
                f.write("-"*40 + "\n")
                
                for priority in [1, 2, 3]:
                    metric_key = f'priority_{priority}_restoration'
                    if self.restoration_metrics[metric_key]:
                        avg_restoration = np.mean(self.restoration_metrics[metric_key])
                        f.write(f"优先级{priority}负荷恢复率: {avg_restoration:.2%}\n")
                
                f.write("\n" + "="*60 + "\n")
                f.write("报告生成完成\n")
                
            print(f"DSR训练报告已保存到: {report_file}")
            
        except Exception as e:
            print(f"生成DSR报告时出错: {e}")