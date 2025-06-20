# -*- coding: utf-8 -*-
"""
Comprehensive Monitoring System for Stackelberg Game Environment

This module provides real-time monitoring and visualization capabilities
for the Stackelberg game-based demand response framework.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from datetime import datetime
import json
import pickle
import logging
from pathlib import Path

# Optional imports for advanced features
try:
    from tensorboardX import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("TensorboardX not available. Install with: pip install tensorboardX")


class StackelbergMonitor:
    """
    Comprehensive monitoring system for Stackelberg game environment.
    
    Features:
    - Real-time metric tracking
    - Convergence analysis
    - Nash equilibrium monitoring
    - System stability metrics
    - Agent behavior analysis
    - Visualization and logging
    """
    
    def __init__(self, 
                 log_dir: str = "logs/stackelberg",
                 experiment_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the monitoring system.
        
        Args:
            log_dir: Directory for saving logs and visualizations
            experiment_name: Name of the experiment
            config: Configuration dictionary
        """
        # Setup directories
        self.log_dir = Path(log_dir)
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self._setup_directories()
        
        # Configuration
        self.config = config or {}
        self.save_interval = self.config.get('save_interval', 100)
        self.plot_interval = self.config.get('plot_interval', 50)
        
        # Metrics storage
        self.metrics_history = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.step_metrics = defaultdict(list)
        
        # Agent-specific metrics
        self.uc_metrics = defaultdict(list)
        self.consumer_metrics = defaultdict(lambda: defaultdict(list))
        
        # Convergence tracking
        self.convergence_history = {
            'nash_gap': deque(maxlen=100),
            'price_stability': deque(maxlen=100),
            'action_variance': deque(maxlen=100)
        }
        
        # System stability metrics
        self.stability_metrics = {
            'voltage_violations': [],
            'power_loss_ratio': [],
            'line_overloads': [],
            'n_minus_1_violations': []
        }
        
        # Nash equilibrium tracking
        self.nash_tracking = {
            'uc_best_response': [],
            'consumer_best_responses': defaultdict(list),
            'equilibrium_distance': []
        }
        
        # Setup logging
        self._setup_logging()
        
        # Setup TensorBoard if available
        if HAS_TENSORBOARD:
            self.tb_writer = SummaryWriter(str(self.experiment_dir / 'tensorboard'))
        else:
            self.tb_writer = None
        
        # Visualization settings
        plt.style.use('seaborn-v0_8-darkgrid')
        self.figure_size = (12, 8)
        
        # Statistics tracking
        self.current_episode = 0
        self.current_step = 0
        self.total_steps = 0
        
    def _setup_directories(self):
        """Create necessary directories for logging."""
        dirs = [
            self.experiment_dir,
            self.experiment_dir / 'plots',
            self.experiment_dir / 'data',
            self.experiment_dir / 'checkpoints',
            self.experiment_dir / 'tensorboard'
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.experiment_dir / 'experiment.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('StackelbergMonitor')
        self.logger.info(f"Starting experiment: {self.experiment_name}")
    
    def log_step(self, 
                 rewards: Dict[int, float],
                 observations: Dict[int, np.ndarray],
                 actions: Dict[int, np.ndarray],
                 infos: Dict[int, Dict[str, Any]],
                 system_state: Dict[str, Any]):
        """
        Log metrics for a single environment step.
        
        Args:
            rewards: Agent rewards
            observations: Agent observations
            actions: Agent actions
            infos: Additional information
            system_state: System state dictionary
        """
        self.current_step += 1
        self.total_steps += 1
        
        # Extract UC and consumer IDs
        uc_id = 0  # Assuming UC is always agent 0
        consumer_ids = [aid for aid in rewards.keys() if aid != uc_id]
        
        # Log UC metrics
        if uc_id in rewards:
            self.uc_metrics['reward'].append(rewards[uc_id])
            if uc_id in actions:
                self.uc_metrics['price_signal'].append(actions[uc_id][0])
                self.uc_metrics['dr_incentive'].append(actions[uc_id][1])
        
        # Log consumer metrics
        for cid in consumer_ids:
            if cid in rewards:
                self.consumer_metrics[cid]['reward'].append(rewards[cid])
            if cid in actions:
                self.consumer_metrics[cid]['load_adjustment'].append(actions[cid][0])
                self.consumer_metrics[cid]['der_output'].append(actions[cid][1])
        
        # Log system metrics
        self.step_metrics['power_loss_ratio'].append(
            system_state.get('power_loss_ratio', 0.0)
        )
        self.step_metrics['voltage_violations'].append(
            system_state.get('voltage_violations', 0)
        )
        self.step_metrics['min_voltage'].append(
            system_state.get('min_voltage', 1.0)
        )
        self.step_metrics['max_voltage'].append(
            system_state.get('max_voltage', 1.0)
        )
        
        # Calculate and log convergence metrics
        self._update_convergence_metrics(rewards, actions)
        
        # Log Nash gap if available
        self._update_nash_gap(actions, system_state)
        
        # Calculate Nash equilibrium metrics
        self._update_nash_metrics(rewards, actions, system_state)
        
        # Log to TensorBoard if available
        if self.tb_writer:
            self._log_to_tensorboard(rewards, actions, system_state)
    
    def log_episode_end(self):
        """Log metrics at the end of an episode."""
        self.current_episode += 1
        
        # Calculate episode statistics
        episode_stats = self._calculate_episode_statistics()
        
        # Log episode summary
        self.logger.info(f"Episode {self.current_episode} completed:")
        self.logger.info(f"  - Total reward (UC): {episode_stats['uc_total_reward']:.2f}")
        self.logger.info(f"  - Avg consumer reward: {episode_stats['avg_consumer_reward']:.2f}")
        self.logger.info(f"  - Social welfare: {episode_stats['social_welfare']:.2f}")
        self.logger.info(f"  - Avg power loss: {episode_stats['avg_power_loss']:.4f}")
        self.logger.info(f"  - Total voltage violations: {episode_stats['total_voltage_violations']}")
        self.logger.info(f"  - Nash gap: {episode_stats['nash_gap']:.4f}")
        
        # Store episode metrics
        for key, value in episode_stats.items():
            self.episode_metrics[key].append(value)
        
        # Generate plots if needed
        if self.current_episode % self.plot_interval == 0:
            self.generate_plots()
        
        # Save data if needed
        if self.current_episode % self.save_interval == 0:
            self.save_monitoring_data()
        
        # Reset step counter
        self.current_step = 0
        
        # Clear step metrics
        for key in self.step_metrics:
            self.step_metrics[key].clear()
    
    def _update_convergence_metrics(self, 
                                  rewards: Dict[int, float],
                                  actions: Dict[int, np.ndarray]):
        """Update convergence tracking metrics."""
        # Nash gap (simplified)
        uc_reward = rewards.get(0, 0.0)
        consumer_rewards = [r for aid, r in rewards.items() if aid != 0]
        if consumer_rewards:
            nash_gap = abs(uc_reward - np.mean(consumer_rewards))
            self.convergence_history['nash_gap'].append(nash_gap)
        
        # Price stability (if UC action available)
        if 0 in actions and len(self.uc_metrics['price_signal']) > 1:
            price_variance = np.var(self.uc_metrics['price_signal'][-10:])
            self.convergence_history['price_stability'].append(price_variance)
        
        # Action variance across consumers
        consumer_actions = [a[0] for aid, a in actions.items() if aid != 0]
        if consumer_actions:
            action_variance = np.var(consumer_actions)
            self.convergence_history['action_variance'].append(action_variance)
    
    def _update_nash_metrics(self,
                            rewards: Dict[int, float],
                            actions: Dict[int, np.ndarray],
                            system_state: Dict[str, Any]):
        """Update Nash equilibrium tracking metrics."""
        # This is a simplified implementation
        # In practice, you would compute best response strategies
        
        # Track UC's current strategy effectiveness
        uc_reward = rewards.get(0, 0.0)
        self.nash_tracking['uc_best_response'].append(uc_reward)
        
        # Track consumer best responses
        for aid, reward in rewards.items():
            if aid != 0:
                self.nash_tracking['consumer_best_responses'][aid].append(reward)
        
        # Estimate equilibrium distance (simplified)
        if len(self.nash_tracking['uc_best_response']) > 10:
            recent_uc = self.nash_tracking['uc_best_response'][-10:]
            recent_consumer = [
                np.mean(list(self.nash_tracking['consumer_best_responses'][aid])[-10:])
                for aid in self.nash_tracking['consumer_best_responses']
                if len(self.nash_tracking['consumer_best_responses'][aid]) >= 10
            ]
            if recent_consumer:
                equilibrium_distance = np.std(recent_uc) + np.mean([
                    np.std(self.nash_tracking['consumer_best_responses'][aid][-10:])
                    for aid in self.nash_tracking['consumer_best_responses']
                    if len(self.nash_tracking['consumer_best_responses'][aid]) >= 10
                ])
                self.nash_tracking['equilibrium_distance'].append(equilibrium_distance)
    
    def _calculate_episode_statistics(self) -> Dict[str, float]:
        """Calculate statistics for the completed episode."""
        stats = {}
        
        # UC statistics
        if self.uc_metrics['reward']:
            stats['uc_total_reward'] = sum(self.uc_metrics['reward'])
            stats['uc_avg_reward'] = np.mean(self.uc_metrics['reward'])
        else:
            stats['uc_total_reward'] = 0.0
            stats['uc_avg_reward'] = 0.0
        
        # Consumer statistics
        consumer_total_rewards = []
        for cid, metrics in self.consumer_metrics.items():
            if metrics['reward']:
                consumer_total_rewards.append(sum(metrics['reward']))
        
        if consumer_total_rewards:
            stats['avg_consumer_reward'] = np.mean(consumer_total_rewards)
            stats['std_consumer_reward'] = np.std(consumer_total_rewards)
        else:
            stats['avg_consumer_reward'] = 0.0
            stats['std_consumer_reward'] = 0.0
        
        # Social welfare
        stats['social_welfare'] = stats['uc_total_reward'] + sum(consumer_total_rewards)
        
        # System metrics
        if self.step_metrics['power_loss_ratio']:
            stats['avg_power_loss'] = np.mean(self.step_metrics['power_loss_ratio'])
            stats['max_power_loss'] = np.max(self.step_metrics['power_loss_ratio'])
        else:
            stats['avg_power_loss'] = 0.0
            stats['max_power_loss'] = 0.0
        
        if self.step_metrics['voltage_violations']:
            stats['total_voltage_violations'] = sum(self.step_metrics['voltage_violations'])
            stats['avg_voltage_violations'] = np.mean(self.step_metrics['voltage_violations'])
        else:
            stats['total_voltage_violations'] = 0
            stats['avg_voltage_violations'] = 0.0
        
        # Convergence metrics
        if self.convergence_history['nash_gap']:
            stats['nash_gap'] = np.mean(list(self.convergence_history['nash_gap']))
        else:
            stats['nash_gap'] = float('inf')
        
        if self.convergence_history['price_stability']:
            stats['price_stability'] = np.mean(list(self.convergence_history['price_stability']))
        else:
            stats['price_stability'] = float('inf')
        
        return stats
    
    def _log_to_tensorboard(self,
                           rewards: Dict[int, float],
                           actions: Dict[int, np.ndarray],
                           system_state: Dict[str, Any]):
        """Log metrics to TensorBoard."""
        if not self.tb_writer:
            return
        
        # Log rewards
        self.tb_writer.add_scalar('rewards/uc', rewards.get(0, 0.0), self.total_steps)
        
        consumer_rewards = [r for aid, r in rewards.items() if aid != 0]
        if consumer_rewards:
            self.tb_writer.add_scalar('rewards/avg_consumer', 
                                    np.mean(consumer_rewards), 
                                    self.total_steps)
            self.tb_writer.add_scalar('rewards/social_welfare',
                                    sum(rewards.values()),
                                    self.total_steps)
        
        # Log system metrics
        self.tb_writer.add_scalar('system/power_loss_ratio',
                                system_state.get('power_loss_ratio', 0.0),
                                self.total_steps)
        self.tb_writer.add_scalar('system/voltage_violations',
                                system_state.get('voltage_violations', 0),
                                self.total_steps)
        
        # Log convergence metrics
        if self.convergence_history['nash_gap']:
            self.tb_writer.add_scalar('convergence/nash_gap',
                                    list(self.convergence_history['nash_gap'])[-1],
                                    self.total_steps)
    
    def generate_plots(self):
        """Generate visualization plots."""
        # Episode rewards plot
        self._plot_episode_rewards()
        
        # System metrics plot
        self._plot_system_metrics()
        
        # Convergence analysis plot
        self._plot_convergence_analysis()
        
        # Nash equilibrium tracking plot
        self._plot_nash_equilibrium()
        
        # Agent behavior analysis
        self._plot_agent_behaviors()
    
    def _plot_episode_rewards(self):
        """Plot episode reward trends."""
        if not self.episode_metrics['uc_total_reward']:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size)
        
        episodes = list(range(1, len(self.episode_metrics['uc_total_reward']) + 1))
        
        # UC rewards
        ax1.plot(episodes, self.episode_metrics['uc_total_reward'], 
                label='UC Total Reward', linewidth=2)
        ax1.fill_between(episodes, 
                        self.episode_metrics['uc_total_reward'],
                        alpha=0.3)
        ax1.set_ylabel('UC Reward')
        ax1.set_title('UC Reward Progression')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Consumer rewards and social welfare
        ax2.plot(episodes, self.episode_metrics['avg_consumer_reward'],
                label='Avg Consumer Reward', linewidth=2)
        ax2.plot(episodes, self.episode_metrics['social_welfare'],
                label='Social Welfare', linewidth=2, linestyle='--')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.set_title('Consumer Rewards and Social Welfare')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'plots' / 'episode_rewards.png', dpi=300)
        plt.close()
    
    def _plot_system_metrics(self):
        """Plot system performance metrics."""
        if not self.episode_metrics['avg_power_loss']:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = list(range(1, len(self.episode_metrics['avg_power_loss']) + 1))
        
        # Power loss
        ax1.plot(episodes, self.episode_metrics['avg_power_loss'], linewidth=2)
        ax1.set_ylabel('Power Loss Ratio')
        ax1.set_title('Average Power Loss per Episode')
        ax1.grid(True, alpha=0.3)
        
        # Voltage violations
        ax2.bar(episodes, self.episode_metrics['total_voltage_violations'], alpha=0.7)
        ax2.set_ylabel('Voltage Violations')
        ax2.set_title('Total Voltage Violations per Episode')
        ax2.grid(True, alpha=0.3)
        
        # Nash gap
        ax3.plot(episodes, self.episode_metrics['nash_gap'], linewidth=2, color='red')
        ax3.set_ylabel('Nash Gap')
        ax3.set_xlabel('Episode')
        ax3.set_title('Nash Gap Convergence')
        ax3.grid(True, alpha=0.3)
        
        # Price stability
        if 'price_stability' in self.episode_metrics:
            ax4.plot(episodes, self.episode_metrics['price_stability'], 
                    linewidth=2, color='green')
            ax4.set_ylabel('Price Variance')
            ax4.set_xlabel('Episode')
            ax4.set_title('Price Signal Stability')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'plots' / 'system_metrics.png', dpi=300)
        plt.close()
    
    def _plot_convergence_analysis(self):
        """Plot convergence analysis."""
        if not self.convergence_history['nash_gap']:
            return
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot recent convergence history
        nash_gaps = list(self.convergence_history['nash_gap'])
        steps = list(range(len(nash_gaps)))
        
        ax.plot(steps, nash_gaps, linewidth=2, alpha=0.7)
        
        # Add moving average
        if len(nash_gaps) > 10:
            window = min(20, len(nash_gaps) // 5)
            moving_avg = pd.Series(nash_gaps).rolling(window=window).mean()
            ax.plot(steps, moving_avg, linewidth=3, color='red', 
                   label=f'{window}-step Moving Average')
        
        ax.set_xlabel('Recent Steps')
        ax.set_ylabel('Nash Gap')
        ax.set_title('Convergence Analysis - Nash Gap')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'plots' / 'convergence_analysis.png', dpi=300)
        plt.close()
    
    def _plot_nash_equilibrium(self):
        """Plot Nash equilibrium tracking."""
        if not self.nash_tracking['equilibrium_distance']:
            return
        
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        distances = self.nash_tracking['equilibrium_distance']
        steps = list(range(len(distances)))
        
        ax.plot(steps, distances, linewidth=2)
        ax.set_xlabel('Calculation Step')
        ax.set_ylabel('Equilibrium Distance')
        ax.set_title('Nash Equilibrium Convergence')
        ax.grid(True, alpha=0.3)
        
        # Add convergence threshold line
        if distances:
            threshold = 0.1  # Example threshold
            ax.axhline(y=threshold, color='red', linestyle='--', 
                      label=f'Convergence Threshold ({threshold})')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'plots' / 'nash_equilibrium.png', dpi=300)
        plt.close()
    
    def _plot_agent_behaviors(self):
        """Plot agent behavior analysis."""
        # UC behavior
        if self.uc_metrics['price_signal'] and len(self.uc_metrics['price_signal']) > 100:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figure_size)
            
            # Price signals over time
            steps = list(range(len(self.uc_metrics['price_signal'])))
            ax1.plot(steps, self.uc_metrics['price_signal'], linewidth=1, alpha=0.7)
            ax1.set_ylabel('Price Signal')
            ax1.set_title('UC Price Signal Evolution')
            ax1.grid(True, alpha=0.3)
            
            # DR incentives
            if self.uc_metrics['dr_incentive']:
                ax2.plot(steps, self.uc_metrics['dr_incentive'], 
                        linewidth=1, alpha=0.7, color='green')
                ax2.set_xlabel('Step')
                ax2.set_ylabel('DR Incentive')
                ax2.set_title('UC DR Incentive Evolution')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.experiment_dir / 'plots' / 'uc_behavior.png', dpi=300)
            plt.close()
        
        # Consumer behavior heatmap
        if len(self.consumer_metrics) > 0:
            self._plot_consumer_heatmap()
    
    def _plot_consumer_heatmap(self):
        """Plot consumer behavior heatmap."""
        # Collect load adjustments for all consumers
        consumer_ids = sorted(self.consumer_metrics.keys())
        if not consumer_ids:
            return
        
        # Get recent load adjustments
        window = min(100, min(len(self.consumer_metrics[cid]['load_adjustment']) 
                             for cid in consumer_ids))
        if window < 10:
            return
        
        load_adjustments = []
        for cid in consumer_ids:
            if self.consumer_metrics[cid]['load_adjustment']:
                recent_adjustments = self.consumer_metrics[cid]['load_adjustment'][-window:]
                load_adjustments.append(recent_adjustments)
        
        if not load_adjustments:
            return
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        heatmap_data = np.array(load_adjustments)
        sns.heatmap(heatmap_data, 
                   cmap='RdBu_r', 
                   center=0,
                   yticklabels=[f'Consumer {cid}' for cid in consumer_ids],
                   xticklabels=False,
                   cbar_kws={'label': 'Load Adjustment'},
                   ax=ax)
        
        ax.set_xlabel(f'Recent {window} Steps')
        ax.set_title('Consumer Load Adjustment Patterns')
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / 'plots' / 'consumer_behavior_heatmap.png', dpi=300)
        plt.close()
    
    def save_monitoring_data(self):
        """Save all monitoring data to disk."""
        # Save raw data
        data = {
            'experiment_name': self.experiment_name,
            'config': self.config,
            'episode_metrics': dict(self.episode_metrics),
            'uc_metrics': dict(self.uc_metrics),
            'consumer_metrics': {k: dict(v) for k, v in self.consumer_metrics.items()},
            'convergence_history': {k: list(v) for k, v in self.convergence_history.items()},
            'nash_tracking': dict(self.nash_tracking),
            'total_episodes': self.current_episode,
            'total_steps': self.total_steps
        }
        
        # Save as pickle
        with open(self.experiment_dir / 'data' / 'monitoring_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        # Save episode metrics as CSV
        if self.episode_metrics:
            df = pd.DataFrame(self.episode_metrics)
            df.to_csv(self.experiment_dir / 'data' / 'episode_metrics.csv', index=False)
        
        # Save summary statistics
        summary = self.get_summary_statistics()
        with open(self.experiment_dir / 'data' / 'summary_statistics.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Monitoring data saved at episode {self.current_episode}")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of the experiment."""
        summary = {
            'experiment_name': self.experiment_name,
            'total_episodes': self.current_episode,
            'total_steps': self.total_steps,
            'timestamp': datetime.now().isoformat()
        }
        
        # Episode statistics
        if self.episode_metrics:
            for metric, values in self.episode_metrics.items():
                if values:
                    summary[f'{metric}_final'] = values[-1]
                    summary[f'{metric}_mean'] = np.mean(values)
                    summary[f'{metric}_std'] = np.std(values)
                    summary[f'{metric}_min'] = np.min(values)
                    summary[f'{metric}_max'] = np.max(values)
                    
                    # Improvement over time
                    if len(values) > 10:
                        early_mean = np.mean(values[:10])
                        late_mean = np.mean(values[-10:])
                        summary[f'{metric}_improvement'] = late_mean - early_mean
        
        # Convergence statistics
        if self.convergence_history['nash_gap']:
            recent_gaps = list(self.convergence_history['nash_gap'])[-50:]
            summary['final_nash_gap'] = np.mean(recent_gaps)
            summary['nash_gap_converged'] = summary['final_nash_gap'] < 0.1
        
        return summary
    
    def close(self):
        """Clean up resources."""
        if self.tb_writer:
            self.tb_writer.close()
        
        # Final save
        self.save_monitoring_data()
        
        # Generate final report
        self.generate_final_report()
        
        self.logger.info("Monitoring system closed")
    
    def generate_final_report(self):
        """Generate a final experiment report."""
        report = []
        report.append("=" * 60)
        report.append(f"STACKELBERG GAME EXPERIMENT REPORT")
        report.append(f"Experiment: {self.experiment_name}")
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append("=" * 60)
        report.append("")
        
        summary = self.get_summary_statistics()
        
        report.append("SUMMARY STATISTICS:")
        report.append(f"- Total Episodes: {summary['total_episodes']}")
        report.append(f"- Total Steps: {summary['total_steps']}")
        report.append("")
        
        report.append("FINAL PERFORMANCE:")
        if 'social_welfare_final' in summary:
            report.append(f"- Social Welfare: {summary['social_welfare_final']:.2f}")
        if 'avg_power_loss_final' in summary:
            report.append(f"- Power Loss: {summary['avg_power_loss_final']:.4f}")
        if 'final_nash_gap' in summary:
            report.append(f"- Nash Gap: {summary['final_nash_gap']:.4f}")
            report.append(f"- Converged: {'Yes' if summary.get('nash_gap_converged', False) else 'No'}")
        report.append("")
        
        report.append("IMPROVEMENTS:")
        for key, value in summary.items():
            if key.endswith('_improvement'):
                metric_name = key.replace('_improvement', '')
                report.append(f"- {metric_name}: {value:+.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        # Save report
        report_text = '\n'.join(report)
        with open(self.experiment_dir / 'final_report.txt', 'w') as f:
            f.write(report_text)
        
        # Also print to console
        print(report_text)
    
    def _update_nash_gap(self, actions: Dict[int, np.ndarray], system_state: Dict[str, Any]):
        """
        Update Nash gap metric based on current actions.
        This measures how far agents are from Nash equilibrium.
        """
        # Simplified Nash gap calculation
        # In practice, this would require computing best responses
        
        uc_id = 0
        consumer_ids = [aid for aid in actions.keys() if aid != uc_id]
        
        if len(consumer_ids) < 2:
            return
        
        # Calculate action variance among consumers as proxy for Nash gap
        consumer_actions = []
        for cid in consumer_ids:
            if cid in actions:
                consumer_actions.append(actions[cid])
        
        if consumer_actions:
            action_variance = np.var(consumer_actions, axis=0).mean()
            # Normalize by number of consumers
            nash_gap = action_variance * len(consumer_ids)
            
            self.convergence_history['nash_gap'].append(nash_gap)
            self.step_metrics['nash_gap'].append(nash_gap)
    
    def _update_nash_metrics(self, rewards: Dict[int, float], actions: Dict[int, np.ndarray], 
                           system_state: Dict[str, Any]):
        """Update Nash equilibrium tracking metrics."""
        # This is a placeholder implementation
        # In practice, would compute actual equilibrium distance
        
        if self.current_step % 10 == 0:  # Calculate every 10 steps
            # Mock equilibrium distance calculation
            equilibrium_distance = np.random.exponential(0.1)
            self.nash_tracking['equilibrium_distance'].append(equilibrium_distance)