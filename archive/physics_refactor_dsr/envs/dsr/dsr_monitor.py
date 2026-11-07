# -*- coding: utf-8 -*-
"""
DSR Environment Monitor
配电网恢复环境监控可视化模块
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import networkx as nx
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import os

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class DSRMonitor:
    """DSR环境监控器，提供实时可视化和分析功能"""
    
    def __init__(self, save_dir: str, update_interval: int = 10):
        """
        初始化监控器
        
        Args:
            save_dir: 保存可视化结果的目录
            update_interval: 更新间隔（步数）
        """
        self.save_dir = save_dir
        self.update_interval = update_interval
        
        # 创建保存目录
        self.vis_dir = os.path.join(save_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # 数据缓存
        self.data_cache = {
            'voltage_profiles': [],
            'line_loadings': [],
            'restoration_progress': [],
            'agent_actions': [],
            'topology_states': [],
        }
        
        # 设置绘图风格
        plt.style.use('seaborn-v0_8-darkgrid')
        self.color_palette = sns.color_palette("husl", 8)
        
    def update(self, info: Dict[str, Any], step: int):
        """更新监控数据"""
        # 缓存数据
        self._cache_data(info, step)
        
        # 定期生成可视化
        if step % self.update_interval == 0:
            self._generate_visualizations(step)
    
    def _cache_data(self, info: Dict[str, Any], step: int):
        """缓存监控数据"""
        # 电压分布
        if 'voltage_profile' in info:
            self.data_cache['voltage_profiles'].append({
                'step': step,
                'profile': info['voltage_profile']
            })
        
        # 线路负载
        if 'line_loading' in info:
            self.data_cache['line_loadings'].append({
                'step': step,
                'loading': info['line_loading']
            })
        
        # 恢复进度
        if 'restoration_progress' in info:
            self.data_cache['restoration_progress'].append({
                'step': step,
                'progress': info['restoration_progress']
            })
        
        # 智能体动作
        if 'agent_behavior' in info:
            self.data_cache['agent_actions'].append({
                'step': step,
                'actions': info['agent_behavior'].get('actions', []),
                'results': info['agent_behavior'].get('action_results', {})
            })
        
        # 拓扑状态
        if 'topology_metrics' in info:
            self.data_cache['topology_states'].append({
                'step': step,
                'metrics': info['topology_metrics']
            })
    
    def _generate_visualizations(self, step: int):
        """生成可视化图表"""
        # 生成系统状态总览
        self._plot_system_overview(step)
        
        # 生成电压热图
        self._plot_voltage_heatmap(step)
        
        # 生成线路负载图
        self._plot_line_loading(step)
        
        # 生成恢复进度图
        self._plot_restoration_progress(step)
        
        # 生成智能体行为分析
        self._plot_agent_behavior(step)
    
    def _plot_system_overview(self, step: int):
        """绘制系统状态总览"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'DSR System Overview - Step {step}', fontsize=16)
        
        # 1. 恢复进度时间线
        if self.data_cache['restoration_progress']:
            ax = axes[0, 0]
            steps = [d['step'] for d in self.data_cache['restoration_progress']]
            ratios = [d['progress']['completion_ratio'] for d in self.data_cache['restoration_progress']]
            ax.plot(steps, ratios, 'b-', linewidth=2)
            ax.fill_between(steps, ratios, alpha=0.3)
            ax.set_xlabel('Step')
            ax.set_ylabel('Restoration Ratio')
            ax.set_title('Load Restoration Progress')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
        
        # 2. 电压违规统计
        if self.data_cache['voltage_profiles']:
            ax = axes[0, 1]
            violations = []
            for data in self.data_cache['voltage_profiles'][-10:]:  # 最近10步
                count = sum(1 for bus_data in data['profile'].values() 
                          if bus_data.get('violation', False))
                violations.append(count)
            
            ax.bar(range(len(violations)), violations, color='red', alpha=0.7)
            ax.set_xlabel('Recent Steps')
            ax.set_ylabel('Voltage Violations')
            ax.set_title('Voltage Violation Count (Last 10 Steps)')
        
        # 3. 拓扑连通性
        if self.data_cache['topology_states']:
            ax = axes[1, 0]
            steps = [d['step'] for d in self.data_cache['topology_states']]
            components = [d['metrics']['total_components'] for d in self.data_cache['topology_states']]
            powered = [d['metrics']['powered_components'] for d in self.data_cache['topology_states']]
            
            ax.plot(steps, components, 'r-', label='Total Components', linewidth=2)
            ax.plot(steps, powered, 'g-', label='Powered Components', linewidth=2)
            ax.fill_between(steps, powered, alpha=0.3, color='green')
            ax.set_xlabel('Step')
            ax.set_ylabel('Component Count')
            ax.set_title('Network Topology Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 4. 智能体动作效率
        if self.data_cache['agent_actions']:
            ax = axes[1, 1]
            recent_actions = self.data_cache['agent_actions'][-20:]  # 最近20步
            
            switch_success = []
            load_success = []
            
            for data in recent_actions:
                results = data['results']
                if 'switch' in results and results['switch']['attempted'] > 0:
                    rate = results['switch']['successful'] / results['switch']['attempted']
                    switch_success.append(rate)
                
                if 'load' in results and results['load']['attempted_restore'] > 0:
                    rate = results['load']['successful_restore'] / results['load']['attempted_restore']
                    load_success.append(rate)
            
            if switch_success:
                ax.plot(switch_success, 'b-o', label='Switch Success Rate', markersize=4)
            if load_success:
                ax.plot(load_success, 'g-s', label='Load Restore Success Rate', markersize=4)
            
            ax.set_xlabel('Recent Actions')
            ax.set_ylabel('Success Rate')
            ax.set_title('Agent Action Success Rates')
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f'system_overview_step_{step}.png'), dpi=150)
        plt.close(fig)
    
    def _plot_voltage_heatmap(self, step: int):
        """绘制电压热图"""
        if not self.data_cache['voltage_profiles']:
            return
        
        latest_data = self.data_cache['voltage_profiles'][-1]
        voltage_profile = latest_data['profile']
        
        # 提取电压数据
        bus_names = list(voltage_profile.keys())[:50]  # 限制显示前50个母线
        voltages = []
        
        for bus in bus_names:
            if 'avg' in voltage_profile[bus]:
                voltages.append(voltage_profile[bus]['avg'])
            else:
                voltages.append(1.0)  # 默认值
        
        # 创建热图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 将电压数组转换为2D以便显示
        voltage_matrix = np.array(voltages).reshape(-1, 1)
        
        # 创建颜色映射
        cmap = plt.cm.RdYlGn
        im = ax.imshow(voltage_matrix.T, cmap=cmap, aspect='auto', 
                      vmin=0.9, vmax=1.1, interpolation='nearest')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Voltage (p.u.)')
        
        # 设置标签
        ax.set_xlabel('Bus Index')
        ax.set_title(f'Bus Voltage Profile - Step {step}')
        ax.set_yticks([])
        
        # 添加违规标记
        for i, (bus, v) in enumerate(zip(bus_names, voltages)):
            if v < 0.95 or v > 1.05:
                ax.axvline(x=i, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f'voltage_heatmap_step_{step}.png'), dpi=150)
        plt.close(fig)
    
    def _plot_line_loading(self, step: int):
        """绘制线路负载分布"""
        if not self.data_cache['line_loadings']:
            return
        
        latest_data = self.data_cache['line_loadings'][-1]
        line_loading = latest_data['loading']
        
        # 提取负载数据
        loading_values = []
        overloaded_lines = []
        
        for line_name, data in line_loading.items():
            if data.get('status') == 'active':
                loading = data.get('loading', 0)
                loading_values.append(loading)
                if loading > 100:
                    overloaded_lines.append((line_name, loading))
        
        if not loading_values:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. 负载分布直方图
        ax1.hist(loading_values, bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax1.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Overload Threshold')
        ax1.set_xlabel('Loading (%)')
        ax1.set_ylabel('Number of Lines')
        ax1.set_title(f'Line Loading Distribution - Step {step}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 过载线路柱状图
        if overloaded_lines:
            overloaded_lines.sort(key=lambda x: x[1], reverse=True)
            top_overloaded = overloaded_lines[:10]  # 显示前10条过载最严重的线路
            
            lines, loadings = zip(*top_overloaded)
            y_pos = np.arange(len(lines))
            
            bars = ax2.barh(y_pos, loadings, color='red', alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels([l[:15] for l in lines])  # 截断长名称
            ax2.set_xlabel('Loading (%)')
            ax2.set_title('Top Overloaded Lines')
            ax2.axvline(x=100, color='black', linestyle='--', linewidth=1)
            
            # 添加数值标签
            for i, (bar, loading) in enumerate(zip(bars, loadings)):
                ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                        f'{loading:.0f}%', va='center')
        else:
            ax2.text(0.5, 0.5, 'No Overloaded Lines', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14)
            ax2.set_title('Top Overloaded Lines')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f'line_loading_step_{step}.png'), dpi=150)
        plt.close(fig)
    
    def _plot_restoration_progress(self, step: int):
        """绘制恢复进度详情"""
        if not self.data_cache['restoration_progress']:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Restoration Progress Analysis - Step {step}', fontsize=16)
        
        # 准备数据
        steps = []
        total_restoration = []
        priority_1 = []
        priority_2 = []
        priority_3 = []
        restoration_rates = []
        
        for data in self.data_cache['restoration_progress']:
            steps.append(data['step'])
            progress = data['progress']
            
            total_restoration.append(progress.get('completion_ratio', 0))
            restoration_rates.append(progress.get('restoration_rate', 0))
            
            if 'priority_restoration' in progress:
                prio_data = progress['priority_restoration']
                priority_1.append(prio_data.get(1, {}).get('restoration_ratio', 0))
                priority_2.append(prio_data.get(2, {}).get('restoration_ratio', 0))
                priority_3.append(prio_data.get(3, {}).get('restoration_ratio', 0))
        
        # 1. 总体恢复进度
        ax1.plot(steps, total_restoration, 'b-', linewidth=3, label='Total')
        ax1.fill_between(steps, total_restoration, alpha=0.3)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Restoration Ratio')
        ax1.set_title('Overall Restoration Progress')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 优先级恢复对比
        if priority_1:
            ax2.plot(steps, priority_1, 'r-', linewidth=2, label='Priority 1')
            ax2.plot(steps, priority_2, 'g-', linewidth=2, label='Priority 2')
            ax2.plot(steps, priority_3, 'b-', linewidth=2, label='Priority 3')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Restoration Ratio')
            ax2.set_title('Restoration by Priority')
            ax2.set_ylim(0, 1.1)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. 恢复速率
        if restoration_rates:
            ax3.bar(steps[-20:], restoration_rates[-20:], color='green', alpha=0.7)
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Loads Restored')
            ax3.set_title('Restoration Rate (Last 20 Steps)')
            ax3.grid(True, alpha=0.3)
        
        # 4. 累计恢复功率
        if self.data_cache['restoration_progress']:
            latest = self.data_cache['restoration_progress'][-1]['progress']
            if 'priority_restoration' in latest:
                priorities = ['Priority 1', 'Priority 2', 'Priority 3']
                restored_kw = []
                total_kw = []
                
                for i in range(1, 4):
                    prio_data = latest['priority_restoration'].get(i, {})
                    restored_kw.append(prio_data.get('restored_kw', 0))
                    total_kw.append(prio_data.get('total_kw', 0))
                
                x = np.arange(len(priorities))
                width = 0.35
                
                bars1 = ax4.bar(x - width/2, restored_kw, width, label='Restored', color='green', alpha=0.7)
                bars2 = ax4.bar(x + width/2, total_kw, width, label='Total', color='blue', alpha=0.7)
                
                ax4.set_xlabel('Priority Level')
                ax4.set_ylabel('Power (kW)')
                ax4.set_title('Power Restoration by Priority')
                ax4.set_xticks(x)
                ax4.set_xticklabels(priorities)
                ax4.legend()
                
                # 添加数值标签
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax4.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.0f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f'restoration_progress_step_{step}.png'), dpi=150)
        plt.close(fig)
    
    def _plot_agent_behavior(self, step: int):
        """绘制智能体行为分析"""
        if not self.data_cache['agent_actions']:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Agent Behavior Analysis - Step {step}', fontsize=16)
        
        # 统计最近的动作
        recent_actions = self.data_cache['agent_actions'][-50:]  # 最近50步
        
        # 1. 动作类型分布
        action_counts = {'switch': 0, 'pv_adjust': 0, 'load_restore': 0}
        
        for data in recent_actions:
            results = data['results']
            if 'switch' in results and results['switch']['attempted'] > 0:
                action_counts['switch'] += 1
            if 'pv' in results and abs(results['pv']['total_power_change']) > 0:
                action_counts['pv_adjust'] += 1
            if 'load' in results and results['load']['attempted_restore'] > 0:
                action_counts['load_restore'] += results['load']['attempted_restore']
        
        # 饼图
        if sum(action_counts.values()) > 0:
            labels = ['Switch Operations', 'PV Adjustments', 'Load Restorations']
            values = [action_counts['switch'], action_counts['pv_adjust'], action_counts['load_restore']]
            colors = ['red', 'yellow', 'green']
            
            wedges, texts, autotexts = ax1.pie(values, labels=labels, colors=colors, 
                                               autopct='%1.1f%%', startangle=90)
            ax1.set_title('Action Type Distribution (Last 50 Steps)')
        else:
            ax1.text(0.5, 0.5, 'No Actions Recorded', 
                    transform=ax1.transAxes, ha='center', va='center', fontsize=14)
            ax1.set_title('Action Type Distribution')
        
        # 2. PV功率调整趋势
        pv_adjustments = []
        action_steps = []
        
        for data in recent_actions:
            results = data['results']
            if 'pv' in results:
                total_change = results['pv'].get('total_power_change', 0)
                if abs(total_change) > 0:
                    pv_adjustments.append(total_change)
                    action_steps.append(data['step'])
        
        if pv_adjustments:
            ax2.plot(action_steps, pv_adjustments, 'yo-', markersize=6, linewidth=2)
            ax2.fill_between(action_steps, pv_adjustments, alpha=0.3, color='yellow')
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Power Change (kW)')
            ax2.set_title('PV Power Adjustments')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No PV Adjustments', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=14)
            ax2.set_title('PV Power Adjustments')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.vis_dir, f'agent_behavior_step_{step}.png'), dpi=150)
        plt.close(fig)
    
    def create_animation(self, output_file: str = 'dsr_restoration.mp4'):
        """创建恢复过程动画"""
        if not self.data_cache['restoration_progress']:
            print("没有足够的数据创建动画")
            return
        
        # 这里可以实现更复杂的动画，展示网络拓扑的演化
        # 由于实现复杂，这里仅提供框架
        print(f"动画功能待实现，将保存到: {output_file}")
    
    def generate_final_report(self):
        """生成最终的可视化报告"""
        print("生成最终可视化报告...")
        
        # 生成最终的综合图表
        if self.data_cache['restoration_progress']:
            last_step = self.data_cache['restoration_progress'][-1]['step']
            self._plot_system_overview(last_step)
            self._plot_restoration_progress(last_step)
            print(f"最终报告已保存到: {self.vis_dir}")