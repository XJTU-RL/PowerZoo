# -*- coding: utf-8 -*-
"""
Episode生成和管理模块
负责从数据生成训练episodes
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from envs.powerzoo_llm.data_process.loadprofile_dss_parser import Constants


class EpisodeGenerator:
    """Episode生成器"""
    
    def __init__(self, dss_folder_path: str, loadshape_path: str, 
                 irradiation_path: str, temperature_path: str,
                 steps: int, load_names: List[str]):
        """
        初始化Episode生成器
        
        Args:
            dss_folder_path: DSS文件夹路径
            loadshape_path: 负载形状路径
            irradiation_path: 辐照度路径
            temperature_path: 温度路径
            steps: 每个episode的步数
            load_names: 负载名称列表
        """
        self.dss_folder_path = dss_folder_path
        self.loadshape_path = loadshape_path
        self.irradiation_path = irradiation_path
        self.temperature_path = temperature_path
        self.steps = steps
        self.load_names = load_names
    
    def generate_from_existing_files(self, target_episodes: int = 30, 
                                    pv_threshold: float = 0.05, scale: float = 1.0) -> int:
        """从loadshape文件夹中的现有数据文件生成episodes
        
        Args:
            target_episodes: 目标episode数量
            pv_threshold: 光伏有效阈值（标准化值，低于此值认为无光伏）
            scale: 负载数据缩放因子，用于调整负载强度
            
        Returns:
            int: 实际生成的episode数量
        """
        print(f"🎯 开始从loadshape文件夹中的现有数据生成 {target_episodes} 个episodes...")
        
        # 1. 自动发现数据文件
        load_files, pv_files, temp_files = self._discover_existing_data_files()
        
        if not load_files:
            print("❌ 未找到负荷数据文件")
            return 0
            
        if not pv_files or not temp_files:
            print("❌ 未找到光伏或温度数据文件")
            return 0
            
        print(f"📁 发现数据文件:")
        print(f"   负荷文件: {len(load_files)}个")
        print(f"   光伏文件: {len(pv_files)}个") 
        print(f"   温度文件: {len(temp_files)}个")
        
        # 2. 分析光伏数据，识别有效时间段
        pv_active_periods = self._identify_pv_active_periods(pv_files, pv_threshold)
        print(f"📊 识别到 {len(pv_active_periods)} 个光伏活跃时段")
        
        if len(pv_active_periods) == 0:
            print("❌ 未找到有效的光伏活跃时段")
            return 0
        
        # 3. 分类采样策略
        episode_samples = self._smart_episode_sampling(pv_active_periods, target_episodes)
        print(f"🎲 采样策略：{len(episode_samples)} 个时间窗口")
        
        # 4. 提取对应的负载、光伏、温度数据
        episodes_created = 0
        for i, sample in enumerate(episode_samples):
            success = self._extract_and_save_episode(
                episode_idx=i,
                sample_info=sample,
                load_files=load_files,
                pv_files=pv_files,
                temp_files=temp_files,
                scale=scale
            )
            if success:
                episodes_created += 1
                if episodes_created % 5 == 0:
                    print(f"✅ 已创建 {episodes_created}/{target_episodes} 个episodes")
        
        print(f"🎉 智能采样完成！成功创建 {episodes_created} 个真实时间对应的episodes")
        return episodes_created
    
    def _discover_existing_data_files(self) -> Tuple[List[str], List[str], List[str]]:
        """自动发现loadshape文件夹中的数据文件"""
        base_path = os.path.join(self.dss_folder_path, 'loadshape')
        
        load_files = []
        pv_files = []
        temp_files = []
        
        if not os.path.exists(base_path):
            return load_files, pv_files, temp_files
            
        for filename in os.listdir(base_path):
            filepath = os.path.join(base_path, filename)
            
            if not filename.endswith('.csv'):
                continue
                
            # 识别负荷数据文件: month_XX_0000_to_2359.csv
            if filename.startswith('month_') and '_0000_to_2359' in filename:
                load_files.append(filepath)
            
            # 识别光伏数据文件: irrad_2025_XX_0500_to_1659.csv
            elif filename.startswith('irrad_') and '_0500_to_1659' in filename:
                pv_files.append(filepath)
            
            # 识别温度数据文件: temperature_2025_XX_0500_to_1659.csv
            elif filename.startswith('temperature_') and '_0500_to_1659' in filename:
                temp_files.append(filepath)
        
        # 排序确保文件顺序一致
        load_files.sort()
        pv_files.sort()
        temp_files.sort()
        
        return load_files, pv_files, temp_files
    
    def _identify_pv_active_periods(self, pv_files: List[str], 
                                   threshold: float) -> List[Dict]:
        """从现有光伏文件中识别活跃时段"""
        active_periods = []
        
        for file_idx, pv_path in enumerate(pv_files):
            if not os.path.exists(pv_path):
                continue
                
            # 从文件名提取月份信息
            filename = os.path.basename(pv_path)
            month_str = filename.split('_')[2] if len(filename.split('_')) >= 3 else f"{file_idx+1:02d}"
            month_num = int(month_str) if month_str.isdigit() else file_idx + 1
                
            # 读取5-17点光伏数据
            pv_data = pd.read_csv(pv_path, header=None).values.flatten()
            
            # 每天720分钟(12小时×60分钟)
            days_in_month = len(pv_data) // 720
            print(f"📅 月份{month_num}: {days_in_month}天数据，共{len(pv_data)}个数据点")
            
            for day in range(days_in_month):
                day_start_idx = day * 720
                day_data = pv_data[day_start_idx:day_start_idx + 720]
                
                # 在5-17点数据中查找6小时连续有效光伏时段
                for start_hour_offset in range(0, 7):  # 0-6小时开始的6小时窗口(对应5-11点)
                    start_minute = start_hour_offset * 60
                    window_data = day_data[start_minute:start_minute + 360]  # 6小时=360分钟
                    
                    if len(window_data) < 360:  # 数据不足
                        continue
                    
                    # 检查光伏有效性
                    valid_ratio = np.sum(window_data > threshold) / len(window_data)
                    avg_irradiance = np.mean(window_data)
                    std_irradiance = np.std(window_data)
                    
                    if valid_ratio > 0.3:  # 至少30%时间有光伏
                        actual_start_hour = 5 + start_hour_offset  # 转回实际小时
                        period_info = {
                            'month': month_num,
                            'day': day + 1,
                            'start_hour': actual_start_hour,
                            'end_hour': actual_start_hour + 6,
                            'start_idx_in_5_17_data': day_start_idx + start_minute,
                            'start_idx_in_full_day': day * 1440 + actual_start_hour * 60,
                            'avg_irradiance': avg_irradiance,
                            'std_irradiance': std_irradiance,
                            'valid_ratio': valid_ratio,
                            'weather_type': self._classify_weather_from_pv(window_data),
                            'file_idx': file_idx  # 文件索引，用于后续数据提取
                        }
                        active_periods.append(period_info)
        
        return active_periods
    
    def _classify_weather_from_pv(self, pv_window: np.ndarray) -> str:
        """根据光伏数据分类天气类型"""
        mean_irr = np.mean(pv_window)
        std_irr = np.std(pv_window)
        
        if mean_irr > 0.8:
            return "sunny" if std_irr < 0.2 else "sunny_variable"
        elif mean_irr > 0.5:
            return "partly_cloudy" if std_irr < 0.3 else "variable"
        else:
            return "overcast"
    
    def _smart_episode_sampling(self, active_periods: List[Dict], 
                               target_episodes: int) -> List[Dict]:
        """智能采样策略"""
        if len(active_periods) <= target_episodes:
            return active_periods
        
        # 按天气类型分层采样
        weather_groups = {}
        for period in active_periods:
            weather = period['weather_type']
            if weather not in weather_groups:
                weather_groups[weather] = []
            weather_groups[weather].append(period)
        
        # 分配采样数量
        samples = []
        remaining_episodes = target_episodes
        
        for weather_type, periods in weather_groups.items():
            # 每种天气类型至少采样1个
            if remaining_episodes > 0:
                sample_count = min(len(periods), max(1, remaining_episodes // len(weather_groups)))
                selected = np.random.choice(periods, sample_count, replace=False)
                samples.extend(selected)
                remaining_episodes -= sample_count
        
        # 如果还有剩余名额，随机补充
        if remaining_episodes > 0:
            all_remaining = [p for p in active_periods if p not in samples]
            additional = np.random.choice(all_remaining, 
                                        min(remaining_episodes, len(all_remaining)), 
                                        replace=False)
            samples.extend(additional)
        
        return samples[:target_episodes]
    
    def _extract_and_save_episode(self, episode_idx: int, sample_info: Dict,
                                 load_files: List[str], pv_files: List[str], 
                                 temp_files: List[str], scale: float = 1.0) -> bool:
        """从现有文件中提取并保存episode数据"""
        try:
            month_num = sample_info['month']
            file_idx = sample_info.get('file_idx', 0)
            
            # 创建episode目录
            episode_folder = str(episode_idx).zfill(Constants.EPISODE_FOLDER_DIGITS)
            
            # 1. 从负荷数据中提取对应的5-17点时段数据
            load_episode_dir = os.path.join(self.loadshape_path, episode_folder)
            os.makedirs(load_episode_dir, exist_ok=True)
            
            # 尝试找到对应月份的负荷文件
            load_file = None
            for lf in load_files:
                if f"month_{month_num:02d}_" in os.path.basename(lf):
                    load_file = lf
                    break
            
            if not load_file and load_files:
                # 如果找不到对应月份，使用第一个可用文件
                load_file = load_files[0]
                print(f"⚠️  未找到月份{month_num}的负荷文件，使用 {os.path.basename(load_file)}")
            
            if load_file:
                full_load_data = pd.read_csv(load_file, header=None).values.flatten()
                full_day_start_idx = sample_info['start_idx_in_full_day']
                
                # 生成负荷数据（对所有负载使用相同的时间序列数据）
                # 从full_day_start_idx开始提取360个点
                base_load_series = full_load_data[full_day_start_idx:full_day_start_idx + self.steps]
                
                if len(base_load_series) < self.steps:
                    # 如果数据不足，尝试循环填充
                    print(f"⚠️  负荷数据不足，需要{self.steps}个点，实际{len(base_load_series)}个，尝试循环填充")
                    if len(base_load_series) > 0:
                        # 循环填充数据
                        repeats = (self.steps // len(base_load_series)) + 1
                        base_load_series = np.tile(base_load_series, repeats)[:self.steps]
                    else:
                        # 没有数据，使用默认值
                        print(f"⚠️  无法获取负荷数据，使用默认值")
                        base_load_series = np.ones(self.steps) * 0.5  # 默认负荷值
                
                # 为每个负载创建数据（可以添加一些随机变化）
                for i, load_name in enumerate(self.load_names):
                    # 为每个负载添加一些随机变化（±10%）
                    variation = 1.0 + (np.random.rand() - 0.5) * 0.2
                    load_series = base_load_series * variation
                    
                    # 应用缩放因子
                    if scale != 1.0:
                        load_series = load_series * scale
                    
                    csv_path = os.path.join(load_episode_dir, f'{load_name}.csv')
                    pd.DataFrame(load_series).to_csv(csv_path, header=False, index=False)
            
            # 2. 从光伏文件中提取数据
            pv_episode_dir = os.path.join(self.irradiation_path, episode_folder)
            os.makedirs(pv_episode_dir, exist_ok=True)
            
            if file_idx < len(pv_files):
                pv_5_17_data = pd.read_csv(pv_files[file_idx], header=None).values.flatten()
                pv_start_idx = sample_info['start_idx_in_5_17_data']
                pv_series = pv_5_17_data[pv_start_idx:pv_start_idx + self.steps]
                
                if len(pv_series) == self.steps:
                    pv_path = os.path.join(pv_episode_dir, '20250307_irradiance.csv')
                    pd.DataFrame(pv_series).to_csv(pv_path, header=False, index=False)
                else:
                    print(f"⚠️  光伏数据不足: 需要{self.steps}个点，实际{len(pv_series)}个")
                    return False
            
            # 3. 从温度文件中提取数据
            temp_episode_dir = os.path.join(self.temperature_path, episode_folder)
            os.makedirs(temp_episode_dir, exist_ok=True)
            
            if file_idx < len(temp_files):
                temp_5_17_data = pd.read_csv(temp_files[file_idx], header=None).values.flatten()
                temp_start_idx = sample_info['start_idx_in_5_17_data']
                temp_series = temp_5_17_data[temp_start_idx:temp_start_idx + self.steps]
                
                if len(temp_series) == self.steps:
                    temp_path = os.path.join(temp_episode_dir, '20250307_temperature.csv')
                    pd.DataFrame(temp_series).to_csv(temp_path, header=False, index=False)
                else:
                    print(f"⚠️  温度数据不足: 需要{self.steps}个点，实际{len(temp_series)}个")
                    return False
            
            print(f"✅ Episode {episode_idx} 创建成功 (月{sample_info['month']}, 天{sample_info['day']}, {sample_info['start_hour']}-{sample_info['end_hour']}点, {sample_info['weather_type']})")
            return True
            
        except Exception as e:
            print(f"❌ Episode {episode_idx} 创建失败: {e}")
            return False