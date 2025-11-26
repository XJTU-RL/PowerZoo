# -*- coding: utf-8 -*-
"""
LoadProfile核心类
精简版的负载配置文件管理器
"""

import os
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

from envs.smartgrid.data_process.loadprofile_dss_parser import Constants, DSSFileParser
from envs.smartgrid.data_process.loadprofile_episode import EpisodeGenerator
from envs.smartgrid.data_process.loadprofile_config import ConfigGenerator


class LoadProfile:
    """负载配置文件管理器 - 精简版"""
    
    def __init__(self, steps: int, dss_folder_path: str, dss_file: str, 
                 worker_idx: Optional[int] = None,
                 pv_data_source: Optional[str] = None, 
                 temperature_data_source: Optional[str] = None,
                 cleanup_on_init: bool = False):
        """
        初始化LoadProfile
        
        Args:
            steps: 每个episode的步数
            dss_folder_path: DSS文件夹路径
            dss_file: DSS文件名
            worker_idx: Worker索引（多进程环境）
            pv_data_source: PV数据源路径
            temperature_data_source: 温度数据源路径
            cleanup_on_init: 是否在初始化时清理临时文件
        """
        self.steps = steps
        self.dss_folder_path = dss_folder_path
        self.worker_idx = worker_idx
        
        # 初始化路径
        self.loadshape_path = os.path.join(dss_folder_path, Constants.LOADSHAPE_FOLDER)
        self.irradiation_path = os.path.join(dss_folder_path, 'irradiation')
        self.temperature_path = os.path.join(dss_folder_path, 'temperature')
        
        # 生成loadshape DSS文件名
        self.loadshape_dss = f'loadshape{"" if worker_idx is None else f"_{worker_idx}"}{Constants.DSS_EXTENSION}'
        
        # 初始化配置生成器
        self.config_generator = ConfigGenerator(dss_folder_path, steps, worker_idx)
        
        # 只有主进程（worker_idx为None或0）才进行清理，避免多进程冲突
        if cleanup_on_init and (worker_idx is None or worker_idx == 0):
            self.config_generator.cleanup_generated_dss_files()
            
        # 所有进程都需要确保默认的pv_data DSS文件存在
        self.config_generator.ensure_default_pv_config()
        
        # PV和温度数据配置
        self.pv_data_source = pv_data_source
        self.temperature_data_source = temperature_data_source
        
        # 查找并初始化负载名称
        self.load_names = self.find_load_names(dss_file)
        
        # 收集所有负载形状CSV文件
        self.csv_files = []
        if os.path.exists(self.loadshape_path):
            self.csv_files = [
                os.path.join(self.loadshape_path, f) 
                for f in os.listdir(self.loadshape_path)
                if 'loadshape' in f.lower() and f.lower().endswith(Constants.CSV_EXTENSION)
            ]
        
        # 初始化Episode生成器
        self.episode_generator = EpisodeGenerator(
            dss_folder_path, self.loadshape_path, 
            self.irradiation_path, self.temperature_path,
            steps, self.load_names
        )
    
    def find_load_names(self, main_dss: str) -> List[str]:
        """从主DSS文件中查找所有负载名称"""
        names = []
        
        # 分析原始DSS文件
        file_path = os.path.join(self.dss_folder_path, main_dss)
        assert os.path.exists(file_path), f'{file_path} not found'
        
        needs_load_duty = False
        duty_mode = False
        
        with open(file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                if DSSFileParser.is_duty_mode_line(line):
                    duty_mode = True
                    continue
                
                is_load, load_name = DSSFileParser.parse_load_line(line)
                if is_load and load_name:
                    names.append(load_name)
                    if not DSSFileParser.has_duty_in_line(line):
                        needs_load_duty = True
        
        # 如果需要添加duty参数，创建增强版文件
        if needs_load_duty or not duty_mode:
            self.create_file_with_duty(main_dss)
            duty_file_name = main_dss[:-4] + '_duty.dss'
            load_file = self.add_redirect_and_mode_at_main_duty_dss(duty_file_name)
            
            # 重新分析
            names.clear()
            self._analyze_dss_file(duty_file_name, names)
            
            if load_file:
                self._analyze_dss_file(load_file, names)
        else:
            load_file = self.find_load_file_from(main_dss)
            if load_file:
                self._analyze_dss_file(load_file, names)
        
        # 修复DSS文件中的错误redirect语句（无论duty模式如何）
        self._fix_redirect_statements(main_dss)
        
        # 验证
        assert len(names) > 0, '未找到任何负载定义'
        assert len(names) == len(set(names)), f'发现重复的负载名称'
        
        return names
    
    def _fix_redirect_statements(self, main_dss: str) -> None:
        """修复DSS文件中错误的redirect语句，确保使用正确的文件名"""
        file_path = os.path.join(self.dss_folder_path, main_dss)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as fin:
                lines = fin.readlines()

            # 检查是否需要修复
            needs_fix = False
            for line in lines:
                if line.strip().startswith('redirect loadshape') and self.loadshape_dss not in line:
                    needs_fix = True
                    break
            
            if not needs_fix:
                return
            
            # 修复错误的redirect语句
            with open(file_path, 'w', encoding='utf-8') as fout:
                for line in lines:
                    if line.strip().startswith('redirect loadshape'):
                        # 替换为正确的文件名
                        fout.write(f'redirect {self.loadshape_dss}\n')
                    else:
                        fout.write(line)
            
            print(f"✅ 修复了DSS文件中的redirect语句: {main_dss}")
            
        except Exception as e:
            print(f"⚠️  修复redirect语句时出错: {e}")
    
    def create_file_with_duty(self, dss_file: str) -> str:
        """为DSS文件中缺少duty参数的负载行添加duty参数"""
        source_path = os.path.join(self.dss_folder_path, dss_file)
        duty_filename = dss_file.replace(Constants.DSS_EXTENSION, f'{Constants.DUTY_SUFFIX}{Constants.DSS_EXTENSION}')
        duty_path = os.path.join(self.dss_folder_path, duty_filename)
        
        with open(source_path, 'r', encoding='utf-8') as fin, \
             open(duty_path, 'w', encoding='utf-8') as fout:
            for line in fin:
                is_load, load_name = DSSFileParser.parse_load_line(line)
                
                if not is_load or DSSFileParser.has_duty_in_line(line):
                    fout.write(line)
                else:
                    clean_line = DSSFileParser.clean_line(line)
                    if load_name:
                        fout.write(f'{clean_line} duty=loadshape_{load_name}\n')
                    else:
                        fout.write(line)
        
        return duty_path
   
    def add_redirect_and_mode_at_main_duty_dss(self, main_duty_dss: str) -> Optional[str]:
        """在主duty DSS文件中添加重定向和设置duty模式"""
        file_path = os.path.join(self.dss_folder_path, main_duty_dss)
        
        with open(file_path, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()

        found_load = False
        load_file = None
        
        with open(file_path, 'w', encoding='utf-8') as fout:
            for line in lines:
                clean_line = DSSFileParser.clean_line(line).lower()
                
                # 检查是否是loadshape redirect语句
                if line.strip().startswith('redirect loadshape'):
                    # 替换为正确的loadshape文件名
                    fout.write(f'redirect {self.loadshape_dss}\n')
                    found_load = True
                    continue
                    
                if not found_load and 'load' in clean_line and not clean_line.startswith('~'):
                    fout.write('! 自动添加加入redirect的负荷 \n')
                    fout.write(f'redirect {self.loadshape_dss}\n\n')
                    found_load = True

                redirect_file = DSSFileParser.parse_redirect_line(line)
                if redirect_file and not load_file:
                    load_file = redirect_file
                    if not redirect_file.endswith('_duty.dss'):
                        duty_file = redirect_file[:-4] + '_duty.dss'
                        fout.write(f'redirect {duty_file}\n')
                    else:
                        fout.write(line)
                else:
                    fout.write(line)
        
            fout.write(Constants.DEFAULT_DUTY_SETTINGS)
        
        assert found_load, f'cannot find load at {main_duty_dss}'
        return load_file

    def find_load_file_from(self, main_dss: str) -> Optional[str]:
        """从主DSS文件中查找重定向的负载文件"""
        file_path = os.path.join(self.dss_folder_path, main_dss)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as fin:
                for line in fin:
                    redirect_file = DSSFileParser.parse_redirect_line(line)
                    if redirect_file:
                        return redirect_file
        except FileNotFoundError:
            raise FileNotFoundError(f"DSS文件不存在: {file_path}")
        
        return None
    
    def _analyze_dss_file(self, fname: str, names: List[str]) -> None:
        """简化的分析方法"""
        file_path = os.path.join(self.dss_folder_path, fname)
        
        with open(file_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                is_load, load_name = DSSFileParser.parse_load_line(line)
                if is_load and load_name:
                    names.append(load_name)
    
    def generate_episodes_from_existing_files(self, target_episodes: int = 30, 
                                             pv_threshold: float = 0.05, scale: float = 1.0) -> int:
        """从loadshape文件夹中的现有数据文件生成episodes
        
        Args:
            target_episodes: 目标episode数量
            pv_threshold: 光伏有效阈值（标准化值，低于此值认为无光伏）
            scale: 负载数据缩放因子，用于调整负载强度
            
        Returns:
            int: 实际生成的episode数量
        """
        return self.episode_generator.generate_from_existing_files(target_episodes, pv_threshold, scale)
    
    def select_load_profile(self, episode_idx: int) -> str:
        """选择指定episode的负载配置文件"""
        if episode_idx is None:
            raise ValueError("episode_idx 不能为空")
        if not isinstance(episode_idx, int):
            raise TypeError(f"episode_idx 必须是 int，当前类型为 {type(episode_idx)}")
        
        episode_folder = str(episode_idx).zfill(Constants.EPISODE_FOLDER_DIGITS)
        assert os.path.exists(os.path.join(self.loadshape_path, episode_folder)), 'episode_idx does not exist'
        
        with open(os.path.join(self.dss_folder_path, self.loadshape_dss), 'w') as fp:
            for load_name in self.load_names:
                fp.write(
                        f'New Loadshape.loadshape_{load_name} '
                        f'npts={self.steps} '
                        f'sinterval=60  '  # 等价 sinterval=60 
                        f'mult=(file=./loadshape/{episode_folder}/{load_name}.csv)\n'
                        )
        return os.path.join(self.loadshape_path, episode_folder)
    
    def select_pv_temperature_profile(self, episode_idx: int) -> bool:
        """选择指定episode的PV和温度配置文件，并生成对应的pv_data DSS文件"""
        return self.config_generator.select_pv_temperature_profile(
            episode_idx, self.irradiation_path, self.temperature_path
        )
    
    def cleanup_generated_dss_files(self, keep_base_files: bool = True) -> int:
        """清理生成的临时DSS文件"""
        return self.config_generator.cleanup_generated_dss_files(keep_base_files)
    
    def get_load_profile_data(self, episode_idx: int) -> pd.DataFrame:
        """获取指定episode的负载配置数据"""
        folder_path = os.path.join(self.loadshape_path, str(episode_idx).zfill(Constants.EPISODE_FOLDER_DIGITS))
        
        load_data = []
        for csv_file in os.listdir(folder_path):
            if csv_file.endswith('.csv'):
                csv_path = os.path.join(folder_path, csv_file)
                load_name = csv_file.split('.')[0]
                load = pd.read_csv(csv_path, header=None, names=[load_name])
                load_data.append(load)

        return pd.concat(load_data, axis=1)
    
    def validate_data_consistency(self, episodes: int) -> Dict[str, Any]:
        """验证负载、PV和温度数据的一致性"""
        validation_result = {
            'status': 'success',
            'load_data_valid': False,
            'pv_data_valid': False,
            'temperature_data_valid': False,
            'warnings': [],
            'errors': []
        }
        
        try:
            # 验证负载数据
            for episode in range(episodes):
                episode_folder = str(episode).zfill(Constants.EPISODE_FOLDER_DIGITS)
                episode_dir = os.path.join(self.loadshape_path, episode_folder)
                
                if os.path.exists(episode_dir):
                    load_files = [f for f in os.listdir(episode_dir) if f.endswith('.csv')]
                    if len(load_files) == len(self.load_names):
                        # 检查每个负载文件的长度
                        for load_file in load_files:
                            file_path = os.path.join(episode_dir, load_file)
                            if os.path.exists(file_path):
                                load_data = pd.read_csv(file_path, header=None)
                                if len(load_data) != self.steps:
                                    validation_result['errors'].append(
                                        f"Episode {episode} 负载文件 {load_file} 长度不匹配: {len(load_data)} vs {self.steps}"
                                    )
                        validation_result['load_data_valid'] = True
                    else:
                        validation_result['errors'].append(f"Episode {episode} 负载文件数量不匹配")
                else:
                    validation_result['errors'].append(f"Episode {episode} 目录不存在")
            
            # 验证PV数据
            for episode in range(episodes):
                episode_folder = str(episode).zfill(Constants.EPISODE_FOLDER_DIGITS)
                pv_episode_dir = os.path.join(self.irradiation_path, episode_folder)
                pv_file = os.path.join(pv_episode_dir, '20250307_irradiance.csv')
                
                if os.path.exists(pv_file):
                    pv_data = pd.read_csv(pv_file, header=None)
                    if len(pv_data) != self.steps:
                        validation_result['errors'].append(
                            f"Episode {episode} PV数据长度不匹配: {len(pv_data)} vs {self.steps}"
                        )
                    else:
                        validation_result['pv_data_valid'] = True
                else:
                    validation_result['warnings'].append(f"Episode {episode} PV数据文件不存在")
            
            # 验证温度数据
            for episode in range(episodes):
                episode_folder = str(episode).zfill(Constants.EPISODE_FOLDER_DIGITS)
                temp_episode_dir = os.path.join(self.temperature_path, episode_folder)
                temp_file = os.path.join(temp_episode_dir, '20250307_temperature.csv')
                
                if os.path.exists(temp_file):
                    temp_data = pd.read_csv(temp_file, header=None)
                    if len(temp_data) != self.steps:
                        validation_result['errors'].append(
                            f"Episode {episode} 温度数据长度不匹配: {len(temp_data)} vs {self.steps}"
                        )
                    else:
                        validation_result['temperature_data_valid'] = True
                else:
                    validation_result['warnings'].append(f"Episode {episode} 温度数据文件不存在")
            
            # 更新验证状态
            if validation_result['errors']:
                validation_result['status'] = 'error'
            elif validation_result['warnings']:
                validation_result['status'] = 'warning'
                
        except Exception as e:
            validation_result['status'] = 'error'
            validation_result['errors'].append(f"验证过程出现异常: {str(e)}")
        
        return validation_result