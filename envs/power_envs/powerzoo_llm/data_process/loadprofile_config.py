# -*- coding: utf-8 -*-
"""
配置文件生成和管理模块
负责生成DSS配置文件和PV/温度配置
"""

import os
import pandas as pd
from typing import List, Optional
import fcntl
import time

from envs.power_envs.powerzoo_llm.data_process.loadprofile_dss_parser import Constants


class ConfigGenerator:
    """配置文件生成器"""
    
    def __init__(self, dss_folder_path: str, steps: int, 
                 worker_idx: Optional[int] = None):
        """
        初始化配置生成器
        
        Args:
            dss_folder_path: DSS文件夹路径
            steps: 每个episode的步数
            worker_idx: Worker索引
        """
        self.dss_folder_path = dss_folder_path
        self.steps = steps
        self.worker_idx = worker_idx
        self.generated_temp_files = []
    
    def select_pv_temperature_profile(self, episode_idx: int, 
                                     irradiation_path: str,
                                     temperature_path: str) -> bool:
        """选择指定episode的PV和温度配置文件"""
        if episode_idx is None:
            raise ValueError("episode_idx 不能为空")
        if not isinstance(episode_idx, int):
            raise TypeError(f"episode_idx 必须是 int，当前类型为 {type(episode_idx)}")
        
        episode_folder = str(episode_idx).zfill(Constants.EPISODE_FOLDER_DIGITS)
        
        # 检查PV和温度数据文件是否存在
        pv_episode_path = os.path.join(irradiation_path, episode_folder)
        temp_episode_path = os.path.join(temperature_path, episode_folder)
        
        pv_file_exists = os.path.exists(os.path.join(pv_episode_path, '20250307_irradiance.csv'))
        temp_file_exists = os.path.exists(os.path.join(temp_episode_path, '20250307_temperature.csv'))
        
        if not (pv_file_exists or temp_file_exists):
            return False  # 如果PV和温度文件都不存在，返回False
        
        # 统一使用简洁的命名格式
        if self.worker_idx is not None:
            pv_dss_filename = f'pv_data_{self.worker_idx}{Constants.DSS_EXTENSION}'
        else:
            pv_dss_filename = f'pv_data{Constants.DSS_EXTENSION}'
        
        pv_dss_path = os.path.join(self.dss_folder_path, pv_dss_filename)
        
        # 记录生成的临时文件
        if pv_dss_filename not in self.generated_temp_files:
            self.generated_temp_files.append(pv_dss_filename)
        
        return self._generate_pv_dss_config(
            pv_dss_path, episode_folder, pv_file_exists, temp_file_exists,
            irradiation_path, temperature_path
        )
    
    def _generate_pv_dss_config(self, pv_dss_path: str, episode_folder: str, 
                               has_pv_data: bool, has_temp_data: bool,
                               irradiation_path: str, temperature_path: str) -> bool:
        """生成PV数据的DSS配置文件"""
        try:
            with open(pv_dss_path, 'w', encoding='utf-8') as fp:
                # 写入文件头注释
                fp.write(f'! PV系统数据配置文件 - Episode {episode_folder}\n')
                fp.write(f'! 自动生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                if self.worker_idx is not None:
                    fp.write(f'! Worker索引: {self.worker_idx}\n')
                fp.write('\n')
                
                # 定义PV温度系数曲线
                fp.write('! 定义PV温度系数曲线\n')
                fp.write('New XYCurve.MyPvsT npts=4 xarray=[0 25 75 100] yarray=[1.2 1.0 0.8 0.6]\n')
                fp.write('\n')
                
                # 定义PV效率曲线
                fp.write('! 定义PV效率曲线\n')
                fp.write('New XYCurve.MyEff npts=4 xarray=[.1 .2 1.0 1.2] yarray=[.86 .9 .98 .99]\n')
                fp.write('\n')
                
                # 配置辐照度LoadShape
                if has_pv_data:
                    fp.write('! 定义辐照度LoadShape\n')
                    # 使用相对路径，从DSS文件夹开始
                    relative_irr_path = os.path.relpath(
                        os.path.join(irradiation_path, episode_folder, '20250307_irradiance.csv'),
                        self.dss_folder_path
                    )
                    fp.write(f'New Loadshape.MyIrrad npts={self.steps} sinterval=60 mult=(file=./{relative_irr_path})\n')
                else:
                    # 如果没有PV数据，使用默认值
                    fp.write('! 使用默认辐照度LoadShape (无数据文件)\n')
                    fp.write(f'New Loadshape.MyIrrad npts={self.steps} sinterval=60 mult=1.0\n')
                fp.write('\n')
                
                # 配置温度TShape
                if has_temp_data:
                    fp.write('! 定义温度TShape\n')
                    # 使用相对路径，从DSS文件夹开始
                    relative_temp_path = os.path.relpath(
                        os.path.join(temperature_path, episode_folder, '20250307_temperature.csv'),
                        self.dss_folder_path
                    )
                    fp.write(f'New TShape.MyTemp npts={self.steps} sinterval=60 temp=(file=./{relative_temp_path})\n')
                else:
                    # 如果没有温度数据，使用默认值
                    fp.write('! 使用默认温度TShape (无数据文件)\n')
                    fp.write(f'New TShape.MyTemp npts={self.steps} sinterval=60 temp=25\n')
                fp.write('\n')
            
            return True
            
        except Exception as e:
            print(f"生成PV DSS配置文件失败: {e}")
            return False
    
    def ensure_default_pv_config(self) -> bool:
        """确保默认的PV配置文件存在，用于系统初始化
        
        使用文件锁避免多进程并发问题
        """
        success_count = 0
        try:
            # 生成当前worker的配置文件
            if self.worker_idx is not None:
                current_filename = f'pv_data_{self.worker_idx}{Constants.DSS_EXTENSION}'
            else:
                current_filename = f'pv_data{Constants.DSS_EXTENSION}'
            
            current_path = os.path.join(self.dss_folder_path, current_filename)
            
            # 使用文件锁避免并发写入
            lock_file = os.path.join(self.dss_folder_path, '.pv_config.lock')
            os.makedirs(self.dss_folder_path, exist_ok=True)
            
            # 尝试获取文件锁，最多等待5秒
            max_wait = 5
            wait_time = 0
            lock_acquired = False
            
            while wait_time < max_wait:
                try:
                    with open(lock_file, 'w') as lock_fd:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        lock_acquired = True
                        
                        # 检查并生成当前worker的配置文件
                        if not os.path.exists(current_path):
                            if self._generate_pv_dss_config(current_path, "000", False, False, "", ""):
                                success_count += 1
                                if current_filename not in self.generated_temp_files:
                                    self.generated_temp_files.append(current_filename)
                        else:
                            success_count += 1  # 文件已存在
                        
                        # 只有worker 0或None生成其他worker的配置文件
                        if self.worker_idx is None or self.worker_idx == 0:
                            for worker_id in range(8):  # 0到7的worker
                                filename = f'pv_data_{worker_id}{Constants.DSS_EXTENSION}'
                                filepath = os.path.join(self.dss_folder_path, filename)
                                
                                if not os.path.exists(filepath):
                                    if self._generate_pv_dss_config(filepath, "000", False, False, "", ""):
                                        success_count += 1
                        
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        break
                        
                except (IOError, OSError):
                    # 文件锁被占用，等待一小段时间后重试
                    time.sleep(0.1)
                    wait_time += 0.1
                    
            if not lock_acquired:
                # 如果无法获取锁，至少确保当前worker的文件存在
                if not os.path.exists(current_path):
                    if self._generate_pv_dss_config(current_path, "000", False, False, "", ""):
                        success_count += 1
                else:
                    success_count += 1
            
            return success_count > 0
            
        except Exception as e:
            print(f"生成默认PV配置失败: {e}")
            # 即使失败也尝试生成当前worker的配置
            try:
                if not os.path.exists(current_path):
                    return self._generate_pv_dss_config(current_path, "000", False, False, "", "")
            except:
                pass
            return False
    
    def cleanup_generated_dss_files(self, keep_base_files: bool = True) -> int:
        """清理生成的临时DSS文件"""
        import re
        
        cleaned_count = 0
        
        if not os.path.exists(self.dss_folder_path):
            return cleaned_count
        
        # 定义需要清理的文件模式
        patterns_to_clean = [
            r'^pv_data_\d+\.dss$',  # pv_data_0.dss, pv_data_1.dss等
            r'^pv_data_\d{3}_\d+\.dss$',  # pv_data_001_1.dss等（旧格式）
            r'^loadshape_\d+\.dss$',  # loadshape_0.dss等
        ]
        
        # 如果不保留基础文件，也清理它们
        if not keep_base_files:
            patterns_to_clean.extend([
                r'^pv_data\.dss$',  # pv_data.dss
                r'^loadshape\.dss$',  # loadshape.dss
            ])
        
        # 遍历目录清理匹配的文件
        for filename in os.listdir(self.dss_folder_path):
            # 检查是否匹配任何清理模式
            for pattern in patterns_to_clean:
                if re.match(pattern, filename):
                    file_path = os.path.join(self.dss_folder_path, filename)
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except Exception as e:
                        print(f"清理文件 {filename} 失败: {e}")
                    break
        
        # 清理记录的临时文件列表
        for temp_file in self.generated_temp_files:
            file_path = os.path.join(self.dss_folder_path, temp_file)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    cleaned_count += 1
                except Exception:
                    pass
        
        self.generated_temp_files.clear()
        
        if cleaned_count > 0:
            print(f"✅ 清理了 {cleaned_count} 个临时DSS文件")
        
        return cleaned_count