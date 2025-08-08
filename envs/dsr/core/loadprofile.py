import numpy as np
import pandas as pd
import os
from pathlib import Path
from fnmatch import fnmatch
from typing import List, Dict, Optional, Any

class LoadProfile:
    """DSR模块的负载配置文件管理类
    
    该类用于管理DSR环境中的负载配置文件，包括负载形状、负载名称等。
    与powerzoo_llm中的LoadProfile类兼容，但针对DSR环境进行了优化。
    """
    
    def __init__(self, steps: int, dss_folder_path: str, dss_file: str, 
                 worker_idx: Optional[int] = None, irrad_dss: Optional[str] = None):
        """初始化LoadProfile
        
        Args:
            steps: 仿真步数
            dss_folder_path: DSS文件夹路径
            dss_file: DSS文件名
            worker_idx: 工作进程索引
            irrad_dss: 辐照度DSS文件
        """
        self.steps = steps
        self.dss_folder_path = dss_folder_path
        self.dss_file = dss_file
        self.worker_idx = worker_idx
        self.irrad_dss = irrad_dss
        
        # 设置负载形状路径
        self.loadshape_path = os.path.join(dss_folder_path, 'loadshape')
        
        # 设置负载形状DSS文件名
        if worker_idx is None:
            self.loadshape_dss = 'loadshape.dss'
        else:
            self.loadshape_dss = f'loadshape_{worker_idx}.dss'
        
        # 初始化负载名称
        self.LOAD_NAMES = self._find_load_names(dss_file)
        
        # 初始化负载文件列表
        self.FILES = self._find_loadshape_files()
        
    def _find_load_names(self, dss_file: str) -> List[str]:
        """从DSS文件中查找负载名称
        
        Args:
            dss_file: DSS文件名
            
        Returns:
            负载名称列表
        """
        load_names = []
        dss_path = os.path.join(self.dss_folder_path, dss_file)
        
        if not os.path.exists(dss_path):
            return load_names
            
        try:
            with open(dss_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip().lower()
                    if line.startswith('new load.'):
                        # 提取负载名称
                        parts = line.split('.')
                        if len(parts) > 1:
                            load_name = parts[1].split()[0]
                            load_names.append(load_name)
        except Exception as e:
            print(f"Warning: Failed to read DSS file {dss_path}: {e}")
            
        return load_names
    
    def _find_loadshape_files(self) -> List[str]:
        """查找负载形状CSV文件
        
        Returns:
            负载形状文件路径列表
        """
        files = []
        
        if not os.path.exists(self.loadshape_path):
            return files
            
        try:
            for f in os.listdir(self.loadshape_path):
                if f.lower().endswith('.csv') and 'loadshape' in f.lower():
                    files.append(os.path.join(self.loadshape_path, f))
        except Exception as e:
            print(f"Warning: Failed to list loadshape directory {self.loadshape_path}: {e}")
            
        return files
    
    def get_load_names(self) -> List[str]:
        """获取负载名称列表
        
        Returns:
            负载名称列表
        """
        return self.LOAD_NAMES.copy()
    
    def get_loadshape_files(self) -> List[str]:
        """获取负载形状文件列表
        
        Returns:
            负载形状文件路径列表
        """
        return self.FILES.copy()
    
    def load_profile_data(self, file_path: str) -> Optional[np.ndarray]:
        """加载负载配置文件数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            负载数据数组，如果加载失败则返回None
        """
        try:
            if file_path.endswith('.csv'):
                data = pd.read_csv(file_path)
                return data.values
            else:
                return np.loadtxt(file_path)
        except Exception as e:
            print(f"Warning: Failed to load profile data from {file_path}: {e}")
            return None
    
    def get_profile_for_step(self, step: int, load_name: str = None) -> float:
        """获取指定步骤的负载配置文件值
        
        Args:
            step: 仿真步骤
            load_name: 负载名称（可选）
            
        Returns:
            负载值
        """
        # 简化实现，返回基于步骤的默认值
        if step < 0 or step >= self.steps:
            return 1.0
            
        # 可以根据实际需求实现更复杂的负载配置文件逻辑
        return 1.0 + 0.1 * np.sin(2 * np.pi * step / 24)  # 24小时周期的简单负载模式
    
    def validate_profiles(self) -> bool:
        """验证负载配置文件的有效性
        
        Returns:
            如果所有配置文件都有效则返回True
        """
        if not self.LOAD_NAMES:
            print("Warning: No load names found")
            return False
            
        if not self.FILES:
            print("Warning: No loadshape files found")
            return False
            
        return True
    
    def __repr__(self) -> str:
        return (f"LoadProfile(steps={self.steps}, "
                f"loads={len(self.LOAD_NAMES)}, "
                f"files={len(self.FILES)}, "
                f"worker_idx={self.worker_idx})")