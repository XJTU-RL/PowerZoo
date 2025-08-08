# -*- coding: utf-8 -*-
"""
统一日志管理器 - 确保所有日志按照层级结构保存
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import time
from datetime import datetime

from envs.power_envs.powerzoo_llm.logging.base_logger import get_logger

logger = get_logger(__name__)


class UnifiedLogManager:
    """
    统一日志路径管理器
    
    负责创建和管理统一的日志目录结构:
    results/
    └── {env_name}/
        └── {system_name}/
            └── {algorithm}/
                └── {experiment_name}/
                    └── seed-{seed}-{timestamp}/
                        ├── logs/
                        ├── models/
                        ├── plots/
                        ├── eval/
                        ├── system_logs/
                        └── training_config.yaml
    """
    
    def __init__(self, 
                 env_name: str,
                 system_name: str,
                 algorithm: str,
                 experiment_name: str,
                 seed: int,
                 base_dir: str = "results",
                 config: Optional[Dict[str, Any]] = None,
                 existing_run_dir: Optional[str] = None):
        """
        初始化统一日志管理器
        
        Args:
            env_name: 环境名称 (如 powerzoo_llm)
            system_name: 系统名称 (如 34Bus_pv)
            algorithm: 算法名称 (如 happo)
            experiment_name: 实验名称
            seed: 随机种子
            base_dir: 基础结果目录
            config: 配置字典（用于保存训练配置）
        """
        self.env_name = env_name
        self.system_name = system_name
        self.algorithm = algorithm
        self.experiment_name = experiment_name
        self.seed = seed
        self.base_dir = base_dir
        self.config = config or {}
        
        # 如果提供了existing_run_dir，使用它并整合到统一结构
        if existing_run_dir and os.path.exists(existing_run_dir):
            self.run_dir = Path(existing_run_dir)
            self.timestamp = self._extract_timestamp_from_path(existing_run_dir)
            # 确保必要的子目录存在
            self._ensure_subdirectories()
            # 迁移现有文件到正确的子目录
            self._migrate_existing_files()
        else:
            # 生成时间戳
            self.timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # 构建完整路径
            self.run_dir = self._create_directory_structure()
        
        # 保存配置文件
        self._save_config()
        
        # 移除旧的扁平结构目录（如果存在）
        self._cleanup_old_structure()
        
    def _create_directory_structure(self) -> Path:
        """创建层级目录结构"""
        # 构建路径：results/{env_name}/{system_name}/{algorithm}/{experiment_name}/seed-{seed}-{timestamp}
        run_path = Path(self.base_dir) / self.env_name / self.system_name / self.algorithm / self.experiment_name
        seed_dir = run_path / f"seed-{self.seed}-{self.timestamp}"
        
        # 创建所有必要的子目录（并集结构）
        subdirs = [
            'logs',           # TensorBoard日志
            'logs/agent0', 'logs/agent1', 'logs/agent2', 'logs/agent3',  # 多智能体日志
            'logs/agent4', 'logs/agent5', 'logs/agent6', 'logs/agent7',
            'models',         # 模型检查点
            'plots',          # 图表
            'eval',           # 评估结果
            'system_logs',    # 系统日志
            'system_params',  # 系统参数记录
            'training_logs'   # 训练日志
        ]
        for subdir in subdirs:
            (seed_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"创建统一日志目录结构: {seed_dir}")
        
        return seed_dir
    
    def _extract_timestamp_from_path(self, path: str) -> str:
        """从路径中提取时间戳"""
        try:
            # 尝试从路径中提取seed-xxx-timestamp格式
            path_parts = Path(path).name.split('-')
            if len(path_parts) >= 4:
                # seed-12345-2025-08-07-21-16-27 格式
                timestamp = '-'.join(path_parts[-6:])
                return timestamp
        except:
            pass
        return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    def _ensure_subdirectories(self):
        """确保所有必要的子目录存在"""
        subdirs = [
            'logs', 'logs/agent0', 'logs/agent1', 'logs/agent2', 'logs/agent3',
            'logs/agent4', 'logs/agent5', 'logs/agent6', 'logs/agent7',
            'models', 'plots', 'eval', 'system_logs', 'system_params', 'training_logs'
        ]
        for subdir in subdirs:
            (self.run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def _migrate_existing_files(self):
        """迁移现有文件到正确的子目录"""
        try:
            # 迁移直接在run_dir下的文件
            for item in self.run_dir.iterdir():
                if item.is_file():
                    # 根据文件类型决定目标目录
                    if item.suffix in ['.pt', '.pth', '.pkl']:
                        target_dir = self.run_dir / 'models'
                    elif item.suffix in ['.png', '.jpg', '.pdf']:
                        target_dir = self.run_dir / 'plots'
                    elif item.suffix in ['.log', '.txt']:
                        target_dir = self.run_dir / 'training_logs'
                    elif item.suffix in ['.json', '.yaml']:
                        continue  # 配置文件保留在根目录
                    else:
                        target_dir = self.run_dir / 'training_logs'
                    
                    if not (target_dir / item.name).exists():
                        shutil.move(str(item), str(target_dir / item.name))
                        logger.debug(f"迁移文件: {item.name} -> {target_dir.name}/")
        except Exception as e:
            logger.warning(f"迁移文件时出错: {e}")
    
    def _save_config(self):
        """保存训练配置到YAML文件"""
        config_path = self.run_dir / "training_config.yaml"
        
        # 添加元信息
        full_config = {
            'meta': {
                'env_name': self.env_name,
                'system_name': self.system_name,
                'algorithm': self.algorithm,
                'experiment_name': self.experiment_name,
                'seed': self.seed,
                'timestamp': self.timestamp,
                'run_dir': str(self.run_dir)
            },
            'config': self.config
        }
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(full_config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置文件保存至: {config_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def _cleanup_old_structure(self):
        """清理旧的扁平结构目录"""
        # 查找并移除旧格式的目录
        old_pattern = Path(self.base_dir) / f"{self.algorithm}_{self.env_name}_*"
        for old_dir in Path(self.base_dir).glob(f"{self.algorithm}_{self.env_name}_*"):
            if old_dir.is_dir():
                # 检查是否为空目录
                subdirs = list(old_dir.iterdir())
                all_empty = all(
                    (subdir.is_dir() and not any(subdir.iterdir())) 
                    for subdir in subdirs 
                    if subdir.is_dir()
                )
                
                if all_empty:
                    try:
                        shutil.rmtree(old_dir)
                        logger.info(f"已删除空的旧格式目录: {old_dir}")
                    except Exception as e:
                        logger.warning(f"删除旧目录失败: {old_dir}, 错误: {e}")
    
    def get_path(self, subdir: str = "") -> Path:
        """
        获取特定子目录的路径
        
        Args:
            subdir: 子目录名称（如 'logs', 'models' 等）
            
        Returns:
            Path对象
        """
        if subdir:
            return self.run_dir / subdir
        return self.run_dir
    
    def get_log_file(self, name: str, subdir: str = "logs") -> Path:
        """
        获取日志文件路径
        
        Args:
            name: 日志文件名
            subdir: 子目录名称
            
        Returns:
            日志文件的完整路径
        """
        log_path = self.get_path(subdir) / name
        return log_path
    
    def migrate_existing_logs(self, old_dir: Path):
        """
        迁移现有的日志文件到新结构
        
        Args:
            old_dir: 旧的日志目录
        """
        if not old_dir.exists():
            return
        
        try:
            # 迁移文件
            for old_file in old_dir.glob("*"):
                if old_file.is_file():
                    # 确定目标子目录
                    if old_file.suffix in ['.log', '.txt']:
                        target_dir = self.get_path('logs')
                    elif old_file.suffix in ['.pt', '.pth', '.pkl']:
                        target_dir = self.get_path('models')
                    elif old_file.suffix in ['.png', '.jpg', '.pdf']:
                        target_dir = self.get_path('plots')
                    else:
                        target_dir = self.run_dir
                    
                    # 复制文件
                    shutil.copy2(old_file, target_dir / old_file.name)
                    logger.debug(f"迁移文件: {old_file} -> {target_dir / old_file.name}")
            
            logger.info(f"成功迁移日志从 {old_dir} 到 {self.run_dir}")
            
        except Exception as e:
            logger.error(f"迁移日志失败: {e}")
    
    def __str__(self) -> str:
        """返回运行目录的字符串表示"""
        return str(self.run_dir)
    
    def __repr__(self) -> str:
        """返回对象的详细表示"""
        return (f"UnifiedLogManager(env='{self.env_name}', system='{self.system_name}', "
                f"algo='{self.algorithm}', exp='{self.experiment_name}', "
                f"seed={self.seed}, path='{self.run_dir}')")


def get_unified_log_manager(args: Dict[str, Any], 
                            algo_args: Dict[str, Any],
                            env_args: Dict[str, Any],
                            existing_run_dir: Optional[str] = None) -> UnifiedLogManager:
    """
    便捷函数：从训练参数创建统一日志管理器
    
    Args:
        args: 主参数字典
        algo_args: 算法参数字典
        env_args: 环境参数字典
        
    Returns:
        UnifiedLogManager实例
    """
    # 提取必要信息
    env_name = env_args.get("env_name", "unknown_env")
    system_name = env_args.get("system_name", "unknown_system")
    algorithm = args.get("algo", "unknown_algo")
    experiment_name = args.get("exp_name", f"{algorithm}_{env_name}_{int(time.time())}")
    seed = args.get("seed", 12345)
    
    # 合并配置
    full_config = {
        'args': args,
        'algo_args': algo_args,
        'env_args': env_args
    }
    
    return UnifiedLogManager(
        env_name=env_name,
        system_name=system_name,
        algorithm=algorithm,
        experiment_name=experiment_name,
        seed=seed,
        config=full_config,
        existing_run_dir=existing_run_dir
    )