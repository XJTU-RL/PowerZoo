#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PowerZoo LLM模型管理器
提供统一的模型保存、加载、版本控制和元数据管理功能
"""

import os
import json
import pickle
import shutil
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import get_device


class ModelMetadata:
    """
    模型元数据类
    """
    
    def __init__(
        self,
        model_name: str,
        algorithm: str,
        version: str,
        creation_time: datetime,
        training_config: Dict[str, Any],
        environment_config: Dict[str, Any],
        performance_metrics: Dict[str, float],
        model_path: str,
        description: str = "",
        tags: List[str] = None
    ):
        self.model_name = model_name
        self.algorithm = algorithm
        self.version = version
        self.creation_time = creation_time
        self.training_config = training_config
        self.environment_config = environment_config
        self.performance_metrics = performance_metrics
        self.model_path = model_path
        self.description = description
        self.tags = tags or []
        self.model_hash = self._calculate_model_hash()
    
    def _calculate_model_hash(self) -> str:
        """
        计算模型文件的哈希值
        """
        try:
            if os.path.exists(self.model_path + ".zip"):
                with open(self.model_path + ".zip", 'rb') as f:
                    return hashlib.md5(f.read()).hexdigest()
        except Exception:
            pass
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        """
        return {
            'model_name': self.model_name,
            'algorithm': self.algorithm,
            'version': self.version,
            'creation_time': self.creation_time.isoformat(),
            'training_config': self.training_config,
            'environment_config': self.environment_config,
            'performance_metrics': self.performance_metrics,
            'model_path': self.model_path,
            'description': self.description,
            'tags': self.tags,
            'model_hash': self.model_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """
        从字典创建ModelMetadata对象
        """
        creation_time = datetime.fromisoformat(data['creation_time'])
        return cls(
            model_name=data['model_name'],
            algorithm=data['algorithm'],
            version=data['version'],
            creation_time=creation_time,
            training_config=data['training_config'],
            environment_config=data['environment_config'],
            performance_metrics=data['performance_metrics'],
            model_path=data['model_path'],
            description=data.get('description', ''),
            tags=data.get('tags', [])
        )


class ModelManager:
    """
    PowerZoo LLM模型管理器
    
    Features:
    - 模型保存和加载
    - 版本控制
    - 元数据管理
    - 性能指标跟踪
    - 模型比较和选择
    - 自动备份和清理
    """
    
    def __init__(
        self,
        base_dir: str,
        max_versions: int = 10,
        auto_backup: bool = True,
        compression: bool = True
    ):
        """
        初始化模型管理器
        
        Args:
            base_dir: 模型存储基础目录
            max_versions: 每个模型保留的最大版本数
            auto_backup: 是否自动备份
            compression: 是否压缩模型文件
        """
        self.base_dir = Path(base_dir)
        self.max_versions = max_versions
        self.auto_backup = auto_backup
        self.compression = compression
        
        # 创建目录结构
        self.models_dir = self.base_dir / "models"
        self.metadata_dir = self.base_dir / "metadata"
        self.backups_dir = self.base_dir / "backups"
        self.logs_dir = self.base_dir / "logs"
        
        for dir_path in [self.models_dir, self.metadata_dir, self.backups_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 加载现有元数据
        self.metadata_registry = self._load_metadata_registry()
    
    def _load_metadata_registry(self) -> Dict[str, List[ModelMetadata]]:
        """
        加载元数据注册表
        """
        registry_file = self.metadata_dir / "registry.json"
        registry = {}
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for model_name, versions in data.items():
                    registry[model_name] = [
                        ModelMetadata.from_dict(version_data)
                        for version_data in versions
                    ]
            except Exception as e:
                print(f"Warning: Failed to load metadata registry: {e}")
        
        return registry
    
    def _save_metadata_registry(self) -> None:
        """
        保存元数据注册表
        """
        registry_file = self.metadata_dir / "registry.json"
        
        try:
            data = {}
            for model_name, versions in self.metadata_registry.items():
                data[model_name] = [version.to_dict() for version in versions]
            
            with open(registry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Failed to save metadata registry: {e}")
    
    def _generate_version(self, model_name: str) -> str:
        """
        生成新的版本号
        """
        if model_name not in self.metadata_registry:
            return "v1.0.0"
        
        versions = self.metadata_registry[model_name]
        if not versions:
            return "v1.0.0"
        
        # 获取最新版本号并递增
        latest_version = versions[-1].version
        try:
            # 解析版本号 (v1.2.3)
            version_parts = latest_version.replace('v', '').split('.')
            major, minor, patch = map(int, version_parts)
            
            # 递增patch版本
            patch += 1
            return f"v{major}.{minor}.{patch}"
        except Exception:
            # 如果解析失败，使用时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"v{timestamp}"
    
    def save_model(
        self,
        model: BaseAlgorithm,
        model_name: str,
        training_config: Dict[str, Any],
        environment_config: Dict[str, Any],
        performance_metrics: Dict[str, float],
        description: str = "",
        tags: List[str] = None,
        version: Optional[str] = None
    ) -> str:
        """
        保存模型及其元数据
        
        Args:
            model: 要保存的模型
            model_name: 模型名称
            training_config: 训练配置
            environment_config: 环境配置
            performance_metrics: 性能指标
            description: 模型描述
            tags: 标签列表
            version: 指定版本号（可选）
        
        Returns:
            保存的模型路径
        """
        # 生成版本号
        if version is None:
            version = self._generate_version(model_name)
        
        # 创建模型目录
        model_dir = self.models_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_path = model_dir / "model"
        try:
            model.save(str(model_path))
            print(f"Model saved to: {model_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
            raise
        
        # 保存训练配置
        config_path = model_dir / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(training_config, f, indent=2, ensure_ascii=False)
        
        # 保存环境配置
        env_config_path = model_dir / "environment_config.json"
        with open(env_config_path, 'w', encoding='utf-8') as f:
            json.dump(environment_config, f, indent=2, ensure_ascii=False)
        
        # 保存性能指标
        metrics_path = model_dir / "performance_metrics.json"
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(performance_metrics, f, indent=2, ensure_ascii=False)
        
        # 创建元数据
        metadata = ModelMetadata(
            model_name=model_name,
            algorithm=model.__class__.__name__,
            version=version,
            creation_time=datetime.now(),
            training_config=training_config,
            environment_config=environment_config,
            performance_metrics=performance_metrics,
            model_path=str(model_path),
            description=description,
            tags=tags or []
        )
        
        # 更新注册表
        if model_name not in self.metadata_registry:
            self.metadata_registry[model_name] = []
        
        self.metadata_registry[model_name].append(metadata)
        
        # 按版本排序
        self.metadata_registry[model_name].sort(key=lambda x: x.creation_time)
        
        # 清理旧版本
        self._cleanup_old_versions(model_name)
        
        # 保存注册表
        self._save_metadata_registry()
        
        # 自动备份
        if self.auto_backup:
            self._backup_model(model_name, version)
        
        print(f"Model '{model_name}' version '{version}' saved successfully")
        return str(model_path)
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        algorithm_class: Optional[type] = None
    ) -> BaseAlgorithm:
        """
        加载模型
        
        Args:
            model_name: 模型名称
            version: 版本号（如果为None，加载最新版本）
            algorithm_class: 算法类（用于类型检查）
        
        Returns:
            加载的模型
        """
        if model_name not in self.metadata_registry:
            raise ValueError(f"Model '{model_name}' not found")
        
        versions = self.metadata_registry[model_name]
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        
        # 选择版本
        if version is None:
            # 加载最新版本
            metadata = versions[-1]
        else:
            # 查找指定版本
            metadata = None
            for v in versions:
                if v.version == version:
                    metadata = v
                    break
            
            if metadata is None:
                raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        # 加载模型
        try:
            from stable_baselines3 import PPO, DQN, SAC, A2C, DDPG, TD3
            
            algorithm_map = {
                'PPO': PPO,
                'DQN': DQN,
                'SAC': SAC,
                'A2C': A2C,
                'DDPG': DDPG,
                'TD3': TD3
            }
            
            if algorithm_class is None:
                algorithm_class = algorithm_map.get(metadata.algorithm)
                if algorithm_class is None:
                    raise ValueError(f"Unknown algorithm: {metadata.algorithm}")
            
            model = algorithm_class.load(metadata.model_path)
            print(f"Model '{model_name}' version '{metadata.version}' loaded successfully")
            return model
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        列出所有模型及其版本
        
        Returns:
            模型列表字典
        """
        result = {}
        for model_name, versions in self.metadata_registry.items():
            result[model_name] = [
                {
                    'version': v.version,
                    'algorithm': v.algorithm,
                    'creation_time': v.creation_time.isoformat(),
                    'performance_metrics': v.performance_metrics,
                    'description': v.description,
                    'tags': v.tags
                }
                for v in versions
            ]
        return result
    
    def get_model_info(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        获取模型详细信息
        
        Args:
            model_name: 模型名称
            version: 版本号（如果为None，返回最新版本信息）
        
        Returns:
            模型信息字典
        """
        if model_name not in self.metadata_registry:
            raise ValueError(f"Model '{model_name}' not found")
        
        versions = self.metadata_registry[model_name]
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        
        # 选择版本
        if version is None:
            metadata = versions[-1]
        else:
            metadata = None
            for v in versions:
                if v.version == version:
                    metadata = v
                    break
            
            if metadata is None:
                raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        return metadata.to_dict()
    
    def compare_models(
        self,
        model_specs: List[tuple],
        metric: str = 'mean_reward'
    ) -> List[Dict[str, Any]]:
        """
        比较多个模型的性能
        
        Args:
            model_specs: 模型规格列表 [(model_name, version), ...]
            metric: 比较指标
        
        Returns:
            排序后的模型比较结果
        """
        results = []
        
        for model_name, version in model_specs:
            try:
                info = self.get_model_info(model_name, version)
                metric_value = info['performance_metrics'].get(metric, float('-inf'))
                
                results.append({
                    'model_name': model_name,
                    'version': version,
                    'algorithm': info['algorithm'],
                    'metric_value': metric_value,
                    'creation_time': info['creation_time'],
                    'description': info['description']
                })
            except Exception as e:
                print(f"Failed to get info for {model_name}:{version}: {e}")
        
        # 按指标值排序（降序）
        results.sort(key=lambda x: x['metric_value'], reverse=True)
        return results
    
    def get_best_model(
        self,
        metric: str = 'mean_reward',
        algorithm: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> tuple:
        """
        获取最佳模型
        
        Args:
            metric: 评估指标
            algorithm: 算法过滤器
            tags: 标签过滤器
        
        Returns:
            (model_name, version) 元组
        """
        best_model = None
        best_value = float('-inf')
        
        for model_name, versions in self.metadata_registry.items():
            for version_metadata in versions:
                # 应用过滤器
                if algorithm and version_metadata.algorithm != algorithm:
                    continue
                
                if tags:
                    if not any(tag in version_metadata.tags for tag in tags):
                        continue
                
                # 检查指标
                metric_value = version_metadata.performance_metrics.get(metric, float('-inf'))
                if metric_value > best_value:
                    best_value = metric_value
                    best_model = (model_name, version_metadata.version)
        
        if best_model is None:
            raise ValueError("No models found matching the criteria")
        
        return best_model
    
    def delete_model(
        self,
        model_name: str,
        version: Optional[str] = None
    ) -> None:
        """
        删除模型
        
        Args:
            model_name: 模型名称
            version: 版本号（如果为None，删除所有版本）
        """
        if model_name not in self.metadata_registry:
            raise ValueError(f"Model '{model_name}' not found")
        
        if version is None:
            # 删除所有版本
            model_dir = self.models_dir / model_name
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            del self.metadata_registry[model_name]
            print(f"All versions of model '{model_name}' deleted")
        else:
            # 删除指定版本
            versions = self.metadata_registry[model_name]
            version_to_remove = None
            
            for i, v in enumerate(versions):
                if v.version == version:
                    version_to_remove = i
                    break
            
            if version_to_remove is None:
                raise ValueError(f"Version '{version}' not found for model '{model_name}'")
            
            # 删除文件
            version_dir = self.models_dir / model_name / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            # 从注册表中移除
            del self.metadata_registry[model_name][version_to_remove]
            
            # 如果没有版本了，删除模型条目
            if not self.metadata_registry[model_name]:
                del self.metadata_registry[model_name]
            
            print(f"Model '{model_name}' version '{version}' deleted")
        
        self._save_metadata_registry()
    
    def _cleanup_old_versions(self, model_name: str) -> None:
        """
        清理旧版本
        """
        if model_name not in self.metadata_registry:
            return
        
        versions = self.metadata_registry[model_name]
        if len(versions) <= self.max_versions:
            return
        
        # 保留最新的max_versions个版本
        versions_to_remove = versions[:-self.max_versions]
        
        for version_metadata in versions_to_remove:
            try:
                version_dir = self.models_dir / model_name / version_metadata.version
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                print(f"Cleaned up old version: {model_name}:{version_metadata.version}")
            except Exception as e:
                print(f"Failed to cleanup version {version_metadata.version}: {e}")
        
        # 更新注册表
        self.metadata_registry[model_name] = versions[-self.max_versions:]
    
    def _backup_model(self, model_name: str, version: str) -> None:
        """
        备份模型
        """
        try:
            source_dir = self.models_dir / model_name / version
            backup_name = f"{model_name}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.backups_dir / f"{backup_name}.tar.gz"
            
            # 创建压缩备份
            shutil.make_archive(
                str(backup_path).replace('.tar.gz', ''),
                'gztar',
                str(source_dir)
            )
            
            print(f"Model backed up to: {backup_path}")
        except Exception as e:
            print(f"Failed to backup model: {e}")
    
    def export_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        export_path: str = None,
        format: str = 'zip'
    ) -> str:
        """
        导出模型
        
        Args:
            model_name: 模型名称
            version: 版本号
            export_path: 导出路径
            format: 导出格式 ('zip', 'tar', 'tar.gz')
        
        Returns:
            导出文件路径
        """
        if model_name not in self.metadata_registry:
            raise ValueError(f"Model '{model_name}' not found")
        
        versions = self.metadata_registry[model_name]
        if version is None:
            metadata = versions[-1]
        else:
            metadata = None
            for v in versions:
                if v.version == version:
                    metadata = v
                    break
            if metadata is None:
                raise ValueError(f"Version '{version}' not found")
        
        # 确定导出路径
        if export_path is None:
            export_path = f"{model_name}_{metadata.version}"
        
        source_dir = self.models_dir / model_name / metadata.version
        
        # 创建导出文件
        if format == 'zip':
            export_file = f"{export_path}.zip"
            shutil.make_archive(export_path, 'zip', str(source_dir))
        elif format == 'tar':
            export_file = f"{export_path}.tar"
            shutil.make_archive(export_path, 'tar', str(source_dir))
        elif format == 'tar.gz':
            export_file = f"{export_path}.tar.gz"
            shutil.make_archive(export_path, 'gztar', str(source_dir))
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Model exported to: {export_file}")
        return export_file