#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""调试训练脚本目录创建问题"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.configs_tools import init_dir

def test_init_dir():
    """测试init_dir函数"""
    env = "powerzoo"
    env_args = {"env_name": "13Bus"}
    algo = "happo"
    exp_name = "debug_test"
    seed = 1
    logger_path = "./results"
    
    print(f"当前工作目录: {os.getcwd()}")
    print(f"logger_path: {logger_path}")
    
    # 调用init_dir
    results_path, log_path, models_path, writter = init_dir(
        env, env_args, algo, exp_name, seed, logger_path
    )
    
    print(f"results_path: {results_path}")
    print(f"results_path exists: {os.path.exists(results_path)}")
    print(f"log_path: {log_path}")
    print(f"log_path exists: {os.path.exists(log_path)}")
    print(f"models_path: {models_path}")
    print(f"models_path exists: {os.path.exists(models_path)}")
    
    # 测试创建文件
    test_file = os.path.join(results_path, "test.txt")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        print(f"成功创建测试文件: {test_file}")
    except Exception as e:
        print(f"创建测试文件失败: {e}")
    
    # 列出目录内容
    if os.path.exists(results_path):
        print(f"\n{results_path} 目录内容:")
        for item in os.listdir(results_path):
            print(f"  - {item}")

if __name__ == "__main__":
    test_init_dir()