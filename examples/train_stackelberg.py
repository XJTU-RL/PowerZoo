# -*- coding: utf-8 -*-
"""
@File      : train_stackelberg.py
@Time      : 2025-05-21
@Author    : Your AI Assistant
@Description: 专用于Stackelberg环境和sn_mappo算法的训练脚本。
"""
import argparse
import sys
import os
import json

# 将项目根目录添加到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.configs_tools import get_defaults_yaml_args, update_args
from runners import RUNNER_REGISTRY

def main():
    """专为Stackelberg环境设计的训练主函数。"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--bus",
        type=int,
        default=13,
        choices=[13, 34, 123],
        help="选择总线系统 (13, 34, or 123)."
    )
    parser.add_argument(
        "--exp_name", type=str, default="stackelberg_test", help="实验名称。"
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="如果需要，可以加载一个已有的实验配置文件。",
    )

    args, unparsed_args = parser.parse_known_args()

    # --- 1. 设置核心参数 ---
    main_args = {
        "algo": "sn_mappo",
        "env": f"stackelberg_{args.bus}bus",
        "exp_name": args.exp_name,
        "load_config": args.load_config,
    }

    # --- 2. 加载和处理配置 ---
    def process(arg):
        try:
            return eval(arg)
        except (NameError, SyntaxError):
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}

    if main_args["load_config"]:
        with open(main_args["load_config"], 'r', encoding='utf-8') as f:
            all_config = json.load(f)
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:
        algo_args, env_args = get_defaults_yaml_args(main_args["algo"], main_args["env"])

    update_args(unparsed_dict, algo_args, env_args)

    # --- 3. 启动训练器 ---
    # 根据算法名称从注册表中获取对应的Runner
    # StackelbergRunner 应该被正确地注册在 RUNNER_REGISTRY 中
    if main_args["algo"] not in RUNNER_REGISTRY:
        raise ValueError(f"错误: 算法 '{main_args['algo']}' 没有在 'runners/__init__.py' 中注册。")
    
    runner = RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args)
    
    # --- 4. 运行与收尾 ---
    runner.run()
    runner.close()

if __name__ == "__main__":
    main() 