# -*- coding: utf-8 -*-
"""
@File      : train.py
@Time      : 2025-04-08 17:38
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 此 Python 文件的主要作用是训练算法。它允许用户通过命令行参数选择算法、环境、实验名称，并加载配置文件。
- 关键组件及职责：
  - `argparse`：处理命令行参数，提供参数选择和默认值。
  - `json`：用于从配置文件中加载配置信息。
  - `get_defaults_yaml_args`：从 yaml 文件中获取默认配置参数。
  - `update_args`：更新参数。
  - `RUNNER_REGISTRY`：根据所选算法运行训练。
- 工作流程：
  1. 解析命令行参数。
  2. 若指定加载配置文件，则从文件加载；否则从 yaml 文件加载。
  3. 更新参数。
  4. 启动相应的训练器进行训练，训练结束后关闭。
"""
"""Train an algorithm."""
import argparse
import json
import yaml
import sys 
import os
# 将项目根目录添加到系统路径中
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from utils.configs_tools import get_defaults_yaml_args, update_args

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 添加算法名称参数，默认为"happo"，可选值为"happo", "hatrpo", "haa2c", "haddpg", "hatd3", "hasac", "had3qn", "maddpg", "matd3", "mappo"
    parser.add_argument(
        "--algo", 
        type=str,
        default="shom",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
            "qmix",
            "shom",
            "sn_mappo",
        ],
        help="算法名称。选择：: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo, shom.",
    )
    # 添加环境名称参数，默认为"powerzoo"，可选值为"smac", "mamujoco", "pettingzoo_mpe", "gym", "football", "dexhands", "smacv2", "lag", "powerzoo"
    parser.add_argument(
        "--env",
        type=str,
        #default="pettingzoo_mpe",
        default="powerzoo",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
            "powerzoo",
            "powerzoo_llm",
            "dsr",
        ],
        help="选择环境: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag, powerzoo,powerzoo_llm, dsr.",
    )
    # 添加实验名称参数，默认为"test"
    parser.add_argument(
        "--exp_name", type=str, default="test", help="Experiment name."
    )
    # 添加加载配置文件参数，默认为空字符串
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="如果设置，则加载现有实验配置文件，而不是从 yaml 配置文件中读取.",
    )
    
    
    
    args, unparsed_args = parser.parse_known_args()

    # 将命令行参数转换为字典
    def process(arg):
        try: 
            return eval(arg)
        except:
            return arg

    # 将命令行参数的键和值分别存储到keys和values中
    keys = [k[2:] for k in unparsed_args[0::2]] 
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # 将args 转换为字典
    # 如果加载配置文件参数不为空，则从配置文件中加载配置
    if args["load_config"] != "":  # 从现有配置文件加载配置
        config_file = args["load_config"]
        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            # 加载YAML配置文件
            with open(config_file, encoding="utf-8") as file:
                all_config = yaml.load(file, Loader=yaml.FullLoader)
            # 从YAML配置中提取算法名称
            if 'algo_name' in all_config:
                args["algo"] = all_config['algo_name']
            # 构建algo_args字典，排除特定的顶级键且只包含字典类型的值
            exclude_keys = {'algo_name', 'env_name', 'env_args','power_system'}
            algo_args = {k: v for k, v in all_config.items() 
                        if k not in exclude_keys and isinstance(v, dict)}
            # 从YAML配置中获取环境参数
            env_args = all_config.get('env_args', {})
        else:
            # 加载JSON配置文件（保持向后兼容）
            with open(config_file, encoding="utf-8") as file:
                all_config = json.load(file)
            args["algo"] = all_config["main_args"]["algo"]
            args["env"] = all_config["main_args"]["env"]
            algo_args = all_config["algo_args"]
            env_args = all_config["env_args"]
    else:  # 从相应的yaml文件加载配置
        # 从yaml文件中加载配置
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    # 更新参数
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line
    for params in algo_args.values():
        print(params)
    # 开始训练
    from runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
