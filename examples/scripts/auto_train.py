# -*- coding: utf-8 -*-
"""
@File      : auto_train.py
@Time      : 2025-04-08 17:39
@Author    : Xiaodong Zheng
@Email     : zxd_xjtu@stu.xjtu.edu.cn
@Description: 该 Python 脚本的主要作用是使用 Optuna 库对强化学习算法的超参数进行优化。
- 关键组件及职责：
  - load_yaml_config：加载 YAML 配置文件。
  - save_yaml_config：保存 YAML 配置文件。
  - objective：Optuna 的目标函数，用于超参数优化，返回加权奖励。
  - main：执行 Optuna 超参数优化的主函数。
- 工作流程：
  1. 通过命令行参数设置环境、试验次数、算法等信息。
  2. 从配置文件或 YAML 文件加载配置，并更新参数。
  3. 创建 Optuna study，进行超参数优化。
  4. 输出最佳试验结果，并保存最佳配置到 YAML 文件。
"""
import argparse
import json
import sys
import os
import optuna
import yaml
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.configs_tools import get_defaults_yaml_args, update_args
from runners import RUNNER_REGISTRY

def load_yaml_config(file_path):
    """加载 YAML 配置文件。"""
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_yaml_config(config, file_path):
    """保存 YAML 配置文件。"""
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

def objective(trial, args, algo_args, env_args):
    """
    Optuna 的目标函数，用于超参数优化。
    """
    # 获取默认的 YAML 配置参数
    algo_args_updated = copy.deepcopy(algo_args)

    # 使用 Optuna 建议修改 algo_args 超参数
    algo_args_updated["algo"]["ppo_epoch"] = trial.suggest_int("ppo_epoch", 3, 10)
    algo_args_updated["algo"]["entropy_coef"] = trial.suggest_float("entropy_coef", 0.001, 0.1)
    algo_args_updated["algo"]["huber_delta"] = trial.suggest_float("huber_delta", 5, 10)

    
    # 开始训练
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args_updated, env_args)
    runner.run()


    result = runner.get_result()  # 返回包含训练奖励和评估奖励的字典，格式如下：
    # {
    #     "train_avg_reward": 平均训练奖励（所有完成的 episode 的奖励平均值，float 类型）,
    #     "eval_avg_reward": 平均评估奖励（所有评估 episode 的奖励平均值，float 类型）,
    #     "eval_powerloss_avg_reward": 平均评估的功率损耗奖励（float 类型）,
    #     "eval_voltage_avg_reward": 平均评估的电压奖励（float 类型）,
    #     "eval_ctrl_avg_reward": 平均评估的控制奖励（float 类型）
    # }
    
    runner.close()
    # 自定义优化目标例如，单独使用评估奖励
    # return result["eval_avg_reward"]

    # 或者结合训练奖励和评估奖励
    # train_weight = trial.suggest_float("train_weight", 0.0, 1.0)
    # eval_weight = 1 - train_weight
    # return train_weight * result["train_avg_reward"] + eval_weight * result["eval_avg_reward"]

    # 获取三种奖励值
    powerloss_avg = result["eval_powerloss_avg_reward"]
    voltage_avg = result["eval_voltage_avg_reward"]
    ctrl_avg = result["eval_ctrl_avg_reward"]
    
    # 加权求和
    powerloss_weight = trial.suggest_float("powerloss_weight", 0.0, 1.0)
    voltage_weight = trial.suggest_float("voltage_weight", 0.0, 1.0)
    ctrl_weight = trial.suggest_float("ctrl_weight", 0.0, 1.0)

    # 归一化权重（确保权重之和为1）
    total_weight = powerloss_weight + voltage_weight + ctrl_weight
    powerloss_weight /= total_weight
    voltage_weight /= total_weight
    ctrl_weight /= total_weight

    # 计算加权和
    balanced_reward = (
        powerloss_weight * powerloss_avg +
        voltage_weight * voltage_avg +
        ctrl_weight * ctrl_avg
    )
    
    return balanced_reward


def main():
    """执行 Optuna 超参数优化的主函数。"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        ],
        help="选择环境: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag,powerzoo.",
    )
    parser.add_argument(
        "--n_trials", 
        type=int,
        default=50, 
        help="Optuna 运行的试验次数。"
    )

    # 添加命令行参数
    parser.add_argument('--algo', 
                        type=str, 
                        default='shom', help='算法名称'
    )
    parser.add_argument('--exp_name', 
                        type=str, 
                        default='optuna_study', help='实验名称'
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

    keys = [k[2:] for k in unparsed_args[0::2]] 
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # 将args 转换为字典

    # 如果加载配置文件参数不为空，则从配置文件中加载配置
    if args["load_config"] != "":  # 从现有配置文件加载配置
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # 从相应的yaml文件加载配置
        # 从yaml文件中加载配置
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])

    # 更新参数
    update_args(unparsed_dict, algo_args, env_args)  # 从命令行更新参数
    
    # 创建一个 Optuna study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, args, algo_args, env_args), n_trials=args['n_trials'])

    # 输出最佳试验结果
    print("最佳试验：")
    trial = study.best_trial

    print(f"  值: {trial.value}")
    print("  参数:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # 保存最佳配置到 YAML 文件
    config = load_yaml_config(args.config)
    config["algo"]["ppo_epoch"] = trial.params.get("ppo_epoch", config["algo"].get("ppo_epoch"))
    config["algo"]["entropy_coef"] = trial.params.get("entropy_coef", config["algo"].get("entropy_coef"))
    config["model"]["hidden_sizes"] = [
        trial.params.get("hidden_size_1", config["model"]["hidden_sizes"][0]),
        trial.params.get("hidden_size_2", config["model"]["hidden_sizes"][1])
    ]
    
    # 保存所有从配置文件中读取的超参数
    save_yaml_config(config, args.config)

if __name__ == "__main__":
    main()
