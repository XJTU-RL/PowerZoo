"""Train an algorithm."""
import argparse
import json
import sys 
import os
# 将当前文件所在目录的上级目录添加到系统路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

#         default="qmix",

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
        ],
        help="选择环境: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag,powerzoo.",
    )
    # 添加实验名称参数，默认为"installtest"
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
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
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line
    for section, params in algo_args.items():
        print(params)
    # 开始训练
    from runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
