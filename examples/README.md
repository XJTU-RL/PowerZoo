# 训练入口

从仓库根目录运行。公共MARL入口为 `python scripts/train.py --config <实验文件>`，
五类环境共用解析、校验、种子准备、快照、worker构造和训练调度。

- [多智能体示例](multi_agent/README.md)
- [SmartGrid集中控制PPO](single_agent/README.md)
- [脚本清点与参数流](../docs/training-entrypoints.md)

环境特定的Python启动器、Shell参数预设和旧单智能体配置已归档到
[archive/legacy-training](../archive/legacy-training/README.md)，不作为可运行示例。
