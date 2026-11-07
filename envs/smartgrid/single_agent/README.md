# SmartGrid 集中控制基线

当前入口：`examples/single_agent/scripts/train_smartgrid.py`，从项目根目录运行：

```powershell
python examples/single_agent/scripts/train_smartgrid.py --steps 240
```

配置复用 `configs/experiments/smartgrid_synthetic_smoke.yaml`，用 `--config` 指定其他 SmartGrid 实验。
适配器为 `envs.smartgrid.single_agent.adapter.SingleAgentVVCEnv`，构造参数使用公共 `SmartGridConfig`。
它与MARL适配器共用runtime_env.Env，不独立实现动作执行、DSS求解或奖励。
默认Box控制将离散设备等距量化；全离散配置支持MultiDiscrete，不枚举联合动作组合。

旧 SingleAgentConfig/SingleAgentTrainingConfig 以及 train_single.py、train_single_agent.py、
train_with_enhanced_callback.py 保留为历史研究代码，不是受支持入口。旧文档中的构造方法不再适用。
完整设计和配置见项目根目录的 `docs/smartgrid-data-runtime.md`。
