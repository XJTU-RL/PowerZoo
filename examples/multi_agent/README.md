# 公共多智能体训练

从仓库根目录执行：

```powershell
python scripts/train.py --config configs/experiments/vvc_13bus_smoke.yaml
python scripts/train.py --config configs/experiments/smartgrid_synthetic_smoke.yaml
python scripts/train.py --config configs/experiments/dsr_13bus_smoke.yaml
python scripts/train.py --config configs/experiments/stackelberg_13bus_smoke.yaml
python scripts/train.py --config configs/experiments/district_dispatch_smoke.yaml
```

这些实验共用 `configs/training/smoke.yaml`。环境选择、系统与环境差异留在各自配置中。
检查最终值使用 `--print-config`，查看来源使用 `--explain env.max_episode_steps`；
显式覆盖使用 `--set model.lr=0.0003`。不需要新增环境专用训练脚本。

`examples/multi_agent/scripts/train.py` 保留为同一CLI的兼容路径；环境专用脚本已归档。
单智能体SB3是独立训练后端，使用共享SmartGrid环境与配置，不伪装成MARL runner。
完整边界及实际验收见 [入口整理说明](../../docs/training-entrypoints.md)。
