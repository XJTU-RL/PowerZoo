# SmartGrid 集中控制示例

受支持入口为 [`scripts/train_smartgrid.py`](scripts/train_smartgrid.py)，从项目根目录运行：

```bash
python examples/single_agent/scripts/train_smartgrid.py --steps 240
```

配置、适配器和旧入口的说明统一维护在
[SmartGrid 集中控制基线](../../envs/smartgrid/single_agent/README.md)。
经典 VVC 的集中控制适配器属于另一任务，见 [`envs/vvc/single_agent.py`](../../envs/vvc/single_agent.py)。
