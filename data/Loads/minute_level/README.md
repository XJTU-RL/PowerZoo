# 分钟级负荷数据说明

本目录用于接收负荷插值输出，CSV、图表和报告均不再纳入版本控制。
原始小时曲线保存在各 `node_systems/` 算例目录，避免维护重复输入。

从仓库根目录重新生成：

```bash
python data/Loads/unified_load_interpolator.py --input-dir node_systems/34Bus/loadshape/data_without_noise --output-dir data/Loads/minute_level
```

输出 `LoadShape1_minute_level.csv`、`LoadShape2_minute_level.csv` 和
`LoadShape3_minute_level.csv`；每列为负荷系数，完整一年为525600行。
换算例或平滑参数后，应重新核对点数、范围和时间索引；插值数据不是实测分钟数据，
历史报告的误差或相关系数也不能代替新输出的检查。

后续分析及目录约定见[负荷工具说明](../README.md)。
