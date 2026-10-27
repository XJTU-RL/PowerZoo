# 配置入口

配置规范、真实参数消费位置、覆盖顺序、扩展示例见 [配置指南](../../docs/configuration.md)。
实际代码审计和遗留入口见 [参数流审计](../../docs/configuration-audit.md)。

环境预设使用 `system_ref` 和 `env:`，训练差异使用 `train/model/algo/eval` 等实际算法节。
实验文件放在 `configs/experiments/`。不再使用伪 curriculum/power_system 参数暗示未实现的功能。

```powershell
python examples/multi_agent/scripts/train.py --config configs/experiments/vvc_13bus_smoke.yaml --print-config
```
