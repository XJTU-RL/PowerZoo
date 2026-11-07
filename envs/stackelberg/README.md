# 顺序需求响应

当前使用`pricing_env.py`与`training.py`：真实价格先公布，消费者看到后响应，再执行一次原生物理区间。
两套正式算例为`stackelberg`小网络和`stackelberg_13bus`；使用`sequential_ppo`独立收益训练。

```powershell
python scripts/train.py --config configs/experiments/stackelberg_smoke.yaml
python scripts/check_pricing.py --train --require-cuda
```

数据、账本、精确阶段响应、训练与checkpoint恢复见[专项说明](../../docs/pricing-physics.md)。
旧实现与GUI已保留在`archive/physics_refactor_stackelberg`，不再作为正式训练路径。
