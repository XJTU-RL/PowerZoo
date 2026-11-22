# 离散 VVC 多智能体基线

公开入口 `scripts/train.py` 支持 `ippo`、`qmix`、`vdn` 和 `coma`。这四种算法使用 VVC 的局部设备观测、EP 团队状态与从零开始的逐设备离散动作。兼容性检查在启动数据和 worker 前执行。训练和确定性推理使用前馈网络。

| 算法 | 策略更新与状态 |
| --- | --- |
| IPPO | 每台设备有独立的 PPO actor、局部价值函数和优势估计，使用同一个环境团队奖励。 |
| QMIX | 每台设备估计局部动作值，全局状态生成非负的混合权重。 |
| VDN | 同样使用局部动作值，团队动作值直接求和。 |
| COMA | 每台设备有局部 actor，集中 critic 固定其他设备的动作，按本设备可用动作及其策略概率计算反事实基线。 |

动作空间声明每台设备的固定槽位。当前 VVC 适配器给出全可用的离散掩码，SOC、功率及其他物理约束由仿真核心处理。算法消费逐帧掩码，并在采样、贪心选择和自举时排除被屏蔽的槽位。QMIX 和 VDN 按 worker 收集完整回合，按回合淘汰，采样时使用一步 TD 更新。自然终止和物理失败切断自举。外部时限截断使用向量环境自动重置前的观测、共享状态和掩码。

## 运行 IEEE 34Bus 冒烟

以下命令只检查配置，不加载仿真。

```bash
python scripts/train.py --config configs/experiments/vvc_34bus_discrete_marl_quickstart.yaml \
  --algo qmix --set algo.batch_size=8 --validate-config
```

在有 OpenDSS 的服务器运行以下短测。测试给公开入口提供一个单独生成的合成训练日，在同一 IEEE 34Bus 算例训练 48 步，检查更新、checkpoint 和确定性推理。数据清单只有训练切分。

```bash
POWERZOO_DISCRETE_VVC_SMOKE=1 python -m pytest -o addopts='' -q \
  tests/test_discrete_marl_vvc_smoke.py -m integration
```

另有获准使用的训练清单时，可在同一 CLI 命令中用
`--set env.data.source=files --set env.data.path=/absolute/path/manifest.json`
指定文件，并去掉 `--validate-config` 开始训练。这份实验配置按 `sunny` 场景名筛选。
QMIX 和 VDN 在训练回合积累到至少 `algo.batch_size` 个样本后开始更新。IPPO 和 COMA 使用各自配置的更新参数。

`--set train.model_dir=/path/to/models/checkpoint_episode_2` 可加载保存的权重和优化器。恢复时复用物理清单检验观测和动作映射。构造 runner 后，`predict(obs, masks)` 返回确定性设备动作。QMIX 和 VDN 的 checkpoint 还保存目标网络和训练计数，重启后从新的完整回合填充回放；checkpoint 适用于策略推理和继续训练，逐步精确续跑需要额外保存回放、worker 和随机数状态。
