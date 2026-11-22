# 集中控制训练与评估

| 后端 | 动作支持 | 续训 |
| --- | --- | --- |
| `sb3_ppo`（默认） | 连续 Box、含离散设备的量化 Box、DSR MultiDiscrete | — |
| `sb3_sac`、`sb3_td3` | 实际设备均为连续动作，策略空间须为一维有界 Box | — |
| `ldcppo` | CBESS 连续 Box，逐步校验 VD、VN、LO 三个约束成本 | 支持回合边界恢复 |

SAC 和 TD3 使用 Stable-Baselines3 的实现及默认超参数。可在实验配置中将
`run.algo` 设为同名后端，并覆盖 `algo.buffer_size`、`algo.learning_starts`、
`algo.batch_size` 等配置。启动时仍以 `--backend` 选择集中控制后端。

VVC/DSR 的新格式 checkpoint 仅支持同一实际场景的策略重载；即使传入另一划分的正确摘要，也不支持跨划分回放。IEEE13 参考验收脚本自行写出的旧 PPO 只校验物理伴随文件，由验收脚本另行核对训练/验证工况是否重复。新 SB3 模型在 ZIP 内标记需要绑定清单：即便残留旧 `.physics.json`，丢失 `.binding.json` 也会拒绝重载。

集中入口在准备运行前按最终配置预检：`--evaluate` 检查策略重载，
`--resume` 检查训练续跑。已识别的 VVC/DSR 新格式元数据若声明不同划分，
在启动仿真前拒绝；同划分仍须在加载时核对实际场景和模型摘要。缺少绑定清单
不会在预检中被认定为旧格式，旧 PPO 兼容仍由加载器验证。

CBESS 的 `env.data_dir` 下按划分存放 `train/` 和可单独准备的 `validation/`。
每个目录包含 `profiles.npz` 和 `manifest.json`。NPZ 使用现有的
`day_keys`、`p_load_kw`、`q_load_kvar`、`pv_available_kw`、
`price_aud_per_mwh` 字段。清单的 `split` 映射必须覆盖该包的日期且全属于选定划分。
`pv_peak_kw` 为按 DER 母线编号索引的静态设备容量输入，两份包应使用同一设备配置。
混合划分的旧归档供研究复现入口使用。集中控制入口使用独立授权交付的剖面包。

```sh
python scripts/train_centralized.py --backend ldcppo \
  --config configs/experiments/cbess_ldcppo_quickstart.yaml \
  --output results/cbess --split train --steps 288

python scripts/train_centralized.py --backend ldcppo --resume \
  --config results/cbess.environment.yaml --output results/cbess --steps 576

python scripts/train_centralized.py --backend ldcppo --evaluate \
  --config results/cbess.environment.yaml --output results/cbess \
  --split validation --eval-scenario-sha256 "$CBESS_VALIDATION_SHA" --eval-episodes 1
```

评估 train checkpoint 的 validation 路线须提供单独授权包的摘要。
数据方可对已授权的包计算摘要：

```sh
python -c 'from envs.cbess.profile_store import ProfileStore; import sys; print(ProfileStore(sys.argv[1], split="validation").fingerprint())' /path/to/profiles/validation
```

SB3 checkpoint 使用 `<checkpoint 文件名>.binding.json` 绑定模型文件摘要，
LDCPPO 将绑定元数据和训练状态写入同一原子 checkpoint。保留相应文件和
`.environment.yaml`，加载时会检查后端、观测和动作语义、物理模型、
实际场景与解析后配置。旧版 PPO 的双参数 `load_model(path, env)` 仍可使用旧
`.physics.json` 校验。LDCPPO 从已保存的回合检查点恢复优化器、对偶乘子、预算和
随机状态。续训使用更大的总步数，可提高 `train.total_episodes` 或调整
`train.checkpoint_every`。训练步数耗尽的最后一批可能短于环境回合，它在收集边界
完成一次更新。重新续训从下一次环境重置开始。自然结束的 CBESS 日回合不自举，
外部截断或收集边界遵循 `algo.bootstrap_at_episode_end`。

轻量回归：

```sh
python -m pytest -q -o addopts="" tests/test_single_agent_backends.py
```

电气冒烟只在南洋的隔离仓库副本上运行。它使用合成 CBESS 训练/验证剖面和
真实 OpenDSS 求解，同时覆盖原有 VVC PPO 的量化动作路径。

```sh
python tests/single_agent_nanyang_smoke.py /path/to/new/smoke-output
```
