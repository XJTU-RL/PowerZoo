# PISTG-LDCPPO 论文复现

复现 Huang, Yang & Wang, "Physics-Informed Graph-Based Safe RL with
Constraint-Specific Critics for Community Battery Coordination in
Distribution Network"（IEEE Transactions on Smart Grid, 2026;
arXiv:2610.33654）的 IEEE 33-bus 算例：OpenDSS 场景 + PISTG 编码器 +
LDCPPO 约束策略优化。

**忠实度原则**：论文明示的参数与公式逐字对齐（映射表见下）；论文未
明说的参数取最接近其语义的默认值，全部记录在
`docs/pistg-repro-deviations.md`，标定依据可追溯。

**论文原文**：本地 PDF `papers-repro/pistg-ldcpo/arXiv2610.33654v1.pdf`
（[目录索引](../../papers-repro/README.md)；线上来源 arXiv:2610.33654）。

## 论文公式 ↔ 代码映射

| 论文位置 | 内容 | 代码 |
|---|---|---|
| Sec. IV-A | IEEE 33-bus、CBESS@6/11/16/31、288×300s | `node_systems/33Bus_CBESS/`（框架 case 库，生成期对拍文献值；`configs/systems/33Bus_CBESS.yaml` 登记） |
| 式(1a-f) | CBESS 能量动态与约束 | `environment/physics.py` |
| 式(2)(3) | PV 弃光与逆变器圆约束 | `physics.map_pv_action` |
| 式(4)-(6) | DistFlow 网络（OpenDSS 全 AC 求解替代） | `environment/cbes_env.py` + `DSSSession` |
| 式(7) | 日经济成本+约束预算优化问题 | `config.ScenarioConfig`（预算标定） |
| 式(8)(9) | W=12 窗口张量/线路特征 | `train.StateEncoder`、`cbes_env._observation` |
| 式(10) | PIAM 物理信息邻接 | `models/pistg.physics_informed_adjacency` |
| 式(11) | GCN×2 传播 | `models/pistg.GCNLayer` |
| 式(12)(13) | 注意力池化 → 图级序列 | `models/pistg.AttentionPooling` |
| 式(14)-(16) | 双 TCN → 拼接 d_STG=80 | `models/pistg.TCN/PISTGEncoder` |
| 式(17)(18) | 编码器/PIAE 组合 | `models/piae.PIAE` |
| 式(19) | L_KCL 节点平衡损失 | `piae.loss`（排除松弛母线） |
| 式(20) | L_VDrop 电压降一致性 | `piae.loss`（p.u. 系统） |
| 式(21)(22) | L_recon 与 β 加权 | `piae.loss`、`pretrain.py` |
| 式(23)(24) | CMDP 元素/奖励/三成本 | `environment/cbes_env.py`、`physics.py` |
| 式(25)(26) | SOC 感知动作映射 | `physics.map_cbess_action` |
| 式(27)-(29) | 折扣回报与约束规划 | `algo/ldcppo.py`（预算机制） |
| 式(30)-(33) | 逐流 GAE、ζ 标准化、critic 回归 | `ldcppo.compute_gae/update` |
| 式(34)(35) | 高斯策略与 clipped surrogate | `algo/networks.GaussianActor` |
| 式(36) | 参数更新与乘子对偶上升 | `ldcppo.update` |
| 表 III/VI/V | 评估口径 | `evaluate.py` |

## 运行

```bash
# 0) 数据：下载 Ausgrid（≈57MB zip，镜像 pierreh.eu；详见 prepare_profiles.py 文档）
#    并生成剖面（1096 天，~1 分钟）
python -m research.pistg_ldcpo.data.prepare_profiles

# 1) Stage 1：PIAE 预训练（论文表 II 超参）
python -m research.pistg_ldcpo.pretrain --episodes 500 --variant pistg

# 2) Stage 2：LDCPPO 训练（--method pistg|stg|flat）
python -m research.pistg_ldcpo.train --method pistg --episodes 2000 \
    --piae results/pistg_ldcpo/piae_pistg_ep500.pt --tag full_pistg

# 3) 评估（100 测试回合，论文表 III/VI/V 口径）
python -m research.pistg_ldcpo.evaluate --run-dir results/pistg_ldcpo/full_pistg

# 全量流水线（PIAE×2 + 训练×3 + 评估×3，约 9-10 小时）
zsh research/pistg_ldcpo/scripts/run_full.sh
```

测试：`python -m pytest tests/test_pistg_repro.py -o addopts="" -q`
（依赖外部数据的测试在数据未生成时显式 skip 并说明原因）

## 目录

```
case/               （已迁出）算例在框架 case 库 node_systems/33Bus_CBESS/
data/               Ausgrid → 5-min 逐日剖面（prepare + 运行时存取）
environment/        CBESS 环境（CMDP）、物理层纯函数、会话工具
models/             PISTG 编码器与 PIAE
algo/               LDCPPO（网络 + 更新规则）
pretrain.py         Stage 1 编码器预训练
train.py            Stage 2 训练（含 STG/flat 基线开关）
evaluate.py         测试期评估
scripts/run_full.sh 全量流水线
```

复现范围（本轮）：PISTG-LDCPPO + LDCPPO（flat）+ STG-LDCPPO 三配置
（论文表 VIII 消融主线）。TD3/SAC/PISTG-CPO/PI-GR 变体与 69-bus 为
第二阶段。
