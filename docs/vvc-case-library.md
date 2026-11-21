# VVC 算例库

首批参考案例为 IEEE 34 和 IEEE 123。登记基线是远端 `main` 的
`97d69b8e19a830ce9ff03aeec3af50eb6997af86`（PR #66）。
[算例清单](../scenarios/vvc/cases.yaml)记录公开来源、文件摘要、设备及核验状态。

## 取得公开原件和生成参考实验

| 层级 | 文件及用途 |
| --- | --- |
| 公开原件 | 从清单中的固定上游版本取得，保留字节和通知。用 `verify-source` 核对已登记的原件。IEEE 来源参见 [EPRI OpenDSS 算例说明](https://opendss.epri.com/ExamplesDocsandTestCases.html)。 |
| PowerZoo 静态底板 | `stage` 从已入库、锁定摘要的静态运行电路抽出研究储能。派生底板保留 PowerZoo 已有的静态负荷和电压基值适配，以及 IEEE 电路中的电容和调压器。 |
| 任务场景 | [34 储能](../scenarios/vvc/34Bus/storage.dss)和 [123 储能](../scenarios/vvc/123Bus/storage.dss)分别声明接入母线、相别和额定值。场景通过 `env.overlays` 显式选用。 |
| 实验配置 | [共用冒烟预设](../configs/experiments/vvc_reference_smoke.yaml)声明算法、步长、合成数据选项和预算。生成配置走现有 `ConfigLoader`。可用 `--algo mappo` 选非共享 MAPPO。 |

```sh
python -m scripts.vvc_case_library list
python -m scripts.vvc_case_library verify-source ieee34 /path/to/upstream/34Bus
python -m scripts.vvc_case_library stage ieee34 results/vvc-reference-34
python -m scripts.vvc_case_library stage ieee123 results/vvc-reference-123
```

每次指定新的输出目录。生成的 `model/network.dss` 可单独编译，实验增加
`model/overlays/storage.dss`。输出同时包含 `experiment.yaml` 和
`manifest.json`，后者记录已入库的源闭包、派生底板、叠加层和实验配置的 SHA-256。
训练入口接受生成的 YAML，例如在南洋隔离目录使用
`python scripts/train.py --config results/vvc-reference-34/experiment.yaml --algo mappo`。
生成配置的模型路径是绝对路径，移动目录后应在目标机器重新生成。

34 的储能接在 890、832，123 接在 33.1、114.1、67 三相、300 三相。各台的
100 kW／200 kWh、50% 初始 SOC、10% 保留和充放电效率 95% 源于 PowerZoo
研究假设。每个电容为单级二态。调压器采用 33 个离散档位，物理端点及执行绕组
从 OpenDSS 读取。储能采用 33 个零点对称的功率档位，实际端口功率受原生铭牌及
当前 SOC 限制。34／123 的原生电容、调压器、储能组合分别是 2／6／2 和 4／7／4，
均无 PV。源电路和两个场景的动作边界各自以系统 YAML 和 DSS 为准，报告记录读回值。

默认 24 个 3600 秒间隔。既有 [时序生成和读取入口](../envs/powergrid/scenario.py)
将分组倍率应用到 DSS Load 铭牌 kW／kvar。显式 `data.load_groups` 优先，其余
由设备名确定性分组。天气为合成辐照度和组件温度，两个参考场景均无光伏出力。
合成种子、分组、单位和文件清单沿用 DataConfig／ScenarioDataset。
`train` 用于拟合，`validation` 用于开发和选择，`test` 保留给最终评估。
本轮物理验收只写一个独立的合成 validation 记录，未打开封存分片。
生成的实验和既有 smoke 实验使用的合成器早已随仓库发布。Mac Studio 逐字节核对了 34／123／4／37／69 的 11 个上游原件。公开原件继续从上游获取。

## 案例状态

| 算例 | 角色与证据 |
| --- | --- |
| IEEE 34 | 第一参考。南洋验证独立叠加层、实测设备动作和一个区间潮流。线路热额定来源待审计。 |
| IEEE 123（PowerZoo 原生） | 第二参考。南洋完成相同的物理验证。126 条线路无来源可核的热限。 |
| IEEE 4（平衡 Δ–Y） | 外部兼容性。历史试验用 n3／n4 的研究电容，电压出现越限。公开原件保留上游版本。 |
| IEEE 37（不平衡 Δ） | 外部兼容性。历史试验使用两台原生调压器并记录通用 VVC 电压越限。合规核验需按同一工况和基准计算 AB／BC／CA 线间相量电压。 |
| Baran–Wu 69（MATPOWER 8.1） | 外部转换兼容性。历史试验用 64／65 的研究电容，电压出现越限。原始及转换后的数据保留在仓库外。 |
| IEEE 8500 | 规模与潮流诊断。本轮一个额定负荷快照收敛且有越限。此前候选分支有固定设备和原生调压的合成日诊断。 |
| PNNL 9500 平衡／不平衡 | 规模与潮流诊断。独立复位后 0.85 倍铭牌点收敛且有越限，1.00 倍点两个变体均首步未收敛。 |
| IEEE 13 | 主线已有 VVC 冒烟案例，见 [既有物理记录](vvc-physics.md)。 |

4／37／69 的 HAPPO、非共享 MAPPO 及集中式 PPO 兼容结论引用只读候选
`sheldon123z/vvc-compact-policy-validation` 的 `a2c364458a5e92ba704e23922117836be8fc1319`
及后续物理核查 `f6746ba3cf6bf85883b678a87015ca19f3192a79`。
历史九个组合各做四步短训、一次更新、恢复与 24 步回放，均出现电压越限。
本分支登记这批证据及输入指纹。复跑需要各自的外部原件，4／69 还需历史研究叠加层。
公开输入、转换方式和静态验收见 [OpenDSS 算例接入记录](https://github.com/XJTU-RL/PowerZoo/blob/a2f0c6a7aded784b5f4ce96761dff15e8011d2ed/docs/opendss-case-intake.md)。
MATPOWER 8.1 的 BSD 代码许可排除了案例数据。IEEE 4／37 的再分发权限仍待确认。
[MATPOWER 8.1 许可](https://github.com/MATPOWER/matpower/blob/8.1/LICENSE)

既有 34Bus_PV 属于 SmartGrid 控制场景。SMART-DS 和 UK MV-LV 保持各自外部接入与
验收状态。OEDI 123 是独立外部时序变体。33Bus_CBESS 属于 CBES 任务。
[SMART-DS](smart-ds-import.md) · [UK MV-LV](uk-mvlv-import.md) ·
[CBES](training-entrypoints.md) ·
[OEDI 123 时序证据](https://github.com/XJTU-RL/PowerZoo/blob/e1a24eb104b8a32c84504cf5355162941053052c/docs/external-temporal-parity.md)

## 南洋物理读回

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout 120 \
  "$HOME/miniconda3/envs/HARL/bin/python" \
  -m scripts.check_vvc_case_library results/vvc-case-library-physics
```

运行环境为南洋隔离目录，Python 3.10.14、DSS-Python 0.15.7、DSS C-API 0.14.5。
34／123 使用同一条 0.70 倍铭牌负荷的一小时合成输入。每个案例对比原有入口与
独立叠加层，复位和动作后每相电压读数逐值相同，调压执行绕组、动作顺序、源端 P/Q
和损耗一致。基础电路装载时仅有原生设备。切换第一台电容、升第一台调压器一档、
请求末台储能满放，DSS 读回电容状态、1.00625 档位及 76.0 kW／约 0 kvar，
SOC 从 50% 到 10%。一小时可用出力由能量与效率限制
`(0.5 − 0.1) × 200 × 0.95 = 76 kWh`。两案有功平衡残差绝对值均低于 0.05 kW。

| 检查点 | 实测求解 | 相电压范围 p.u. | 低于 0.95／高于 1.05 p.u. 的相节点数 |
| --- | --- | --- | --- |
| 34 动作后 | 收敛 | 0.82644–1.00000 | 63／0 |
| 123 动作后 | 收敛 | 0.94782–1.00624 | 4／0 |
| 8500，1.00 倍 | 19 次迭代收敛 | 0.00000–1.05000 | 4863／0 |
| 9500 不平衡，0.85 倍 | 16 次迭代收敛 | 0.93971–1.07037 | 9／53 |
| 9500 平衡，0.85 倍 | 16 次迭代收敛 | 0.95082–1.07043 | 0／63 |
| 9500 两版，1.00 倍 | 各达 100 次迭代后未收敛，时钟仍为 0 s | — | — |

123 的 126 条线路逐条读回相电流，端点最大值约 451.46 A。闭包无明示线路或
LineCode 热限，400 A 是软件默认读数，来源明确的过载计数保留为空。

规模检查固定编译后的调压器档位、零辐照度和 20°C，独立实例施加负荷倍率。
8500 的三个 0 p.u. 相节点各只接一条常开联络线的开路端，端口定位引用候选
`f83ded295d544c5b974c2cf99462976d43b76207`。9500 两个独立入口均经
`direct` 编译初始化，运行时读回 `NormalSolve` 快照。所有负荷均为 Model 1
（平衡 1275，不平衡 2550），两版各有 178 个原模型 PV。两个收敛的轻载诊断点
仍有越限，额定负荷失败点的电压和损耗留空。9500 的稳定整日工况和
0.85–1.00 倍之间的收敛边界尚无验收证据。

南洋 `report.json` 的 SHA-256 为
`60fe7701eab29dfd506f8f6de010bb47fe93a6c5d7cb0e90985c6176da2b11ff`。

## 集成边界

现有 `34Bus`／`123Bus` 默认入口继续供既有实验使用。独立场景的底板路径、
文件集合与 overlay 指纹不同。统一评测如采用独立场景，应登记其闭包与物理指纹，
为新旧指纹各绑定对应的模型工件。旧 checkpoint 继续使用原配置。
本分支以主线配置契约实现，没有引入候选分支中的策略、回放工具或统一评测改动。
本轮未训练、未评估算法排名、未读取封存数据。
