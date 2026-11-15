# 外部算例导入与物理基线：英国 MV-LV 联合网络（第一阶段）

状态：**第一阶段完成，等待审查**。本阶段只建立可复现导入、结构清单与原生物理基线；
不含 RL 训练、不改 9500 现有实验语义。基线提交：`147f90e7`（main，PR #57 合并后），
任务分支：`codex/uk-mvlv-import`。

## 1. 数据来源与许可（不可再分发）

- 上游：<https://github.com/deakinmt/uk-mvlv-models>，master 分支，最新提交
  `e97e260ed46b2ad818f1a701ba2b95f269c9c12c`（2021-12-11，此后未更新）。
  接入对象为 `HV_UG_full.zip`（9,257,503 字节，
  SHA-256 `23ab52a9bc3d678ae2e9e3d0c0376a005aacd4471bb3498d8318a4cf64074808`）。
- 出处论文：M. Deakin, D. Greenwood, S. Walker, P. C. Taylor, "Hybrid European
  MV-LV Network Models for Smart Distribution Network Modelling", 2021 IEEE
  Madrid PowerTech（arXiv:2009.14240）。
- **许可：上游仓库无 LICENSE 文件**（GitHub API `license=null`，默认保留所有
  权利），其素材来源 UKGDS（MV 网）与 ENW LVNS（LV 网）另有条款。因此本仓库
  **不提交**压缩包、展开模型或完整派生模型；`data/external/` 已加入
  `.gitignore`，只提交来源登记（`utils/external_cases.py`）、下载脚本、适配
  配置、清单工具、测试与本文档。如需公开分发派生数据，必须先取得作者书面
  许可——在此之前保留为明确阻塞。

获取与校验（默认落到仓库 `data/external/`，可用 `POWERZOO_EXTERNAL_CASES`
或 `--data-dir` 改到仓库外）：

```bash
python scripts/download_external_case.py --case uk_mvlv_hv_ug          # 下载+校验+解压+写 manifest
python scripts/download_external_case.py --case uk_mvlv_hv_ug --verify # 复核
```

下载脚本逐成员检查压缩包路径（拒绝绝对路径与 `..`），校验 SHA-256 与字节数，
任何不符即删除产物报错；来源（URL、上游提交、SHA-256、下载时间、引用、许可
说明）写入 `data/external/uk_mvlv_hv_ug/manifest.json`，与代码内登记值双向核对。

## 2. 模型结构与接入约定（不修改上游文件）

模型组织：`master_mvlv.dss` 为唯一入口，redirect 引用 `transformers/regcontrols/
lines/loads.dss`（MV 网）与 `redirect_lv_ntwx.dss`（75 个 LV **slave** 子网，
经文本 redirect 挂入同一电路；各子网目录里的 `master_network_*.dss` 是可独立
运行版，**不得**遍历执行）。`lds_edit.dss` 把 75 个 11kV 中压集中负荷全部
`enabled=False`——需求改由 LV 明细网承载，**重新启用会重复计负荷**，禁止。
`generators.dss`（22 台 DG，全部 `enabled=False`）未被主入口引用，不进电路。
主入口末尾自带 `solve` 与 `energymeter`；`set voltagebases="33.0,11.0,0.3984"`。

- **零修改声明**：本阶段对上游模型文件不做任何修改（无 overlay、无设备重写）。
  适配层 = 来源登记 + 系统 YAML + 清单/对照工具。DSSSession 把模型目录整体复制
  到临时目录后在副本上编译，源目录只读（沿用 9500/8500 的既有机制）。
- 反斜杠 redirect（`lvNetworks\...`）在本仓 DSS C-API 0.14.5 引擎（macOS 实测）
  可直接编译，无需改路径；该结论随 A/B 对照逐点一致共同验证。
- 联合网络经主入口引用 LV 子网：引擎实测 75 个 slave 网全部进同一电路
  （Generator 计数 0 印证 generators.dss 未被引用）。

## 3. 结构清单（引擎实测，见 `scripts/inspect_external_case.py`）

运行 `python scripts/inspect_external_case.py --case uk_mvlv_hv_ug`，报告写入
`results/validation/external/uk_mvlv_hv_ug_inventory.json`。要点：

| 项 | 值 |
| --- | --- |
| 电路/频率 | `hv_ug_mvlv`，50 Hz |
| 电压基值 | 33 / 11 / 0.3984 kV（LV 基值=230V×√3） |
| 母线数（NumBuses） | **37,629**（全部为三相节点齐备母线） |
| 电气节点（NumNodes） | **112,887**（即相节点数；与任务书参考值一致） |
| 负荷 | 19,072 个：18,997 启用（19,031 kW / 6,255 kvar）+ 75 禁用（铭牌 24,274 kW）；18,980 单相 230V + 92 三相 |
| 线路 | 37,552 条，全部启用、无开断 |
| 变压器 | 77 台（2×33/11kV 26.4MVA 主变 + 75 个 LV 站的 11/0.416kV 配变） |
| 调压器 | 2 个原生 RegControl（vreg=104、band=1.3、ptratio≈63.51、NumTaps=19、tap 范围 0.9708–1.1764，MaxTapChange=16），绑定 2 台主变第 2 绕组 |
| 电容/PV/储能/电容控制 | **全部为 0**（模型无此类设备） |
| 依赖闭包 | 从入口经 redirect 展开 1,159 个唯一脚本文件（.dss/.txt），逐文件 SHA-256；缺失依赖报 `文件:行号` |
| 源 | Vsource 33kV，pu=1.02，经近零阻抗 sourceZ 线到 301 母线 |

「设备存在」与「由智能体控制」是两个口径：清单只报告设备存在与原生控制器
绑定；智能体控制范围由 `configs/systems/UKMVLV_HVUG.yaml` 的 `devices:` 段
声明（本阶段登记调压器 `action_num: 20`——NumTaps=19 为素数，动作间隔必须
整除 19，故 20 档；无电容/储能/PV 可控项）。

## 4. 原生基线与 A/B 对照（`scripts/check_uk_mvlv.py`）

- **A1**：DSS 引擎按模型原文编译（入口自带 solve；求解器为引擎默认
  15 次迭代 / 1e-4 容差，静态控制模式）。
- **A2**：同引擎编译后显式对齐 DSSSession 求解设置（snapshot、controlmode=
  static、maxiterations=100、tolerance=1e-8）再求解——与 B 严格同口径。
- **B**：`envs/powergrid.DSSSession` 加载同一模型（临时副本 + 全文件 SHA-256
  身份 `model_sha256`），`solve(native_controls=True)` **保留原生 RegControl**
  参与求解（会话默认路径关闭控制、由智能体直接改分接，本基线不用）。

结果（同引擎 dss-python 0.15.7 / C-API 0.14.5，同源电压 pu=1.02，同负荷）：

| 对照 | 逐节点电压 max\|Δ\| | 源端 P/Q 相对差 | 损耗相对差 | 元件数/档位/开断 |
| --- | --- | --- | --- | --- |
| **A2 vs B** | **0.0（112,887 节点逐点全等）** | **0.0** | **0.0** | 全等，match=通过 |
| A1 vs A2 | 7.2e-6 pu（均值 6.7e-7） | 6.2e-6 | 1.4e-5 | 全等 |

- 公差依据：同引擎同模型两处编译的数值路径一致，A2/B 应达浮点噪声量级
  （电压 ≤2e-6 pu、功率 ≤1e-6），实测为**精确零差**；超出即判失败退出非零。
- A1→A2 的差异即「引擎默认 vs 会话设置」的求解器灵敏度（1e-4→1e-8 容差），
  单独量化报告，不与 A/B 对照混同。
- 原始快照收敛（Newton 14 次迭代 / 3 次控制迭代）但**存在原始越限，按原样
  保留报告**：max 1.0825 pu、min 0.9657 pu、55,253/112,887（49%）节点高于
  1.05 pu（源 1.02 pu + vreg 104 的整定结果）。本阶段不设校准工况、不改源模型。
- 功率口径：源端注入 19,957.3 kW / 9,042.5 kvar（TotalPower 取负=送出）；
  损耗 760.57 kW / 2,732.8 kvar（引擎 `Losses` 单位为 W，除以 1000 记 kW；
  逐支路求和与 `Circuit.Losses` 双口径**完全一致**）。

## 5. 性能量测（一次性，不代表 RL 训练吞吐）

机器：Apple Silicon arm64（macOS 26.4），10 核，Python 3.12.8；计时
`time.perf_counter` 单进程顺序量测；峰值内存 `ru_maxrss`（进程单调峰值）。

| 段 | 耗时 |
| --- | --- |
| A 引擎编译（含入口自带首次 solve，含控制迭代） | 2.35 s |
| A2/A1 静态再求解（热启动，设置对齐） | 0.062 s |
| 扰动求解（`loadmult=1.01` 触发真实重解，3 次牛顿迭代） | 0.055 s |
| `AllBusVmagPu` 提取（112,887 节点） | 0.002 s |
| B DSSSession 构造（copytree 3,282 文件 + 全文件 SHA-256 + 编译 + 连接快照 + 设备发现 + 拓扑建图） | 6.37 s |
| B 原生控制求解 | 0.064 s |
| B `read_state`（37,629 母线逐相电压 + 37,629 支路电流/损耗/负载率全量遍历） | **2.82 s** |
| B 满帧负荷写入探针（19,072 条逐负荷 Edit，等效 `apply_operating_row`） | 0.44 s |
| 峰值内存 | 1.43 GB |

读数说明：入口 `solve` 后不改变量测的重复求解（≈0.012s）只是增量重解，
不代表求解成本；上表「扰动求解」才是真实重解成本。**热启动静态求解时间
不能当成 RL 训练吞吐**。按现有会话实现推算的单步成本下限约为
0.44（负荷写入）+0.06（求解）+2.82（read_state）≈ 3.3 s/步（VVC 环境实测
单步 4.0 s，含观测组装）——这是第二阶段的性能优化目标（见 §8），不是本阶段
的验收项。

## 6. 会话层扩展：停用元件的节点序（唯一代码改动于共享层）

上游模型含 75 个编译期即禁用的负荷。引擎对停用元件不构建节点序表，
`DSSSession.reset()` 原实现读 `NodeOrder` 会报 "Nodes are not initialized"，
导致合法模型无法加载（9500/8500 无停用元件，从未触发）。扩展
（`envs/powergrid/session.py`）：启用元件仍读原生 `NodeOrder`（存量行为
逐字节不变）；停用元件按 `BusNames` 语义推导节点序（显式相别后缀优先，
无后缀取 1..NumPhases，不足 NumConductors 补中性导体 0）。该推导已在
UK/8500/9500 共 70,842 个启用元件上与原生 `NodeOrder` 逐一对拍全等；
合成小模型测试覆盖停用元件加载与 A/B 一致性。

## 7. 测试与验证记录

本机（上述机器，dss-python 0.15.7 / 引擎 C-API 0.14.5 [OpenDSS SVN 3723] /
opendssdirect 0.9.4 / numpy 2.2.4 / torch 2.9.1 / pandas 2.3.3 / gymnasium
1.1.1）实测：

- `tests/test_external_cases.py`：**13 项全部通过**（16.8 s）——合成小模型
  11 项（来源校验/闭包缺失定位/清单计数/会话加载停用元件/A-B 一致/下载校验
  往返/篡改拒绝/越界成员拒绝）+ 真实 UK 2 项（清单钉扎值、A2vsB 逐点一致）。
  大数据测试标记 `requires_external_case`，未下载数据时明确 skip 并给出下载
  命令；**CI 无数据即不运行，不假装已跑**。
- 既有回归：`test_powergrid_session`（含 model_sha256/路径隔离）、
  `test_divergence_semantics`、`test_observation_projection`、
  `test_large_systems`（8500/9500/bal 构造+单步+投影，即既有算例固定动作
  单步回放）、`test_configuration_contract/pipeline/unified`：全部通过。
- 固定动作回放记录：既有算例沿用 `test_large_systems` 的固定基线动作单步
  （通过）；UK 算例 VVC 探针（2 调压器智能体取 20 档中位动作）单步
  reward=-1994.36、观测有限（见 §5 耗时）。
- `python -m pip check`：PowerZoo 运行链（dss/torch/numpy/pandas/gymnasium）
  无冲突；本机共享 base 环境有 3 条与本项目无关的既有冲突
  （pymilvus/langgraph/langchain 的传递依赖版本），非本任务引入。
- `python scripts/check_runtime.py --train`：**POWERZOO_RUNTIME_OK**（DSS 双绑定
  验证、VVC 单/双 worker、HAPPO 24 步短训练全过，CPU 模式）。

VVC 环境端到端可构造性（UK 算例，超出最低要求的验证）：
`ConfigLoader().load("happo","vvc",system_ref="UKMVLV_HVUG",
values={"env":{"observation_neighborhood":2}})` → 构造 15.5 s、reset 8.9 s、
观测 46 维（k=2 邻域投影）、2 智能体、固定动作单步 4.0 s 通过。这验证了
配置→会话→观测投影链路，**不构成训练验收**。

## 8. 已知限制与下一阶段建议

1. **未做 RL 训练**（本阶段范围外）。单步 ~4 s 是第二阶段性能门槛：
   `read_state` 全量遍历（2.8 s）与逐负荷 Edit 风暴（0.44 s）需按
   9500 先例批量化（LoadShape/负载抽样/观测按需提取）后再评估训练可行性。
2. **原生控制器的长期处置未决**：本基线经 `native_controls=True` 保留
   RegControl 且 A/B 一致；正式 VVC 训练路径会关闭原生控制、由智能体接管
   分接（既有语义）。两模式的切换语义在 UK 37k 母线规模下的奖励/电压影响
   未评估，属第二阶段。
3. `base_load_kw` 汇总含 75 个停用负荷的铭牌功率（24,274 kW）——只是
   正值校验的基准，不影响物理（停用负荷不出力），但第二阶段做负荷基准
   统计时需改为仅启用负荷。
4. 模型无电容/PV/储能：VVC 在 UK 算例只有 2 个调压器智能体，可控维度低；
   如需 DER 场景（SMART-DS 或 UK 加 PV），须先按「适配层 overlay」设计
   显式增补设备并单独验收，不改上游文件。
5. 上游 2021 年后不再维护、无负荷时序（纯静态快照）；时序场景需配
   PowerZoo 场景管线（`DataConfig`）合成或外部实测曲线。
6. 数据再分发许可未确认（§1）——公开分发任何派生模型前必须停下解决。
7. 下一阶段（经审查后）：SMART-DS 时序导入、UK 算例性能批量化、
   逐负荷分组（`load_groups` 显式映射，替代名称哈希分组）、UK 光伏 overlay
   设计与验收。

## 9. 修改文件清单

| 文件 | 内容 |
| --- | --- |
| `utils/external_cases.py` | 新增：外部算例登记（URL/提交/SHA-256/许可）、下载/校验/定位、manifest |
| `scripts/download_external_case.py` | 新增：下载 CLI |
| `envs/powergrid/case_inventory.py` | 新增：依赖闭包、引擎清单、A/B 对照 |
| `scripts/inspect_external_case.py` | 新增：清单 CLI（JSON 报告） |
| `scripts/check_uk_mvlv.py` | 新增：A1/A2/B 基线对照 + 性能探针 |
| `envs/powergrid/session.py` | 修改：停用元件节点序推导（§6，唯一共享层改动） |
| `configs/systems/UKMVLV_HVUG.yaml`、`_registry.yaml` | 新增/登记：系统配置 |
| `tests/test_external_cases.py`、`tests/conftest.py`、`pytest.ini` | 新增测试与 marker |
| `.gitignore` | 忽略 `data/external/` |
| `docs/uk-mvlv-import.md` | 本文档 |

复现命令：

```bash
python scripts/download_external_case.py --case uk_mvlv_hv_ug
python scripts/inspect_external_case.py --case uk_mvlv_hv_ug
python scripts/check_uk_mvlv.py
python -m pytest tests/test_external_cases.py -o addopts="" -q
```
