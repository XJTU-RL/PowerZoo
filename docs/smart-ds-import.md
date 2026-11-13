# SMART-DS 时序与光伏控制基线（第二阶段）

状态：**小馈线与单日验证完成，等待审查**。本阶段接入 NREL SMART-DS 时序数据
与 PV 控制权机制，完成 p4udt4 小馈线 24 小时四模式基线；不含变电站级数据、
大规模训练或算法收敛性结论。任务分支：`codex/smart-ds-timeseries`（基于
任务 1 的 `codex/uk-mvlv-import`）。

## 1. 数据来源、版本与许可

- 数据集：SMART-DS v1.0（2017 年），NREL 合成配电网，OEDI submission
  [2981](https://data.openei.org/submissions/2981)。**许可 CC BY 4.0**（提交
  页 License 字段）：下载、内部派生与再分发均允许，须署名并注明修改；本仓库
  仍按文件清单从上游获取、不打包原始数据。
- 引用：B. Palmintier et al., "SMART-DS Synthetic Electrical Network Data
  OpenDSS Models for SFO, GSO, and AUS," OEDI, 2020；方法论文 EPSR 190,
  106665 (2021), doi:10.1016/j.epsr.2020.106665。
- 接入算例（文件清单型登记，逐文件 SHA-256 钉扎，清单 JSON 提交仓库）：
  - `smart_ds_sfo_p4udt4`：SFO/P4U 的 **p4udt4** 馈线（4kV 城市馈线），含
    base_timeseries（无 PV 对照）与 solar_high_batteries_none（6 台 PV）两个
    入口；71 文件 / 39.5MB（`configs/external/smart_ds_sfo_p4udt4_files.json`）。
  - `smart_ds_gso_rdt1534`：GSO rural 的 rhs0_1247 变电站 **rdt1534** 馈线
    （12.47kV 农村馈线，472 台 PV）；452 文件
    （`configs/external/smart_ds_gso_rdt1534_files.json`）。
- 下载/发现/校验：`scripts/download_external_case.py --case <id> [--discover]`。
  `--discover` 联网从 S3 列举+解析 LoadShape 引用自动生成最小清单（不含未被
  引用的曲线与兄弟馈线）；日常下载只依赖已提交清单，逐文件校验 SHA-256，
  篡改即报错。**不下载完整区域。**

### 派生修改（逐条登记于 manifest，原始 SHA-256 同步记录）

当前 DSS C-API 0.14.5 与 2017 年模型的参数差异，以精确文本替换修补并记录：

| 文件 | 替换 | 理由 |
| --- | --- | --- |
| PVSystems.dss（两算例） | `kvarlimit=` → `kvarMax=` | 旧参数在当前引擎已拆分为 kvarMax/kvarMaxAbs；映射到发出侧上限，吸收侧随引擎默认（=kvarMax） |
| PVSystems.dss（两算例） | `VV_RefReactivePower` → `RefReactivePower` | InvControl 参数 2020 年官方改名（引擎报错提示） |

除以上两条外**零修改**；上游文件原始字节可从清单 SHA-256 复核。

## 2. 模型结构与官方指标交叉核对（以实际编译清单为准）

引擎实测（solar_high_batteries_none 变体，`scripts/inspect_external_case.py`
同款清单机制）：

| 项 | p4udt4 | rdt1534 |
| --- | --- | --- |
| 母线/电气节点 | 71 / 156 | 2150 / 4173 |
| 负荷对象 | 36（14 对中心抽头单相 + 4 三相 + 独立单相） | 1440 |
| PVSystem | 6（两相 240V；pf∈{1,±0.95}；kVA<Pmpp 的容量受限台；kvarMax=1.1 非对称台） | 472（218 台 kvarlimit、217 台绑定 InvControl） |
| 原生 InvControl | 1（VOLTVAR，6 点 IEEE1547 风格曲线） | 217 |
| 调压器/电容/储能 | 0 / 0 / 0（batteries_none 变体） | 0 / 0 / 0 |

官方 metrics.csv 的 p4udt4 行（16 单相 LV 负荷 + 4 三相、0 PV、0 电容）与
编译清单交叉一致（"16 单相负荷"即 8 个中心抽头用户 × 2 侧；PV 只在 solar_*
变体存在，base_timeseries 行的 Number of PVs=0 与实测一致）。**以编译清单
为准**的原则下无冲突。

## 3. 忠实时序（`envs/powergrid/smartds_timeseries.py`）

- **数据语义（已实测闭环验证）**：负荷 pu 曲线（35040 点单列）= 负荷对象
  年峰值归一，`Loads.dss` 的 kW 即该对象年峰值；PV 曲线单位 **kW/m²**
  （辐照度，非 pu）；温度取 solar_data 的 Temperature 列（环境 °C）。
- **时间基准**：当地标准时（SFO=UTC-8 / GSO=UTC-5，全年无夏令时；实测夏至
  PoA 峰在 12:15 且 DST 切换日无跳变）。**按行号对齐**（上游 parquet 为区间
  末端、solar_data 为区间起点标记，官方示例统一按行号）。
- **默认 15 分钟原始采样不插值不重采样**；步长恒 900s，24h=96 步。本阶段
  未实现任何插值选项（需要时必须显式配置并说明假设——当前没有）。
- **中心抽头分摊（重点核实）**：一个中心抽头用户 = 两个 Load 对象（`_1`/`_2`
  各挂一根相线），共享同一 yearly 曲线、**铭牌各为用户总额定的一半**。官方
  指南明确按 0.5 系数写入并警告勿双计；实测 parquet 年峰 5.0025 kW == DSS
  两对象铭牌之和（2×2.5012），逐对象写入即无重复计负荷。
  `SmartDSTimeseries.bind` 与测试锁定该口径（成对共享曲线 + 峰值=铭牌）。
- **曲线无重复作用**：模型自带 `yearly=` LoadShape 在会话 snapshot 模式下
  不参与（OpenDSS 语义：形状只在时序模式生效）；时序数据由接口按"铭牌×pu"
  或"辐照直写"独立注入，二者不叠加。新增 **LoadMult==1 栅栏**
  （`session.reset()`）：snapshot 下全局负荷乘子仍生效，若模型设置了非 1
  乘子会与逐设备写入重复计负荷，直接报错拒绝。
- profiles pu 曲线与 load_data parquet 的关系（如实记录）：kvar 侧逐点精确
  一致、kW 侧年均精确一致但约 2% 的 15 分钟点存在 ±20% 内重分布（上游生成
  管线差异）；接口以官方发布物 profiles（模型实际引用）为准。
- 环境级（gym env）逐设备时序注入尚未接入（现有 `DataConfig`/episode 结构
  与 15 分钟逐设备数据的融合设计留待下一阶段）；本阶段时序接口由基线脚本
  直接驱动会话，物理链路与训练路径共用 `session.apply`/`set_pv_conditions`。

## 4. PV 控制权与逆变器（共享设备/会话层扩展，不新建物理）

- **既有 PV 物理核对**：PVDevice（Pmpp/kVA/kvarMax/kvarMaxAbs/P-TCurve）、
  双层限幅（设备层 `prepare` 的 kVA 容量圆 + 非对称 ±Q 上限为唯一权威）保持
  不变；新增 `devices.pv_command_request` 把"fraction/qcommand 动作编码 →
  kW/kvar 请求"抽为共享函数，训练环境与基线脚本同源，不重写限幅。
- **三态控制权**（`DeviceControlEnv` 构造参数，`legacy` 为默认逐字节保持
  旧行为）：
  - `legacy`：布尔 `pv_control_enabled` 语义，全量探测且保持探测设定
    （既有训练路径，零回归）。
  - `rl`：只控 `pv_control_devices`（缺省全部）；**探测仅限接管名单且探测
    前快照、探测后恢复设定并补一次同工况求解**（探测对会话完全透明，
    read_state 立即可用）；未接管 PV 不写不探测，保持模型 pf/kvar 设定。
  - `fixed`：PV 不进动作空间、不探测、不写（模型静态 pf 运行）。
  - 动作格式：`pv_action_mode="pq"`（连续 Box(-1,1,(2,))，语义不变）或
    `"q_only"`（Box(-1,1,(1,))；有功不控=当步可用功率满发，仅无功命令）。
    控制数量、设备顺序（排序名单）与动作维度回合内稳定；夜间/低出力由
    可用功率探测值为 0 表达（不伪造出力），q_only 在可用功率为 0 时自然
    退化为纯电压响应边界。
  - **唯一写入方守卫**：rl 模式发现期扫描原生 InvControl 绑定
    （`devices.invcontrol_bindings`，已处理引擎返回带类前缀的全名），绑定
    被接管 PV 的 InvControl 在每次重编译后显式 `enabled=no` 并记入
    manifest（`pv_control` 段：mode/action_mode/controlled/uncontrolled/
    disabled_invcontrols）；fixed 模式保留 InvControl 原状（未接管）。
  - 逐 PV 辐照/温度：`set_pv_conditions(..., mapping={PV: (kW/m², °C)})`
    只写映射内设备，映射含不存在设备立即报错；缺省保持全网单标量旧行为。
- **夜间/缺测、动作方向与限幅验证**：夜间（辐照 0）可用功率恒 0（官方曲线
  与引擎一致，测试锁定）；正/负 Q 命令方向、非对称 kvarMax/kvarMaxAbs、
  kVA 容量圆边界由单测与 24h 基线逐台 P/Q 实测验证（见 §6）。**不凭容量圆
  假设动作可行**——所有下发经设备层限幅，"请求/限幅下发/实测 P/Q"三口径
  全部记录（trace.applied vs injections）。
- **可用功率探测副作用核查**：子集探测恢复设定且不触碰未列入设备；探测只
  调 SolveSnap（不推进时间/能量，`FinishTimeStep` 仍每区间恰一次，既有锁定
  测试不变）；探测后补解使量测栅栏立即可读。
- SMART-DS 系统注册守卫：非 34Bus_PV 系统在 SmartGrid 家族下必须显式
  `pv_plan: none`（防止家族预设把 34Bus 的 PV overlay 静默叠加到外部模型）。

## 5. 24 小时基线（p4udt4，2017-01-01，96×15min，四模式）

`python scripts/check_smartds.py`（输出
`results/validation/external/smart_ds_sfo_p4udt4_day0_baseline.json`）。
四模式同一时序、同一引擎；RL 两模式为**示例固定策略**（q_only：满可用功率
+Q 命令 +0.5；pq：有功分数 0.75、Q 命令 -0.3），仅验证控制权与物理链路，
**不代表算法收敛或泛化**。

| 指标 | native（原生控制） | fixed（静态 pf） | rl_q_only | rl_pq |
| --- | --- | --- | --- | --- |
| 求解失败 | 0/96 | 0/96 | 0/96 | 0/96 |
| 网损 kWh | 162.4 | 162.4 | 229.8 | 172.9 |
| PV 实际发电 kWh | 1480.6 | 1480.6 | 1488.0 | 1112.1 |
| PV 可用 kWh（探测口径） | —（不探测） | —（不探测） | 1482.8 | 1482.8 |
| 限发 kWh | — | — | 0 | 370.7 |
| 电压 min–max pu | 1.017–1.091 | 1.017–1.091 | 1.017–1.113 | 0.999–1.061 |
| 越限步数（>1.05 或 <0.95） | 28/96 | 28/96 | 96/96 | 15/96 |
| 最大越限 pu | 0.041 | 0.041 | 0.063 | 0.011 |
| 最大支路负载率 | 3.53 | 3.53 | 3.33 | 2.64 |
| 单步均耗时 | 4ms | 4ms | 4ms | 4ms |

自洽性核对：rl_pq 实际 1112.07 + 限发 370.69 = 1482.76 ≈ 可用 1482.77
（fraction=0.75 精确生效）；rl_q_only 实际 1488.0 略超探测可用 1482.8
（0.4%，源于探测时零 Q 与实际带 Q 的电压差，属电压依赖的正常波动）。
native/fixed 的 PV 实际发电一致（1480.6），同样源于 §5.1 的引擎限制。
逐台逐时刻明细（请求/限幅下发/实测 P/Q）见 JSON。

**关键发现与已知限制：**

1. **native 与 fixed 逐位一致的原因已查明**：本仓引擎（C-API 0.14.5 /
   OpenDSS SVN 3723）在 InvControl 携带 `RefReactivePower`（任意取值，
   含官方示例 VARMAX_VARS/VARMAX）时 VOLTVAR 控制几乎不动作（kvar 冻结在
   compile 期遗留值；对照实验：去掉该参数的最小 InvControl 正常渐变动作，
   电压推至 1.10pu>1.08 阈值仍不动即为证据）。这是引擎与 2017 模型的第三处
   兼容性问题，且**静默改变控制行为**（不报错）。处理：**不 patch 该参数**
   （删除会改变上游 Volt-Var 参照语义），如实报告——在本引擎上"原生
   Volt-Var 基线"等效于 InvControl 静止；换用其他引擎版本复验原生控制属
   未解决问题，留给审查决策。
2. 馈线原始过电压（v_max≈1.09pu、源 pu=1.03 整定）与 LV 支路高负载率
   （≈3.5pu，合成数据 240V 单相大 PV/集中负荷所致）为模型固有工况，四模式
   一致，按原样报告。
3. q_only 示例策略（+Q 满命令）在过压馈线上抬高电压（v_max 1.113）——方向
   性正确（正 Q 命令→发出→抬压），说明动作链路与符号约定一致。

## 6. 测试与验证记录

- `tests/test_smartds_import.py`：**16 项全过**（rdt1534 下载后含其绑定测试：
  会话构造 133s——大头是 master 内嵌的 35040 步全年求解、时序绑定 10s/
  180 条曲线、午间 472 台 PV 辐照全为正）（合成小模型 13 项：闭包
  datafile 依赖与缺失定位、中心抽头共享曲线/铭牌各半/年峰=铭牌、PV 曲线
  kW/m² 与温度、年峰损坏拒绝、逐 PV 辐照映射只写点名设备、子集探测恢复+
  量测透明、LoadMult 栅栏、InvControl 绑定前缀剥离、pq/q_only 动作限幅、
  文件清单下载补丁/未命中拒绝/篡改检出；真实 p4udt4 2 项：清单+时序+夜间
  可用 0+午间可用>0、三态端到端（rl 停用 InvControl 引擎侧生效、fixed 不
  写 PV 且奖励有限））。
- 既有回归（本任务改动的回归保护）：test_powergrid_session、
  test_smartgrid_physics、test_smartgrid_dataset_runtime、
  test_observation_projection、test_vvc_physics、test_envs、
  test_configuration_contract/config_pipeline/unified_config、
  test_large_systems（8500/9500）、test_external_cases（任务 1）全过——
  完整清单见提交说明。
- rdt1534（GSO rural）：登记+清单（452 文件）+下载+编译/时序/472 台 PV
  绑定测试通过（构造 133s 偏慢：主因 master 内嵌全年求解；优化需派生
  master 去掉内嵌 solve，属派生修改，待审查决策）。**未对 rdt1534 跑 24h 基线**（任务约定先小馈线），
  留待审查后与变电站级扩展一并做。
- 复现命令：
  ```bash
  python scripts/download_external_case.py --case smart_ds_sfo_p4udt4
  python scripts/download_external_case.py --case smart_ds_gso_rdt1534
  python scripts/check_smartds.py                 # 24h 四模式基线
  python -m pytest tests/test_smartds_import.py -o addopts="" -q
  ```

## 7. 修改文件清单

| 文件 | 内容 |
| --- | --- |
| `utils/external_cases.py` | 新增 FileSetCase/FilePatch/CaseEntry、文件清单下载校验、S3 listing；两算例登记 |
| `scripts/download_external_case.py` | 文件清单下载 + `--discover` 清单发现 |
| `configs/external/smart_ds_{sfo_p4udt4,gso_rdt1534}_files.json` | 提交仓库的逐文件 SHA-256 清单 |
| `envs/powergrid/case_inventory.py` | 依赖闭包扩展 LoadShape `file=` 数据依赖（kind=datafile） |
| `envs/powergrid/smartds_timeseries.py` | 新增：SMART-DS 忠实时序接口（逐负荷 P/Q、逐 PV 辐照/温度、中心抽头口径、单位/时间基准记录） |
| `envs/powergrid/session.py` | set_pv_conditions 逐 PV 映射；available_pv_power 子集探测+快照恢复+补解透明化；LoadMult 栅栏；read_state 拒绝原因细化 |
| `envs/powergrid/devices.py` | invcontrol_bindings（前缀剥离）；pv_command_request 共享动作映射 |
| `envs/powergrid/control_env.py` | PV 三态控制权 + q_only + 接管名单 + InvControl 守卫（legacy 零回归）；manifest pv_control 段 |
| `envs/smartgrid/base_env/{env_config,runtime_env,env_register}.py` | 三态参数透传与校验；非 34Bus_PV 系统 pv_plan 守卫 |
| `configs/systems/SMARTDS_P4UDT4.yaml`、`_registry.yaml` | 系统登记（布局根 + 馈线相对入口） |
| `scripts/check_smartds.py` | 24h 四模式基线 |
| `tests/test_smartds_import.py` | 合成+真实测试 |
| `docs/smart-ds-import.md` | 本文档 |

## 8. 已知限制与下一阶段建议

1. **引擎-模型兼容第三项（RefReactivePower 静默失效）未解决**（§5.1）——
   原生 Volt-Var 基线在本引擎上等效于控制器静止；建议：a) 在其他引擎版本
   （如 opendssdirect 更新构建）复验该参数行为后决定是否登记第三条补丁；
   b) 或接受"native==fixed"结论并在论文/实验中注明。
2. 环境级（gym env）逐设备时序注入未接入：需要 `DataConfig`/episode 结构
   与 15 分钟逐设备数据融合的显式设计（新 schema 版本或独立数据集类型），
   属下一阶段；本阶段基线脚本已验证全部物理与控制权链路。
3. rdt1534 仅完成数据接入与绑定验证；变电站级（rhs0_1247 全部 3 馈线 +
   subtransmission）与大规模训练在审查后进行。
4. `fixed` 模式当前语义 = 静态 pf（controlmode=off 下 InvControl 不动作）；
   若需要"原生 InvControl 参与的 fixed 运行"，应走 native 模式（受 §5.1
   引擎限制约束）。
5. SMART-DS parquet（用户级建筑分解数据）未接入（可选依赖 pyarrow 才能读），
   本阶段以官方 LoadShape 链路为准，parquet 差异已记录（§3）。
