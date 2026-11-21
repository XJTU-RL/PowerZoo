# IEEE34 / IEEE123 双馈线 LFC 工程平台

同一工作树运行三机九节点 Bus8 与原生 IEEE34 或 IEEE123 的实测频率、配网调节和实测 PCC 双向返回。IEEE34 工作点、柔性负荷、原闭环收集器来自 `66071605`，IEEE123 原闭环与共用原生接收器来自 `e4417692`。在同一 checkout 验证原 34 馈线树为 `8fbc1a9a3b7a2a5da636224908298a256b4c2b43`，直接使用公共电路。IEEE123 用 `b58012bf` 已实测的四台双向储能物理检查和评分。两台馈线的原始物理回路、断链对照及输入输出计量均可独立执行。工程自检采用 2 kW 有符号继电反馈，只为让两种符号都在真实端口可见。

所有工程入口均在本仓库内。运行无需 M2/M3 控制器工作树、学习权重、开发实验目录或联合性能验证清单。原生频率保留 0.2 s 时间戳，1 s 交换，1/60 s 输电步长，2/60 s 为写入护栏。IEEE34 配网端以 890 的柔性负荷调节进出口；IEEE123 新工程自检等分到四台储能，使用独立 DSS 上下文做一步签名功率、时钟和能量预验，再实际步进并比较。配网端电源沿用既有固定边界，工程证据只覆盖频率、资源和 PCC P/Q 的双向链。

## 运行条件

南洋已验证 HARL Python 3.10、DSS-Python 0.15.7、MATLAB R2025b、Xvfb 及仓库内原生模型。IEEE34 上游静态工作点输入 `pair-packaged.json` 的 SHA-256 必须为 `d709b9522d21256fb6b3d2fab4e1b17c07c8c9afb07459f1b2cf12b0127fb666`。运行时复查六个分接、95 个相电压身份、实际工作点、绕组回读和输入树。该私有原始基准不随分支分发。IEEE123 历史闭环需要独立开放接收侧报告，既可在南洋使用已验收的原件，也可先用同一分支重新生成并通过其指纹及正负 P 读回检查。归档核验还需下文两个原始闭环报告。受限依赖和原始资料的位置均限于现有南洋环境。

## 快速验证和复跑

从 PowerZoo checkout 内执行。以下环境变量应指向南洋已获准使用的私有输入和见证目录。新输出一律独占，避免覆盖。带既有证据的物理与时序评分通常只需数秒。

```sh
cd "$(git rev-parse --show-toplevel)" || exit
export PATH="$HOME/matlab_2025b/bin:$PATH" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
py="$HOME/miniconda3/envs/HARL/bin/python"
ref="${LFC34_REFERENCE:?需要已验收的私有 IEEE34 工作点}"
open="${LFC123_OPEN_REPORT:?需要 IEEE123 开放标定记录}"
a34="${LFC34_ARCHIVED_LOOP:?需要 IEEE34 原始物理验收记录}"
a123="${LFC123_ARCHIVED_LOOP:?需要 IEEE123 原始物理验收记录}"
w="${LFC_ARCHIVED_WITNESSES:?需要已归档的工程自检目录}"
d=$(mktemp -d "$HOME/exps/lfc-platform-check.XXXXXX") || exit
"$py" -m scripts.verify_lfc_platform_evidence --archived --loop34 "$a34" --loop123 "$a123" --reference34 "$ref" --open123 "$open" --storage "$w/fresh-storage.jsonl" --out "$d/archive.json" || exit
"$py" -m scripts.check_lfc_platform_witness verify "$w/integrated-123-retry.jsonl" || exit
"$py" -m scripts.check_lfc_platform_witness verify "$w/integrated-34.jsonl"
```

重新采集只需下面这些工程脚本。前两项只用原生 DSS，通常数秒。双馈线工程自检一次仅启一个新引擎，每个约 2–3 分钟，超时 360 s。正负 17 kW 外部脉冲只施加于 Bus8；继电反馈仅使用写入前的实测 Bus8 频率。静置参考取写入前两秒内两次通信边界的最新真实频率，完整波形和 RoCoF 仍使用原生 0.2 s 序列。两方向实际 PCC 支援均需超过 1.5 kW，PCC 与 2 kW 指令的差限 0.75 kW，Bus8 实测增量与目标增量各通道差限 1.5 kW/kvar。时钟、设备保护或测量失效即停止。该小自检独立检验集成工作树接线，断链因果则由另有独立对照的原生记录检验。

```sh
"$py" -m scripts.check_lfc34_flexible_load "$ref" "$d/34-flex.jsonl" || exit
"$py" -m scripts.check_lfc123_storage_absorption "$d/storage.jsonl" || exit
"$py" -m scripts.score_lfc123_storage_absorption "$d/storage.jsonl" "$d/storage-score" || exit
timeout --kill-after=15s 360s xvfb-run -a "$py" -m scripts.check_lfc_platform_witness run "$d/witness-123.jsonl" --feeder 123 || exit
timeout --kill-after=15s 360s xvfb-run -a "$py" -m scripts.check_lfc_platform_witness run "$d/witness-34.jsonl" --feeder 34 --reference34 "$ref"
```

如需重新采集原物理验收的完整对照（约 35–50 分钟，串行），从同一工作树执行下列命令。生成数据使用对应来源进行完整性、窗口、断链、连续交付和顺序复核；`--archived` 另要求与旧归档逐指标及 SHA-256 完全匹配，只用于原件。整个过程没有算法性能晋级入口。

```sh
timeout --kill-after=15s 2100s xvfb-run -a "$py" -m scripts.check_lfc34_bus8_loop "$PWD" "$ref" "$d/full-34.jsonl" || exit
timeout --kill-after=15s 1500s xvfb-run -a "$py" -m scripts.check_lfc_123_transfer --report "$d/open-123.jsonl" || exit
timeout --kill-after=15s 1080s xvfb-run -a "$py" -m scripts.check_lfc_123_feedback_engineering "$d/open-123.jsonl" "$d/full-123.jsonl" || exit
"$py" -m scripts.verify_lfc_platform_evidence --loop34 "$d/full-34.jsonl" --loop123 "$d/full-123.jsonl" --reference34 "$ref" --open123 "$d/open-123.jsonl" --storage "$d/storage.jsonl" --out "$d/full-evidence.json"
```

评分器拒绝未完成、带失败标记、不可用数值、来源不匹配、时钟未对齐、读数不新鲜以及无实际回传的记录。归档候选在南洋通过 13 项相关轻量测试。线路热额定缺少独立来源，ACE 未定义；设备或绕组的模型内检查不能补足这些缺口。工程候选不包含未通过的协调算法，不提供任何算法优越性、全域频率稳定性或外部部署结论。原始报告、引擎日志与模型副本留在南洋私有目录。

## 南洋整合验证结果

已有 IEEE34 的 8 臂与 IEEE123 的 3 臂完整反馈记录重新评分后，逐项匹配原归档及原始摘要。切断回传时，两条馈线在 PCC 均有实际调节，但相对无配网控制的 Bus8 频率差低于辨识阈值。恢复回传后，IEEE34 与 IEEE123 分别恢复 3 帧、7 帧连续交付。原始无控制、断链和闭环的指标表仍在本分支的对应文件夹。

| 馈线 | 历史首次动作 / 实测 Sensor7 P / 可辨 Bus8 频差 s | 新自检实测 PCC 吸收 / 支援 kW | 新自检 P/Q 回传最大增量残差 kW 或 kvar |
|---|---|---:|---:|
| IEEE34 | 23.033333 / 23.050 / 23.400 | -2.2407 / +2.2384 | 0.08094 |
| IEEE123 | 23.033333 / 23.050 / 23.200 | -2.0792 / +2.0790 | 0.08084 |

新自检各在一个独立引擎完成 15 个连续真实交换帧，观察两种方向的实测 PCC 交付；它们用于核实整合后入口可运行。另在该工作树重新完成 34 节点六臂、48 帧 DSS 资源检查，以及 123 节点静置、充放电及撤回六臂检查。

第一次 IEEE123 启动采用了后期接口才有的参考回调，原生接收器在载入引擎前报参数错误。保留了失败指纹。修正为从工程版已有通信边界读回取得静置参考后，使用新文件重新执行并通过；扰动幅值、设备保护和通过界限原样保留。运行提交、完整度和摘要见 `docs/lfc-results/platform-engineering-check.json`。本轮没有重新采集原来的 11 个长对照臂。

可收尾范围为在已授权南洋环境可重跑的工程平台与测量工具。当前工作没有验证脱离南洋私有工作点文件的全新安装。两项控制原型和任何联合算法性能结论都不在此候选内。


## 发布来源与复现范围

工程文件经隔离移植自归档 `472b11bddf81267e844c73bb8bfac1ff4759884d`。公开接收器保留原生测量、时钟、储能账本与设备边界。新采集的数据记录当前工程配置指纹；归档标定按已验收原件的完整文件指纹校验。历史 13Bus 与三机九节点论文复现仍可独立使用。IEEE34 的完整复跑需要同一私有工作点、MATLAB/Simulink 与 DSS 环境。协调控制器的联合性能验收未通过。
