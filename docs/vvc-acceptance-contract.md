# VVC 接入与验收

本接口基于远端 `main` 的 `97d69b8e19a830ce9ff03aeec3af50eb6997af86`（PR #66）。
集中控制和 MARL 各用自己的训练器，共用配置解析、设备身份和验收证据格式。

## 静态预检

```sh
python scripts/check_capabilities.py --list
python scripts/check_capabilities.py --config configs/experiments/vvc_13bus_smoke.yaml \
  --backend sb3_ppo --checkpoint policy_reload --independent-replay
python scripts/check_capabilities.py --env vvc --algo matd3
python scripts/check_capabilities.py --env vvc --mask feasible
```

后两条命令应以非零状态退出。预检使用 `ConfigLoader` 解析同一套实验配置。
Python 调用使用 `utils.capabilities.discover_capabilities()` 查询声明，
以及 `preflight(config, backend=..., layout=...)` 预检已解析配置。
`backend` 选择集中控制入口的后端，省略时从配置读取 MARL 算法。
`compatible` 表示已知配置可继续。存在 `deferred` 时，调用方应在构造对应环境后使用
`device_layout(wrapper)` 和同一份配置再检查一次。以 `report.stages.preflight` 判断预检，
整份静态报告的 `status` 为 `not_run`，因为运行阶段尚未开始。

| 能力 | 接入约定 |
| --- | --- |
| 动作与设备身份 | 策略端 `action_kinds` 与物理端 `device_action_kinds` 分别声明。VVC 可有离散、连续和混合设备动作，确切类型与顺序使用 `physics_manifest.action_schema` 和实际空间。默认 VVC 集中入口用 Box 编码并量化离散设备。DSR 集中入口使用 MultiDiscrete，分组 ID 取清单的 `group`。 |
| 观测与团队状态 | MARL 的 actor 可使用局部或全局观测，集中 critic 的 EP 团队状态来自全局字段，采用配置指定的母线投影。集中控制消费单个全局向量。 |
| 策略共享 | `independent`、`group`、`all` 表示逐设备、同构组和全队共享。全队共享复核每个设备的动作及观测空间。集中控制使用 `single`。 |
| 掩码与物理限值 | `structural` 描述离散索引槽位。VVC 的全 1 掩码不包含 SOC、实际分接档位、潮流结果或电压限制。物理限值以原生引擎的请求、限幅下发和读回为准。DSR 的 `feasible` 掩码检查逐组单一选择的当前操作约束，联合动作仍经过环境校验和求解。 |
| 成本通道 | LDCPPO 的约束能力要求 CBESS 原生提供 `VD`、`VN`、`LO`。VVC 的奖励分项不声明为这些成本。 |
| 终止与截断 | VVC 的任务末步和原生求解失败均为终止。PettingZoo 边界分别传递 `terminated` 和 `truncated`。LDCPPO 在基线中按旧 done 处理，当前适用声明限于无外部截断的 CBESS 有限时域。 |
| 检查点与评估 | `policy_reload` 用于恢复策略，`training_resume` 表示有训练续跑入口。独立评估需要分开的 validation 工况和 checkpoint 物理绑定。基线 CBESS 的公开 reset 只使用 train，LDCPPO 暂无统一的公开策略重载接口。 |

当前能力声明覆盖 VVC、DSR 和 CBESS 中表列的入口。其它已注册环境在发现结果中保留其交互类型，
预检对未声明的环境/算法组合给出明确错误。主线旧 QMIX 显示为未接入公共八节配置。
算法分支可在合入时为各自真实入口补声明，随后接入同一预检函数。
现有训练 CLI 保留原入口行为，能力检查可作为 CI、界面和训练入口的独立前置调用。

## 阶段验收

`utils.acceptance.run_acceptance(preflight, checks)` 接收一组四个零参数回调。
运行器负责构造自己的环境、记录本次观测并关闭资源。回调接续共享实例，评估阶段应使用新环境
和重载的检查点。按顺序执行，首个失败或阻塞会使后续阶段标为 `not_run`。

| 阶段 | 证据与通过判定 |
| --- | --- |
| `preflight` | 解析后的环境、系统、算法、动作和共享类型，及待运行期确认项。 |
| `physical_readback` | `converged`，有功残差和容差，稳定有序的 `device_ids`，以及每个 ID 的 `quantity`、`requested`、`applied`、`measured`、`tolerance`、`matched`。按设备定义校验限幅，并核对下发值与实测值。 |
| `minimal_update` | 实际 `steps`、`updates`，不同的 `parameters_before_sha256` 和 `parameters_after_sha256`，以及从真实训练交互提取的 `training_scenarios`（工况 ID 到内容摘要）。 |
| `save_reload` | 模型工件 `checkpoint_sha256` 和同输入上的 `prediction_match`。回调负责用训练器自身的绑定校验重载检查点。 |
| `independent_replay` | validation 的 `scenario_id`、`scenario_sha256` 与训练证据均无重复，并使用同一个检查点摘要。完成预期步数并正常终止，记录 `terminated`、`truncated` 和 `physical_failure`。 |

报告的 `schema_version` 为 1。顶层 `stages` 按表中的顺序列出阶段。
每段含 `status`，可附 `evidence` 或 `detail`。因此通过和失败均可审计。
`AcceptanceBlocked` 用于缺少执行依赖或必要输入，顶层 `status` 依次采用
`failed`、`blocked`、`not_run`、`passed` 的优先级。JSON 使用有限数值，回调可补充
时间步、物理摘要和运行器指标等字段。脚本返回非零码表示完整运行未通过。

### 南洋代表性运行

下列命令用于隔离的南洋源码目录，输出路径必须是新目录。

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/check_vvc_acceptance.py /path/to/new-report
```

参考脚本使用 IEEE13 和 SB3 PPO，复用现有 VVC 工厂、集中控制适配器和物理绑定检查。
它用四个采样步得到一次更新，保存权重，再从一个新环境完成 24 步 validation 回放。
数据是为本次运行新生成的合成夹具，实际消费的两个分片为 train 和 validation。
报告与模型均留在隔离工作区。回放越限按逐步逐相计数；报告通过表示接入链路完整。

## PettingZoo Parallel 边界

`envs.vvc.parallel_adapter.VVCParallelAdapter` 接受并独占一个已构造的 `VVCEnv`，
使用原生设备 ID 和稳定空间，按清单顺序组装动作，一次提交一个完整同步帧。
`state()` 返回 EP 团队全局状态。团队终止或截断后所有 agent 同时退出，reset 才重新加入；
动作不足或超出声明空间时在调用物理环境前失败。离散槽位掩码置于
`infos[agent]["structural_action_mask"]`，训练器可据需要自行消费。
该适配器实现 Parallel 协议并可选运行官方 `parallel_api_test`，无需安装 PettingZoo 即可导入。
顺序定价任务采用各自的阶段采样入口。协议见 [PettingZoo Parallel API](https://pettingzoo.farama.org/api/parallel/)。

## 并行接入依赖

- 单智能体后端可将实际支持的动作编码、成本、终止处理、重载及评估声明纳入目录。SAC/TD3
  应在真实且适用的连续动作入口就绪后登记，避免将 VVC 离散量化当作原生连续设备。
- MARL 接入在各自分支添加公开配置和采集器后，登记 IPPO、QMIX、VDN、COMA 的实际空间、团队状态及掩码要求。
  MADDPG/MATD3 需要具有原生连续动作的已绑定设备；IEEE13 默认离散配置会在静态阶段拒绝。
- 各 lane 可把现有短训、真机物理读回、检查点和独立工况回放包装成验收回调。候选分支在本轮仅作只读对照。
