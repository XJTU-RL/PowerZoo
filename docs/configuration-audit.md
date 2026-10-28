> 后续按实际消费者精简字段、合并单次引用预设，并修正 District 电价/SOC/邻居开关。
> 当前字段及优先级以[配置说明](configuration.md)为准；本文是前期历史审计。

> 后续入口清理已完成，旧训练脚本及旧单智能体配置见归档，当前公共链路见
> [训练入口整理](training-entrypoints.md)。下文保留前期审计原文。

> 后续更新：`codex/smartgrid-data-runtime` 已解决本文记录的 SmartGrid 数据入口、重复奖励计算、
> 原生context隔离及单/多智能体仿真分叉。新增支持的SB3入口为 train_smartgrid.py，
> 旧研究脚本不再维护。下文是配置重构阶段的历史审计；当前实现与验收见
> [SmartGrid 数据与运行设计](smartgrid-data-runtime.md)。

# 参数流代码审计与验收记录

本次从环境修复提交 `28e25a7c` 建立独立 `codex/config-pipeline` 分支。
Git 提交身份只配置在本仓库；不自动合并或推送。以下基于实际 Python 调用链，
不以旧 README 中的功能列表作为实现证据。

## 发现的问题与已完成的改动

| 原位置 | 实际问题 | 当前处理 |
| --- | --- | --- |
| `examples/multi_agent/scripts/train.py` | `eval()` 解析 CLI；不成对参数被 zip 截断；自定义环境文件绕过引用；UI 文件形状与读取器不符 | 入口缩为公共 `utils/train_cli.py`，安全 YAML、完整路径、同一解析器 |
| `utils/configs_tools.py` | 递归按叶子名覆盖多个位置，未知键静默忽略；绝对系统名使输出逃离日志目录 | 旧接口要求唯一匹配；目录只使用系统 basename，时间戳增加微秒 |
| `utils/unified_config_loader.py` | 缺失引用后退回默认；缓存修改前的 YAML；系统设备与环境参数分散 | 只保留兼容导出，实际解析移入单个 `config_pipeline.py`，无内容缓存 |
| 环境 `from_env_args()` | 多份别名映射；VVC 丢设备配置；SmartGrid 初始化默认对象后再 setattr，校验不针对最终值 | 一个兼容规范化边界；正式入口严格按消费方 dataclass 构造 |
| `utils/envs_tools.py` | train/eval/render 各自复制五类环境构造，先用另一个 seed 创建环境 | 一个 `make_power_env()`；构造前注入 worker seed |
| VVC `env_register.py` / `vvc_env.py` | 自定义目录与关闭清理使用不同来源 | 构造和清理均使用实际配置路径 |
| VVC `Env.MyReward` | 电压阈值写死为 0.95/1.05 | 从最终配置读取；有改变实际奖励的测试 |
| SmartGrid `VVCReward` | YAML CRBP 权重传成旧名称，CMDP 奖励读取另一套名称；电压范围也未接通 | 配置的完整奖励字典与电压范围传到公式，保留明确的 CMDP 死区参数 |
| District wrapper | 约束、奖励、市场价只从嵌套 `env_args` 取，公共入口传顶层导致丢值 | 统一 dataclass 构造，顶层和嵌套值均可进入兼容边界 |
| District core | 重新读取系统 YAML，容量和设备数可能与预览不同；效率未传给储能单元 | 拓扑随最终配置传递，数量核对，容量和效率进入单元 |
| Stackelberg wrapper/base | ESS 等真实参数未传递；base 忽略消费者数量与模型目录；自放电损失率误作保留率 | 参数透传；使用最终模型目录和消费者数；明确转换 retention_factor |
| Stackelberg 旧 YAML | 大量市场、动作、奖励、N-1 选项并未实现消费 | 删除默认文件中的伪选项；正式入口拒绝已知未实现项 |
| DSR/District 旧 YAML | 训练/网络/PPO 参数平铺在环境文件，实际未控制算法 | 迁至真实 train/model/algo/eval 节，并校验 RNN 分块等组合 |
| `training_ui` | 独立合并流程；预览不解析引用；环境表单未加入启动 overrides；重复维护数百个参数映射 | 环境页改为直接编辑 YAML；解析、预览和启动调用同一入口；保存的是可重放最终配置 |
| `train_vvc.py` / `train_stackelberg.py` / `train_dsr_aggregation.py` | 独立解析/覆盖和硬编码规则 | 变成同一 CLI 的快捷入口，旧专用参数需要迁移到实际路径 |
| 包 `__init__.py` | 读配置就提前导入 SB3/Torch/DSS，可能加载无关环境 | 公共 API 延迟导入；纯配置检查不依赖原生运行时 |

旧环境文件中平铺键的原始清单见 `configuration-legacy-fields.json`。
这个文件记录迁移前的键，**不表示清单里所有键都被删除**：训练和网络字段已移入真实消费节，
没有消费方的字段才移除。PV 方案文件的设备与指标建议原本没有对应的运行期读者，
实际 PV 设备仍由 DSS 方案定义，改写为只保留真实可用字段。

## 当前最终参数经过哪些位置

公共流程为：

```text
CLI / UI / Python
  -> ConfigLoader.load()
     -> 读取算法字段表与环境 dataclass 默认值
     -> 确定最终 system_ref / training_ref 并展开一次
     -> 顺序应用环境预设、专用覆盖、实验覆盖、CLI
     -> 类型与组合校验，记录值/来源/文件摘要
  -> runner_args + algo_args + env_args（仅分包，不改值）
  -> runner：set_seed / init_device / save_config
  -> make_power_env：仅加入 rank 和实际 worker seed，构造 dataclass
  -> 各环境适配器 -> core / circuit / reward
```

VVC 仍需 `to_env_info()/to_sys_info()` 给旧 core；SmartGrid 仍需 `to_info_dict()`。
它们是明确的消费方适配边界，不再负责读取 YAML 或发明覆盖顺序。
Stackelberg base 仍使用字典，wrapper 只转交已知实际参数。District core 不再读取 YAML。
`--env dsr_13bus` 等先选择预设文件，运行时 `run.env` 归一为环境家族，避免 logger/factory
把同一类环境当作不同实现。

## 尚未完成适配的独立研究入口

这些路径仍需后续专项开发，不能用本次公共配置测试声称它们已经工作：

1. **SB3 单智能体**：`train_single.py`、`train_single_agent.py`、
   `envs/smartgrid/single_agent/single_agent_training_config.py` 仍各自读取或组装配置。
   其中 `config_root` 指向包内而实际默认 YAML 在仓库 `configs/`，缺失后静默回退；
   CLI 默认值覆盖文件；`seed or ...` 丢失种子 0。
   更深处 `SingleAgentVVCEnv -> core_env.Env` 仍以旧签名调用新版 LoadProfile 和 Circuits，
   单纯把 YAML 接到新解析器会掩盖接口不兼容。下一步应实现明确的 SB3 适配器，
   共用解析阶段后将 SB3 超参数直接传给 SB3，并单独验收动作空间与 Gymnasium 的五元返回。
2. **DAN-HAPPO**：专用脚本还使用 argparse Namespace、独立 `DSRDANRunner` 与 buffer；
   YAML 是平铺格式，公共注册表却指向另一套通用 runner。actor 还有对字典使用 getattr
   导致 DAN 超参数取默认值的实际问题。需要统一该算法的运行器契约后再迁移。
3. **SN-MAPPO / QMix / 2TS**：现有 YAML 与公共八节配置和构造签名不一致。
   2TS runner 的签名甚至是 `(env, cfg, exp_name)`，不能把三字典传进去。
   正式公共入口已经在导入 runner/创建环境前拒绝这些组合。
4. **SmartGrid 旧 Python helper**：`from_env_name/from_dict` 等兼容入口仍可供旧脚本直接调用；
   新训练不走其历史兜底路径。直接构造底层 Env 的研究脚本不享有公共入口的完整来源记录。

这里将“完整统一的公共 MARL 配置链”与“仓库中全部研究脚本已经迁移”明确区分。
不通过批量删除旧研究算法或伪造训练成功来宣称全项目问题已彻底解决。

## 后续应优先解决的代码问题

- **动作是否真正作用到 DSS**：Stackelberg 原生一步交互虽返回成功，仍报告
  `Invalid property name kW for Transformer.xfm1`，说明 active element 选择/负荷命令存在问题。
  应先以原始 DSS 元件功率前后对比验收，不以 reward 有数字判断控制有效。
- **DSR 的设备与智能体规模**：`DSRConfig.get_agent_config()` 仍按系统名估算负荷数、限制
  PV/开关数量，`DSRCoreEnv` 中还有前 N 个母线作为 PV 节点的简化。
  当前原生13节点返回19个智能体；要通用扩展应从电路拓扑派生映射并记录实际映射。
- **终止与截断语义**：核对 episode 结束、求解失败、time-limit 与 GAE bootstrap；
  独立验收后再改，避免与配置迁移混为一种训练结果。
- **DSS 原生状态与数据**：检查并行 context、CWD、异常求解和 worker 临时文件隔离。
  DSS 编译可能改变进程 CWD，临时目录验证需在释放前恢复 CWD。
- **奖励计算副作用**：SmartGrid 的 `get_reward_info()` 与主奖励有重复计算，平滑历史可能
  被重复更新；这是奖励执行顺序问题，当前参数接通不代表该公式已完成数学审查。
- **研究数据完整性**：SmartGrid 缺少完整 month/irrad/temperature 时序，不能以合成数据
  替代输入然后宣称复现。逐个环境、算法、输入组合建立验收记录。

## 本机验证（Windows x64 / Python 3.10.22）

- `pip check`：无依赖冲突。
- 配置/Windows 相关 pytest：107 passed，无 skip；包括54项原配置测试和50项新契约测试。
- `check_runtime.py --train --require-cuda`：两套 DSS binding 原生求解通过；
  VVC 单 worker、两个 spawn worker；CPU 与 RTX3080 CUDA 各完成 HAPPO 24步/一次更新。
- 临时复制模型的额外探针：DSR13（19 agents）、Stackelberg13（9 agents）、
  District（3 agents）均完成初始化、reset、一步六元交互；不代表收敛或全功能正确。
- 公共 YAML CLI 实验（临时 IEEE13 副本）：240步、10次 HAPPO 更新完成，保存 checkpoint 与最终配置。
- 全部9个环境预设快照往返一致；UI builder 的最终文档与 CLI 读取结果相同。
- UI 的 Gradio 页面尚未在本环境安装可选依赖并进行浏览器交互验证；公共 builder 已测。
- DSS-Python Windows 原生库初始化可能让 pytest faulthandler 打印 `0xe0465043`，
  以最终退出码与原生求解结果为准；不隐藏该输出。

原始运行日志位于当前 worktree 的 `results/validation/`（忽略提交）。
更详细的环境基线见 [Windows 运行记录](windows-runtime.md)。
