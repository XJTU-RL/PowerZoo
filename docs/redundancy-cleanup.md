# 冗余清理记录（2026-11-06）

本轮从 `e9666130` 清理现役代码和数据生成物，未纳入 `papers/` 的并行整理。

## 已清理

- 删除12个退役代码文件：旧 DAN buffer/runner、无消费者的共享 buffer 和异常定义、
  5个旧训练 UI 组件、未使用的 VVC 静态报告绘图、占位奖励标定工具。
- 奖励包移除 `RewardCalibrator`、`quick_calibrate` 和 `load_calibration` 导出。
  旧工具使用固定动作差异，策略分支仍返回随机动作，不能视作已实现的自动标定能力。
  仓库外调用者应停止使用这些入口；现役奖励计算和参数未修改。
- 移除318个数据生成文件，约34.7 MB，包括每日/月度 PV DSS 与统计输出、
  批量可视化、负荷分析结果和温度对比图。补充精确的输出忽略规则和重建命令。
- 重写根环境导航，更新 `CLAUDE.md`，合并重复单智能体说明的维护入口；
  历史计划增加状态提示，保留审计证据。
- 删除 QMix 配置中被后一个 `seed` 映射覆盖的旧标量键。
  修改前后的普通 YAML 解析结果相同，严格解析及训练 UI 构建恢复正常。
- HF 共享 JSON、SVG 与依赖改为指定来源生成副本，并在 CI 校验一致性。
  每个 Space 仍保留独立部署文件。中文发布需要配置已确认的 `HF_ZH_SPACE`，
  未配置时明确跳过，详见[维护说明](../huggingface_spaces/README.md)。

## 保留边界

保留 VVC 集中控制入口、旧路径兼容门面、SmartGridCircuit、命令行分析工具、
模型工具、全部 archive、各算例自包含 CSV 和 PV 原始 TXT。
温度扩展脚本、指南及示意 DSS 保留并标为历史合成示例。
本轮没有迁移 Git LFS 或重写历史，因此删除文件不会缩小已有 Git pack。

## 验证

- 相关配置、训练入口、归档契约、VVC、runner、buffer、奖励、集中适配与 HF 测试：
  172 passed，0 skipped。
- `python -m pip check` 通过。
- `python scripts/check_runtime.py --train` 通过原生 DSS、单 worker、双 spawn worker
  和24步 HAPPO CPU训练；输出全部使用临时模型副本和目录。
- 训练 UI 实际构建得到 `Blocks`；所有算法 YAML 通过严格解析。
- 临时目录从真实 PV 输入的64行切片重建 CSV/DSS/报告，并列出日期；
  从保留的 LoadShape1.CSV 重建525600行分钟数据。
- HF 共享资源检查为0差异，工作流通过 actionlint；独立审查通过。

验证环境为 macOS ARM64、Python 3.11.15、NumPy 1.26.4、PyTorch 2.10.0 CPU、
DSS-Python 0.15.7、OpenDSSDirect 0.9.4。依赖补充仅在任务 worktree 的忽略环境内完成，
未改依赖清单。该结果不等同于锁定的 Python 3.10/PyTorch 2.6.0、Windows 或 CUDA 验证，
也未验证浏览器完整交互或实际 HF 发布。
