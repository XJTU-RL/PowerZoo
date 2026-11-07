# 退役训练入口归档

本目录保存从提交 `3f47f9f4` 清点后退役的脚本、专属配置和旧API测试。
原目录结构保留，文件统一追加 `.txt`，不会被Python导入、pytest收集或当作活动配置加载。
不保证这些历史脚本可执行；恢复研究思路时先迁移到当前公共配置契约。

逐文件原路径、归档路径、按LF换行规范化后的SHA256、退役原因和替代入口见 [manifest.json](manifest.json)。
算法、模型和仍在注册表使用的DAN研究实现没有因旧启动器失效而删除。
数据处理、渲染、部署、监控和绘图脚本也不因不参与训练而删除。

当前入口：

- MARL：`python scripts/train.py --config configs/experiments/<experiment>.yaml`
- SmartGrid集中控制PPO：`python examples/single_agent/scripts/train_smartgrid.py`
- 旧通用 `examples/multi_agent/scripts/train.py` 仅作为同一公共CLI的兼容路径保留。

旧单智能体测试随其退役API归档；活动测试改为验证共享配置、真实内核、Gymnasium契约和动作效果，
没有把历史研究接口的测试通过情况当作新系统可运行的依据。
