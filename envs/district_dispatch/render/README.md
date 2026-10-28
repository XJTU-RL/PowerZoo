# 正式区域回放

`engine/episode_runner.py`使用正式DistrictDispatchCore、局部actor观测和公共原生快照。
策略通过`envs.powergrid.checkpoint.load_checkpoint(..., expected_family="district_dispatch")`恢复，
不能猜测观测/动作维度。`engine/manual_override.py`使用实际核心实例绑定控制ID。

原八页Gradio界面、旧量测/导出与压力扫描依赖虚拟交换、EV动作和错误尺寸，
已原样归档于`archive/physics_refactor_district/`，尚未完成GUI迁移。
当前运行与验证方式见`docs/district-physics.md`；公共快照可作为后续页面的数据入口。
