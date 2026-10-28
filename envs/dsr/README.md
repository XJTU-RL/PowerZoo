# 有源静态恢复

正式DSR使用公共DSSSession、真实故障资产/开关/负荷和相供电判定。
当前算例为RestorationSmall与IEEE13派生恢复网络，支持合作控制组及集中式对照。
训练入口scripts/train.py，专项验收scripts/check_dsr.py；数据为data/restoration下的故障manifest。
原黑启动/DG/PV计数、Enabled恢复率、旧量测与GUI已归档于archive/physics_refactor_dsr。
静态潮流不证明黑启动、频率动态或保护协调；完整定义与验收见docs/dsr-physics.md。
