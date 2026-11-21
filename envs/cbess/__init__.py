"""CBESS 协调环境家族：PISTG-LDCPPO 论文（arXiv:2610.33654）场景入库。

- `cbes_env.py`：CMDP 核心（OpenDSS 全 AC，式(24)）
- `physics.py`：纯函数物理层（式(1)-(3)、(25)(26)、(24c-f)）
- `task_env.py`：注册表包装器（gymnasium 风格单智能体接口 + cost 通道）
- `config.py`：CBESSConfig（= research 侧 ScenarioConfig）
- `profile_store.py`：Ausgrid 逐日剖面运行时存取
复现入口与 PIAE/PISTG 研究代码在 `research/pistg_ldcpo/`。
"""
