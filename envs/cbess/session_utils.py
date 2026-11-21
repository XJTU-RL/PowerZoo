"""DSSSession 的 33-bus 专用补充设置（不改共享物理层）。

12.66 kV 非引擎标准基值档，模型文件里的 ``set voltagebases`` 在 compile
后不存活，DSSSession.reset 的 CalcVoltageBases 会把基值吸附到标准
12.47 kV 档，p.u. 电压系统性偏大 1.53%（已实测）。因此每次 reset 后用
公开 command() 接口重新登记基值并重算，见 test_pistg_repro.py 的对拍。
"""

from __future__ import annotations

from envs.powergrid.session import DSSSession

BASE_KV_LL = 12.66


def rebase_voltage(session: DSSSession, base_kv_ll: float = BASE_KV_LL) -> None:
    """在 reset 之后、区间开始之前调用：登记非标准电压基值并重算。"""
    session.command(f"set voltagebases=[{base_kv_ll:g}]")
    session.command("calcvoltagebases")
