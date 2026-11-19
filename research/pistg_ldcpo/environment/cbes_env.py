"""CBESS 协调环境：论文式(24) CMDP 的 OpenDSS 实现（单智能体集中控制）。

每回合一天 288 步 × 300 s。动作 60 维（论文式(24b)）：
[α_P×4 CBESS | α_Q×4 CBESS | α_κ×26 PV 弃光 | α_q×26 PV 无功]，全部
∈[-1,1]，经 physics.py 映射为可行功率指令后写入 DSS 求解。

状态侧输出逐步母线特征（式(8b)：p_load、q_load、p_PV、V、E）与电价，
由上层训练循环组装 W=12 滚动窗口。V 为最近一次求解电压（伪量测口径：
决策 t 时窗口内的 V_t 用 t-1 步解得的电压，论文未区分此半步，偏离文档
记录）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from envs.powergrid.session import DSSSession

from ..data.profile_store import DayProfile, ProfileStore
from .physics import (
    COST_KEYS,
    CBESSParams,
    cbess_energy_update,
    constraint_costs,
    economic_reward,
    map_cbess_action,
    map_pv_action,
)
from .session_utils import rebase_voltage

DT_HOURS = 300.0 / 3600.0
SQRT3 = np.sqrt(3.0)


class CBESEnv:
    """IEEE 33-bus + 4 CBESS + 26 DER 母线（论文 Sec. IV-A）。"""

    def __init__(
        self,
        case_folder: Path,
        store: ProfileStore,
        scenario,  # research.pistg_ldcpo.config.ScenarioConfig
    ):
        self.case_folder = Path(case_folder)
        self.store = store
        self.cfg = scenario
        meta = json.loads(
            (self.case_folder / "case_meta.json").read_text(encoding="utf-8")
        )
        self.lines = meta["lines"]
        self.der_buses = list(meta["der_buses"])
        self.cbess_buses = list(meta["cbess_buses"])
        self.base_kv_ll = float(meta["base_kv_ll"])
        self.n_bus = int(meta["num_buses"])
        # 线路视在容量 kVA = normamps × 线电压 × √3。
        self.branch_limit_kva = np.array(
            [ln["normamps"] * self.base_kv_ll * SQRT3 for ln in self.lines]
        )
        self.branch_names = [ln["name"] for ln in self.lines]
        # PV 逆变器容量 = 超配 × PV 峰值（manifest 静态容量）。
        pv_peak = np.array(
            [store.manifest["pv_peak_kw"][str(b)] for b in self.der_buses], float
        )
        self.pv_s_max_kva = scenario.pv_inverter_oversize * pv_peak
        self.cbess = CBESSParams(
            capacity_kwh=scenario.cbess_capacity_kwh,
            soc_min=scenario.cbess_soc_min,
            soc_max=scenario.cbess_soc_max,
            p_limit_kw=scenario.cbess_p_limit_kw,
            q_limit_kvar=scenario.cbess_q_limit_kvar,
            efficiency=scenario.cbess_efficiency,
            s_rating_kva=scenario.cbess_s_rating_kva,
        )
        self.action_dim = 2 * len(self.cbess_buses) + 2 * len(self.der_buses)
        self.steps_per_day = scenario.steps_per_day
        self._session = None
        self._last_v_pu = np.full(self.n_bus, 1.0)

    # ------------------------------------------------------------------
    def close(self):
        if self._session is not None:
            self._session.close()
            self._session = None

    def _ensure_session(self) -> DSSSession:
        if self._session is None:
            self._session = DSSSession(
                self.case_folder,
                "network.dss",
                overlays=("der_overlay.dss",),
                step_seconds=self.cfg.step_seconds,
            )
        return self._session

    # ------------------------------------------------------------------
    def reset(self, rng: np.random.Generator, split: str = "train") -> dict:
        """采样一天并复位：DSS 重编译、CBESS 能量日初值（式(1f)）。"""
        session = self._ensure_session()
        session.reset()
        rebase_voltage(session)
        self.profile: DayProfile = self.store.sample_episode(rng, split)
        self.t = 0
        self._last_v_pu = np.full(self.n_bus, 1.0)
        self.energy_kwh = np.full(
            len(self.cbess_buses),
            self.cbess.capacity_kwh * self.cfg.cbess_soc_init,
        )
        self.day_metrics = {
            "reward": 0.0,
            **{k: 0.0 for k in COST_KEYS},
            "curtailed_kwh": 0.0,
        }
        return self._observation()

    # ------------------------------------------------------------------
    def step(self, action: np.ndarray) -> tuple[dict, float, dict, bool, dict]:
        """执行式(24)-(26)：映射动作 → 写 DSS → 求解 → 奖励与三成本。"""
        if self.t >= self.steps_per_day:
            raise RuntimeError("回合已结束，须先 reset")
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        n_m, n_der = len(self.cbess_buses), len(self.der_buses)
        alpha_p = action[:n_m]
        alpha_q_cb = action[n_m : 2 * n_m]
        alpha_kappa = action[2 * n_m : 2 * n_m + n_der]
        alpha_q_pv = action[2 * n_m + n_der :]

        pv_avail = self.profile.pv_available_kw[self.t]  # (26,)
        p_cbess = np.zeros(n_m)
        q_cbess = np.zeros(n_m)
        p_pv_der = np.zeros(n_der)
        q_pv_der = np.zeros(n_der)
        kappa = np.zeros(n_der)
        for m in range(n_m):
            p_cbess[m], q_cbess[m] = map_cbess_action(
                alpha_p[m], alpha_q_cb[m], self.energy_kwh[m], self.cbess, DT_HOURS
            )
        for j in range(n_der):
            p_pv_der[j], q_pv_der[j], kappa[j] = map_pv_action(
                alpha_kappa[j], alpha_q_pv[j], pv_avail[j], self.pv_s_max_kva[j]
            )

        p_load = self.profile.p_load_kw[self.t]  # (33,)
        q_load = self.profile.q_load_kvar[self.t]

        session = self._ensure_session()
        session.begin_interval(self.t * self.cfg.step_seconds)
        self._write_setpoints(p_load, q_load, p_pv_der, q_pv_der, p_cbess, q_cbess)
        try:
            session.solve()
        except RuntimeError:
            # 发散兜底：无功清零重解一次；仍失败则如实抛出（不伪造状态）。
            self._write_setpoints(
                p_load,
                q_load,
                p_pv_der,
                np.zeros_like(q_pv_der),
                p_cbess,
                np.zeros_like(q_cbess),
            )
            session.solve()

        v_pu = self._read_voltages()
        self._last_v_pu = v_pu
        branch_kva = self._read_branch_flows()
        costs = constraint_costs(
            v_pu,
            branch_kva,
            self.branch_limit_kva,
            self.cfg.voltage_min,
            self.cfg.voltage_max,
        )
        price = float(self.profile.price_aud_per_mwh[self.t]) * self.cfg.price_level
        reward = economic_reward(
            p_load,
            float(p_pv_der.sum()),
            p_cbess,
            price,
            self.cfg.cbess_degradation_aud_per_kwh,
            DT_HOURS,
        )
        for m in range(n_m):
            self.energy_kwh[m] = cbess_energy_update(
                self.energy_kwh[m], p_cbess[m], self.cbess, DT_HOURS
            )
        self.day_metrics["reward"] += reward
        for key in COST_KEYS:
            self.day_metrics[key] += costs[key]
        self.day_metrics["curtailed_kwh"] += float(
            (kappa * pv_avail).sum() * DT_HOURS
        )

        self.t += 1
        done = self.t >= self.steps_per_day
        info = {
            "v_pu": v_pu,
            "p_cbess_kw": p_cbess,
            "q_cbess_kvar": q_cbess,
            "p_pv_kw": p_pv_der,
            "kappa": kappa,
            "pv_available_kw": pv_avail,
            "soc": self.energy_kwh / self.cbess.capacity_kwh,
            "price_aud_per_mwh": price,
            "curtailed_kwh_step": float((kappa * pv_avail).sum() * DT_HOURS),
        }
        session.finish_interval()
        return self._observation(), reward, costs, done, info

    # ------------------------------------------------------------------
    def _write_setpoints(self, p_load, q_load, p_pv, q_pv, p_cbess, q_cbess):
        session = self._ensure_session()
        commands = []
        for bus in range(2, self.n_bus + 1):
            commands.append(
                f"Edit Load.load_b{bus} kW={p_load[bus - 1]:.6f}"
                f" kvar={q_load[bus - 1]:.6f}"
            )
        for j, bus in enumerate(self.der_buses):
            commands.append(
                f"Edit Generator.pv_b{bus} kW={p_pv[j]:.6f} kvar={q_pv[j]:.6f}"
            )
        for m, bus in enumerate(self.cbess_buses):
            commands.append(
                f"Edit Generator.cbes_b{bus} kW={p_cbess[m]:.6f}"
                f" kvar={q_cbess[m]:.6f}"
            )
        for command in commands:
            session.command(command)

    def _read_voltages(self) -> np.ndarray:
        state = self._ensure_session().read_state(light=True)
        v_pu = np.zeros(self.n_bus)
        for i in range(1, self.n_bus + 1):
            values = state.voltages_pu[f"bus{i}"]
            v_pu[i - 1] = float(np.mean(list(values.values())))
        return v_pu

    def _read_branch_flows(self) -> np.ndarray:
        """各线路 parent 端视在功率 kVA（式(24f) 的 |S_ij|）。"""
        session = self._ensure_session()
        s_kva = np.zeros(len(self.branch_names))
        for k, name in enumerate(self.branch_names):
            p, q = session.powers(f"Line.{name}")
            s_kva[k] = float(np.hypot(p, q))
        return s_kva

    def _observation(self) -> dict:
        """逐步观测（式(8) 窗口的原料）：母线特征 (33,5) + 电价。

        回合终态（done）外生剖面用最后一步（t 已达 288 时钳回 287），
        V/E 用终值，作为 GAE 自举的 s_T。
        """
        t = min(self.t, self.steps_per_day - 1)
        p_load = self.profile.p_load_kw[t]
        q_load = self.profile.q_load_kvar[t]
        pv_avail = self.profile.pv_available_kw[t]
        p_pv = np.zeros(self.n_bus)
        p_pv[np.array(self.der_buses) - 1] = pv_avail
        energy = np.zeros(self.n_bus)
        energy[np.array(self.cbess_buses) - 1] = self.energy_kwh
        bus_features = np.stack(
            [p_load, q_load, p_pv, self._last_v_pu, energy], axis=1
        )
        return {
            "bus_features": bus_features.astype(np.float32),
            "price": np.float32(
                self.profile.price_aud_per_mwh[t] * self.cfg.price_level
            ),
        }
