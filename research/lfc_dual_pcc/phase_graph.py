"""Three-phase DSS graph and pre-action telemetry for the two IEEE34 contexts."""
from collections import Counter
import hashlib
import json

import numpy as np
import torch

from envs.lfc.diagnostics.ieee34_bus8 import RATINGS


def node(bus, phase):
    return f"{bus.lower().split('.')[0]}.{int(phase)}"


def terminal_injections(devices):
    """Positive P/Q means absorption. Exclude source so PCC is an independent meter."""
    result = {}
    for device, terminals in devices.items():
        if device.startswith("vsource."):
            continue
        for terminal in terminals:
            for conductor in terminal["conductors"]:
                if conductor["node"] not in (1, 2, 3):
                    continue
                key = node(terminal["bus"], conductor["node"])
                pq = result.setdefault(key, [0., 0.])
                pq[0] += conductor["p_kw"]
                pq[1] += conductor["q_kvar"]
    return result


def complex_matrix(data):
    a = np.asarray(data)
    if not np.iscomplexobj(a):
        a = a.reshape(-1, 2)
        a = a[:, 0] + 1j * a[:, 1]
    n = int(np.sqrt(a.size))
    return a.reshape(n, n, order="F")


class PhaseGraph:
    def __init__(self, plant):
        c = plant.c
        self.nodes = sorted(plant.last["voltage"])
        pos = {name: i for i, name in enumerate(self.nodes)}
        kv = np.array([c.Buses[k.split(".")[0]].kVBase for k in self.nodes])
        self.static = np.zeros((len(pos), 9))
        adj = np.zeros((3, len(pos), len(pos)))
        self.devices = {}
        counts = Counter()
        for name in c.AllElementNames:
            c.SetActiveElement(name)
            e = c.ActiveCktElement
            family, label = name.lower().split(".", 1)
            if not e.Enabled or family not in ("line", "transformer", "storage", "pvsystem"):
                continue
            ns = np.asarray(e.NodeOrder).reshape(e.NumTerminals, e.NumConductors)
            terminals = [[pos.get(node(bus, ph)) for ph in phases] for bus, phases in zip(e.BusNames, ns)]
            if family in ("storage", "pvsystem"):
                self.devices[name.lower()] = [i for i in terminals[0] if i is not None]
                continue
            # Complex primitive in Siemens includes the actual line length, mutual
            # impedances and transformer connections/taps. Preserve phase cross terms.
            y = complex_matrix(e.Yprim)
            nc = e.NumConductors
            for a, left in enumerate(terminals):
                rating = RATINGS.get(f"{label}:{a+1}")
                for i in left:
                    if i is not None and rating:
                        self.static[i, 4] = max(self.static[i, 4], np.log1p(rating[1]) / 10)
                for b, right in enumerate(terminals):
                    if a == b:
                        continue
                    for j, i in enumerate(left):
                        for k, h in enumerate(right):
                            if i is None or h is None:
                                continue
                            v = y[a*nc+j, b*nc+k] * kv[i] * kv[h]  # 1 MVA base
                            if abs(v) > 1e-10:
                                adj[:, i, h] += (np.sign(v.real)*np.log1p(abs(v.real)),
                                                 np.sign(v.imag)*np.log1p(abs(v.imag)), np.log1p(abs(v)))
            counts[family] += 1
        st = plant.storage
        self.batteries = [self.devices[d.lower()] for d in st["devices"]]
        self.pv = self.devices[plant.pv["device"].lower()]
        for i, key in enumerate(self.nodes):
            self.static[i, int(key[-1])-1] = 1.
        self.static[:, 3] = kv / 40
        for ix in self.batteries:
            self.static[ix, 5:7] = [st["kw"]/100, st["kwh"]/200]
        self.static[self.pv, 7:] = [plant.pv["kw"]/10, plant.pv["kva"]/10]
        degree = adj[2].sum(axis=1)
        reachable = {0}
        for _ in pos:
            reachable.update(np.flatnonzero(adj[2, list(reachable)].sum(0)).tolist())
        if (len(reachable) != len(pos) or len(pos) != 95 or min(degree) <= 0
                or not counts["transformer"] or not counts["line"] or not all(self.batteries) or not self.pv):
            raise ValueError("incomplete native electrical graph")
        self.adj = torch.tensor(adj / degree[None, :, None], dtype=torch.float32)
        self.pools = [sorted(set(sum(self.batteries, []))), self.pv]
        self.summary = dict(nodes=len(pos), elements=dict(counts), directed_phase_edges=int((adj[2]>0).sum()),
                            devices=self.devices, names=self.nodes, adjacency_sha256=hashlib.sha256(
                                self.adj.numpy().tobytes()).hexdigest())

    def features(self, plant, target):
        r = plant.last
        soc = plant.soc()
        x = np.zeros((len(self.nodes), 9))
        for i, k in enumerate(self.nodes):
            v = r["voltage"][k]
            p, q = r["phase_pq"].get(k, (0., 0.))  # no local shunt/device
            x[i, :4] = [(v-1)*20, min(v-.95, 1.05-v)*20, p/100, q/100]
        for j, ix in enumerate(self.batteries):
            x[ix, 4] = soc[j]
            x[ix, 5] = r["battery_kw"][j]/10
        x[self.pv, 6] = r["pv_kw"]/10
        x[self.pv, 7] = (plant.pv["kw"]-r["pv_kw"])/10  # fixed study irradiance 1
        x[:, 8] = target/10
        full = np.column_stack([self.static, x])
        if not np.isfinite(full).all():
            raise ValueError("non-finite native graph")
        return torch.tensor(full, dtype=torch.float32)
