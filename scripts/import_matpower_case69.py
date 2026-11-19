#!/usr/bin/env python3
"""Offline converter for the pinned MATPOWER 8.1 Baran–Wu case69 snapshot."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

GIT_BLOB_SHA1 = "5fbd163c0c3f2c32e3fd998e4100b2e3f786fc5e"
SOURCE_URL = "https://github.com/MATPOWER/matpower/blob/8.1/data/case69.m"
SOURCE_DOI = "10.1109/61.19265"


def matrix(source, field, columns):
    match = re.search(rf"\bmpc\.{field}\s*=\s*\[(.*?)\]\s*;", source, re.S)
    if not match:
        raise ValueError(f"missing {field}")
    rows = [list(map(float, row.split())) for row in match[1].split(";") if row.strip()]
    if any(len(row) != columns or not all(map(math.isfinite, row)) for row in rows):
        raise ValueError(f"unexpected {field} dimensions or values")
    return rows


def read_pinned_case(path):
    data = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    if blob != GIT_BLOB_SHA1:
        raise ValueError("reference snapshot does not match the pinned MATPOWER 8.1 blob")
    source = re.sub(r"%[^\n]*", "", data.decode("utf-8"))
    buses, branches = matrix(source, "bus", 13), matrix(source, "branch", 13)
    if len(buses) != 69 or len(branches) != 68:
        raise ValueError("expected 69 buses and 68 branches")
    for index, row in enumerate(buses, 1):
        if row[0] != index or row[1] != (3 if index == 1 else 1):
            raise ValueError("unexpected slack or bus ordering")
        if row[4] or row[5] or row[9] != 12.66:
            raise ValueError("unexpected shunts or voltage bases")
    children = {}
    adjacency = {i: [] for i in range(1, 70)}
    for row in branches:
        f, t = int(row[0]), int(row[1])
        if row[4:10] != [0] * 6 or row[10] != 1:
            raise ValueError("unsupported branch limits, charging, taps or status")
        if f not in adjacency or t not in adjacency or t in children or t == 1:
            raise ValueError("not the expected outward radial topology")
        children[t] = (f, complex(row[2], row[3]))
        adjacency[f].append(t)
    order = [1]
    for parent in order:
        order.extend(adjacency[parent])
    if set(order) != set(adjacency) or len(order) != len(adjacency):
        raise ValueError("network is not a connected radial feeder")
    return buses, branches, children, order, hashlib.sha256(data).hexdigest()


def solve_reference(buses, children, order):
    """Independent balanced backward/forward sweep, ignoring external source impedance."""
    kv, base_mva = buses[0][9], 10.0
    zbase = kv**2 / base_mva
    s = {int(row[0]): complex(row[2], row[3]) / (1000 * base_mva) for row in buses}
    voltage = {i: complex(1) for i in order}
    for iteration in range(1, 501):
        current = {i: (s[i] / voltage[i]).conjugate() for i in order}
        for i in reversed(order[1:]):
            parent, _ = children[i]
            current[parent] += current[i]
        update = {1: complex(1)}
        for i in order[1:]:
            parent, impedance = children[i]
            update[i] = update[parent] - impedance / zbase * current[i]
        converged = max(abs(update[i] - voltage[i]) for i in order) < 1e-12
        voltage = update
        if converged:
            break
    else:
        raise ValueError("independent reference solver did not converge")
    loss_kw = sum(children[i][1].real / zbase * abs(current[i])**2 for i in order[1:]) * base_mva * 1000
    voltages = {f"b{i}": abs(v) for i, v in sorted(voltage.items())}
    return voltages, loss_kw, iteration


def dss_text(buses, branches):
    kv = buses[0][9]
    lines = [
        "! Balanced, static Baran-Wu case69 using the pinned reference input.",
        "! No device controls, time series, or validated thermal ratings are supplied.",
        "Clear",
        f"New Circuit.baran_wu_69 bus1=b1.1.2.3 phases=3 basekv={kv} pu=1 frequency=60",
        "~ r1=1e-6 x1=1e-6 r0=1e-6 x0=1e-6",
    ]
    for f, t, r, x, *_ in branches:
        i, j = int(f), int(t)
        lines.append(
            f"New Line.l{i}_{j} phases=3 bus1=b{i}.1.2.3 bus2=b{j}.1.2.3 length=1 units=none "
            f"r1={r:.12g} x1={x:.12g} r0={r:.12g} x0={x:.12g} c1=0 c0=0"
        )
    for index, _, p, q, *_ in buses:
        if p or q:
            lines.append(
                f"New Load.ld{int(index)} phases=3 bus1=b{int(index)}.1.2.3 conn=wye "
                f"kv={kv} kw={p:.12g} kvar={q:.12g} model=1 vminpu=0.8 vmaxpu=1.2"
            )
    return "\n".join(lines + [f"Set VoltageBases=[{kv}]", "CalcVoltageBases", "Set ControlMode=Off", ""])


def verify_native(path, reference, reference_loss_kw):
    import opendssdirect as dss
    ctx = dss.NewContext()
    ctx.Text.Command(f'Compile "{path.resolve()}"')
    ctx.Solution.SolveNoControl()
    if not ctx.Solution.Converged() or ctx.Circuit.NumBuses() != 69 or ctx.Circuit.NumNodes() != 207:
        raise ValueError("OpenDSS did not converge with the expected 69 buses / 207 phase nodes")
    deviations = []
    for bus, magnitude in reference.items():
        ctx.Circuit.SetActiveBus(bus)
        values = ctx.Bus.puVmagAngle()[::2]
        if len(values) != 3 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"invalid phase voltages at {bus}")
        deviations.extend(abs(v - magnitude) for v in values)
    loss_kw = ctx.Circuit.Losses()[0] / 1000
    voltage_error, loss_error = max(deviations), abs(loss_kw - reference_loss_kw)
    if not math.isfinite(loss_kw) or voltage_error > 1e-5 or loss_error > 0.1:
        raise ValueError(f"native power flow parity failed: {voltage_error:.6g} p.u., {loss_error:.6g} kW")
    return dict(buses=69, phase_nodes=207, power_loss_kw=loss_kw,
                max_voltage_error_pu=voltage_error, power_loss_error_kw=loss_error)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path, help="New or empty output directory; keep generated data local.")
    ap.add_argument("--verify-native", action="store_true")
    args = ap.parse_args()
    buses, branches, children, order, sha256 = read_pinned_case(args.source)
    v, loss, iters = solve_reference(buses, children, order)
    if not (min(v.values()) > 0.8 and max(v.values()) < 1.2):
        raise ValueError("reference voltages fall outside the modeled constant-power region")
    # The native DSS compiler can change the process working directory.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("refusing to overwrite a nonempty output directory")
    deck = output / "network.dss"
    deck.write_text(dss_text(buses, branches), encoding="utf-8")
    report = dict(source=SOURCE_URL, doi=SOURCE_DOI, source_sha256=sha256, buses=len(buses),
                  branches=len(branches), total_p_kw=sum(row[2] for row in buses),
                  total_q_kvar=sum(row[3] for row in buses),
                  min_voltage_bus=min(v, key=v.get), min_voltage_pu=min(v.values()),
                  reference_loss_kw=loss, reference_iterations=iters,
                  classification="static_powerflow_candidate")
    if args.verify_native:
        report["native_check"] = verify_native(deck, v, loss)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (output / "verification.json").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
