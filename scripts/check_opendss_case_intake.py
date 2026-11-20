"""Re-run pinned IEEE4/37 native and PowerZoo static parity, and Case69 shunt screens."""
import argparse
import cmath
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

if __package__:
    from .import_matpower_case69 import dss_text, read_pinned_case
else:
    from import_matpower_case69 import dss_text, read_pinned_case

PINS = {
    "4Bus-DY-Bal/4Bus-DY-Bal.DSS": "8fae57ff4f8c2abfa0f7f6d2915996b1204d2cab2290d6a4858963b9c3ea3011",
    "37Bus/ieee37.dss": "97c2571c1698be56c164d3b4c3a1c7a536df976631d3dba4d5108b8b44498b81",
    "37Bus/IEEELineCodes.DSS": "4bc8e3a0b24ad7b1cb4321f1c3b6c8458b646dd3a83b3cd11cccc02444624cb2",
    "37Bus/IEEE37_BusXY.csv": "e1af4a2e8667938db07ad8bd5704605d77bfea70678ecf89b255b2405f65afce",
}


def readings(circuit, query):
    output = {}
    for bus, kind in query.items():
        if circuit.SetActiveBus(bus) < 0:
            raise ValueError(f"Missing bus {bus}")
        raw = circuit.ActiveBus.VMagAngle
        phasors = {
            n: cmath.rect(raw[2 * i], math.radians(raw[2 * i + 1]))
            for i, n in enumerate(circuit.ActiveBus.Nodes)
        }
        if not {1, 2, 3}.issubset(phasors):
            raise ValueError(f"Missing phase at {bus}")
        pairs = ((1, 2), (2, 3), (3, 1))
        output[bus] = (
            {
                k: abs(phasors[i] - phasors[j])
                for k, (i, j) in zip(("AB", "BC", "CA"), pairs)
            }
            if kind == "LL"
            else {chr(64 + i): abs(phasors[i]) for i in (1, 2, 3)}
        )
    return output


def snapshot(circuit, query):
    taps = {}
    regs = circuit.RegControls
    for name in regs.AllNames:
        if name.lower() == "none":
            continue
        regs.Name = name
        taps[name] = int(regs.TapNumber)
    voltages = {
        str(k).lower(): float(v)
        for k, v in zip(circuit.AllNodeNames, circuit.AllBusVmagPu)
    }
    if len(voltages) != circuit.NumNodes or not all(
        math.isfinite(v) for v in voltages.values()
    ):
        raise ValueError("Invalid phase voltage set")
    loss = float(circuit.Losses[0]) / 1000
    measured = readings(circuit, query)
    if not math.isfinite(loss) or not all(
        math.isfinite(v) for p in measured.values() for v in p.values()
    ):
        raise ValueError("Non-finite loss or voltage reading")
    return dict(
        buses=circuit.NumBuses,
        nodes=circuit.NumNodes,
        phase_pu=voltages,
        loss_kw=loss,
        taps=taps,
        voltages_v=measured,
    )


def compare(folder, master, query, controls, capacitor_site=None):
    import dss
    from envs.powergrid.session import DSSSession

    with tempfile.TemporaryDirectory(prefix="pz-native-static-") as tmp:
        copy = Path(tmp) / "model"
        shutil.copytree(folder, copy)
        context = dss.DSS.NewContext()
        context.Text.Command = f'Compile "{copy/master}"'
        c = context.ActiveCircuit
        c.Solution.Tolerance = 1e-8
        c.Solution.ControlMode = 0 if controls else -1
        c.Solution.SolveSnap()
        if not c.Solution.Converged or (controls and not c.Solution.ControlActionsDone):
            raise RuntimeError("Native solve/control failed")
        native = snapshot(c, query)

        session = DSSSession(folder, master, step_seconds=3600)
        try:
            session.native.Solution.Tolerance = 1e-8
            session.solve(native_controls=controls)
            actual = snapshot(session.native, query)
            if capacitor_site is not None:
                # OpenDSS Powers are consumed terminal kW/kvar, so injection
                # uses the opposite sign. Use terminal 1 of the shunt only.
                actual["capacitor_kvar_injected"] = -float(
                    session.powers("Capacitor.screen")[1]
                )
        finally:
            session.close()
        for k in ("buses", "nodes", "taps"):
            if native[k] != actual[k]:
                raise ValueError(
                    f"Native parity failed at {k}: {native[k]} vs {actual[k]}"
                )
        a, b = native.pop("phase_pu"), actual["phase_pu"]
        if a.keys() != b.keys():
            raise ValueError("Native phase sets differ")
        ev = max(abs(a[n] - b[n]) for n in a)
        ep = abs(native["loss_kw"] - actual["loss_kw"])
        er = max(
            (
                abs(v - actual["voltages_v"][bus][phase])
                for bus, ph in native["voltages_v"].items()
                for phase, v in ph.items()
            ),
            default=0,
        )
        if ev > 1e-8 or ep > 1e-5 or er > 1e-4:
            raise ValueError(f"Native parity failed: {ev:g} pu, {ep:g} kW, {er:g} V")
        return actual, dict(pu=ev, loss_kw=ep, observed_v=er)


def independent_case69(buses, children, order, multiplier, candidate):
    base_mva = 10.0
    base_z = buses[0][9] ** 2 / base_mva
    loads = {
        int(r[0]): multiplier * complex(r[2], r[3]) / (1000 * base_mva) for r in buses
    }
    v = {i: 1 + 0j for i in order}
    for _ in range(500):
        currents = {i: (loads[i] / v[i]).conjugate() for i in order}
        if candidate is not None:
            currents[candidate] += 1j * (100 / (1000 * base_mva)) * v[candidate]
        for i in reversed(order[1:]):
            currents[children[i][0]] += currents[i]
        update = {1: 1 + 0j}
        for i in order[1:]:
            parent, impedance = children[i]
            update[i] = update[parent] - impedance / base_z * currents[i]
        error = max(abs(update[i] - v[i]) for i in order)
        v = update
        if error < 1e-12:
            break
    else:
        raise ValueError("Independent solver failed")
    loss = (
        sum(children[i][1].real / base_z * abs(currents[i]) ** 2 for i in order[1:])
        * base_mva
        * 1000
    )
    return {f"b{i}": abs(s) for i, s in v.items()}, loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("ieee_root", type=Path)
    p.add_argument("matpower_case69", type=Path)
    args = p.parse_args()
    args.ieee_root = args.ieee_root.resolve()
    args.matpower_case69 = args.matpower_case69.resolve()
    for relative, expected in PINS.items():
        if (
            hashlib.sha256((args.ieee_root / relative).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError(f"Pinned file mismatch: {relative}")
    results = {}
    for key, sub, master, query in (
        ("IEEE4", "4Bus-DY-Bal", "4Bus-DY-Bal.DSS", {"n2": "LL", "n4": "LN"}),
        ("IEEE37", "37Bus", "ieee37.dss", {"799": "LL", "701": "LL", "740": "LL"}),
    ):
        with tempfile.TemporaryDirectory(prefix="pz-ieee-static-") as tmp:
            folder = Path(tmp)
            for rel in PINS:
                if Path(rel).parent.as_posix() == sub:
                    shutil.copyfile(args.ieee_root / rel, folder / Path(rel).name)
            actual, errors = compare(folder, master, query, True)
        actual.pop("phase_pu")
        actual["native_error"] = errors
        results[key] = actual
    ref4 = {
        "n2": dict(AB=12340, BC=12349, CA=12318),
        "n4": dict(A=1920, B=2054, C=1986),
    }
    r = results["IEEE4"]
    r["reference_spot_signed_error_v"] = {
        bus: {phase: r["voltages_v"][bus][phase] - v for phase, v in phases.items()}
        for bus, phases in ref4.items()
    }
    if any(
        abs(v) > 3.0
        for values in r["reference_spot_signed_error_v"].values()
        for v in values.values()
    ):
        raise ValueError("IEEE4 3 V reference spot-check regression failed")
    results["IEEE37"][
        "ieee_reference_not_comparable"
    ] = "Kersting ideal 1 p.u. source at bus 799 differs from original DSS upstream transformer"
    stock37 = {
        "799": (4432.856, 4551.986, 4431.463),
        "701": (4762.765, 4889.082, 4694.662),
        "740": (4601.057, 4797.155, 4532.344),
    }
    for bus, values in stock37.items():
        for phase, expected_v in zip(("AB", "BC", "CA"), values):
            if abs(results["IEEE37"]["voltages_v"][bus][phase] - expected_v) > 0.03:
                raise ValueError(
                    f"IEEE37 stock voltage regression failed at {bus}.{phase}"
                )
    expected = {
        "IEEE4": (4, 12, 559.6193, {}),
        "IEEE37": (39, 117, 152.3446, {"creg1a": 16, "creg1c": 14}),
    }
    for label, (buses_expected, nodes_expected, loss, taps) in expected.items():
        row = results[label]
        if (row["buses"], row["nodes"], row["taps"]) != (
            buses_expected,
            nodes_expected,
            taps,
        ) or abs(loss - row["loss_kw"]) > 0.03:
            raise ValueError(
                f"Unexpected pinned snapshot topology, taps or loss for {label}: {row}"
            )
    buses, branches, children, order, sha256 = read_pinned_case(args.matpower_case69)
    results["Case69_source_sha256"] = sha256
    rows = []
    for scale in (0.5, 1, 1.25):
        for site in (None, 61, 64, 65):
            ref, ref_loss = independent_case69(buses, children, order, scale, site)
            with tempfile.TemporaryDirectory(prefix="pz-case69-static-") as tmp:
                folder = Path(tmp)
                # Scale nameplates in the disposable case only. The PowerZoo
                # session disallows a second global load multiplier by design.
                scaled_buses = [
                    row[:2] + [scale * row[2], scale * row[3]] + row[4:]
                    for row in buses
                ]
                deck = dss_text(scaled_buses, branches)
                if site is not None:
                    deck += f"New Capacitor.screen bus1=b{site}.1.2.3 phases=3 conn=wye kV=12.66 kvar=100\n"
                (folder / "network.dss").write_text(deck, encoding="utf-8")
                actual, native_error = compare(folder, "network.dss", {}, False, site)
                if actual["buses"] != 69 or actual["nodes"] != 207:
                    raise ValueError("Unexpected case69 topology")
                phase_pu = actual["phase_pu"]
                if not all(0.8 < v < 1.2 for v in phase_pu.values()):
                    raise ValueError("Outside native constant-power band")
                max_error = max(
                    abs(ref[node.split(".")[0]] - v) for node, v in phase_pu.items()
                )
                loss_error = abs(actual["loss_kw"] - ref_loss)
                if max_error > 1e-6 or loss_error > 1e-3:
                    raise ValueError(
                        f"Case69 independent parity failed at {scale}/{site}: {max_error} p.u., {loss_error} kW"
                    )
                bus_min = min(phase_pu, key=phase_pu.get).split(".")[0]
                q_expected = (
                    None
                    if site is None
                    else 100 * sum(phase_pu[f"b{site}.{i}"] ** 2 for i in (1, 2, 3)) / 3
                )
                measured = actual.get("capacitor_kvar_injected")
                if site is not None and (
                    not math.isfinite(measured) or abs(measured - q_expected) > 1e-5
                ):
                    raise ValueError(
                        f"Case69 capacitor kVAr mismatch at {scale}/{site}"
                    )
                rows.append(
                    dict(
                        load_multiplier=scale,
                        capacitor_bus=site,
                        loss_kw=actual["loss_kw"],
                        min_bus=bus_min,
                        min_voltage_pu=min(phase_pu.values()),
                        cap_kvar_from_voltage=q_expected,
                        cap_kvar_injected=measured,
                        native_parity=native_error,
                        independent_error_pu=max_error,
                        independent_error_loss_kw=loss_error,
                    )
                )
    results["Case69"] = rows
    print(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
