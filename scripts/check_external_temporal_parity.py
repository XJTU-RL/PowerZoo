"""Compare installed DSS native Yearly and PowerZoo snapshot inputs for one day."""

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import numpy as np

INTERVAL_SECONDS = 900
STEPS_PER_DAY = 96
POINTS_PER_YEAR = 35040
OEDI_PIN = "7c8bcca06708ea2dd54b822821d637814ef08dc4"
OEDI_INPUT_SHA256 = "b1dd51b8b5e97bd2fc924d30f5e8b351daed17a584d3ccb4f4e813f0b5c148be"
OEDI_UNSHAPED = {"s48": (210.0, 150.0), "s49c": (35.0, 20.0)}
OEDI_SCRIPTS = ("IEEE123Pv.dss", "IEEE123PvShapes.dss", "IEEE123LoadShapes.dss")
OEDI_PV7 = {
    "kv": "4.16",
    "kva": "50",
    "pmpp": "50",
    "yearly": "pvshape_7",
    "tyearly": "constant_temperature",
    "phases": "3",
}


def day_indices(day):
    if type(day) is not int or not 0 <= day < POINTS_PER_YEAR // STEPS_PER_DAY:
        raise ValueError("day must be a zero-based index in [0, 365)")
    return range(day * STEPS_PER_DAY, (day + 1) * STEPS_PER_DAY)


def oedi_input_digest(root):
    """Match DSSSession's relocation-stable model-tree hash before compiling."""
    paths = sorted(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("symlinks are not accepted in the pinned OEDI input tree")
    file_hashes = {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }
    return sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()


def shape_definitions(root):
    """Parse the known OEDI deck declarations, preserving case-sensitive paths."""
    root = Path(root).resolve()
    result = {}
    for script in OEDI_SCRIPTS:
        for line in (root / "qsts" / script).read_text().splitlines():
            clean = re.split(r"!|//", line, maxsplit=1)[0]
            match = re.match(r"\s*new\s+(tshape|loadshape)\.(\S+)\s+(.*)", clean, re.I)
            if not match:
                continue
            kind, name, properties = match.groups()
            name = name.lower()
            if name in result:
                raise ValueError(f"duplicate profile: {name}")
            if not re.search(r"\bnpts=35040\b", properties, re.I) or not re.search(
                r"\binterval=0\.25\b", properties, re.I
            ):
                raise ValueError(f"unexpected profile interval or length: {name}")
            refs = dict(
                (key.lower(), ref.strip().strip('"'))
                for key, ref in re.findall(
                    r"\b(temp|mult|qmult)=\(file=([^)]+)\)", properties, re.I
                )
            )
            if kind.lower() == "tshape":
                if set(refs) != {"temp"}:
                    raise ValueError(f"unexpected temperature declaration: {name}")
            elif set(refs) not in ({"mult"}, {"mult", "qmult"}):
                raise ValueError(f"unexpected loadshape declaration: {name}")
            paths = {k: (root / "qsts" / v).resolve() for k, v in refs.items()}
            if any(not path.is_relative_to(root) for path in paths.values()):
                raise ValueError(f"profile escapes input root: {name}")
            result[name] = paths
    if not result:
        raise ValueError("no OEDI time series declarations")
    return result


def property_value(circuit, element, key):
    if circuit.SetActiveElement(element) < 0:
        raise ValueError(f"missing element {element}")
    return circuit.ActiveDSSElement.Properties(key).Val.strip()


def verify_unshaped(actual):
    if actual != OEDI_UNSHAPED:
        raise ValueError(f"OEDI unshaped load inventory changed: {actual}")


@dataclass
class OEDIInputs:
    root: Path
    session: object

    def __post_init__(self):
        definitions = shape_definitions(self.root)
        cached = {}

        def curve(name, column):
            path = definitions[name][column]
            if path not in cached:
                data = np.loadtxt(path)
                if data.shape != (POINTS_PER_YEAR,) or not np.isfinite(data).all():
                    raise ValueError(f"invalid OEDI data profile: {path.name}")
                cached[path] = data
            data = cached[path]
            if column != "temp" and (data < 0).any():
                raise ValueError(f"negative OEDI multiplier: {path.name}")
            return data

        circuit = self.session.native
        self.loads, self.pv, self.unshaped = {}, {}, {}
        for n, dev in self.session.loads.items():
            shape = property_value(circuit, f"Load.{n}", "yearly").lower()
            if shape in ("", "none"):
                self.unshaped[n] = (dev.kw, dev.kvar)
            else:
                self.loads[n] = (curve(shape, "mult"), curve(shape, "qmult"))
        verify_unshaped(self.unshaped)
        for n in self.session.pvs:
            elt = f"PVSystem.{n}"
            shape = property_value(circuit, elt, "yearly").lower()
            temp = property_value(circuit, elt, "tyearly").lower()
            self.pv[n] = (curve(shape, "mult"), curve(temp, "temp"))

    def apply(self, session, idx):
        session.apply(
            {
                f"Load.{n}": {
                    "kw": dev.kw * self.loads[n][0][idx],
                    "kvar": dev.kvar * self.loads[n][1][idx],
                }
                for n, dev in session.loads.items()
                if n in self.loads
            }
        )
        session.set_pv_conditions(
            0.0,
            0.0,
            mapping={n: (x[idx], t[idx]) for n, (x, t) in self.pv.items()},
        )


def system_state(circuit, pv_names):
    names = list(circuit.AllNodeNames)
    volts = np.asarray(circuit.AllBusVmagPu)
    source = -np.asarray(circuit.TotalPower)
    pv = {}
    for n in pv_names:
        if circuit.SetActiveElement(f"PVSystem.{n}") < 0:
            raise ValueError(f"missing PV {n}")
        pv[n] = -np.asarray(circuit.ActiveCktElement.TotalPowers[:2])
    if len(names) != len(set(names)) or len(names) != len(volts):
        raise ValueError("invalid native phase node inventory")
    ensure_finite(volts, source, *pv.values())
    if source.shape != (2,) or any(value.shape != (2,) for value in pv.values()):
        raise ValueError("invalid native P/Q pair")
    return dict(zip(names, volts)), source, pv


def ensure_finite(*values):
    if any(not np.isfinite(value).all() for value in values):
        raise ValueError("nonfinite electrical measurement")


def regulator_taps(circuit):
    controls = circuit.RegControls
    taps = {}
    for name in controls.AllNames if controls.Count else ():
        controls.Name = name
        taps[name] = int(controls.TapNumber)
    return taps


def max_tap_error(a, b):
    if a.keys() != b.keys():
        raise ValueError("regulator inventory differs")
    return max((abs(a[n] - b[n]) for n in a), default=0)


def max_phase_error(a, b):
    if not a or a.keys() != b.keys():
        raise ValueError("native and PowerZoo phase nodes differ")
    return float(max(abs(a[n] - b[n]) for n in a))


def aligned_endpoint(index, time_s):
    if not math.isfinite(time_s) or abs(time_s - (index + 1) * INTERVAL_SECONDS) > 1e-6:
        raise ValueError("native Yearly clock did not finish the expected interval")


def run(case, root, day, native_controls):
    root = Path(root).resolve()
    if case != "oedi123":
        return _run_verified(case, root, day, native_controls)
    if oedi_input_digest(root) != OEDI_INPUT_SHA256:
        raise ValueError("OEDI input tree does not match the pinned qsts/profiles")
    with TemporaryDirectory(prefix="powerzoo-oedi-parity-") as folder:
        staged = Path(folder) / "case"
        # Never give either compiler the user input folder; validate the copy too.
        shutil.copytree(root, staged, symlinks=True)
        if oedi_input_digest(staged) != OEDI_INPUT_SHA256:
            raise ValueError("staged OEDI input no longer matches the pin")
        return _run_verified(case, staged, day, native_controls)


def _run_verified(case, root, day, native_controls):
    import dss
    from envs.powergrid.session import DSSSession

    indices = day_indices(day)
    root = Path(root).resolve()
    if case == "oedi123":
        model = "qsts/master.dss"
    else:
        from utils.external_cases import locate_entry
        from envs.powergrid.smartds_timeseries import SmartDSTimeseries

        folder, master = locate_entry("smart_ds_sfo_p4udt4", "solar_high_none", root)
        root = folder.parents[4]
        model = (folder / master).relative_to(root).as_posix()
    native = dss.DSS.NewContext()
    native.AllowChangeDir = False
    native.Text.Command = f'compile "{root / model}"'
    circuit = native.ActiveCircuit
    details = {}
    if case == "oedi123":
        actual = {k: property_value(circuit, "PVSystem.7", k) for k in OEDI_PV7}
        if actual != OEDI_PV7:
            raise ValueError(f"PV7 readback differs from the stock case: {actual}")
        details["pv7_readback"] = actual
    session = DSSSession(root, model, step_seconds=INTERVAL_SECONDS)
    try:
        if case == "oedi123":
            if session.model_sha256 != OEDI_INPUT_SHA256:
                raise ValueError("OEDI input tree does not match the pinned qsts/profiles")
            details["upstream_revision"] = OEDI_PIN
            series = OEDIInputs(root, session)
            details["unshaped_kW_kvar"] = series.unshaped
        else:
            series = SmartDSTimeseries(root, model).bind(session)

        solution = circuit.Solution
        solution.Mode = 2  # OpenDSS Yearly; Solve advances to the next shape sample.
        solution.ControlMode = 0 if native_controls else -1
        solution.Number = 1
        solution.StepSize = INTERVAL_SECONDS
        solution.MaxIterations = 100
        solution.Tolerance = 1e-8
        solution.dblHour = indices.start / 4
        session.reset(seconds=indices.start * INTERVAL_SECONDS)
        rows = []
        for idx in indices:
            solution.Solve()
            aligned_endpoint(idx, solution.dblHour * 3600)
            if not solution.Converged or (
                native_controls and not solution.ControlActionsDone
            ):
                raise RuntimeError(f"native QSTS nonconvergence at interval {idx}")
            a, src_a, pv_a = system_state(circuit, session.pvs)
            session.begin_interval(idx * INTERVAL_SECONDS)
            series.apply(session, idx)
            session.solve(native_controls=native_controls)
            b, src_b, pv_b = system_state(session.native, session.pvs)
            session.finish_interval()
            aligned_endpoint(idx, session.time_s)
            deltas = [np.abs(pv_a[n] - pv_b[n]) for n in pv_a]
            tap_error = (
                max_tap_error(regulator_taps(circuit), regulator_taps(session.native))
                if native_controls
                else None
            )
            rows.append(
                {
                    "index": idx,
                    "max_phase_pu_error": max_phase_error(a, b),
                    "max_regulator_tap_index_error": tap_error,
                    "source_p_error_kw": abs(float(src_a[0] - src_b[0])),
                    "source_q_error_kvar": abs(float(src_a[1] - src_b[1])),
                    "max_pv_p_error_kw": max((float(x[0]) for x in deltas), default=0),
                    "max_pv_q_error_kvar": max((float(x[1]) for x in deltas), default=0),
                }
            )
        limits = (1e-6, 1e-3, 1e-3, 1e-3, 1e-3)
        maxima = {
            key: max(row[key] for row in rows)
            for key in rows[0]
            if key not in ("index", "max_regulator_tap_index_error")
        }
        ensure_finite(list(maxima.values()))
        max_tap = (
            max(row["max_regulator_tap_index_error"] for row in rows)
            if native_controls
            else None
        )
        return {
            "case": case,
            "day_index": day,
            "interval_seconds": INTERVAL_SECONDS,
            "native_controls": native_controls,
            "backend_version": dss.__version__,
            "model_sha256": session.model_sha256,
            "loads": len(session.loads),
            "pvs": len(session.pvs),
            "matched_phase_nodes": len(a),
            "parity_passed": all(v <= tol for v, tol in zip(maxima.values(), limits))
            and (max_tap is None or max_tap == 0),
            "max_regulator_tap_index_error": max_tap,
            "max_errors": maxima,
            "intervals": rows,
            **details,
        }
    finally:
        session.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", choices=("oedi123", "smartds_p4udt4"), required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--day", type=int, default=2)
    p.add_argument("--native-controls", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = run(args.case, args.data_dir, args.day, args.native_controls)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload, end="")
    return 0 if result["parity_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
