# External feeder temporal parity

These diagnostics compare the same OpenDSS engine in two contexts: its native
Yearly solver and PowerZoo's `DSSSession` snapshot solver with explicit source
profile assignments. Passing demonstrates equivalent temporal input and solver
integration for one day, not independent physical validation, a VVC/dispatch
benchmark, or an RL training result.

## Inputs (outside Git)

- **OEDI IEEE123 ZIP variant:** [OEDI deposit](https://doi.org/10.25984/2228282),
  [upstream revision](https://github.com/openEDI/oedisi-ieee123/tree/7c8bcca06708ea2dd54b822821d637814ef08dc4).
  Place its unchanged `qsts/` and `profiles/` directories under one external
  directory. Use `qsts/master.dss`, not `snapshot/master.dss`. The pinned QSTS
  entry requires local line codes, regulator and load/PV definitions, and the
  sibling profile trees. Keep the source revision and rights distinct: upstream
  GitHub BSD-3-Clause notice and OEDI data deposit CC BY 4.0. Do not add the
  deck or its external traces to this branch.
- **SMART-DS SFO P4UDT4:** [OEDI deposit](https://data.openei.org/submissions/2981).
  Use the existing minimal file manifest with the `solar_high_none` entry:
  `python scripts/download_external_case.py --case smart_ds_sfo_p4udt4 --data-dir /path/external`.
  Pass that same external directory to the diagnostic.

```sh
python -m scripts.check_external_temporal_parity --case oedi123 \
  --data-dir /path/oedi --day 2 --native-controls --output /path/oedi-report.json
python -m scripts.check_external_temporal_parity --case smartds_p4udt4 \
  --data-dir /path/external --day 2 --output /path/smartds-report.json
```

Day 2 is zero-based. The native Yearly solve reads the i-th zero-based shape
sample after incrementing its clock to `(i+1)*900` seconds. PowerZoo writes that
same sample to snapshot mode at `i*900`, solves, and advances its clock exactly
once. Do not combine a Yearly load shape and explicit scaling in one solve.
Both contexts use matching disabled or native STATIC controls (as requested),
solver limits, and tolerance. Compare a keyed phase-node inventory, source
active/reactive power, and each PV's measured terminal power, not commands.

OEDI uses IEEE feeder nominal powers with ZIP coefficients and separate
multiplier/temperature files. It has 91 compiled loads; `s48`
(210 kW, 150 kvar) and `s49c` (35 kW, 20 kvar) have no yearly shape and remain
at their base setpoints; the validator refuses any extra unshaped load. Nine
PV systems are present, not all 14 declared PV shapes. For stock `PVSystem.7`,
DSS-Python 0.15.7 compiled the extra token without changing the read-back:
3 phases, 4.16 kV, 50 kVA, 50 kW Pmpp, `pvshape_7`, and
`constant_temperature`. This is engine-specific evidence, not permission
to silently fix the upstream file. The [DOE workflow](https://www.energy.gov/sites/default/files/2024-01/November%202023%20Data%20Preprocessing%20by%20ANL.pdf)
describes mapping processed measured inputs onto a test network; the final
profiles are not measurements from this IEEE feeder.

## Day 2 on DSS-Python 0.15.7

| Input / control | Converged intervals | Phase nodes | Max voltage error p.u. | Max source P/Q error kW/kvar | Max per-PV P or Q error |
| --- | ---: | ---: | ---: | ---: | ---: |
| OEDI / OFF | 96 | 278 | 5.60e-10 | 5.68e-6 / 1.79e-6 | 1.51e-7 |
| OEDI / STATIC | 96 | 278 | 8.15e-10 | 6.49e-6 / 3.34e-6 | 8.99e-8 |
| SMART-DS / OFF | 96 | 156 | 9.78e-11 | 3.83e-6 / 3.07e-6 | 7.12e-8 |

Raw reports remain outside Git. The SMART-DS case has 36 compiled loads and 6
PV systems. Other days and engines require a separate check; the diagnostic
does not certify time-series measurements or task readiness.
