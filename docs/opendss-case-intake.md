# OpenDSS case intake

A case is not an RL benchmark merely because it compiles. Keep four stages
separate: source and rights, native DSS compilation, independent physical
agreement, and a task-specific rollout with verified controllable devices.

| Case | Intake status | Intended use |
| --- | --- | --- |
| Baran–Wu 69-bus | Converted from the pinned MATPOWER 8.1 source; static OpenDSS and PowerZoo DSS backend checked | Balanced radial power flow and solver regression |
| IEEE 4-node | Pinned [native delta–wye deck][ieee4]; native and PowerZoo static solves checked; numerical reference and redistribution pending | Balanced transformer and connection checks |
| IEEE 37-node | Pinned [native delta feeder][ieee37]; native and PowerZoo static solves checked; numerical reference and redistribution pending | Unbalanced load flow, line-to-line voltages and regulator checks |
| IEEE 13, 34, 123 and 8500-node | Already tracked by PowerZoo; not imported again here | Existing models; task readiness remains case/config-specific |

[ieee4]: https://github.com/dss-extensions/electricdss-tst/blob/2f65ebd3ea21546102fc96345fc0cee7bcf33537/Version8/Distrib/IEEETestCases/4Bus-DY-Bal/4Bus-DY-Bal.DSS
[ieee37]: https://github.com/dss-extensions/electricdss-tst/blob/2f65ebd3ea21546102fc96345fc0cee7bcf33537/Version8/Distrib/IEEETestCases/37Bus/ieee37.dss

Both DSS candidates are in the mirror of the unmodified OpenDSS SVN data.
The IEEE 37-node master also requires `IEEELineCodes.DSS` and
`IEEE37_BusXY.csv` from its directory. Preserve filename case on Linux.
For this three-wire delta case compare line-to-line voltages, not the
usual line-to-ground magnitudes. First verify the open-delta regulators
under native controls before any agent takes ownership of their taps.

Pinned file bytes (preserve upstream line endings):

| Repository path within `Version8/Distrib/IEEETestCases/` | SHA-256 |
| --- | --- |
| `4Bus-DY-Bal/4Bus-DY-Bal.DSS` | `8fae57ff4f8c2abfa0f7f6d2915996b1204d2cab2290d6a4858963b9c3ea3011` |
| `37Bus/ieee37.dss` | `97c2571c1698be56c164d3b4c3a1c7a536df976631d3dba4d5108b8b44498b81` |
| `37Bus/IEEELineCodes.DSS` | `4bc8e3a0b24ad7b1cb4321f1c3b6c8458b646dd3a83b3cd11cccc02444624cb2` |
| `37Bus/IEEE37_BusXY.csv` | `e1af4a2e8667938db07ad8bd5704605d77bfea70678ecf89b255b2405f65afce` |

See the [IEEE working group's summary](https://cmte.ieee.org/pes-testfeeders/wp-content/uploads/sites/167/2017/08/testfeeders.pdf)
and the [EPRI reference for line-to-line output](https://opendss.epri.com/Show2.html).
Do not claim agreement with a numerical reference until its exact operating
point, transformer connections, phase order, and tap settings are matched.

## IEEE 4-node and 37-node static evidence

On DSS C-API 0.14.5 with DSS-Python 0.15.7, native OpenDSS and PowerZoo
`DSSSession` agreed for each pinned deck with the same solver and control
settings. This is a check of the PowerZoo integration with the same engine,
not an independent worksheet comparison.

- **IEEE 4-node D–Y balanced:** 4 compiled buses, 12 phase nodes and
  559.6193 kW of active loss. This is the 12.47/4.16 kV delta–grounded-wye
  transformer variant, with unity taps; it is not an unbalanced test case.
- **IEEE 37-node:** 39 compiled buses and 117 phase nodes in this deck, even
  though the published feeder name is 37-node. Active loss was 152.3446 kW.
  The native open-delta regulator control loop completed with tap ratios
  1.1000 and 1.0875. The 111 phase-pair magnitudes on the nominal 4.8 kV
  line-to-line base ranged from 4.4315 to 4.9412 kV. Compare the same phase
  pairs and operating point; this range is not a voltage compliance result.

These observations establish static execution only. Numerical comparison
against the original IEEE worksheets and permission to redistribute either
deck are still unverified.

## Reproduce the Baran–Wu 69-bus intake

Obtain the original `data/case69.m` from the [MATPOWER 8.1
release](https://github.com/MATPOWER/matpower/blob/8.1/data/case69.m)
and save it locally. The source pins the git blob, does not execute MATLAB, and
refuses any other input. The upstream
[license notice](https://matpower.org/license/) excludes the case files from
the code license. Do not commit the original or generated data. The example
destination is already excluded by PowerZoo's gitignore.

```sh
python scripts/import_matpower_case69.py /path/to/case69.m \
    data/external/baran_wu_69 --verify-native
```

The destination must be empty. The converter writes `network.dss` and
`verification.json`. Native verification uses OpenDSSDirect.py and compares all
three phase magnitudes at all 69 buses to an independent balanced
backward/forward sweep. It checks topology and balances the comparison in
physical units; it does not adjust loads or impedances to match the solver.
The model has 68 branches, 3,802.1 kW and 2,694.7 kvar of nominal loads,
about 225.0 kW of active loss, and a minimum voltage near 0.90919 p.u.
at bus 65. The independent sweep models an ideal slack; the OpenDSS deck
has small finite source impedance, so its absolute tolerances are printed
as measured errors in the report. The converted deck also passed a static
check through the PowerZoo `DSSSession` backend. A rerun with the final
relative destination produced outputs identical to the previous verified run.

The source is [Baran and Wu (1989)](https://doi.org/10.1109/61.19265), via
MATPOWER. This conversion is balanced and static. It is **not** a native IEEE
PES feeder, a VVC-ready scenario, or evidence of any thermal limits. No
capacitors, tap controls, storage, PV devices, or time series are invented.
Any such additions belong in an explicitly named and separately validated
variant, not in the original-case converter.

## Gate for future cases

1. Prefer [IEEE PES and EPRI's published test-feeder material](https://opendss.epri.com/ExamplesDocsandTestCases.html).
   Pin the exact revision and primary citation, record file hashes and rights,
   and check all nested redirects and data dependencies. Without confirmed
   redistribution rights use local inputs, not repository assets.
2. Compile and solve on a clean DSS context. Check topology and phase counts,
   load and loss units, voltage extremes, and upstream reference agreement.
   Fail on bad inputs or nonconvergence; do not treat DSS default equipment
   ratings as validated data.
3. Recheck the same circuit in PowerZoo. Publish the evidence per task;
   success in one task does not establish readiness for another.

| Task capability | What must pass before marking it ready |
| --- | --- |
| Static power flow | Native and PowerZoo execution with aligned settings; topology, phases, convergence and power balance. For physical agreement, compare against an independent reference. |
| Three-phase / transformer | Verify phase mapping, winding connections, voltage bases and taps. Match the reference phase pairs and operating point; a balanced case does not validate imbalance. |
| Time series | Pin the traces and actual interval duration; verify alignment, time advancement and repeated solver convergence. |
| Dispatch | In addition to traces, verify available resources, rating and energy limits, and feasible scheduling constraints. |
| Volt/VAR control | Identify controllable devices, verify their action mappings and measured response, and enforce one writer per device between native controls and the policy. |
| Switching / restoration | Identify the valid switches and outage states; verify feasible transitions, energization, topology and load recovery after switching. |

Keep each original deck distinct from any variant with added devices or
scenarios. Use evidenced equipment and limits only. Freeze the case manifest,
data and device/control assumptions before final testing; do not repair a
failed final benchmark by changing the original-case model.
