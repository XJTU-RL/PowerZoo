# OpenDSS case intake

A case is not an RL benchmark merely because it compiles. Keep four stages
separate: source and rights, native DSS compilation, independent physical
agreement, and a task-specific rollout with verified controllable devices.

| Case | Intake status | Intended use |
| --- | --- | --- |
| Baran–Wu 69-bus | Converted from the pinned MATPOWER 8.1 source; static OpenDSS and PowerZoo DSS backend checked | Balanced radial power flow and solver regression |
| IEEE 4-node | Pinned [native delta–wye deck][ieee4]; native execution and redistribution scope pending | Transformer and connection checks |
| IEEE 37-node | Pinned [native delta feeder][ieee37]; native execution and redistribution scope pending | Unbalanced load flow, line-to-line voltages and regulator checks |
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
check through the PowerZoo `DSSSession` backend.

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
3. Recheck the same circuit in PowerZoo. If suitable for a control task, verify
   devices, action mapping and exactly one control owner per device before a
   short deterministic rollout. Record which steps actually passed; only then
   label the case ready for that task.
