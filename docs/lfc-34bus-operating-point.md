# IEEE 34-bus LFC operating-point audit

This diagnostic uses the existing **scaled legacy research circuit** unchanged.
The SubXF 15,000 kVA nameplate and 230/75/12.5 kV bases are properties of the
repository model, not field ratings for the IEEE feeder. There is no calibrated
operating point in this patch and no MATLAB or policy run.

From a clean root checkout, with DSS-Python and the existing LFC dependencies:

```sh
PYTHONPATH=. python scripts/check_lfc34_operating_point.py . --mode raw --vmin 0.95 --vmax 1.05
PYTHONPATH=. python scripts/check_lfc34_operating_point.py . --mode no-der --vmin 0.95 --vmax 1.05
```

The prepared rank-0 worker and generated loadshape files exist only in a
temporary directory. The report identifies the source commit and file hashes.
Mode `raw` preserves constructor and reset behavior separately. Mode `no-der`
disables all 16 PV units and all four Generator-backed batteries in memory,
runs count-one OpenDSS Duty/SolveNoControl with profile index 0 and the
existing hold state of taps/compensation. This is structural removal, **not a
zero action** and not an RL baseline. The fixed source, taps and capacitors
have no transmission feedback. Introducing a voltage-control owner and its
timing would be a distinct scenario, not a correction to the raw case.

Observed with DSS-Python 0.15.7 at main `7a2c5f44`, using the illustrative
0.95–1.05 pu screening band (not a compliance claim):

| Snapshot | PCC P (MW) | PCC Q (MVAr) | Max SubXF net kVA/rating | Out-of-band phases |
|---|---:|---:|---:|---:|
| Raw constructor | +8.116 | −35.459 | 2.425 | 97 / 99 |
| Raw reset | −6.312 | +9.342 | 0.752 | 75 / 99 |
| No DER / fixed voltage control | +98.719 | +59.425 | 7.682 | 77 / 95 |

Positive PCC values mean import to the feeder. All three converged; all
three failed the declared global complex-power accounting tolerance.
The no-DER residual is +0.215 kW/+0.079 kvar and the minimum phase is
0.82078 pu. All four raw wrapper SOCs are 8.0 (800,000/100,000 kWh),
not a measured state of a native Storage element. Raw PV826 and PV838
declare three phases but only B is supplied by L12 and L31 respectively.
The four extra phase nodes disappear in the no-DER diagnostic.
The script reports conductor currents at transformer terminals and the
nominal terminal line-current equivalent from the three-phase nameplate;
it does not infer delta winding-coil current from terminal conductor current.

Before a calibrated case can be called feasible, pin a physical 34-bus
source and source any replacement ratings, document DER connection by
phase, designate one voltage-control owner and one simulation clock, then
verify source-to-load balance, loaded phase voltages, measured conductor
currents, and winding capacity. Separate upstream equivalent design from
feeder calibration. If mapping a validated feeder to the original Bus8
operating point, measure the feeder PCC at the accepted initial clock and
set the background to reference Bus8 P/Q minus that PCC contribution in
consistent sign and units, or explicitly initialize a different transmission
equilibrium. No such substitution or automatic rating adjustment occurs here.

## Separate standard-scale case

```sh
PYTHONPATH=. python scripts/check_lfc34_operating_point.py . --mode standard-off --vmin 0.95 --vmax 1.05
PYTHONPATH=. python scripts/check_lfc34_operating_point.py . --mode standard-static --vmin 0.95 --vmax 1.05
```

These use the separately pinned `node_systems/34Bus/network.dss` tree. They
disable the two research Storage units without editing the input and use
snapshot rather than paper-case Duty mode. OFF holds taps; STATIC allows
the six native regulator controls to finish before measurement (the
phase/tap-specific output records the final positions). STATIC is a
single steady-state control owner, not a dynamic or causal voltage
feedback controller for Simulink.

| Standard snapshot | PCC P (MW) | PCC Q (MVAr) | SubXF primary net MVA | Min phase pu | Out-of-band phases |
|---|---:|---:|---:|---:|---:|
| OFF | 1.678 | 0.304 | 1.705 | 0.75107 | 77 / 95 |
| STATIC | 2.038 | 0.302 | 2.061 | 0.89307 | 8 / 95 |

The deck has 69/24.9 kV SubXF with **25 MVA research nameplate and
deliberately stiff impedance**, not evidence of 25 MVA equipment in
the physical feeder. STATIC leaves under-voltage phases at buses 812, 814, 852 and
890 despite settled regulation, with the minimum at 814.1. Both snapshots converge but fail the illustrative voltage
screen and the stated complex-power residual tolerance; the STATIC
residual is about 0.0557 kW/0.0304 kvar. This is not an accepted
calibrated operating point. Report absolute measurements rather than
claiming physical hosting capacity from the research rating.

To separate native numerical residue from bad electrical bookkeeping, repeat the
unchanged standard snapshots with `--standard-tolerance 1e-9`. Both tighter
runs accounted for the same 117 enabled elements, summing measured terminal
complex power once per element (thus including transformer dissipation without
double-counting it). Compared with the native default 1e-4, OFF closes to
9.2e-9 kW / 2.9e-8 kvar and STATIC to 4.8e-7 kW / 3.0e-7 kvar; both meet the
unchanged accounting gate. Their voltage classifications remain 77/95 and
8/95 respectively. This is a global power residual check, not a per-node
Kirchhoff-current test.

## IEEE-reference boundary and frozen load screen

The [June 2004 IEEE worksheet, public mirror](https://studylib.net/doc/7612638/ieee-34-node-test-feeder)
reports bus 800 at 1.0500 pu on all phases, with 814.A at 0.9467
and 890.A/B/C near 0.917/0.924/0.918. This is a legitimate historical
stress operating point even when it fails this project's 0.95–1.05 pu
screen. The 832–888 transformer is 24.9/4.16 kV, grounded-wye to
grounded-wye, 500 kVA, and feeds the 10,560-foot 888–890 line, type 300.
The nominal 890 load is 450 kW/225 kvar, delta constant current.
This does not make the PowerZoo model identical to the worksheet.

```sh
PYTHONPATH=. python scripts/check_lfc34_operating_point.py . \
  --mode standard-boundary-matrix --vmin .95 --vmax 1.05
```

This scenario unit freezes the synthetic load multipliers 0.50/0.75/1.00,
a 1.05 pu source command, native STATIC controls, a 1e-9 solve tolerance,
and the existing no-DER standard deck. The realized voltage at node 800
is separately measured from the upstream source command. The JSON keeps all
three modes, including failures, and records the actual winding currents and
apparent power against the **research-deck** ratings.

| Synthetic load | PCC kW/kvar | Primary/secondary SubXF kVA, each rated 25,000 | Min/max phase pu | Strict screen violations |
|---|---:|---:|---:|---|
| 0.50 | +965.355/−381.347 | 1037.947/1037.957 | 0.971710/1.050002345 | sourcebus A/B/C, 800 A/B/C high |
| 0.75 | +1480.449/−67.904 | 1482.005/1482.006 | 0.947445/1.049999455 | 890.C low |
| 1.00 | +2039.876/+281.881 | 2059.260/2059.253 | 0.918700/1.049996863 | 814.A, 890.A/B/C low |

All 95 phase nodes were energized in each scenario; native controls
completed and all three met the same global power residual gate. Both
winding terminal phase currents were below their research nameplate current
equivalents, not direct delta coil current or field thermal certifications.
At light load the fixed boundary sits at the upper edge and slight reactive
export accompanies a tiny voltage rise. The strict comparison reports
those boundary overages rather than rounding them away or relaxing the band.
**None of the three rows clears all gates; no calibrated coupling point
was selected.** No changes to base files, device settings or ratings were
used to pass the screen.

## Mid-load local reactive-support sensitivity

```sh
PYTHONPATH=. python scripts/check_lfc34_operating_point.py . \
  --mode standard-midload-shunt --vmin .95 --vmax 1.05
```

Only the frozen 0.75-load, 1.05-source, STATIC, no-original-DER scenario
is used here. The separate temporary layer holds 0/75/150 kvar of nominal
three-phase delta shunt at bus 890, 4.16 kV, one terminal with three
conductors. These are synthetic 0, one-third and two-thirds fractions of
the historical 890 load's 225 kvar nominal demand, not equipment selections.
Actual reactive injection is measured from the capacitor terminal.

| Nominal shunt kvar | Injected kvar | PCC kW / kvar | Feeder loss kW | 890 min L-N pu (C) | 890 AB / BC / CA kV | Strict voltage failure |
|---:|---:|---:|---:|---:|---|---|
| 0 | 0 | +1480.449 / −67.904 | 146.361 | 0.947444959 | 3.9862 / 3.9411 / 3.9510 | 890.C low |
| 75 | 69.5821 | +1481.245 / −137.497 | 144.388 | 0.957597585 | 4.0352 / 3.9928 / 3.9927 | 800.C high, 1.050000013965 |
| 150 | 143.0618 | +1488.427 / −209.205 | 145.928 | 0.972200834 | 4.0846 / 4.0477 / 4.0555 | 800.C high, 1.050000612452 |

All three cases converge and the native controls complete.
L32 has two three-conductor terminals; XFM1 and SubXF each have two
four-conductor terminals. The JSON retains measured phase/neutral terminal
currents and kVA for each, along with every regulator tap. Peak L32
phase current is 52.490/48.829/47.037 A respectively; peak XFM1
low-side phase current is 52.474/48.824/47.037 A. This does not certify
winding-coil current or field thermal capability. Greater shunt support
increased reactive export at the PCC and did not monotonically reduce loss.

No row met the strict voltage screen, despite small upper-limit overages
in the two supported cases. The study stops here: no relaxed limits,
rating edits, additional capacity or source setpoints were selected. A
separately predeclared controllable source boundary with headroom (for example
1.045 pu) would be a different experimental assumption, not a repair or
reproduction of the IEEE reference. It has not been run here.
