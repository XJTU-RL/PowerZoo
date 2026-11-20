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
