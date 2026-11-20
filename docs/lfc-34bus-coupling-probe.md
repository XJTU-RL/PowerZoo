# IEEE34 → IEEE9 Bus8 one-way perturbation: draft contract

This is an offline mapping and acceptance draft only. No new OpenDSS run,
MATLAB/Simulink run, controller training, closed-loop validation or equipment
rating claim. The original 230/75 kV paper deck and 13Bus defaults are not used
or changed.

- **Feeder:** pin the accepted 75 kvar row of [the controlled-boundary study](lfc-34bus-operating-point.md),
  commit `fdfb8680a15e76fd87b4801c1cb5f0c85172a3ac`: PowerZoo
  `node_systems/34Bus/network.dss`, source command 1.045 pu,
  `LoadMult=0.75`, 75 kvar delta shunt at 890, no original DER or storage,
  native STATIC regulators, solve tolerance 1e-9. SubXF 69/24.9 kV, 25 MVA,
  XFM1 24.9/4.16 kV, 500 kVA are model ratings.
- **Mapping:** reference feeder import roughly +1486.417 kW / −138.427 kvar,
  re-measured at runtime; negate source terminal-one into-element P/Q once.
  Negative Q means physical feeder export. Write `(anchor + import − reference)
  × 1000` to Bus8 `load1p/load1q` in W/var (unity gain). Report measured
  feeder P/Q, measured feeder deltas, actual demand command and measured
  receiving terminal power separately. The reference reproduces the existing
  13Bus reset anchor +7851.739821876062 kW/+12942.530488711882 kvar.
  Preserve the fixed parallel Load C 100 MW/35 Mvar. These are anchored
  perturbations, not substitution of the feeder for the transmission load.
- **Receiving endpoint:** reuse the verified
  [123 probe implementation](https://github.com/XJTU-RL/PowerZoo/tree/24dddb130f6eaae546d81bf373fabe73f6592c83)
  if later integrated: existing Bus8 dynamic load, power sensor, frequency,
  model hashes, measured clock/tail checks; no new SLX files or permanent
  topology/logging changes. No transmission bus voltage is fed back to DSS.

## Evidence required before any live probe

The adapter first requires externally supplied bytes of the **actual** accepted
static native JSON report, verified against the pinned SHA-256
`d709b9522d21256fb6b3d2fab4e1b17c07c8c9afb07459f1b2cf12b0127fb666`.
The unchanged report was checked read-only and identifies capture commit
`43717ce767cea6950df632414afcfea4bbd7afc7`, not the later publication tip.
The feeder tree is the same at both revisions. The parser verifies capture
commit/deck tree, scenario/source/solver parameters, one
admitted row, measured reference P/Q, six named regulator tap readings and the
published winding/phase evidence. The repository’s synthetic unit-test fixture
uses deliberately dummy phase IDs. It is not real evidence and must not be
approved for a live run. The validated report is stored externally, not in Git. No native-reader
wiring is included in this draft.

Every input snapshot must carry runtime-read source/tree and scenario binding,
the exact phase keys from that observed report (not just a count), the four
named winding terminals with matching bus and model ratings, keyed ABC and
neutral current inventory, and the six matching named regulator taps. The
future runner must independently read and prove no active Storage/PV/Generator
or active loadshape binding, enforce the named shunt and source configuration,
and record both source-terminal flow and global accounting. Typed records do
not by themselves establish that a solver produced the values.

The stateful admission sequence requires neutral reset at index 0, then
strictly next one-second slot once only. Any repeat, skip, invalid native
measurement or downstream failure invalidates the episode. The runner must call
`invalidate()` on engine failure (even if only one P/Q write succeeded), reset
the actual engine to a verified baseline, and then use the adapter's reset.
The Python gate cannot reset or attest an engine itself.

## Proposed bounded probe, not yet run

1. Reproduce the static supported row using the published setup, including the
   six measured taps. Prepare independent frozen snapshots: neutral; add a
   synthetic three-phase constant-PQ delta load at `890.1.2.3`,
   4.16 kV, with separate arms (20 kW, 0 kvar), (0 kW, +15 kvar), and
   (0 kW, −15 kvar) in load-import sign. Verify constant-PQ mode and terminal
   delivery, including negative Q, rather than substituting a voltage-dependent
   shunt or crediting the setpoint. Keep profiles, source and original devices fixed. This is not an
   RL action, field-device proposal or storage test.
2. Check native solver convergence and controls settled, identical tap policy
   and reported tap state, all 95 phase voltages in 0.95–1.05 pu, all four
   named transformer terminal kVA and ABC current ratios under their model
   bounds, L32 current within documented deck limits where available,
   global complex-power residual, and original storage disabled (no inferred
   SOC). Verify trustworthy nonzero signed PCC deltas within proposed
   ±50 kW/±40 kvar; discard failed candidates, don't retune silently.
3. Only after the feeder gate, a new reviewed, timestamped run may apply the
   actual deltas to the existing Bus8 anchor after settling. Use the same
   initial state and one-second hold clock for each arm; verify zero-delta,
   consume each admitted snapshot once in order, and capture matched actual
   terminal P/Q and clock, Bus8 frequency and three-machine output on
   trustworthy native timestamps. This draft's clock check permits at most
   two 1/60 s callback ticks and does not attest signal freshness or write
   ordering. A trustworthy advancing measured response is required, not
   just successful parameter assignment. Feeder times here are exchange
   slots; DSS snapshots are not one-second energy integrations.

## Bounded native OpenDSS input audit

The OpenDSS-only collector at commit
`6f76ce12774f0f737e71e4f2df370a62160b6ab6` was run on Linux
with DSS-Python 0.15.7 and the exact archived feeder tree and external
reference digest declared above. Rebuild from a clean checkout at that
commit, from the repository root, with its ordinary Python dependencies
and the separately retained original `pair-packaged.json` (do not replace
it with a newly generated report or a synthetic fixture):

```bash
sha256sum "$PAIR_REPORT"
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --kill-after=15s 90s python -m scripts.check_lfc34_coupling_inputs \
  "$PAIR_REPORT" --repo "$PWD" > arms.json 2> stderr.log
```

Record the exit status, commit, feeder tree SHA, reference SHA and test
checkout status. Keep JSON/log files outside Git (or redirect to a scratch
path). Per-arm output records input request and pass/fail; accepted arms
include measured terminal, phase and winding records. The observed complete
batch exited **1** with two rejected arms, not a timeout:

| OpenDSS arm at bus 890 | Measured pulse kW/kvar | PCC import kW/kvar | Delta at PCC kW/kvar | Result |
| --- | ---: | ---: | ---: | --- |
| Hold (no pulse) | n/a | +1486.416708 / −138.426778 | 0 / 0 | Accepted |
| +20 kW | not retained | not retained | not retained | Rejected: tap comparison |
| +15 kvar | ≈0 / +15.000000 | +1484.571648 / −122.601355 | −1.845060 / +15.825423 | Accepted |
| −15 kvar | not retained | not retained | not retained | Rejected: tap comparison |

The 95 observed phase voltages span 0.964375–1.045000 pu in the hold and
0.960971–1.045000 pu in the accepted +Q arm. The six named taps match the
reference in both, and the modeled SubXF/XFM1 terminal screens pass. The shunt
injects +70.002502 and +69.514427 kvar respectively; active losses are
145.804854 and 145.777284 kW. Negative PCC Q still denotes feeder export:
the accepted +Q pulse reduces that export, rather than producing positive
total Q import. L32 currents are measured, with no thermal-clearance claim.
For each rejected arm the collector retained the generic tap-comparison
failure, not the identities, positions or other physical measurements.
Thus the evidence does not establish a signed-pulse suite or Bus8 response.
No MATLAB, policy or energy integration was run.

## Proposed separate seconds-scale tap-hold variant

Do not rerun or reinterpret the rejected STATIC arms as passes.
A distinct opt-in study could first reproduce the accepted nominal STATIC
solution, save the six named mechanical regulator tap positions, then lock
only that mechanical control owner at those measured starting positions
before applying independently declared seconds-scale pulses. Recheck each
tap and its controller readback before and after every arm. Report this
separately from the original STATIC re-settling comparison: whether an
actual device would hold over that time is a study assumption, not a claim
about the historical case. A future diagnostic should retain each mismatched
tap ID, expected/actual positions, and pre/post stepping evidence on failure.
All source/shunt/phase, terminal, accounting and no-active-DER checks must
remain enforced. This variant is not implemented or run here.
