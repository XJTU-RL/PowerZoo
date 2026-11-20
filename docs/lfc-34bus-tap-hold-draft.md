# Separate seconds-scale IEEE34 tap-hold draft

This is distinct from the retained STATIC four-arm comparison. That run
accepted hold and +15 kvar, and rejected +20 kW and −15 kvar because the
taps changed. No past result is overwritten or recategorized.

The proposed collector reproduces the exact pinned case and settles the native
STATIC controls once at neutral. It validates all six named regulator
controllers/winding mappings, records each native `TapNumber` and
transformer-secondary ratio, and sets ControlMode OFF without setting any tap.
All four original requests are then solved as snapshots with SolveNoControl,
matching actual taps before and after every arm. A disabled temporary pulse and
another no-control solve must recover the measured neutral condition after each
non-neutral arm. Failed or missing tap readback records exact names and
expected/observed values; failed neutral recovery stops the sequence.

The phase, winding-ABC/model-rating, shunt, time-series and DER binding,
source, pulse terminal, and residual checks are the same; no voltage-band or
other threshold is changed. The held report names ControlMode OFF separately
from the initial STATIC settlement. This is one opt-in mechanical hold
assumption for a short timescale, not proof of physical switching delays or
a different way to pass the old STATIC test. No MATLAB/Bus8, storage SOC,
dynamics, or energy evaluation occurs. L32 current remains a measurement, not
a cleared rating. The bounded native result is recorded separately below.


Arm ordinal is not a time reading. Native DSS Hour, Seconds and dblHour are
reported and must remain consistent and unchanged from the admitted nominal
STATIC snapshot. OFF has no per-arm control settlement claim even if the
engine reports its completion flag. A failed arm remains failed; direct
clock, tap and source readbacks are retained when possible in the local
audit, with non-finite diagnostic values marked unavailable. Do not commit
raw native audit output.

## Verified OpenDSS hold results

Exact collector commit `7788b69caae2b1b427b7ba999332ddb0f97cf642`,
deck tree `8fbc1a9a3b7a2a5da636224908298a256b4c2b43`, and the
approved external `pair-packaged.json` digest
`d709b9522d21256fb6b3d2fab4e1b17c07c8c9afb07459f1b2cf12b0127fb666`
were used. Linux DSS-Python 0.15.7; exact clean checkout, one thread, bounded
OpenDSS-only execution. Record the commit/tree/report hashes, exit marker,
and local JSON separately; do not add raw reports to Git.

```bash
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --kill-after=15s 90s python -m scripts.check_lfc34_tap_hold \
  "$PAIR_REPORT" > "$AUDIT_DIR/arms.json" 2> "$AUDIT_DIR/stderr.log"
```

The collector exited **0**, with four accepted arms:

| Arm at 890 | Measured pulse kW/kvar | PCC import kW/kvar | PCC change kW/kvar | Phase voltage min/max pu |
| --- | ---: | ---: | ---: | ---: |
| Hold | no pulse | +1486.416708 / −138.426778 | near 0 / near 0 | 0.964375 / 1.045000 |
| +20 kW | +20.000000 / ≈0 | +1508.812561 / −131.645351 | +22.395854 / +6.781427 | 0.958675 / 1.045000 |
| +15 kvar | ≈0 / +15.000000 | +1484.571648 / −122.601355 | −1.845060 / +15.825423 | 0.960971 / 1.045000 |
| −15 kvar | ≈0 / −15.000000 | +1488.371949 / −154.118663 | +1.955242 / −15.691885 | 0.967741 / 1.045309 |

Each checks the same 95 phase IDs and SubXF/XFM1 modeled winding-terminal
screens; no field thermal-clearance claim. The initial nominal STATIC
settlement read native indices 7/1/2 at reg1a/b/c and 10/9/9 at reg2a/b/c
(ratios 1.04375/1.00625/1.0125 and 1.0625/1.05625/1.05625).
All OFF arms preserved these taps (both ratio and index); all three pulse
arms recovered the initial neutral PCC within the declared tolerance. DSS
Hour, Seconds, dblHour and absolute seconds were all **0** throughout.
Arm order is not time progression. Negative absolute PCC Q still means
reactive export. Earlier STATIC +20 kW and −15 kvar remain rejected on tap
changes; this separately posed held study does not reinterpret them.
No MATLAB, Bus8 dynamics, time-domain electrical coupling, energy
integration or training was run.
