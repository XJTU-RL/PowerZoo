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
a cleared rating. Only mock and pure checks have run.
