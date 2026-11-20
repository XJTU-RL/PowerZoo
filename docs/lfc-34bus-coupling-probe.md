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

## Proposed bounded probe, not yet run

1. Reproduce the static supported row using the published setup, including the
   six measured taps. Prepare independent frozen snapshots: neutral; add a
   synthetic three-phase 20 kW / 0 kvar constant-power pulse at bus 890; add
   0 kW with either +15 or −15 kvar via a declared signed reactive injection
   there. Keep profiles, source and original devices fixed. This is not an
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
