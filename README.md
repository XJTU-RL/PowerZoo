# LFC research archive: feeder delivery capability estimation

**Research question.** Can a small supervised model estimate deliverable active power, time to delivery and sustainable duration better than a conservative lookup table?

The study constructs labels from finite simulator trajectories and keeps whole operating-condition groups separate for training, validation and test. The MLP and lookup table use the same information. Finite safe trajectories are witnesses, not a proof of maximum deliverable power.

**Evidence.** On the independent test groups, the derated MLP had a power MAE of 3.198 kW versus 2.761 kW for lookup. It had fewer duration overestimates but did not win overall. This estimator was not integrated into the controller.

**Start here:** [study, limitations and rerun](docs/case-studies/lfc-dual-pcc/grouped-estimator.md) · [estimator implementation](research/lfc_dual_pcc/grouped_estimator.py).

This branch is a static research snapshot. Do not develop on it or delete it automatically.
