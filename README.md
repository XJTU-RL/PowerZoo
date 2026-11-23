# LFC research archive: multi-feeder capacity allocation

**Research question.** How should a transmission system allocate frequency support across feeders with different deliverable capacities?

This study compares a centralized two-feeder allocation policy with shared local offer policies and capped clearing for more feeders. The capacity learners use linear policies and policy-gradient training. The native 2- and 3-feeder capacity experiments hold the fast device controller fixed to a feasible rule; the 4- and 6-feeder experiments use only the reduced model.

**Evidence.** The reduced-model slow learner lowered tracking error for the archived holdout cases. In the native three-feeder case, the learned and reference allocations coincided, so it does not demonstrate a learned coordination advantage. A separate conditional-credit PPO prototype is present, without a native conditional-update performance result. This archive is not a validation of integrated multi-feeder graph control.

**Start here:** [experiment notes](docs/case-studies/lfc-dual-pcc/questions.md) · [capacity study](research/lfc_dual_pcc/run_capacity_cases.py) · [native closed loop](research/lfc_dual_pcc/run_capacity_closed_loop.py) · [saved outputs](research/lfc_dual_pcc/results/).

This branch is a static research snapshot. Do not develop on it or delete it automatically.
