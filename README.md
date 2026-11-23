# LFC research archive: two-feeder clocked graph control

**Research question.** Can a learned transmission coordinator and feeder-level device policies train and execute in a bidirectional frequency-control loop?

The slow policy reads timestamped delivery and capability reports from two IEEE 34-bus feeders and dispatches at 2 s intervals. Feeder and device-type policies act at 1 s intervals. The storage and PV actors use a phase-coupled network graph; two storage units in each group split the selected output equally. Training uses a clipped policy-gradient update.

**Evidence.** The native 60 Hz pretrial on a three-machine, nine-bus transmission model updated seven actor/critic pairs on one development event and evaluated frozen weights on a separate event. This demonstrates trainability and closed-loop execution, not superiority over a matched baseline or transfer to more feeders. Dispatch is discrete and two-feeder-specific. The formal 50 Hz initialization gate was not met.

**Start here:** [experiment notes, commands and audit](docs/case-studies/lfc-dual-pcc/questions.md) · [training and evaluation entry point](research/lfc_dual_pcc/run_clock_graph.py).

This branch is a static research snapshot. Do not develop on it or delete it automatically.
