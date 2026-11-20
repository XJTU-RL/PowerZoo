# VVC reproduction: SHOM (IEEE Transactions on Smart Grid)

This is a paper-specific entry under **Volt-Var Control (VVC)**, not a new
environment family or a result of the generic VVC benchmark.

- Paper: [IEEE TSG, DOI 10.1109/TSG.2025.3540416](https://doi.org/10.1109/TSG.2025.3540416).
- Authors' materials: [SHOM repository at `f1339d1`](https://github.com/sheldon123z/SHOM/tree/f1339d1259b2c920b7912589ff96055638e40bb9).
- Existing reviewed plotting entry: [PowerZoo `524b868e`](https://github.com/XJTU-RL/PowerZoo/tree/524b868e6f8f62515ba3c240dbb73c7293bbb9cc/reproduction/shom).
  That commit supplies the [guide](https://github.com/XJTU-RL/PowerZoo/blob/524b868e6f8f62515ba3c240dbb73c7293bbb9cc/reproduction/shom/README.md)
  and [replot script](https://github.com/XJTU-RL/PowerZoo/blob/524b868e6f8f62515ba3c240dbb73c7293bbb9cc/reproduction/shom/plot_paper_panels.py); this entry does not copy them.

## Preserve the paper protocol

The paper studies modified 13-, 34- and 123-bus feeders and heterogeneous
capacitor, regulator and battery actions under centralized training and
decentralized execution. Section V reports hourly control; the 13-bus
comparison uses 24 steps per episode and reports 24,000 training steps
across four parallel processes. The 34-bus noisy evaluation adds zero-mean
Gaussian noise with 0.02 p.u. standard deviation; the deployment comparison
uses 24 hourly samples. These are documented paper settings, not claims
that any currently available traces, synthetic profiles or checkpoints
came from that protocol.

Algorithm 1 and Eqs. (27)–(30) are the sensitivity-order reference: compute
the V–Q sensitivity, aggregate the absolute row contributions, weight by
the per-step reward relative to the episode total, and prioritize higher
sensitivity in the actor-update order. A generic VVC smoke test, a fixed-order
fallback, or a newer training run cannot establish faithful SHOM ordering.
Do not infer the original feeder modifications, checkpoint identity, random
seeds or figure transforms from a current default or a similar run.

## Verified panel-to-script crosswalk

| Published panel | Recorded quantity | Existing script argument |
| --- | --- | --- |
| 7 | 123-bus evaluation episode rewards | `7` |
| 8(a) | 13-bus evaluation episode rewards | `8a` |
| 8(b) | Cross-network evaluation episode rewards; plotted series labeled 13/34/123 bus | `8b` |
| 9(a) | 34-bus noisy evaluation: per-episode power-loss ratio | `9a` |
| 9(b) | 34-bus noisy evaluation: capacitor control actions | `9b` |
| 9(c) | 34-bus noisy evaluation: regulator control actions | `9c` |
| 9(d) | 34-bus noisy evaluation: battery discharge actions | `9d` |
| 12(a) | 34-bus one-day deployment: regulator actions | `12a` |
| 12(b) | 34-bus one-day deployment: capacitor actions | `12b` |

The mapping was checked against stored notebook outputs; it is not proof that
replotting supplied scalars reproduces the published numbers. For Fig. 8(b), the published plot and the preserved notebook label the
series `34BUS/SHOM`, consistent with the paper's discussion. The historical
subcaption's reference to 37 bus is a typographical discrepancy. The exact
run configuration and weights have not been independently recovered, so
the plotted label does not establish the feeder configuration or checkpoint.

Historical notebook sources have optional scaling for Figs. 7 and
8(a); it is unknown whether that scaling affected the published traces. The
reviewed script preserves recorded values and does not apply it. Other
figures are not claimed as mapped.

## What can be reproduced

| Level | Current status |
| --- | --- |
| Paper-to-plot mapping | Nine panels verified as above. |
| Replotting from supplied traces | Supported by the linked script; synthetic inputs check parsing and rendering only. |
| Historical evaluation replay | Not verified. No released weight set is tied to an exact paper run, seed, scenario or panel. |
| Numerical match or end-to-end retraining | Not verified. Requires the original feeder/configuration and scenario splits, device/agent mapping, validated sensitivity update order, trace provenance and the paper's evaluation rules. |

Before releasing any further SHOM artifacts, review the exact figure/trace
and checkpoint/experiment crosswalk with the author. A candidate set needs
a matching run configuration, checkpoint fingerprint, evaluation scenario
and agent order, plus an attributable trace or replay. Unverified weights,
historical raw results, figures and private research files are not included.
