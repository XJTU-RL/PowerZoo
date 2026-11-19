# SHOM paper: figure replotting

Paper: [Sensitivity-Based Heterogeneous Ordered Multiagent Reinforcement Learning for Distributed Volt-Var Control in Active Distribution Networks](https://doi.org/10.1109/TSG.2025.3540416).

This package maps nine panels in the published paper to outputs saved in the original research notebook. It **replots supplied scalar traces**; it does not train agents or reproduce the paper end to end. No historical logs, result files, or model checkpoints are bundled because their exact correspondence to the published runs has not been verified.

## Plot a panel

From the PowerZoo repository root:

```sh
python -m pip install matplotlib==3.10.8
python reproduction/shom/plot_paper_panels.py 7 \
  --series SHOM path/to/your-run.csv --out fig7.png
```

Pass one `--series LABEL FILE` per trace. Each CSV must have `Step` and `Value` columns. To pass a TensorBoard event file instead, install `tensorboard` and use a single file containing the scalar tag below. The script sorts by recorded step, rejects duplicate steps and non-finite samples, and does not rescale, smooth, shift, interpolate, or discard points. PNG, PDF, and SVG are supported.

| Panel | Trace | TensorBoard scalar tag |
| --- | --- | --- |
| 7, 8(a), 8(b) | 123-bus reward, 13-bus reward, and cross-network reward comparison | `eval_average_episode_rewards` |
| 9(a) | Power-loss ratio | `eval_powerloss_average_episode_rewards` |
| 9(b) | Capacitor control | `eval_capacitor_control` |
| 9(c) | Regulator control | `eval_regulator_control` |
| 9(d) | Battery control | `eval_discharge_control` |
| 12(a) | Regulator control during deployment | `render_regulator_control` |
| 12(b) | Capacitor control during deployment | `render_capacitor_control` |

**Verification:** Panel correspondence was checked against the published figures and saved notebook outputs. Synthetic CSV tests check plotting and input validation, not numerical agreement with the paper. Historical notebook code for Fig. 7 and Fig. 8(a) includes optional scaling; whether it was used for the published curves has not been established. The script does not implement it.
