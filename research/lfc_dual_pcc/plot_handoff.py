"""Plot a scored native handoff pair on the common Simulink and DSS time axis."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .compare import load
from .score_handoff import audit, signals


def plot(neutral, issued, delivered, out):
    scores = audit(neutral, issued, delivered)
    reference, _ = signals(neutral)
    colors = {"A": "#1463a0", "B": "#b04d10"}
    fig, axes = plt.subplots(3, 2, figsize=(10, 6.6), sharex=True, sharey="row",
                             constrained_layout=True)
    for col, (mode, path) in enumerate((("issued", issued), ("delivered", delivered))):
        frames = load(path)["frames"]
        raw, frequency = signals(path)
        metric = scores[mode]
        start, stop = metric["window_s"]
        action = [f["write_s"] for f in frames]
        ends = [f["end_s"] for f in frames]
        onset = next(f["write_s"] for f in frames if f["slot"] == metric["release_slot"])
        top, middle, lower = axes[:, col]
        for site, color in colors.items():
            target = [f["request"][site] for f in frames]
            actual = [f["delivered"][site] for f in frames]
            top.step(action, target, where="post", color=color, linewidth=1.2, label=f"{site} request")
            top.plot(ends, actual, linestyle="none", marker="o", markersize=3.5,
                     color=color, markeredgewidth=.7, markerfacecolor="white", label=f"{site} PCC at end")
        times = [t for t, _ in raw["A"]]
        support = [sum(reference[s][i][1] - raw[s][i][1] for s in colors) for i in range(len(times))]
        middle.plot(times, support, color="#3e4451", linewidth=1.1, label="Paired sensor P")
        middle.hlines(8., start, stop, colors="#1463a0", linewidth=1.2, label="8 kW service target")
        lower.plot([t for t, _ in frequency["samples"]],
                   [(v - frequency["reference_hz"]) * 1e6 for _, v in frequency["samples"]],
                   color="#3e4451", linewidth=1.2, marker=".", markersize=2.5)
        lower.axhspan(-50, 50, color="#8995a0", alpha=.13)
        for ax in (top, middle, lower):
            ax.axvspan(start, stop, color="#d9eddb", alpha=.36, zorder=-2)
            ax.axvline(onset, color="#8d286b", linewidth=.9, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#dee2e6", linewidth=.5)
        lower.set_xlim(start-5, stop+4)
        lower.set_xlabel("Simulink time (s)")
        top.set_title({"issued": "Withdraw A on B command",
                       "delivered": "Withdraw A on confirmed B PCC"}[mode])
    axes[0, 0].set_ylabel("Feeder support (kW)")
    axes[1, 0].set_ylabel("Main-grid support (kW)")
    axes[2, 0].set_ylabel("Frequency deviation (µHz)")
    axes[0, 0].legend(loc="upper right", fontsize=7, ncol=2)
    axes[1, 1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Case A • dashed line: A withdrawal starts   |   green: common 6 s scoring window", fontsize=11)
    if out.exists():
        raise FileExistsError(out)
    fig.savefig(out, dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("neutral", "issued", "delivered", "out"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    plot(args.neutral, args.issued, args.delivered, args.out)
