"""Replot recorded scalar traces for selected SHOM paper panels."""

import argparse
import csv
import math
from pathlib import Path


PANELS = {
    "7": ("eval_average_episode_rewards", "Episode reward"),
    "8a": ("eval_average_episode_rewards", "Episode reward"),
    "8b": ("eval_average_episode_rewards", "Episode reward"),
    "9a": ("eval_powerloss_average_episode_rewards", "Power-loss ratio"),
    "9b": ("eval_capacitor_control", "Capacitor control"),
    "9c": ("eval_regulator_control", "Regulator control"),
    "9d": ("eval_discharge_control", "Battery control"),
    "12a": ("render_regulator_control", "Regulator control"),
    "12b": ("render_capacitor_control", "Capacitor control"),
}

# Colors and line patterns distinguish traces without modifying their samples.
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#8A6300", "#333333")
LINE_STYLES = ("-", "--", "-.", ":")


def read_series(path, tag):
    """Load one run, sort by step, and reject missing or ambiguous samples."""
    if not path.is_file():
        raise ValueError(f"{path}: expected a single input file")

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if not {"Step", "Value"}.issubset(reader.fieldnames or ()):
                raise ValueError(f"{path}: expected Step and Value columns")
            rows = []
            for row in reader:
                try:
                    rows.append((float(row["Step"]), float(row["Value"])))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{reader.line_num}: invalid sample") from exc
    else:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        accumulator = EventAccumulator(
            str(path), size_guidance={"scalars": 0}, purge_orphaned_data=False
        )
        accumulator.Reload()
        if tag not in accumulator.Tags()["scalars"]:
            raise ValueError(f"{path}: scalar tag not found: {tag}")
        rows = [(item.step, item.value) for item in accumulator.Scalars(tag)]

    if not rows or any(not (math.isfinite(x) and math.isfinite(y)) for x, y in rows):
        raise ValueError(f"{path}: expected nonempty, finite samples")
    rows.sort(key=lambda pair: pair[0])
    if any(a[0] == b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"{path}: duplicate steps; provide one non-overlapping run")
    return rows


def plot_series(panel, series, output):
    """Render exactly the loaded series; style affects presentation only."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    with plt.rc_context({"path.simplify": False, "font.size": 10}):
        fig, ax = plt.subplots(figsize=(7.2, 4.1), layout="constrained")
        try:
            for index, (label, rows) in enumerate(series):
                step, value = zip(*rows)
                ax.plot(
                    step,
                    value,
                    label=label,
                    color=COLORS[index % len(COLORS)],
                    linestyle=LINE_STYLES[(index + index // len(COLORS)) % len(LINE_STYLES)],
                    linewidth=1.5,
                    marker=".",
                    markersize=2,
                )
            ax.set(xlabel="Recorded step", ylabel=PANELS[panel][1], title=f"SHOM · Fig. {panel}")
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False)
            fig.savefig(output, dpi=200)
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panel", choices=PANELS)
    parser.add_argument(
        "--series", nargs=2, action="append", required=True, metavar=("LABEL", "LOG"),
        help="Plot one labeled CSV or TensorBoard event file; repeat for each trace.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output PNG, PDF, or SVG.")
    args = parser.parse_args()
    if args.out.suffix.lower() not in {".png", ".pdf", ".svg"}:
        parser.error("--out must end in .png, .pdf, or .svg")
    if len({label for label, _ in args.series}) != len(args.series):
        parser.error("Series labels must be unique")
    try:
        inputs = [(label, Path(path)) for label, path in args.series]
        if any(path.resolve() == args.out.resolve() for _, path in inputs):
            raise ValueError("Output must not overwrite any input")
        tag = PANELS[args.panel][0]
        series = [(label, read_series(path, tag)) for label, path in inputs]
        plot_series(args.panel, series, args.out)
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
