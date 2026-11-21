"""Render paired native Bus8 observations from one private calibration report."""
import argparse
import json
import math
from pathlib import Path


def plot(source, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    traces = {}
    start = release = None
    with source.open() as handle:
        for line in handle:
            row = json.loads(line)
            phase = row["phase"]
            if phase in ("inputs","manifest","design"):
                design = row.get("design",row.get("config"))
                clock = design.get("clock", design)
            name = row.get("arm",row.get("case"))
            if phase == "native_frequency":
                traces[name] = [(s["time_s"],s["hz"]) for s in row["data"]]
            if phase == "native_logs":
                traces[name] = row["signals"]["frequency_hz"]
            if phase in ("hold_command","write"):
                rec = row.get("record",row)
                slot = rec.get("slot",rec.get("start_s"))
                if slot == clock["event_start_s"]:
                    start = rec["pause_s"]
                if slot == clock["event_stop_s"]:
                    release = rec["pause_s"]
    if start is None or release is None or not traces:
        raise ValueError("missing native write boundaries or frequency")
    reference = "zero_a" if "zero_a" in traces else "uncontrolled"
    ax = plt.subplots(figsize=(8.2,3.7), layout="constrained")[1]
    ax.axvspan(start, release, color="0.9", label="applied disturbance")
    for name, points in traces.items():
        if name == reference:
            continue
        seconds, mhz = [], []
        for (t,v),(t0,v0) in zip(points,traces[reference],strict=True):
            if abs(t-t0)>1e-7 or not all(map(math.isfinite,(t,v,t0,v0))):
                raise ValueError("invalid native time pair")
            if t >= start-2:
                seconds.append(t)
                mhz.append(1000*(v-v0))
        ax.plot(seconds,mhz,marker=".",markersize=2.6,linewidth=1,label=name)
    ax.set(xlabel="Native simulation time (s)",ylabel=f"Bus 8 frequency minus {reference} (mHz)")
    ax.grid(alpha=.22)
    ax.legend(loc="upper left",bbox_to_anchor=(1,1),frameon=False)
    with destination.open("xb") as file:
        ax.figure.savefig(file,format="png",dpi=180)
    plt.close(ax.figure)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report",type=Path)
    parser.add_argument("png",type=Path)
    args=parser.parse_args()
    plot(args.report,args.png)
