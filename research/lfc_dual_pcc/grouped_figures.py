"""Produce the frozen group design, held-out comparisons and native handoff diagnostic plots."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from .grouped_source import PROTOCOL, read
from .driver import dump


def groups_plot(out):
    study=read(PROTOCOL)
    groups=study["groups"]
    names=[g for ids in groups.values() for g in ids]
    levels=["L","M","H"]
    fig,ax=plt.subplots(figsize=(6.4,5.2),layout="constrained")
    mat=np.array([[levels.index(x) for x in g] for g in names])
    ax.imshow(mat,aspect="auto",cmap=ListedColormap(["#d9ebf6","#9ccdbb","#438f76"]),vmin=0,vmax=2)
    for i,g in enumerate(names):
        for j,v in enumerate(g):ax.text(j,i,v,ha="center",va="center")
    ax.set_yticks(range(len(names)),[next(k for k,v in groups.items() if g in v)+"   "+g for g in names])
    ax.set_xticks([0,1,2],["Storage SOC","Load multiplier","PV irradiance"]);ax.xaxis.tick_top()
    ax.axhline(5.5,color="white",lw=3);ax.axhline(7.5,color="white",lw=3)
    ax.set_title("Independent Case A reset groups",pad=20)
    fig.savefig(out,dpi=155);plt.close(fig)


def heldout_plot(predictions,out):
    fig,axes=plt.subplots(1,2,figsize=(9.2,4),sharex=True,sharey=True,layout="constrained")
    records=read(predictions); marks={"LHL":"o","HMH":"s","MMM":"^"};colors=["#1463a0","#a35416","#388260"]
    for ax,m in zip(axes,("lookup","mlp"),strict=True):
        unique={}
        for row in records:
            if row["model"]==m:unique[tuple(row["key"][:4])]=row
        for color,(g,marker) in zip(colors,marks.items(),strict=True):
            rows=[v for k,v in unique.items() if k[0]==g]
            ax.scatter([v["witness_kw"] for v in rows],[v["derated_kw"] for v in rows],s=38,marker=marker,
                       facecolors="none",edgecolors=color,label=g,linewidths=1.2)
        ax.plot([0,12],[0,12],linestyle="--",color="#888888",linewidth=.8)
        ax.set_xlim(-.3,12.1);ax.set_ylim(-.3,12.1);ax.set_aspect("equal");ax.set_xlabel("Witnessed power (kW)")
        ax.set_title("Two-group lookup" if m=="lookup" else "Small MLP")
    axes[0].set_ylabel("Validation-derated prediction (kW)");axes[1].legend(fontsize=8)
    fig.suptitle("Held-out test • 3 independent conditions, 36 state–horizon targets")
    fig.savefig(out,dpi=155);plt.close(fig)


def span_plot(path,out):
    pair=read(path);fig,axes=plt.subplots(2,1,figsize=(7.2,5),sharex=True,layout="constrained")
    for mode,color in (("issued","#155b86"),("delivered","#ac521b")):
        rows=pair[mode]["timeline"];t=[v["time_s"] for v in rows];name="On B command" if mode=="issued" else "On B delivery"
        axes[0].plot(t,[v["offset_uhz"] for v in rows],color=color,label=name,marker=".",markersize=3)
        for k in ("frequency_peak","frequency_trough"):
            v=pair[mode][k]
            axes[0].scatter([v["time_s"]],[v["offset_uhz"]],facecolors="none",edgecolor=color,s=80)
        axes[1].plot(t,[v["sum_support_kw"] for v in rows],color=color,label=name,marker=".",markersize=3)
    axes[0].legend();axes[0].set_ylabel("Frequency offset (µHz)")
    axes[1].set_ylabel("Measured grid support (kW)");axes[1].set_xlabel("Simulink time (s)")
    axes[0].set_title("Common handoff window • native frequency and paired sensor support")
    fig.savefig(out,dpi=155);plt.close(fig)


def main(root,out):
    out.mkdir(parents=True,exist_ok=False)
    groups_plot(out/"groups.png")
    span_plot(root/"handoff-span.json",out/"handoff-span.png")
    heldout_plot(root/"evaluation/predictions.json",out/"heldout-power.png")
    study=read(PROTOCOL);table=[]
    for split,ids in study["groups"].items():
        for g in ids:
            native=read(root/"labels"/g/"group.json");rs=native["rows"]
            limit={t:sum(x["rejected_targets"].get(t,0) for x in rs) for t in
                   sorted({reason for x in rs for reason in x["rejected_targets"]})}
            table.append({"group":g,"split":split,"states":len(rs),"source_sha256":native["source_sha256"],
                    "condition_sha256":native["condition_sha256"],"native_seconds":native["native_elapsed_s"],
                    "verified_candidate_paths":sum(x["independent_witness_paths"] for x in rs),
                    "rejected_candidate_reason_counts":limit,"native_accepted_violations":sum(x["actual_accepted_violations"] for x in rs),
                    "voltage_range_pu":[min(x["voltage_range_pu"][0] for x in rs),max(x["voltage_range_pu"][1] for x in rs)],
                    "min_soc_pu":min(x["min_soc_pu"] for x in rs)})
    dump(out/"groups.json",table)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path);p.add_argument("out",type=Path)
    a=p.parse_args();main(a.root,a.out)
