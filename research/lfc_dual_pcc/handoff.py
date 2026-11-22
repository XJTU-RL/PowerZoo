"""Case A deterministic hand-off with only the A release trigger varied."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .case_a_plant import CONFIG
from .case_a_runtime import native_episode
from .driver import gates,report,dump,setup,sha
from .hierarchy import Hierarchy
from .metrics import measure

HANDOFF=Path(__file__).with_name("handoff.json")


class Release:
    def __init__(self,mode,cfg):
        if mode not in ("issued","delivered"): raise ValueError(mode)
        self.mode,self.cfg=mode,cfg
        self.start=self.issue=self.release_at=None
        self.reason=None
        self.ready=self.low=self.backup=0
        self.a=0.
        self.last_report=None

    def dispatch(self,slot,error,used,enabled):
        c=self.cfg
        if enabled and self.start is None and error>c["activation_error_hz"]:
            self.start=slot
            self.a=c["pickup_kw"]
        if self.start is not None and self.issue is None and slot-self.start>=c["issue_b_after_ticks"]:
            self.issue=slot
        fresh=False
        if used is not None:
            src=used["slot"]
            if self.last_report is not None and src!=self.last_report+1:
                raise ValueError("expired or missing native handoff report")
            self.last_report=src
            fresh=self.issue is not None and src>=self.issue
            if fresh:
                self.ready=self.ready+1 if used["delivered"]["B"]>=c["b_ready_kw"] else 0
                short=c["pickup_kw"]-sum(used["delivered"].values())
                self.low=self.low+1 if short>=c["backup_shortfall_kw"] else 0
        service=self.issue is not None and slot<self.issue+c["service_ticks_after_issue"]
        if self.release_at is None and self.issue is not None:
            if self.mode=="issued": self.reason="B_request_issued"
            elif fresh and self.ready>=c["consecutive_ready_reports"]: self.reason="B_PCC_confirmed"
            elif slot-self.issue>=c["max_wait_after_issue_s"]: self.reason="common_deadline"
            if self.reason is not None: self.release_at=slot
        if self.release_at is not None:
            self.a=max(0.,self.a-c["release_step_kw"])
        extra=0.
        if (service and fresh and slot-self.issue>=c["backup_delay_ticks"] and
                self.low>=c["backup_required_reports"] and self.backup<c["backup_max_ticks"]):
            extra=c["backup_kw"];self.backup+=1
        requests={"A":self.a if enabled else 0.,"B":c["pickup_kw"]+extra if service and enabled else 0.}
        return requests,dict(slot=slot,pickup_slot=self.start,b_issue_slot=self.issue,release_slot=self.release_at,
                            release_reason=self.reason,used_slot=None if used is None else used["slot"],
                            ready_count=self.ready,short_count=self.low,backup_kw=extra,backup_ticks=self.backup,
                            service_kw=c["pickup_kw"] if service else 0.,requests=requests)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode",choices=["issued","delivered"],required=True)
    p.add_argument("--stage0",type=Path,required=True)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--mechanism",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(); a.dry_run=False
    cfg=json.loads(CONFIG.read_bytes()); hc=json.loads(HANDOFF.read_bytes())
    gates(a,cfg)
    setup(cfg["study"]["eval_seed"])
    os.umask(0o077)
    a.out.mkdir(parents=True,exist_ok=False)
    t=time.monotonic();git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    with report(a.out/(a.mode+".jsonl")) as emit:
        emit(dict(phase="manifest",mode="native",arm=a.mode,method="handoff",git=git,
                  config_sha256=sha(CONFIG),handoff_sha256=sha(HANDOFF),reference_sha256=sha(a.reference),
                  seed=cfg["study"]["eval_seed"]))
        _,_,frames,hz,sensors,ref=native_episode(Hierarchy(cfg),cfg,"fixed",cfg["events"]["evaluation"],
                            a.reference,cfg["study"]["eval_seed"],True,emit,handoff=Release(a.mode,hc))
        m=measure(frames,hz,sensors,ref,cfg["events"]["evaluation"])
        r=dict(metrics=m,training=False,seed=cfg["study"]["eval_seed"],event_split="evaluation",
               config_sha256=sha(CONFIG),elapsed_s=time.monotonic()-t)
        emit(dict(phase="complete",**r))
    dump(a.out/(a.mode+".json"),r)


if __name__=="__main__":
    main()
