"""Shared provenance, gates, deterministic seeds and checkpoint I/O for Case A."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .case_a_plant import CONFIG
from .case_a_runtime import native_episode
from .hierarchy import Hierarchy,attribution,update
from .metrics import measure
from .score_stage0 import score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(seed):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def dump(path,body):
    with path.open("x") as f:
        json.dump(body,f,indent=2,allow_nan=False)
        f.write("\n")


def state_digest(model):
    h=hashlib.sha256()
    for key,t in model.state_dict().items():
        h.update(key.encode()); h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@contextmanager
def report(path):
    with path.open("x") as f:
        def emit(r):
            f.write(json.dumps(r,allow_nan=False)+"\n"); f.flush()
        try:
            yield emit
        except Exception as exc:
            emit(dict(phase="failed",reason=repr(exc)))
            raise


def dry_episode(model,mode,seed,cfg):
    # Synthetic plumbing for the same five heads, credit function and optimizer.
    rng=np.random.default_rng(seed)
    rows=[]
    windows={}
    for w in (22,24,26):
        observed=[.5,.4,.2,.1,.4,.8,.5,.12,.1,.3,.05,1.]
        sites={}
        for site in ("A","B"):
            cap=5. if site=="A" else 8.
            sites[site]=dict(**attribution(8.,cap,3.),certified=True)
        windows[str(w)]=dict(frequency_reward=-.5,sites=sites)
        for role,head in model.heads.items():
            n=head.policy.out_features
            _,rec=model.action(role,observed,[True]*n,[0.]*n,rng,mode=="eval")
            rows.append(dict(**rec,window=w,at_s=float(w)))
    return windows,rows


def parser(label):
    p=argparse.ArgumentParser(description=label)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--stage0",type=Path)
    p.add_argument("--reference",type=Path)
    p.add_argument("--mechanism",type=Path)
    return p


def gates(args,cfg):
    if args.dry_run: return
    if not args.stage0 or not args.reference or not args.mechanism:
        raise ValueError("physical execution requires stage0 raw directory, reference and native mechanism report")
    checked=score(args.stage0)
    if not checked["stage0_passed"]: raise ValueError("dual PCC gate")
    check=json.loads(args.mechanism.read_bytes())
    if (not check["passed"] or check["config_sha256"]!=sha(CONFIG) or check["reference_sha256"]!=sha(args.reference)):
        raise ValueError("Case A native mechanism gate")


def run(args,cfg,method,arm,model,training):
    dev=training or getattr(args,"development",False)
    seed=cfg["study"]["seed" if dev else "eval_seed"]
    filename=args.out/(arm+".jsonl")
    git=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    start=time.monotonic()
    with report(filename) as emit:
        before=state_digest(model)
        emit(dict(phase="manifest",git=git,seed=seed,mode="synthetic-dry-run" if args.dry_run else "native",
                  arm=arm,method=method,training=training,config_sha256=sha(CONFIG),parameter_sha256=before,
                  reference_sha256=None if args.dry_run else sha(args.reference)))
        if args.dry_run:
            wins,actions=dry_episode(model,"train" if training else "eval",seed,cfg)
            for key,val in wins.items(): emit(dict(phase="synthetic_window",slot=key,**val))
            for a in actions: emit(dict(phase="synthetic_decision",**a))
            metrics=None
        else:
            event=cfg["events"]["development" if dev else "evaluation"]
            wins,actions,frames,hz,sensors,ref=native_episode(model,cfg,method,event,args.reference,seed,not training,emit)
            metrics=measure(frames,hz,sensors,ref,event)
        if training:
            n=sum(s["certified"] for w in wins.values() for s in w["sites"].values())
            if arm=="attribution_on" and not n:
                raise ValueError("no certified windows for attribution training")
            opt=update(model,actions,wins,cfg,arm=="attribution_on")
            emit(dict(phase="optimizer",**opt))
        else:
            opt=None
        after=state_digest(model)
        if training==(before==after): raise ValueError("parameter mutation disagrees with run mode")
        result=dict(arm=arm,method=method,seed=seed,config_sha256=sha(CONFIG),source_git=git,training=training,
                    event_split="development" if dev else "evaluation",dry_run=args.dry_run,model_before=before,model_after=after,metrics=metrics,optimizer=opt,
                    elapsed_s=time.monotonic()-start)
        if training:
            checkpoint=args.out/(arm+".pt")
            with checkpoint.open("xb") as handle:
                torch.save(dict(model=model.state_dict(),config_sha256=sha(CONFIG),dry_run=args.dry_run,
                                role=arm,model_sha256=after,seed=seed,git=git),handle)
            loaded=torch.load(checkpoint,map_location="cpu",weights_only=True)
            verify=Hierarchy(cfg); verify.load_state_dict(loaded["model"])
            if state_digest(verify)!=after: raise ValueError("checkpoint readback differs")
            result["checkpoint_sha256"]=sha(checkpoint)
        emit(dict(phase="complete",**result))
    result["raw_sha256"]=sha(filename)
    dump(args.out/(arm+".json"),result)
    return result
