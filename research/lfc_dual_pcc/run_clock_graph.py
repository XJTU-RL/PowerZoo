"""One bounded native on-policy rollout or a separated frozen checkpoint evaluation."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import torch

from .case_a_plant import CONFIG
from .case_a_runtime import native_episode
from .clock_graph import ClockGraphPolicy, Coordinator, update
from .driver import dump, gates, report, setup, sha, state_digest
from .metrics import measure
from .replay_collected import recover
from .stage0 import CONFIG as WIRING


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage0", type=Path, required=True)
    parser.add_argument("--mechanism", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume-collected", type=Path)
    parser.add_argument("--frequency", type=int, required=True, choices=(50, 60))
    args = parser.parse_args()
    train = args.mode == "train"
    if train == bool(args.checkpoint):
        parser.error("only evaluation requires a checkpoint")
    if not train and args.resume_collected:
        parser.error("only unoptimized training trajectories may be recovered")
    if args.frequency == 50:
        parser.error("50 Hz requires reinitialized machine, grid and two feeder operating points plus a new native gate")
    git = subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=normal"], text=True).strip():
        raise RuntimeError("research runtime requires a clean checkout")
    cfg = json.loads(CONFIG.read_bytes())
    args.dry_run = False
    gates(args, cfg)
    cfg["phase_graph"] = True
    seed = cfg["study"]["seed" if train else "eval_seed"]
    event = cfg["events"]["development" if train else "evaluation"]
    setup(seed)
    key = dict(git=git, config=sha(CONFIG), wiring=sha(WIRING), mechanism=sha(args.mechanism),
               reference=sha(args.reference), frequency_hz=args.frequency,
               versions={k:version(k) for k in ("dss-python", "torch", "matlabengine", "numpy")})
    model = ClockGraphPolicy()
    if not train:
        checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
        if checkpoint["key"] != key:
            raise ValueError("source, configuration, physics, or runtime differs from training")
        model.load_state_dict(checkpoint["weights"])
        if state_digest(model) != checkpoint["digest"]:
            raise ValueError("model readback")
    before = state_digest(model)
    control = Coordinator(model, cfg, not train)
    args.out.mkdir(parents=True, exist_ok=False)
    with report(args.out/"raw.jsonl") as emit:
        emit(dict(phase="manifest", mode=args.mode, seed=seed, event=event, key=key, before=before,
                  argv=sys.argv, study="60 Hz engineering pretrial"))
        with torch.set_grad_enabled(train):
            if args.resume_collected:
                frames, trace, sensors, ref = recover(args.resume_collected,control,cfg,key,args.reference,event,emit)
            else:
                _, _, frames, trace, sensors, ref = native_episode(model,cfg,"clock_graph",event,args.reference,
                                                                    seed,not train,emit,coordinator=control)
            # The physical warm-up gate is already enforced by the native receiver.
            # Verify the initialization sample here. The loaded equilibrium has droop.
            if abs(trace[0][0])>1e-8 or abs(trace[0][1]-args.frequency)>.01:
                raise ValueError("native initialization sample disagrees with requested nominal frequency")
            groups = control.returns(frames,trace,ref)
            opt = update(model,groups,cfg["study"]["learning_rate"]) if train else None
        counts = {k:len(v) for k,v in groups.items()}
        if set(counts)!=set(model.heads) or any(n!=(8 if k=="master" else 16) for k,n in counts.items()):
            raise ValueError("missing aligned feedback")
        after = state_digest(model)
        if train == (before==after):
            raise ValueError("unexpected parameter state")
        if train:
            path = args.out/"model.pt"
            with path.open("xb") as f:
                torch.save(dict(weights=model.state_dict(),key=key,digest=after),f)
            snap = torch.load(path,weights_only=True,map_location="cpu")
            restored = ClockGraphPolicy()
            restored.load_state_dict(snap["weights"])
            if state_digest(restored)!=after:
                raise ValueError("checkpoint readback")
        metrics = measure(frames,trace,sensors,ref,event)
        # Positive is additional grid import vs two measured pre-event sensor baselines.
        base = {s:sum(row[0] for row in z[:2])/2 for s,z in sensors.items()}
        metrics["measured_grid_positive_excess_kwh"] = sum(
            max(0.,sum(sensors[s][i][0]-base[s] for s in sensors))*(row["end_s"]-row["write_s"])/3600
            for i,row in enumerate(frames))
        summary = dict(key=key, mode=args.mode, seed=seed, event=event, before=before, after=after,
                       recovered_raw_sha256=sha(args.resume_collected) if args.resume_collected else None,
                       counts=counts, optimizer=opt, metrics=metrics, formal_50hz=False,
                       checkpoint_sha256=sha(args.out/"model.pt" if train else args.checkpoint))
        emit(dict(phase="complete", **summary))
    dump(args.out/"summary.json",dict(raw_sha256=sha(args.out/"raw.jsonl"), **summary))


if __name__ == "__main__":
    main()
