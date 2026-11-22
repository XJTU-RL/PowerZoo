"""Separate frozen native evaluation or deterministic commissioning controls."""
import json
import os
from pathlib import Path
import torch
from .case_a_plant import CONFIG
from .driver import gates,parser,run,setup,state_digest,sha
from .hierarchy import Hierarchy


def main():
    p=parser(__doc__)
    p.add_argument("--models",type=Path)
    p.add_argument("--arm",choices=["none","fixed","droop","rule","attribution_off","attribution_on"],required=True)
    p.add_argument("--development",action="store_true",help="run a deterministic control on the commissioning event")
    args=p.parse_args()
    if args.development and args.arm.startswith("attribution_"):
        raise ValueError("development selection is for rule commissioning only")
    cfg=json.loads(CONFIG.read_bytes()); gates(args,cfg)
    os.umask(0o077); args.out.mkdir(parents=True,exist_ok=False)
    setup(cfg["study"]["seed"]); model=Hierarchy(cfg)
    method=args.arm
    if method.startswith("attribution_"):
        if args.models is None: raise ValueError("learned evaluation requires persisted checkpoint")
        ck=torch.load(args.models/(args.arm+".pt"),map_location="cpu",weights_only=True)
        if ck["dry_run"]!=args.dry_run or ck["role"]!=args.arm or ck["config_sha256"]!=sha(CONFIG):
            raise ValueError("checkpoint scope or config mismatch")
        model.load_state_dict(ck["model"])
        if state_digest(model)!=ck["model_sha256"]: raise ValueError("checkpoint weights differ")
        method="learned"
    model.eval()
    run(args,cfg,method,args.arm,model,False)


if __name__=="__main__":
    main()
