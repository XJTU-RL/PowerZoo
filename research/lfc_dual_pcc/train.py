"""One native development rollout and one parameter update per attribution arm."""
import json
import os
from .case_a_plant import CONFIG
from .driver import gates,parser,run,setup,dump
from .hierarchy import Hierarchy


def main():
    args=parser(__doc__).parse_args()
    cfg=json.loads(CONFIG.read_bytes())
    gates(args,cfg)
    os.umask(0o077)
    args.out.mkdir(parents=True,exist_ok=False)
    results=[]
    for arm in ("attribution_off","attribution_on"):
        setup(cfg["study"]["seed"])
        model=Hierarchy(cfg)
        results.append(run(args,cfg,"learned",arm,model,True))
    if results[0]["model_before"]!=results[1]["model_before"]:
        raise ValueError("paired initial models differ")
    dump(args.out/"paired.json",{"dry_run":args.dry_run,"initial":results[0]["model_before"],
         "end":[r["model_after"] for r in results],"seed":cfg["study"]["seed"],"episodes_per_arm":1})


if __name__=="__main__":
    main()
