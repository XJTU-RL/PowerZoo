"""Nominal-frequency inventory of the pinned Simulink archive and two live DSS contexts."""
import argparse
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from scripts.check_lfc_123_coupling import SCENARIO
from .case_a_plant import CONFIG, feeder
from .driver import dump, sha
from .stage0 import CONFIG as WIRING


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    model = ROOT/json.loads(SCENARIO.read_bytes())["model"]
    if sha(model)!=json.loads(WIRING.read_bytes())["model_sha256"]:
        raise ValueError("model fingerprint")
    fields = {"FRated","Freq0","fMechanical0","fElectrical0","freq","frequency","frequencyindice","fundamental","f0","Fn"}
    rows = []
    with ZipFile(model) as z:
        block = ET.fromstring(z.read("simulink/blockdiagram.xml"))
        preload = [p.text for p in block.iter("P") if p.get("Name")=="PreLoadFcn"]
        for f in z.namelist():
            if not f.startswith("simulink/systems/") or not f.endswith(".xml"):
                continue
            for b in ET.fromstring(z.read(f)).iter("Block"):
                params = {p.get("Name"):p.text for p in b.iter("P") if p.get("Name") in fields}
                if params:
                    rows.append(dict(file=f, sid=b.get("SID"), block=b.get("Name"), parameters=params))
    cfg = json.loads(CONFIG.read_bytes())
    ref = ObservedReference.read_report(args.reference.read_bytes())
    sites = {}
    with tempfile.TemporaryDirectory() as tmp:
        deck = native.archived_deck(ROOT,Path(tmp))
        for s in cfg["sites"]:
            with feeder(deck,ref,cfg,s) as p:
                sites[s] = dict(solution_hz=p.c.Solution.Frequency,
                                source_hz=float(native._prop(p.c,"Vsource.source","frequency")),
                                source_base_hz=float(native._prop(p.c,"Vsource.source","basefreq")),
                                pcc=p.base)
    if not rows:
        raise ValueError("empty frequency inventory")
    dump(args.out, dict(model_sha256=sha(model),preload=preload,parameters=rows,actual_dss=sites,
                        formal_50hz_verified=False,initialization_attempted=False,
                        outstanding="50 Hz native machine field/torque and no-event equilibrium, two 50 Hz feeder anchors and dual-port gate"))


if __name__ == "__main__":
    main()
