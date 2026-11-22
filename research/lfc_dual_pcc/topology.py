"""Verify the exact physical edges and the independent native bases in the shipped SLX."""
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


def check(model_path: Path, config):
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != config["model_sha256"]:
        raise ValueError("model fingerprint mismatch")
    with ZipFile(model_path) as archive:
        root = ET.fromstring(archive.read("simulink/systems/system_root.xml"))
    blocks = {b.attrib["SID"]: b for b in root.findall("Block")}
    edges = []
    for line in root.findall("Line"):
        edges.append({p.text for p in line.iter("P") if p.get("Name") in {"Src", "Dst"}})
    result = {}
    # SIDs and composite electrical terminals are taken from the native drawing.
    for port, ids in {"A": ("1890", "3652", "3580", "3581", "3583", "3582", "3584"),
                      "B": ("3496", "3647", "3681", "3684", "3685", "3682", "3683")}.items():
        bus, sensor, load, p, q, pc, qc = ids
        spec = config["ports"][port]
        if blocks[bus].get("Name").split()[0] != f"Bus{spec['bus']}":
            raise ValueError("bus identity mismatch")
        if blocks[sensor].get("Name").replace("\n", " ") != spec["sensor"]:
            raise ValueError("physical sensor identity mismatch")
        if [blocks[k].get("Name") for k in (p, q)] != spec["blocks"]:
            raise ValueError("controlled input identity mismatch")
        bus_pin = "rconn:1" if port == "A" else "rconn:2"
        required = [(bus+"#"+bus_pin, sensor+"#lconn:1"), (sensor+"#rconn:1", load+"#lconn:1"),
                    (p+"#out:1", pc+"#in:1"), (q+"#out:1", qc+"#in:1"),
                    (pc+"#rconn:1", load+"#lconn:2"), (qc+"#rconn:1", load+"#lconn:3")]
        if any(edges.count(set(pair)) != 1 for pair in required):
            raise ValueError(f"{port}: physical port or P/Q command path changed")
        result[port] = {"bus": spec["bus"], "sensor_sid": sensor, "load_sid": load,
                        "verified_edges": required}
    def prop(sid, name):
        return blocks[sid].findtext(f".//P[@Name='{name}']")
    if (prop("1932","Commented") != "on" or prop("1933","Commented") == "on"
            or [float(prop(s,"Value"))/1000 for s in ("3684","3685")] != config["ports"]["B"]["anchor_kw_kvar"]
            or float(prop("3579","Pbase")) != config["base_mva"] * 1e6):
        raise ValueError("unexpected active loads or power base")
    return {"model_sha256": digest, "base_mva": config["base_mva"], "ports": result,
            "bus6_fixed_load_commented": True, "bus8_fixed_load_active": True}
