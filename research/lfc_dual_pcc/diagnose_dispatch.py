"""Trace capacity projection and replay frozen local candidate sets, without training."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import subprocess
import tempfile

import torch

from envs.lfc.diagnostics.ieee34_bus8 import ObservedReference
from scripts.check_lfc34_flexible_load import ROOT, native
from research.lfc_dual_pcc.case_a_plant import CONFIG, feeder
from research.lfc_dual_pcc.clock_graph import ClockGraphPolicy, Message, tensor
from research.lfc_dual_pcc.driver import sha, state_digest
from research.lfc_dual_pcc.master_dispatch import SITES, allocate_support, bid_limits, feasible_actions
from research.lfc_dual_pcc.phase_graph import PhaseGraph


def close(found, expected, label, tol=1e-7):
    if not math.isclose(found, expected, rel_tol=0., abs_tol=tol):
        raise ValueError(f"{label}: {found} != {expected}")


def equivalents(rows):
    distinct = []
    for values in rows:
        if not any(all(math.isclose(a, b, rel_tol=0., abs_tol=1e-8)
                       for a, b in zip(values, other, strict=True)) for other in distinct):
            distinct.append(values)
    return len(distinct)


def slow_windows(holds, policies, packets, commands, ref, cfg):
    results = {}
    for slot, cmd in commands.items():
        master = policies[slot, "master"]
        now = master["consumed_s"]
        reports = {s: packets.get(master["reports"][s]) for s in SITES}
        if any(p is not None and not p.valid(now) for p in reports.values()):
            raise ValueError("master consumed a report outside its time window")
        limits = bid_limits(reports, cfg["fast_s"])
        if limits != cmd["bid_limits_kw"]:
            raise ValueError("allocation caps disagree with recorded telemetry")
        error = ref - holds[slot]["frequency"][1]
        alternatives = []
        for action in range(9):
            projected, request = allocate_support(
                error, action, reports, limits, cfg["max_request_kw"], cfg["droop_kw_hz"],
            )
            fraction = (.2, .5, .8)[action//3]
            alternatives.append(dict(action=action, request_kw=request,
                                     preferred_kw=[request*fraction, request*(1.-fraction)],
                                     projected_kw=list(projected.values())))
        chosen = alternatives[master["action"]]
        close(chosen["request_kw"], cmd["request_kw"], "master request")
        for site, value in zip(SITES, chosen["projected_kw"], strict=True):
            close(value, cmd["data"][site], "projected dispatch")
        results[slot] = dict(chosen, limits_kw=limits, distinct_preferred=equivalents(
            a["preferred_kw"] for a in alternatives), distinct_projected=equivalents(
            a["projected_kw"] for a in alternatives), alternatives=alternatives, fast=[])
    return results


def logits(head, inputs, policy):
    mask = torch.tensor(policy["mask"], dtype=torch.bool)
    scores = head.actor(*inputs)
    if "logit_offsets" in policy:
        scores = scores + scores.new_tensor(policy["logit_offsets"])
    allowed = scores.masked_fill(~mask, -torch.inf)
    if int(allowed.argmax()) != policy["action"]:
        raise ValueError(f"frozen argmax differs for {policy['role']} at {policy['slot']}")
    return dict(mask=policy["mask"], chosen=policy["action"], raw=scores.tolist())


def replay(directory, cfg, reference, reference_sha, deck, model, key, checkpoint_sha, arm):
    summary = json.loads((directory/"summary.json").read_bytes())
    raw = directory/"raw.jsonl"
    if (summary["key"] != key or summary["raw_sha256"] != sha(raw)
            or summary["checkpoint_sha256"] != checkpoint_sha or summary["master_policy"] != arm
            or summary["mode"] != "eval" or summary["before"] != summary["after"]
            or summary["formal_50hz"] is not False or key["reference"] != reference_sha):
        raise ValueError("frozen run metadata mismatch")
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    if rows[-1]["phase"] != "complete" or any(r["phase"] == "failed" for r in rows):
        raise ValueError("requires a complete native trajectory")
    holds = {r["slot"]: r for r in rows if r["phase"] == "hold"}
    policies = {(r["slot"], r["role"]): r for r in rows if r["phase"] == "policy"}
    inputs = {(r["slot"], r["site"]): r for r in rows if r["phase"] == "device_input"}
    packets = {r["sequence"]: Message(**{k: v for k, v in r.items() if k != "phase"})
               for r in rows if r["phase"] == "telemetry"}
    commands = {r["sequence"]: r for r in rows if r["phase"] == "dispatch"}
    graph_record = next(r["graphs"] for r in rows if r["phase"] == "graphs")
    windows = slow_windows(holds, policies, packets, commands, summary["metrics"]["reference_hz"], cfg)
    table = cfg["actions_battery_pv_kw"]
    with ExitStack() as stack:
        plants = {s: stack.enter_context(feeder(deck, reference, cfg, s)) for s in SITES}
        graphs = {s: PhaseGraph(p) for s, p in plants.items()}
        if any(g.summary != graph_record[s] for s, g in graphs.items()):
            raise ValueError("reconstructed graph does not match frozen graph")
        for slot, hold in holds.items():
            for site, plant in plants.items():
                selected = hold["actions"][site]
                if (slot, site) in inputs:
                    source = inputs[slot, site]
                    command = source["command"]
                    req = hold["request"][site]
                    close(req, commands[command]["data"][site], "effective request")
                    coordinator = policies[slot, "feeder"+site]
                    target = req*(.65, 1., 1.25)[coordinator["action"]//2]
                    close(target/10, source["context"][0], "device target")
                    candidates = plant.candidates()
                    eligible = feasible_actions(table, candidates, req)
                    options = {i: dict(veto=r["veto"], pcc=r.get("pcc"),
                                       support_kw=plant.base[0]-r["pcc"][0] if "pcc" in r else None)
                               for i, r in enumerate(candidates)}
                    battery, pv = (policies[slot, role+site] for role in ("battery", "pv"))
                    b = (0, 4, 8)[battery["action"]]
                    if battery["mask"] != [any(table[i][0] == k for i in eligible) for k in (0, 4, 8)]:
                        raise ValueError("reconstructed battery mask disagrees with frozen mask")
                    if pv["mask"] != [any(table[i] == [b, k] for i in eligible) for k in (0, 2)]:
                        raise ValueError("reconstructed PV mask disagrees with frozen mask")
                    if selected not in eligible or table[selected] != [b, (0, 2)[pv["action"]]]:
                        raise ValueError("frozen joint action differs from replay")
                    graph = graphs[site]
                    xb, context = tensor(source["x"]), tensor(source["context"])
                    action_scores = {
                        "feeder": logits(model.heads["feeder"+site], (tensor(coordinator["x"]),), coordinator),
                        "battery": logits(model.heads["battery"+site], (xb, graph.adj, graph.pools[0], context), battery),
                        "pv": logits(model.heads["pv"+site], (xb, graph.adj, graph.pools[1], context), pv),
                    }
                    nearest = min(eligible, key=lambda i: (abs(options[i]["support_kw"]-target), i))
                    windows[command]["fast"].append(dict(
                        slot=slot, site=site, request_kw=req, target_kw=target, chosen=selected,
                        delivered_kw=hold["delivered"][site], eligible=eligible, nearest=nearest,
                        candidates=options, scores=action_scores, soc=plant.soc(), previous_kw=plant.previous,
                    ))
                else:
                    plant.probe(0)
                actual = plant.execute(selected)
                for field in ("pcc", "soc", "soc_after"):
                    for a, b in zip(actual[field], hold["actual"][site][field], strict=True):
                        close(a, b, f"{site} native replay {field}")
                close(actual["delivered_kw"], hold["delivered"][site], f"{site} native replay delivery")
                # The native receiver tried all candidates again when publishing the end-of-hold offer.
                plant.candidates()
    return summary, windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("learned", "fixed", "checkpoint", "reference"):
        parser.add_argument(arg, type=Path)
    args = parser.parse_args()
    model = ClockGraphPolicy()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["weights"])
    model.eval()
    key = ckpt["key"]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config_sha, reference_sha, checkpoint_sha = sha(CONFIG), sha(args.reference), sha(args.checkpoint)
    digest = state_digest(model)
    if (key["git"] != head or key["config"] != config_sha or key["frequency_hz"] != 60
            or key["reference"] != reference_sha or digest != ckpt["digest"]):
        raise ValueError("run the diagnostic from the frozen 60 Hz source and matching inputs")
    cfg = dict(json.loads(CONFIG.read_bytes()), phase_graph=True)
    reference = ObservedReference.read_report(args.reference.read_bytes())
    result = {}
    with torch.no_grad(), tempfile.TemporaryDirectory() as scratch:
        deck = native.archived_deck(ROOT, Path(scratch))
        summaries = []
        for arm in ("learned", "fixed"):
            summary, windows = replay(getattr(args, arm), cfg, reference, reference_sha, deck,
                                      model, key, checkpoint_sha, arm)
            if digest != summary["after"]:
                raise ValueError("activity used a different frozen model")
            result[arm] = windows
            summaries.append(summary)
        if any(summaries[0][k] != summaries[1][k] for k in ("event", "seed", "key")):
            raise ValueError("unpaired held-out runs")
    print(json.dumps(dict(commit=head, checkpoint_sha256=checkpoint_sha,
                          raw_sha256={k: s["raw_sha256"] for k, s in zip(result, summaries, strict=True)},
                          arms=result), allow_nan=False))


if __name__ == "__main__":
    main()
