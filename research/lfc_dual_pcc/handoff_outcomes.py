"""Separate the external grid step, common bridge target and local request tracking."""
import argparse
from pathlib import Path

from .compare import load
from .driver import dump, sha
from .score_handoff import audit, interval_area, signals


def score(neutral, issued, delivered):
    paired = audit(neutral, issued, delivered)
    neutral_p, _ = signals(neutral)
    results = {}
    for mode, path in (("issued", issued), ("delivered", delivered)):
        trial = load(path)
        trace, hz = signals(path)
        combined = []
        for pos, (timestamp, _) in enumerate(trace["A"]):
            kw = sum(neutral_p[s][pos][1] - trace[s][pos][1] for s in ("A", "B"))
            combined.append((timestamp, kw))
        frames = trial["frames"]
        event = [row for row in frames if row["event_kw"]]
        sizes = {row["event_kw"] for row in event}
        if len(sizes) != 1:
            raise ValueError("a single fixed-amplitude step is required")
        start, stop = event[0]["write_s"], event[-1]["end_s"]
        step, = sizes
        post_start, post_end = paired[mode]["window_s"]
        if not start <= post_start < post_end <= stop:
            raise ValueError("handoff comparison crosses an external step edge")
        metrics = trial["result"]["metrics"]
        tracking = {s: metrics["per_port"][s]["local_tracking_shortfall_kwh"] for s in ("A", "B")}
        results[mode] = {
            "raw_sha256": sha(path), "source_git": trial["header"]["git"],
            "step_kw": step, "step_interval_s": [start, stop],
            "system_whole_step_vs_no_control": interval_area(combined, start, stop, step),
            "system_post_window_vs_no_control": interval_area(combined, post_start, post_end, step),
            "bridge_window_s": [post_start, post_end],
            "bridge_target_kw": 8.,
            "bridge_window_shortfall_kwh": paired[mode]["shortfall_kwh"],
            "local_request_tracking_shortfall_kwh": tracking,
            "event_secondary_dip": metrics["second_dip"],
            "event_recovery_s": metrics["recovery_s"],
            "release_positive_rebound_hz": metrics["positive_rebound_hz"],
            "reference_hz": hz["reference_hz"], "full_event_metrics": metrics
        }
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("neutral", "issued", "delivered", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    dump(args.out, score(args.neutral, args.issued, args.delivered))
