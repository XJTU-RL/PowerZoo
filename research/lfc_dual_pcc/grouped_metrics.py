"""Score power, time and duration against witnessed local native service envelopes."""
from collections import Counter, defaultdict
import math
import statistics


def summary(pairs,rows,offset=0.):
    lookup={}
    groups=defaultdict(dict)
    for nominal,pred in zip(pairs,rows,strict=True):
        g,slot,site,h,r=nominal["key"]
        kw=max(0.,pred["power_kw"]-offset)
        time=pred["time_class"]
        item={"group":g,"slot":slot,"site":site,"h":h,"request":r,"kw":nominal["kw"],
              "pred_kw":kw,"actual_time":nominal["time_class"],"raw_time":time,
              "pred_time":time if kw+1e-8>=r else 0,"p_witness":pred["p_witness"]}
        lookup[(g,slot,site,h,r)]=item
        groups[(g,slot,site,r)][h]=item
    entries=list(lookup.values()); unique={}
    for a in entries:
        k=(a["group"],a["slot"],a["site"],a["h"])
        if k in unique and abs(unique[k]["pred_kw"]-a["pred_kw"])>1e-5:
            raise ValueError("capacity estimate changed with requested power")
        unique[k]=a
    diffs=[a["pred_kw"]-a["kw"] for a in unique.values()]
    arrival=[abs(a["pred_time"]-a["actual_time"]) for a in entries if a["pred_time"] and a["actual_time"]]
    over5=only5=over=miss=exact=0
    duration_abs=[]
    confusion=Counter()
    for v in groups.values():
        actual=max((h for h,a in v.items() if a["actual_time"]),default=0)
        claimed=0
        for h in (2,5,10):
            if not v[h]["pred_time"]:break
            claimed=h
        confusion[(actual,claimed)]+=1
        exact+=actual==claimed
        over+=claimed>actual
        miss+=claimed<actual
        only5+=actual==5;over5+=actual==5 and claimed==10
        duration_abs.append(abs(claimed-actual))
    mw=statistics.mean
    return {"unique_capacity_targets":len(unique),"requested_time_rows":len(entries),"duration_queries":len(groups),
        "power_mae_kw":mw(map(abs,diffs)),"power_bias_kw":mw(diffs),
        "overpromise_rate_beyond_resolution":mw(int(d>.1+1e-8) for d in diffs),
        "mean_positive_power_excess_kw":mw(max(0.,d) for d in diffs),
        "max_positive_power_excess_kw":max(0.,max(diffs)),
        "arrival_mae_s_on_mutual_claims":mw(arrival) if arrival else None,"mutual_time_count":len(arrival),
        "false_arrival_claims":sum(bool(a["pred_time"]) and not a["actual_time"] for a in entries),
        "missed_arrivals":sum(bool(a["actual_time"]) and not a["pred_time"] for a in entries),
        "time_class_accuracy_including_no_witness":mw(int(a["pred_time"]==a["actual_time"]) for a in entries),
        "duration_accuracy":exact/len(groups),"mean_duration_absolute_error_s":mw(duration_abs),
        "duration_overcommit_rate":over/len(groups),"duration_underclaim_rate":miss/len(groups),
        "actual_certified_5":only5,"certified_5_claimed_10":over5,
        "five_to_ten_rate":over5/only5 if only5 else None,
        "duration_confusion_count":{f"{a}->{b}":n for (a,b),n in sorted(confusion.items())},
        "raw_witness_brier":mw((a["p_witness"]-bool(a["actual_time"]))**2 for a in entries),
        "power_errors_kw":list(diffs)}


def calibration(pairs,rows,grid,goal):
    choices=[(x,summary(pairs,rows,x)["overpromise_rate_beyond_resolution"]) for x in grid]
    ok=[(a,b) for a,b in choices if b<=goal]
    selected=min(ok)[0] if ok else min(choices,key=lambda v:(v[1],v[0]))[0]
    errors=sorted(map(abs,summary(pairs,rows,selected)["power_errors_kw"]))
    return {"derate_kw":selected,"validation_grid":choices,
            "val_p90_abs_error_kw":errors[min(len(errors)-1,math.ceil(.9*len(errors))-1)]}
