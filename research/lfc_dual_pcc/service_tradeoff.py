"""Registered 60 Hz service trade-off and an action head on witnessed splits."""
from itertools import product
import math

from torch import nn

from .clock_graph import MLP, PolicyValue
from .continuous_service import allocation_interval
from .equal_service import SERVICE_SLOTS, EqualServicePolicy
from .residual_service import ResidualServiceCoordinator
from .revision_protocol import REVISED_KW

# Use the equal split for ties and fallback. The other coordinates are the
# development probes, independent of either fitted policy.
PROBES = (.5, .125, .875)
GEOMETRY = (.625, 1.875)
TOLERANCE_KW = .05
PEAK_UNMET_KW = .40
UNMET_BUDGET_KWH = .00125
MIN_EXCESS_REDUCTION_KWH = TOLERANCE_KW/3600.
IAE_MARGIN_HZ_S = .00025
NADIR_MARGIN_HZ = .00002
HOLDOUT = dict(start=25, stop=36, kw=30.)
HOLDOUT_SEED = 4701


def service_terms(request, delivery):
    return max(0., request-delivery), max(0., delivery-request-TOLERANCE_KW)


def allowed_splits(total, lo, hi, reports, error, scale):
    witnessed = (
        math.isclose(total, REVISED_KW, abs_tol=1e-9)
        and all(math.isclose(a, b, rel_tol=0., abs_tol=1e-9) for a, b in zip((lo,hi), GEOMETRY))
        and all(report is not None for report in reports.values())
        and abs(error) <= 2*scale
    )
    return [True, witnessed, witnessed]


def summarize_frontier(precheck):
    """Separate instantaneous feasible offers from witnessed two-step splits."""
    if not precheck["passed"] or [w["slot"] for w in precheck["windows"]] != list(SERVICE_SLOTS):
        raise ValueError("incomplete development evidence")
    eligible = [True]*len(PROBES)
    score = [0.]*len(PROBES)
    windows = []
    for w in precheck["windows"]:
        if len(w["alternatives"])!=len(PROBES) or tuple(w["interval_kw"])!=GEOMETRY:
            raise ValueError("unregistered probe geometry")
        points = []
        for u in PROBES:
            trial = next(t for t in w["alternatives"] if t["latent"]==u)
            if not math.isclose(sum(trial["request_kw"].values()), REVISED_KW, abs_tol=1e-9):
                raise ValueError("native probe changed the total request")
            deliveries = [sum(sites.values()) for sites in trial["delivered_kw"]]
            penalties = [service_terms(REVISED_KW, y) for y in deliveries]
            action = len(points)
            peak = max(short for short, _ in penalties)
            cost = sum(short+over for short, over in penalties)/len(penalties)
            score[action] += cost
            eligible[action] &= peak <= PEAK_UNMET_KW
            points.append(dict(action=action, delivered_kw=deliveries, peak_unmet_kw=peak,
                               mean_unmet_kw=sum(s for s,_ in penalties)/len(penalties),
                               mean_excess_over_tolerance_kw=sum(o for _,o in penalties)/len(penalties),
                               mean_cost_kw=cost))
        for point in points:
            loss = point["mean_unmet_kw"], point["mean_excess_over_tolerance_kw"]
            point["pareto"] = not any(
                (p["mean_unmet_kw"] <= loss[0]+1e-5 and p["mean_excess_over_tolerance_kw"] <= loss[1]+1e-5)
                and (p["mean_unmet_kw"] < loss[0]-1e-5 or p["mean_excess_over_tolerance_kw"] < loss[1]-1e-5)
                for p in points
            )
        support = sorted(sum(x) for x in product(*w["native_one_step_support_kw"].values()))
        if len(support)==0:
            raise ValueError("missing native feasible candidates")
        windows.append(dict(slot=w["slot"], probes=points, native_one_step_combinations=len(support),
                            exact_within_tolerance=any(abs(v-REVISED_KW)<=TOLERANCE_KW for v in support),
                            closest_lower_kw=max((v for v in support if v<=REVISED_KW), default=None),
                            closest_upper_kw=min((v for v in support if v>=REVISED_KW), default=None)))
    available = [i for i in range(len(PROBES)) if eligible[i]]
    if not available:
        raise ValueError("no two-step probe meets the registered shortfall limit")
    # DSS repeatability is coarser than the left/right numerical differences.
    rule = min(available, key=lambda i: (round(score[i]/len(windows), 5), i))
    separates = sum(w["alternatives"][0]["actions"]!=w["alternatives"][1]["actions"]
                    and w["distinct_native_pcc"]>=2 for w in precheck["windows"])
    rule_separates = sum(abs(w["probes"][0]["delivered_kw"][0]-w["probes"][rule]["delivered_kw"][0])
                         > TOLERANCE_KW for w in windows)
    return dict(passed=separates>=4 and rule_separates>=4, rule_action=rule,
                rule_development_mean_cost_kw=[v/len(windows) for v in score],
                rule_bounded_peak_shortfall=eligible, windows=windows,
                scope="native one-step independent feasible masks; three restored two-step command probes")


def compare_risk(old, revised):
    a,b = old["metrics"],revised["metrics"]
    diff_iae = b["iae_hz_s"]-a["iae_hz_s"]
    nadir_loss = max(0.,a["nadir_hz"]-b["nadir_hz"])
    excess_change = b["excess_external_kwh"]-a["excess_external_kwh"]
    flags = dict(excess_reduction=excess_change <= -MIN_EXCESS_REDUCTION_KWH,
                 iae=diff_iae<=IAE_MARGIN_HZ_S, nadir=nadir_loss<=NADIR_MARGIN_HZ,
                 cumulative_shortfall=b["unmet_external_kwh"]<=UNMET_BUDGET_KWH,
                 peak_shortfall=b["peak_unmet_external_kw"]<=PEAK_UNMET_KW)
    return dict(passed=all(flags.values()),checks=flags,iae_increase_hz_s=diff_iae,
                nadir_loss_hz=nadir_loss,excess_change_kwh=excess_change)


class ProbeMaster(PolicyValue):
    def __init__(self):
        super().__init__(MLP(37,len(PROBES)), MLP(37,1))
        nn.init.zeros_(self.actor.layers[-1].weight)
        nn.init.zeros_(self.actor.layers[-1].bias)

    def evaluate_sample(self, sample):
        if int(sample["mask"].sum())==1:
            return None, self.critic(*sample["args"]).squeeze()
        return super().evaluate_sample(sample)


class TradeoffPolicy(EqualServicePolicy):
    def __init__(self):
        super().__init__()
        self.heads["master"] = ProbeMaster()


class TradeoffCoordinator(ResidualServiceCoordinator):
    def __init__(self, model, cfg, greedy, *, rule_action=None):
        super().__init__(model, cfg, greedy)
        self.rule_action = rule_action
        self.penalize_excess = False

    def service_penalty_kw(self, requested, delivered):
        unmet, excess = service_terms(requested, delivered)
        return unmet+excess if self.penalize_excess else unmet

    def master_allocation(self, error, reports, limits, slot, now, freq_time, sequences):
        request = self.demand_for(slot)
        total, lo, hi = self.policy_interval(request, limits)
        x, confidence = self.master_observation(error,reports,now,request,total,lo,hi)
        mask = allowed_splits(total,lo,hi,reports,error,self.cfg["frequency_scale_hz"])
        head = self.model.heads["master"]
        if self.rule_action is None:
            action, record = head.decision((x,), mask, self.greedy)
        else:
            action = self.rule_action if mask[self.rule_action] else 0
            _, record = head.decision((x,), [i==action for i in range(len(mask))], True)
        a = lo+PROBES[action]*(hi-lo)
        if not mask[1]:
            # Geometry or observation moved outside the frozen development support.
            _, safe_lo, safe_hi = allocation_interval(request, limits)
            a = min(safe_hi, max(safe_lo,total*.5))
        self.records.append(dict(role="master",slot=slot,command=slot,outputs=record))
        self.emit(dict(phase="policy", role="master", slot=slot, command=slot, consumed_s=now,
                       frequency_sample_s=freq_time, reports=sequences, one_step_freshness=confidence,
                       action=action, latent_u=PROBES[action] if mask[1] else None, mask=mask,
                       behavior_logp=float(record["old_logp"]), x=x.tolist(),
                       feasible_a_kw=[lo,hi], allocatable_kw=total, requested_kw=request,
                       capacity_shortfall_kw=request-total, candidate_a_kw=a))
        return dict(A=a, B=min(limits["B"],max(0.,total-a))), request
