"""Rule-anchored bounded residual under the registered small-service profile."""
from torch import nn

from .continuous_service import ContinuousServiceCoordinator, ContinuousServicePolicy, allocation_interval
from .revision_protocol import REVISED_KW, demand


def residual_interval(request, limits):
    total,lo,hi = allocation_interval(request,limits)
    center=min(hi,max(lo,total*.5))
    radius=min(center-lo,hi-center,request/4.)
    return total,center-radius,center+radius


class ResidualServicePolicy(ContinuousServicePolicy):
    def __init__(self):
        super().__init__()
        actor=self.heads["master"].actor.layers[-1]
        nn.init.zeros_(actor.weight)
        nn.init.zeros_(actor.bias)


class ResidualServiceCoordinator(ContinuousServiceCoordinator):
    demand_for=staticmethod(demand)
    deficit_scale_kw=REVISED_KW

    def policy_interval(self,request,limits):
        return residual_interval(request,limits)
