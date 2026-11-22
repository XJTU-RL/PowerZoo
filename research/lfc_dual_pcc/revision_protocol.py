"""Frozen development-window and new held-out protocol for the engineering revision."""
from .equal_service import SERVICE_SLOTS


REVISED_KW = 2.5
HOLDOUT_SEED = 3701
HOLDOUT = dict(start=24,stop=35,kw=29)
RESIDUAL_PROBES = (.125,.5,.875)
MIN_PHYSICAL_WINDOWS = 4


def demand(slot):
    return REVISED_KW if slot in SERVICE_SLOTS else 0.
