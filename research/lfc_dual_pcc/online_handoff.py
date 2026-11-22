"""Causal admission of a conditional sustained offer and the common release guard."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SustainedOffer:
    sampled: float
    arrival: float
    effective: float
    expiry: float
    valid_through: float
    command: int
    state_key: str
    capacity_kw: float

    def capacity_at(self, now, deadline, command, state_key):
        """Return the witnessed cap for the remaining interval, or no usable offer."""
        scalars = (self.sampled, self.arrival, self.effective, self.expiry,
                   self.valid_through, self.capacity_kw, now, deadline)
        if (not all(math.isfinite(v) for v in scalars) or self.capacity_kw < 0.
                or not self.sampled <= min(self.arrival, self.effective)
                or not self.arrival < now or not self.effective <= now < self.expiry
                or not now < deadline <= min(self.valid_through, self.expiry)
                or self.command != command or not self.state_key or self.state_key != state_key):
            return None
        return self.capacity_kw


def limit_handoff(candidates_kw, request_kw, ceiling_kw):
    """Select from currently masked native candidates and report unavoidable excess."""
    if (not candidates_kw or not all(math.isfinite(v) for v in candidates_kw.values())
            or not 0. <= request_kw <= ceiling_kw < math.inf):
        raise ValueError("invalid delivery candidates or limits")
    under_ceiling = [index for index, kw in candidates_kw.items() if kw <= ceiling_kw]
    if under_ceiling:
        chosen = min(under_ceiling, key=lambda i: (abs(candidates_kw[i]-request_kw), candidates_kw[i], i))
    else:
        chosen = min(candidates_kw, key=lambda i: (candidates_kw[i], i))
    return chosen, max(0., candidates_kw[chosen]-ceiling_kw)
