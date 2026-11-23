"""Bounded capacity commitments and conditional, discrete PCC execution policies.

All public policy inputs have leading axes (episode, site). The final axis of
the device inputs indexes the six native Case A actuator combinations.
"""
import numpy as np

from .case_a_capacity_constants import ACTIONS

FRACTIONS = np.linspace(0., 1., 5)
BIDS = np.array([.4, 1., 2.5])


def clear_capacity(request, limits, weights):
    """Capped proportional allocation, including scarcity and saturated stations."""
    request, limits, weights = (np.asarray(x, dtype=float) for x in (request, limits, weights))
    if (limits.ndim != 2 or not limits.shape[1] or weights.shape != limits.shape
            or request.shape != (limits.shape[0],)):
        raise ValueError("expected one demand per row and one weight per station")
    if (not all(np.isfinite(x).all() for x in (request, limits, weights))
            or (request < 0.).any() or (limits < 0.).any() or (weights <= 0.).any()):
        raise ValueError("finite nonnegative demand and caps and positive weights required")
    total = np.minimum(request, limits.sum(axis=1))
    left = total.copy()
    capacity = np.zeros_like(limits, dtype=float)
    for _ in range(limits.shape[1]):
        room = np.maximum(limits - capacity, 0.)
        # Ignore roundoff remnants of filled stations while retaining very small offers.
        active = weights * (room > np.minimum(1.e-12, limits * .5))
        share = left[:, None] * active / np.where(active.any(axis=1, keepdims=True), active.sum(axis=1, keepdims=True), 1.)
        capacity += np.minimum(share, room)
        left = np.maximum(total - capacity.sum(axis=1), 0.)
    if not np.allclose(capacity.sum(axis=1), total, atol=1.e-9):
        raise ArithmeticError("capacity clearing failed conservation")
    return capacity


def _choice(features, weights, prior, valid, rng):
    logits = np.where(valid, prior + features @ weights, -np.inf)
    if not np.isfinite(logits.max(axis=-1)).all():
        raise ValueError("empty feasible action set")
    probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    if rng is None:
        return logits.argmax(axis=-1), None
    cumulative = probabilities.cumsum(axis=-1)
    # Normalize the CDF at its last entry, including any trailing masked actions.
    cumulative /= cumulative[..., -1:]
    draw = rng.random(logits.shape[:-1])
    if not ((draw >= 0.) & (draw < 1.)).all():
        raise ValueError("action samples must lie in [0, 1)")
    sample = (cumulative <= draw[..., None]).sum(axis=-1)
    selected = np.take_along_axis(features, sample[..., None, None], axis=-2).squeeze(axis=-2)
    score_gradient = selected - (probabilities[..., None] * features).sum(axis=-2)
    return sample, score_gradient


class CapacityPolicy:
    """Linear softmax policies, score functions and frozen same-information rule."""

    def __init__(self, n_sites, weights=None):
        if n_sites < 2:
            raise ValueError("at least two sites are required")
        self.n_sites = n_sites
        self.per_site_credit = False
        self.weights = {"upper": np.zeros(5), "lower": np.zeros(6)}
        if weights is not None:
            for key, value in self.weights.items():
                saved = np.asarray(weights[key], dtype=float)
                if saved.shape != value.shape or not np.isfinite(saved).all():
                    raise ValueError(f"invalid {key} capacity policy weights")
                value[:] = saved

    def slow(self, request, energy, ramp, previous, limits, *, rule=False, rng=None, force=None):
        """At window opening, commit available capacity until the next slow decision."""
        if force is not None and (rng is not None or rule):
            raise ValueError("capacity interventions require deterministic learned inference")
        if (request < 0).any() or (limits < 0).any():
            raise ValueError("upward requests and offers must be nonnegative")
        total = np.minimum(request, limits.sum(axis=1))
        # Each window lasts 2 seconds. The PV increment is renewable at each step.
        sustainable = np.minimum(limits, 2.135 + .95 * energy / 2.)
        importance = np.maximum(sustainable, .001)
        heuristic = clear_capacity(total, limits, importance)
        if rule:
            return heuristic, None

        if self.n_sites == 2:
            lo = np.maximum(0., total - limits[:, 1])
            hi = np.minimum(total, limits[:, 0])
            first = lo[:, None] + (hi - lo)[:, None] * FRACTIONS
            pairs = np.stack([first, total[:, None] - first], axis=-1)
            # Exchangeable features aggregate station-local consequences of a joint split.
            share = pairs / 10.
            headroom = energy[:, None, :] / 48.
            up = ramp[:, None, :] / 10.
            delivered = previous[:, None, :] / 10.
            x = np.stack([(share**2).sum(-1), (share * headroom).sum(-1),
                          (share * up).sum(-1), (share * delivered).sum(-1),
                          np.maximum(pairs - sustainable[:, None, :], 0.).sum(-1) / 10.], axis=-1)
            prior = -np.abs(first - heuristic[:, :1]) / 2.
            action, grad = _choice(x, self.weights["upper"], prior, np.ones_like(prior, bool), rng)
            if force is not None:
                action = np.broadcast_to(force, action.shape)
            return pairs[np.arange(len(action)), action], grad

        # Each agent independently samples a bounded bid multiplier. No actor reads
        # another station's energy, action or bid. Only the clearing uses all bids.
        z = np.log(BIDS)
        inputs = np.stack([np.ones_like(energy), energy / 48., ramp / 10.,
                           previous / 10., limits / 10.], axis=-1)
        x = inputs[..., None, :] * z[:, None]
        prior = np.broadcast_to(-.35 * z**2, x.shape[:-1])
        bids, grad = _choice(x, self.weights["upper"], prior, np.ones_like(prior, bool), rng)
        if force is not None:
            # A full intervention can hold the other stations' simultaneous bids fixed.
            bids = np.broadcast_to(force, bids.shape)
        capacity = clear_capacity(total, limits, importance * BIDS[bids])
        return capacity, grad if (grad is None or self.per_site_credit) else grad.sum(axis=1)

    def fast(self, target, capacity, energy, ramp, previous, valid, delivery, *, rule=False, rng=None):
        """Each station acts from its own target, measured state and admissible actions."""
        delivered = np.broadcast_to(delivery, valid.shape)
        error = (delivered - target[..., None]) / 4.
        usage = np.broadcast_to(ACTIONS[:, 0] / 8., error.shape)
        x = np.stack([np.maximum(error, 0.), np.maximum(-error, 0.), usage,
                      usage / (1. + energy[..., None] / 8.),
                      np.abs(delivered - previous[..., None]) / (1. + ramp[..., None]),
                      (delivered / 10.) * (capacity[..., None] / 10.)], axis=-1)
        prior = -2.8 * np.abs(error)
        if rule:
            if not valid.any(axis=-1).all():
                raise ValueError("empty feasible action set")
            return np.where(valid, prior, -np.inf).argmax(-1), None
        action, grad = _choice(x, self.weights["lower"], prior, valid, rng)
        return action, grad if (grad is None or self.per_site_credit) else grad.sum(axis=1)
