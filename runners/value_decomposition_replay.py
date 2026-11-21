"""Whole-episode retention and one-step sampling for the VVC value baselines."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayFrame:
    obs: tuple[np.ndarray, ...]
    state: np.ndarray
    available: tuple[np.ndarray, ...]

    @classmethod
    def from_worker(cls, obs, states, available):
        if not np.all(np.asarray(states) == states[0]):
            raise ValueError(
                "QMIX/VDN requires an identical shared state for all devices"
            )
        return cls(
            tuple(np.array(local, dtype=np.float32, copy=True) for local in obs),
            np.array(states[0], dtype=np.float32, copy=True),
            tuple(np.array(mask, copy=True) for mask in available),
        )


@dataclass(frozen=True)
class Transition:
    current: ReplayFrame
    actions: np.ndarray
    reward: float
    following: ReplayFrame
    terminated: bool


class EpisodeReplay:
    """Stage per worker and evict complete episodes from a bounded transition ring."""

    def __init__(self, capacity, workers):
        self.pending = [[] for _ in range(workers)]
        self._slots = [None] * capacity
        self._episode_sizes = deque()
        self._start = 0
        self.size = 0

    def append(self, worker, transition, *, ended):
        episode = self.pending[worker]
        episode.append(transition)
        if len(episode) > len(self._slots):
            raise ValueError("Replay capacity must fit a complete environment episode")
        if not ended:
            return

        while self.size + len(episode) > len(self._slots):
            evicted = self._episode_sizes.popleft()
            for index in range(evicted):
                self._slots[(self._start + index) % len(self._slots)] = None
            self._start = (self._start + evicted) % len(self._slots)
            self.size -= evicted
        for item in episode:
            self._slots[(self._start + self.size) % len(self._slots)] = item
            self.size += 1
        self._episode_sizes.append(len(episode))
        episode.clear()

    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return [
            self._slots[(self._start + index) % len(self._slots)] for index in indices
        ]
