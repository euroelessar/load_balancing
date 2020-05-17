from __future__ import annotations

import heapq
import random

from argparse import ArgumentParser
from typing import Protocol, Sequence

from dataclasses import dataclass
from tabulate import tabulate


@dataclass
class Stats(object):
    duration: float
    max_inflight_requests: int


@dataclass
class Backend(object):
    latency: float
    inflight_requests: int = 0

    def __lt__(self, other: Backend):
        return False


class LoadBalancer(Protocol):
    def pick(self) -> Backend:
        pass


class RandomBalancer(object):
    def __init__(self, backends: Sequence[Backend], rand: random.Random) -> None:
        self._backends = backends
        self._rand = rand

    def pick(self) -> Backend:
        return self._rand.choice(self._backends)


class RoundRobin(object):
    def __init__(self, backends: Sequence[Backend]) -> None:
        self._backends = backends
        self._index = 0

    def pick(self) -> Backend:
        self._index = (self._index + 1) % len(self._backends)
        return self._backends[self._index]


class LeastRequests(object):
    def __init__(self, backends: Sequence[Backend]) -> None:
        self._backends = backends

    def attempt_backends(self) -> Sequence[Backend]:
        return self._backends

    def pick(self) -> Backend:
        return min(
            self.attempt_backends(), key=lambda backend: backend.inflight_requests
        )


class LeastRequestsIndependent(LeastRequests):
    def __init__(
        self, backends: Sequence[Backend], rand: random.Random, choices: int
    ) -> None:
        super().__init__(backends)
        self._rand = rand
        self._choices = min(len(backends), choices)

    def attempt_backends(self) -> Sequence[Backend]:
        return self._rand.choices(self._backends, k=self._choices)


class LeastRequestsUnique(LeastRequests):
    def __init__(
        self, backends: Sequence[Backend], rand: random.Random, choices: int
    ) -> None:
        super().__init__(backends)
        self._rand = rand
        self._choices = min(len(backends), choices)

    def attempt_backends(self) -> Sequence[Backend]:
        return self._rand.sample(self._backends, k=self._choices)


LOAD_BALANCERS = {
    "random": lambda backends, rand: RandomBalancer(backends, rand),
    "round_robin": lambda backends, rand: RoundRobin(backends),
    "least_requests_full_scan": lambda backends, rand: LeastRequests(backends),
    "least_requests_independent_2pc": lambda backends, rand: LeastRequestsIndependent(
        backends, rand, choices=2
    ),
    "least_requests_unique_2pc": lambda backends, rand: LeastRequestsUnique(
        backends, rand, choices=2
    ),
    "least_requests_unique_4pc": lambda backends, rand: LeastRequestsUnique(
        backends, rand, choices=4
    ),
    "least_requests_independent_4pc": lambda backends, rand: LeastRequestsIndependent(
        backends, rand, choices=4
    ),
    "least_requests_1_random": lambda backends, rand: LeastRequestsIndependent(
        backends, rand, choices=1
    ),
}


def simulate(
    *, lb_name: str, backends_arg: str, num_requests: int, concurrency: int
) -> Stats:
    rand = random.Random(x=123)
    if "," in backends_arg:
        backends = [
            Backend(latency=float(latency_str))
            for latency_str in backends_arg.split(",")
        ]
    else:
        backends = [
            Backend(latency=rand.lognormvariate(mu=1.0, sigma=1.5))
            for _ in range(int(backends_arg))
        ]
    balancer_rand = random.Random(x=321)
    balancer = LOAD_BALANCERS[lb_name](backends, balancer_rand)

    inflight_requests = []
    max_inflight_requests = 0
    current_time = 0.0
    for _ in range(num_requests):
        assert len(inflight_requests) <= concurrency
        if len(inflight_requests) == concurrency:
            current_time, backend = heapq.heappop(inflight_requests)
            backend.inflight_requests -= 1
        backend = balancer.pick()
        backend.inflight_requests += 1
        max_inflight_requests = max(max_inflight_requests, backend.inflight_requests)
        heapq.heappush(inflight_requests, (current_time + backend.latency, backend))

    for end_time, _ in inflight_requests:
        current_time = max(current_time, end_time)

    return Stats(duration=current_time, max_inflight_requests=max_inflight_requests)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--backends", default="10")
    parser.add_argument("--requests", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()

    rows = []
    for lb_name in LOAD_BALANCERS.keys():
        stats = simulate(
            lb_name=lb_name,
            backends_arg=args.backends,
            num_requests=args.requests,
            concurrency=args.concurrency,
        )
        rows.append([lb_name, stats.duration, stats.max_inflight_requests])

    rows = sorted(rows, key=lambda row: (row[1], row[0], row))
    print(tabulate(rows, headers=["Name", "Duration", "Max Inflight Requests"]))


if __name__ == "__main__":
    main()
