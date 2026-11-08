"""Online semantic caching: the cache is updated while queries arrive (panels (c)-(f)).

Every round a query x_t arrives. The algorithm first updates its cache, then either serves x_t
from the cache or sends it to the LLM, in which case it observes the LLM cost (with noise).
The regret of a round is the loss of the current cache over all queries seen so far minus that
of the cache Reverse Greedy picks with the true costs and frequencies of those queries.
"""
import math
import time

import numpy as np

from caching import KernelRidge, average_loss, distances, reverse_greedy


def stage_ended(now, past, horizon, m):
    """Low-switching rule: a stage ends once its count reaches 1 + sqrt(T * past / m)."""
    return now >= 1.0 + np.sqrt(horizon * past / max(m, 1))


class QueryArms:
    """Bookkeeping over the distinct queries seen so far (the "arms" of the discrete methods).

    A query keeps the embedding of its first arrival. Costs are the true costs of its arrivals.
    """

    def __init__(self):
        self.index, self.embeddings, self.arrivals, self.cost_sum = {}, [], [], []

    def arrive(self, query_id, x, cost):
        if query_id not in self.index:
            self.index[query_id] = len(self.embeddings)
            self.embeddings.append(np.array(x, dtype=float))
            self.arrivals.append(0)
            self.cost_sum.append(0.0)
        q = self.index[query_id]
        self.arrivals[q] += 1
        self.cost_sum[q] += cost
        return q

    def __len__(self):
        return len(self.embeddings)


class OnlinePolicy:
    """Base class. `cache` holds indices into `points()`; `switches` counts cache recomputations."""

    learns = False  # whether the policy sends queries to the LLM to learn their costs

    def __init__(self, k, horizon, phi):
        self.k, self.horizon, self.phi = k, horizon, phi
        self.cache = np.array([], dtype=int)
        self.switches = 0

    def cached_points(self):
        return self.points()[self.cache] if len(self.cache) else np.zeros((0, 1))


class LowSwitchingCLCB(OnlinePolicy):
    """Shared by CLCB-SC-LS-Cont and CLCB-SC-LS. Each arm (epsilon-net center or distinct
    query) has a lower confidence bound `lcb` on its cost; an arriving query is sent to the LLM
    when the bound of its arm is below the mismatch cost of the cache.

    The cache is only recomputed when a stage ends: the cost stage of an arm ends once its
    number of LLM observations in the stage reaches 1 + sqrt(T * N / m), where N counts the
    observations of earlier stages and m the number of arms, and the arrival stage ends once
    its number of arrivals reaches 1 + sqrt(T * N), with N the arrivals of earlier stages.
    """

    learns = True

    def __init__(self, k, horizon, phi):
        super().__init__(k, horizon, phi)
        self.lcb = np.zeros(0)
        self.cost_now, self.cost_past = np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        self.arrival_now, self.arrival_past = 0, 0

    def add_arm(self):
        self.lcb = np.append(self.lcb, 0.0)
        self.cost_now, self.cost_past = np.append(self.cost_now, 0), np.append(self.cost_past, 0)

    def end_stages(self, t):
        self.arrival_now += 1
        cost_stages = stage_ended(self.cost_now, self.cost_past, self.horizon, len(self.lcb))
        arrival_stage = stage_ended(self.arrival_now, self.arrival_past, self.horizon, 1)
        if cost_stages.any() or arrival_stage:
            self.cost_past[cost_stages] += self.cost_now[cost_stages]
            self.cost_now[cost_stages] = 0
            if arrival_stage:
                self.arrival_past += self.arrival_now
                self.arrival_now = 0
            self.recompute(t)
            self.switches += 1

    def wants_llm(self, mismatch):
        return self.lcb[self.arm] < mismatch


class CLCBContinuous(LowSwitchingCLCB):
    """CLCB-SC-LS-Cont. Arriving queries are grouped into an online epsilon-net: a query farther
    than epsilon from every center becomes a new center. The cost of every center is estimated by
    kernel ridge regression on all LLM observations, and Reverse Greedy runs on the centers with
    the lower confidence bounds of their costs and their empirical arrival frequencies.
    """

    def __init__(self, k, horizon, phi, epsilon, delta, ridge, B, R):
        super().__init__(k, horizon, phi)
        self.epsilon, self.delta, self.ridge, self.B, self.R = epsilon, delta, ridge, B, R
        self.centers = np.zeros((0, 1))
        self.arrivals = np.zeros(0, dtype=int)
        self.X_obs, self.y_obs = [], []

    def points(self):
        return self.centers

    def arrive(self, t, x, query_id, cost):
        x = np.array(x, dtype=float)
        d = distances(x[None, :], self.centers)[0] if len(self.centers) else np.zeros(0)
        if len(d) and d.min() <= self.epsilon:
            self.arm = int(np.argmin(d))
        else:  # x becomes a new center
            self.centers = np.vstack([self.centers, x]) if len(self.centers) else x[None, :]
            self.arrivals = np.append(self.arrivals, 0)
            self.add_arm()
            self.arm = len(self.centers) - 1
        self.arrivals[self.arm] += 1
        self.end_stages(t)

    def recompute(self, t):
        m = len(self.centers)
        if self.X_obs:
            model = KernelRidge(np.stack(self.X_obs), np.array(self.y_obs), self.ridge, normalize_inputs=True)
            c_hat, var = model.predict(self.centers)
        else:
            c_hat, var = np.zeros(m), np.ones(m)
        c_hat = np.maximum(0.0, c_hat)
        beta = self.B + self.R * math.sqrt(2.0 * (math.log(max(2, t + 1)) + math.log(1.0 / self.delta)))
        self.lcb = np.maximum(0.0, c_hat - beta * np.sqrt(var))
        weights = self.arrivals.astype(float) / max(1, self.arrivals.sum())
        D = distances(self.centers, self.centers, rescale=True)
        self.cache = reverse_greedy(weights, self.lcb, D, self.k, self.phi)

    def observe(self, x, cost):
        self.X_obs.append(np.array(x, dtype=float))
        self.y_obs.append(cost)
        self.cost_now[self.arm] += 1


class CLCBDiscrete(LowSwitchingCLCB):
    """CLCB-SC-LS: every distinct query is its own arm, with its empirical cost and arrival
    frequency; Reverse Greedy runs on the queries seen so far."""

    def __init__(self, k, horizon, phi, delta, R):
        super().__init__(k, horizon, phi)
        self.delta, self.R = delta, R
        self.arms = QueryArms()
        self.observations, self.observed_sum = np.zeros(0, dtype=int), np.zeros(0)

    def points(self):
        return np.stack(self.arms.embeddings)

    def arrive(self, t, x, query_id, cost):
        self.arm = self.arms.arrive(query_id, x, cost)
        if self.arm == len(self.lcb):  # first arrival of this query
            self.observations, self.observed_sum = np.append(self.observations, 0), np.append(self.observed_sum, 0.0)
            self.add_arm()
        self.end_stages(t)

    def recompute(self, t):
        m, rounds = len(self.arms), max(1, t + 1)
        n = self.observations.astype(float)
        seen = n > 0
        log_term = math.log(max(4.0 * m * rounds / self.delta, 2.0))
        c_hat = np.divide(self.observed_sum, n, out=np.zeros(m), where=seen)
        bonus = self.R * np.sqrt(2.0 * log_term / np.where(seen, n, 1.0))
        self.lcb = np.where(seen, np.maximum(c_hat - bonus, 0.0), 0.0)
        p_hat = np.array(self.arms.arrivals) / rounds
        D = distances(self.points(), self.points(), rescale=True)
        self.cache = reverse_greedy(p_hat, self.lcb, D, self.k, self.phi)

    def observe(self, x, cost):
        self.observations[self.arm] += 1
        self.observed_sum[self.arm] += cost
        self.cost_now[self.arm] += 1


class FrequencyCostGreedy(OnlinePolicy):
    """Greedy: every `every` rounds, cache the k queries with the highest score
    0.75 * frequency / max frequency + 0.25 * mean cost (ties: the query seen first). With
    probability `explore` (Epsilon-Greedy) the cache is instead k random queries seen so far."""

    def __init__(self, k, horizon, phi, every, explore=0.0, seed=0):
        super().__init__(k, horizon, phi)
        self.every, self.explore, self.seed = every, explore, seed
        self.arms = QueryArms()

    def points(self):
        return np.stack(self.arms.embeddings)

    def arrive(self, t, x, query_id, cost):
        self.arms.arrive(query_id, x, cost)
        if t % self.every:
            return
        freq = np.array(self.arms.arrivals, dtype=float)
        mean_cost = np.array(self.arms.cost_sum) / freq
        scores = 0.75 * (freq / max(np.max(freq), 1.0)) + 0.25 * mean_cost
        k = min(self.k, len(freq))
        rng = np.random.default_rng(self.seed + t)
        if self.explore > 0 and rng.random() < self.explore:
            self.cache = np.sort(rng.choice(len(freq), size=k, replace=False))
        else:
            self.cache = np.sort(np.argsort(-scores, kind="stable")[:k])
        self.switches += 1


class LFU(OnlinePolicy):
    """Every `every` rounds, cache the k most frequent queries seen so far (ties: the query seen first)."""

    def __init__(self, k, horizon, phi, every):
        super().__init__(k, horizon, phi)
        self.every = every
        self.arms = QueryArms()

    def points(self):
        return np.stack(self.arms.embeddings)

    def arrive(self, t, x, query_id, cost):
        self.arms.arrive(query_id, x, cost)
        if t % self.every == 0:
            freq = np.array(self.arms.arrivals, dtype=float)
            self.cache = np.sort(np.argsort(-freq, kind="stable")[: self.k])
            self.switches += 1


def optimal_losses(X, cost, k, phi):
    """For every round t, the loss over the first t queries of the cache Reverse Greedy picks
    from those queries, using their true costs and empirical frequencies."""
    losses = []
    for t in range(1, len(X) + 1):
        unique, first, counts = np.unique(X[:t], axis=0, return_index=True, return_counts=True)
        D = distances(unique, unique)
        best = reverse_greedy(counts / t, cost[:t][first], D, k, phi)
        losses.append(average_loss(X[:t], cost[:t], unique[best], phi))
    return np.array(losses)


def run_online(policy, X, cost, ids, optimum, noise, seed):
    """Run `policy` on the stream (X, cost, ids). Returns the regret of every round, the number of
    cache switches and the time the policy spent (excluding the regret evaluation)."""
    rng = np.random.RandomState(seed)
    regret, runtime = [], 0.0
    for t in range(len(X)):
        start = time.perf_counter()
        policy.arrive(t, X[t], ids[t], cost[t])
        if policy.learns:
            cached = policy.cached_points()
            if len(cached):
                mismatch = float(policy.phi(np.min(distances(X[t][None, :], cached, rescale=True))))
            else:
                mismatch = np.inf
            if policy.wants_llm(mismatch):
                policy.observe(X[t], float(max(0.0, rng.normal(loc=cost[t], scale=noise))))
        runtime += time.perf_counter() - start
        regret.append(average_loss(X[: t + 1], cost[: t + 1], policy.cached_points(), policy.phi) - optimum[t])
    return np.array(regret), policy.switches, runtime


# ---------------------------------------------------------------------------------------------
# Query streams
# ---------------------------------------------------------------------------------------------

def jittered_stream(universe, length, max_jitter, seed):
    """Uniformly random universe queries whose embeddings get Gaussian noise (std 0.05) clipped
    to +-max_jitter in every coordinate, so no query repeats exactly."""
    rng = np.random.default_rng(seed)
    m, dim = universe.X.shape
    ids = rng.integers(0, m, size=length)
    noise = rng.normal(0.0, 0.05, size=(length, dim))
    rng.random(length)  # unused draw, kept so seeds reproduce the paper's streams
    X = universe.X[ids] + np.clip(noise, -max_jitter, max_jitter)
    return X, universe.cost[ids], ids


def bursty_stream(universe, source, length, rng, burst=10):
    """Bursts of 5 to 19 queries from one dataset (NQ or TriviaQA, each with probability 1/2).
    Within a dataset, query popularities are log-normal."""
    pools = [np.flatnonzero(source == name) for name in ("NQ", "TriviaQA")]
    popularity = [rng.lognormal(0.0, 1.0, size=len(pool)) for pool in pools]
    probs = [p / p.sum() for p in popularity]
    ids = []
    while len(ids) < length:
        which = 0 if rng.random() < 0.5 else 1
        size = rng.integers(burst // 2, burst * 2)
        ids.extend(rng.choice(pools[which], size=size, replace=True, p=probs[which]).tolist())
    ids = np.array(ids[:length])
    return universe.X[ids], universe.cost[ids], ids
