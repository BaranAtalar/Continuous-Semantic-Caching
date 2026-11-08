"""Offline semantic caching: learn a cache from a logged stream of queries (panels (a) and (b)).

Logged queries are noisy copies of the universe queries, so they are never seen exactly again.
A logged query's LLM cost is only recorded (with noise) when it was not served from the cache.
Each method picks k universe queries; its suboptimality gap is the true loss of that cache,
with every universe query equally likely, minus the loss of a reference cache.
"""
import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from caching import KernelRidge, distances, reverse_greedy

MAX_CELLS = 50  # epsilon-nets with more cells are thinned out by farthest-point sampling


@dataclass
class LoggedData:
    X: np.ndarray         # embeddings of the logged queries
    observed: np.ndarray  # observed LLM cost, or 0 when the query was answered from the cache


def logged_stream(universe, length, rng, jitter=0.05, jitter_prob=0.6, noise=0.1):
    """Uniformly random universe queries; with probability jitter_prob the embedding gets
    Gaussian noise (std jitter). Half of the queries are cache misses, whose LLM cost is logged
    with Gaussian noise (std noise) and clipped to [0, 1]."""
    m, dim = universe.X.shape
    ids = rng.randint(0, m, size=length)
    perturbation = rng.normal(0.0, jitter, size=(length, dim))
    jittered = (rng.rand(length) < jitter_prob).astype(float)[:, None]
    X = universe.X[ids] + jittered * perturbation
    miss = rng.rand(length) < 0.5
    observed = np.where(miss, np.clip(universe.cost[ids] + rng.normal(0.0, noise, size=length), 0.0, 1.0), 0.0)
    return LoggedData(X, observed)


class Evaluator:
    """True loss of a cache of universe queries, with all universe queries equally likely."""

    def __init__(self, universe):
        self.universe = universe
        self.D = distances(universe.X, rescale=True)

    def loss(self, cache):
        nearest = np.min(self.D[:, cache], axis=1)
        return float(np.mean(np.minimum(self.universe.phi(nearest), self.universe.cost)))

    def optimal_loss(self, k, batch=20_000):
        """Loss of the best cache of size k, by brute force over all subsets."""
        best = np.inf
        subsets = combinations(range(len(self.D)), k)
        while True:
            chunk = np.array([s for _, s in zip(range(batch), subsets)])
            if not len(chunk):
                return best
            nearest = np.ascontiguousarray(self.D[:, chunk].min(axis=2).T)  # (subsets, queries)
            losses = np.mean(np.minimum(self.universe.phi(nearest), self.universe.cost), axis=1)
            best = min(best, float(losses.min()))

    def greedy_loss(self, k):
        """Loss of the cache built by adding, k times, the query that lowers the loss the most."""
        mismatch = self.universe.phi(self.D)
        nearest = np.full(len(self.D), np.inf)
        remaining = list(range(len(self.D)))
        for _ in range(k):
            losses = [np.mean(np.minimum(np.minimum(nearest, mismatch[:, j]), self.universe.cost)) for j in remaining]
            nearest = np.minimum(nearest, mismatch[:, remaining.pop(int(np.argmin(losses)))])
        return float(np.mean(np.minimum(nearest, self.universe.cost)))

    def nearest_universe(self, X):
        """Index of the closest universe query of every row of X."""
        return np.argmin(distances(X, self.universe.X, rescale=True), axis=1)


def epsilon_net(X, eps):
    """Greedy epsilon-net over the logged queries (distances rescaled to [0, 1]). Returns the
    indices of the centers and the cell of every query (its closest center among those that
    cover it)."""
    n = len(X)
    D = distances(X, rescale=True)
    covered = np.zeros(n, dtype=bool)
    centers, cell = [], -np.ones(n, dtype=int)
    for i in range(n):
        if covered[i]:
            continue
        centers.append(i)
        within = np.flatnonzero(D[i] <= eps)
        covered[within] = True
        for j in within:
            if cell[j] == -1 or D[j, i] < D[j, centers[cell[j]]]:
                cell[j] = len(centers) - 1
    return np.array(centers), cell


def farthest_point_sampling(X, size, rng):
    chosen = [rng.randint(0, len(X))]
    D = distances(X, X[chosen], rescale=True)[:, 0]
    while len(chosen) < size:
        chosen.append(int(np.argmax(D)))
        D = np.minimum(D, distances(X, X[[chosen[-1]]], rescale=True)[:, 0])
    return np.array(chosen)


def cucb_sc_cont(data, evaluator, k, eps, delta, ridge, beta_scale, B, rng):
    """CUCB-SC-Cont. Cover the logged queries with an epsilon-net, estimate the cost of each
    center by kernel ridge regression on the logged misses, add a pessimistic confidence bonus,
    and run Reverse Greedy on the centers weighted by the size of their cells. Each kept center
    is then represented by its closest universe query."""
    X, n = data.X, len(data.X)
    centers, cell = epsilon_net(X, eps)
    if len(centers) > MAX_CELLS:
        centers = centers[farthest_point_sampling(X[centers], MAX_CELLS, rng)]
        cell = np.argmin(distances(X, X[centers], rescale=True), axis=1)
    Xc, m = X[centers], len(centers)
    weights = np.bincount(cell, minlength=m).astype(float) / n

    miss = data.observed > 0
    if miss.any():
        c_hat, var = KernelRidge(X[miss], data.observed[miss], ridge, normalize_inputs=False).predict(Xc)
    else:
        c_hat, var = np.zeros(m), np.ones(m)
    beta = beta_scale * math.sqrt(max(1e-12, math.log(max(2 * m, 2) / delta)))
    ucb = np.clip(c_hat + (B + beta * np.sqrt(var)), 0.0, 1.0)

    kept = reverse_greedy(weights, ucb, distances(Xc, rescale=True), k, evaluator.universe.phi)
    return np.unique(evaluator.nearest_universe(Xc[kept]))


def cucb_sc(data, evaluator, k, delta, beta_scale):
    """CUCB-SC (discrete): map every logged query to its closest universe query, estimate each
    query's cost from its logged misses with an upper confidence bound (1 if none), and run
    Reverse Greedy on the queries seen, weighted by their frequencies."""
    nearest = evaluator.nearest_universe(data.X)
    queries, counts = np.unique(nearest, return_counts=True)
    m, n = len(queries), len(nearest)
    ucb = np.ones(m)
    log_term = math.log(max(4.0 * m * n / delta, 2.0))
    for i, q in enumerate(queries):
        misses = data.observed[(nearest == q) & (data.observed > 0)]
        if len(misses):
            bonus = beta_scale * math.sqrt(2.0 * log_term / len(misses))
            ucb[i] = np.clip(np.mean(misses) + 2 * bonus, 0.0, 1.0)
    D = distances(evaluator.universe.X[queries], rescale=True)
    return queries[reverse_greedy(counts / n, ucb, D, k, evaluator.universe.phi)]


def greedy(data, evaluator, k, explore, rng):
    """Greedy: cache the k universe queries with the highest 0.75 * frequency / max frequency
    + 0.25 * mean logged cost (0.5 if never logged), ties to the lowest index. Epsilon-Greedy
    (explore > 0) instead caches k random universe queries with probability explore."""
    m = len(evaluator.universe.X)
    nearest = evaluator.nearest_universe(data.X)
    mean_cost, observations = np.full(m, 0.5), np.zeros(m)
    for q, cost in zip(nearest, data.observed):
        if cost > 0:
            mean_cost[q] = (mean_cost[q] * observations[q] + cost) / (observations[q] + 1)
            observations[q] += 1
    freq = np.bincount(nearest, minlength=m)
    scores = 0.75 * (freq / max(np.max(freq), 1)) + 0.25 * np.clip(mean_cost, 0.0, 1.0)
    if rng.rand() < explore:
        return np.sort(rng.choice(m, size=k, replace=False))
    return np.sort(np.argsort(-scores, kind="stable")[:k])


def lfu(data, evaluator, k):
    """LFU: cache the k universe queries closest to the most logged queries (ties to the lowest index)."""
    freq = np.bincount(evaluator.nearest_universe(data.X), minlength=len(evaluator.universe.X))
    return np.sort(np.argsort(-freq, kind="stable")[:k])


def epsilon_grid(X, points=20):
    """20 values of epsilon from the 1st percentile of the rescaled pairwise distances of a pilot
    stream to 1.25 times their 95th percentile."""
    D = distances(X, rescale=True)
    pairs = D[np.triu_indices_from(D, k=1)]
    lo = max(float(np.quantile(pairs, 0.01)), 0.01)
    hi = max(float(np.quantile(pairs, 0.95)) * 1.25, lo + 0.1)
    return np.linspace(lo, hi, points)
