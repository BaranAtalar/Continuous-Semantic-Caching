"""Building blocks shared by the offline and online algorithms.

A cache holds k representative queries. An arriving query x is answered with the response of
its closest cached query at mismatch cost phi(d(x, M)), or sent to the LLM at cost c(x),
whichever is cheaper. The loss of a cache M over a set of queries with weights w is

    L(M) = sum_x w(x) * min(phi(d(x, M)), c(x)),

where d is the Euclidean distance between L2-normalized sentence embeddings.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------------------------
# Query universes
# ---------------------------------------------------------------------------------------------

class Universe:
    """The distinct queries a stream is drawn from: embeddings X, LLM costs c and mismatch phi."""

    def __init__(self, X, cost, phi):
        self.X, self.cost, self.phi = X, cost, phi


def clipped_mismatch(d):
    """phi(d) = min(max(d, 0), 1); used with costs in [0, 1]."""
    return np.clip(d, 0.0, 1.0)


def scaled_mismatch(c_max):
    """phi(d) = c_max * max(d, 0), which puts mismatch and LLM costs on the same scale."""
    return lambda d: np.maximum(d, 0.0) * c_max


@lru_cache(maxsize=None)
def synthetic_universe():
    """The 50 queries in queries.txt. The LLM cost is the GPT-2 token count, min-max normalized."""
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    queries = ROOT.joinpath("queries.txt").read_text().splitlines()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokens = np.array([len(tokenizer.encode(q)) for q in queries], dtype=float)
    cost = (tokens - tokens.min()) / (tokens.max() - tokens.min() + 1e-12)
    X = SentenceTransformer("paraphrase-MiniLM-L6-v2").encode(queries, convert_to_numpy=True)
    return Universe(X, cost, clipped_mismatch)


@lru_cache(maxsize=None)
def nq_triviaqa_universe():
    """2,500 Natural Questions and 2,500 TriviaQA questions (data/nq_triviaqa.npz).

    The LLM cost is tokens^1.5 divided by its mean, so cheap and expensive queries differ more.
    Returns the universe and the dataset each query comes from.
    """
    data = np.load(ROOT / "data" / "nq_triviaqa.npz")
    stretched = data["tokens"].astype(float) ** 1.5
    cost = stretched / np.mean(stretched)
    return Universe(data["embeddings"], cost, scaled_mismatch(float(np.max(cost)))), data["source"]


# ---------------------------------------------------------------------------------------------
# Distances, loss and Reverse Greedy
# ---------------------------------------------------------------------------------------------

def normalize_rows(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def distances(X, Y=None, rescale=False):
    """Euclidean distances between the L2-normalized rows of X and of Y (default: X itself).

    With rescale=True the distances are divided by their maximum, which maps them to [0, 1].
    """
    Xn = normalize_rows(X)
    D = euclidean_distances(Xn, Xn if Y is None else normalize_rows(Y))
    if rescale:
        top = float(D.max()) if D.size else 1.0
        if top > 1e-12:
            D = D / top
    return D


def average_loss(X, cost, cached, phi):
    """Mean over the queries X of min(phi(distance to the closest cached embedding), cost)."""
    if len(cached) == 0:
        return float(np.mean(cost))
    mismatch = phi(np.min(distances(X, cached), axis=1))
    return float(np.mean(np.minimum(mismatch, cost)))


def reverse_greedy(weights, cost, D, k, phi):
    """Start from all m points and repeatedly drop the one whose removal increases the loss
    sum_i weights[i] * min(phi(min_{j in cache} D[i, j]), cost[i]) the least (ties: lowest index).

    Returns the sorted indices of the k points kept.
    """
    m = len(weights)
    if k >= m:
        return np.arange(m)
    keep = np.arange(m)
    rows = np.arange(m)
    for _ in range(m - k):
        # Distance of every point to the cache without keep[j]: its nearest cached point
        # unless that point is keep[j], in which case its second nearest.
        sub = D[:, keep]
        nearest = np.argmin(sub, axis=1)
        d1 = sub[rows, nearest]
        sub[rows, nearest] = np.inf
        d2 = sub.min(axis=1)
        without = np.where(nearest[None, :] == np.arange(len(keep))[:, None], d2, d1)
        losses = np.sum(weights * np.minimum(phi(without), cost), axis=1)
        keep = np.delete(keep, np.argmin(losses))
    return keep


# ---------------------------------------------------------------------------------------------
# Kernel ridge regression of the LLM cost
# ---------------------------------------------------------------------------------------------

def rbf_kernel(A, B, gamma):
    return np.exp(-gamma * euclidean_distances(A, B, squared=True))


class KernelRidge:
    """Kernel ridge regression with an RBF kernel and a GP-style posterior variance.

    The kernel width comes from the median heuristic, gamma = 1 / median^2 of the pairwise
    distances between the (L2-normalized) training points. With normalize_inputs=True the
    kernel is also evaluated on L2-normalized embeddings; otherwise on the raw embeddings.
    """

    def __init__(self, X, y, lam, normalize_inputs):
        self.normalize_inputs = normalize_inputs
        Xn = normalize_rows(np.asarray(X, dtype=float))
        D = euclidean_distances(Xn, Xn)
        pairs = D[np.triu_indices_from(D, k=1)]
        pairs = pairs[pairs > 0]
        self.gamma = 1.0 / (float(np.median(pairs)) ** 2 + 1e-12) if pairs.size else 1.0
        self.X = self._features(X)
        K = rbf_kernel(self.X, self.X, self.gamma) + lam * np.eye(len(self.X))
        self.alpha = np.linalg.solve(K, y)
        self.K_inv = np.linalg.inv(K)

    def _features(self, X):
        X = np.asarray(X, dtype=float)
        return normalize_rows(X) if self.normalize_inputs else X

    def predict(self, X):
        """Posterior mean and variance (clipped to [0, 1]) at the rows of X."""
        Kx = rbf_kernel(self.X, self._features(X), self.gamma)
        mean = Kx.T @ self.alpha
        var = np.clip(1.0 - np.sum(Kx * (self.K_inv @ Kx), axis=0), 0.0, 1.0)
        return mean, var
