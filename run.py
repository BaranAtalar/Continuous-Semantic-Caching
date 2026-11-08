"""Run the paper's experiments, save the results to results/<experiment>.json and draw the figure.

    python run.py               # every experiment, then results/figure.pdf
    python run.py online_real   # only some experiments
"""
import argparse
import json
from pathlib import Path

import numpy as np

from caching import nq_triviaqa_universe, synthetic_universe
from offline import Evaluator, cucb_sc, cucb_sc_cont, epsilon_grid, greedy, logged_stream, lfu
from online import (CLCBContinuous, CLCBDiscrete, FrequencyCostGreedy, LFU, bursty_stream, jittered_stream,
                    optimal_losses, run_online)

K = 5  # cache size

# Offline, panel (a): suboptimality gap vs length of the logged stream
STREAM_LENGTHS = list(range(100, 2501, 100))
LENGTH_SETTINGS = dict(eps=0.4, delta=0.8, ridge=0.1, beta_scale=0.1, B=0.0)
# Offline, panel (b): suboptimality gap vs epsilon-net radius, with logged streams of 1000 queries
EPSILON_SETTINGS = dict(length=1000, delta=0.95, ridge=0.1, beta_scale=0.1, B=0.0)
EXPLORE = 0.4  # Epsilon-Greedy's exploration probability (offline)

# Online, panels (c)-(f): horizon T = 300
T = 300
SYNTHETIC = dict(epsilon=0.3, delta=0.01, ridge=1e-2, B=1.0, R=0.25, R_discrete=0.25, every=25, noise=0.1)
REAL = dict(epsilon=0.5, delta=0.01, ridge=0.1, B=1.0, R=0.25, R_discrete=0.35, every=10, noise=0.0)
ONLINE_EXPLORE = 0.3  # Epsilon-Greedy's exploration probability (online)


def stream_length(runs):
    """Panel (a). The reference cache is built greedily with the true costs."""
    universe = synthetic_universe()
    evaluator = Evaluator(universe)
    reference = evaluator.greedy_loss(K)
    s = LENGTH_SETTINGS
    gaps = {name: [] for name in ("CUCB-SC-Cont", "CUCB-SC", "Epsilon-Greedy", "LFU", "Greedy")}
    for length in STREAM_LENGTHS:
        for name in gaps:
            gaps[name].append([])
        for seed in range(runs):
            rng = np.random.RandomState(seed)
            data = logged_stream(universe, length, rng)
            caches = {
                "CUCB-SC-Cont": cucb_sc_cont(data, evaluator, K, s["eps"], s["delta"], s["ridge"],
                                             s["beta_scale"], s["B"], rng),
                "CUCB-SC": cucb_sc(data, evaluator, K, s["delta"], s["beta_scale"]),
                "Epsilon-Greedy": greedy(data, evaluator, K, EXPLORE, rng),
                "LFU": lfu(data, evaluator, K),
                "Greedy": greedy(data, evaluator, K, 0.0, rng),
            }
            for name, cache in caches.items():
                gaps[name][-1].append(evaluator.loss(cache) - reference)
    return {"lengths": STREAM_LENGTHS, "gaps": gaps}


def epsilon(runs):
    """Panel (b). The reference cache is the best one (brute force over all caches of size k)."""
    universe = synthetic_universe()
    evaluator = Evaluator(universe)
    optimum = evaluator.optimal_loss(K)
    s = EPSILON_SETTINGS
    grid = epsilon_grid(logged_stream(universe, s["length"], np.random.RandomState(42)).X)
    names = ("CUCB-SC-Cont", "CUCB-SC", "Greedy", "Epsilon-Greedy", "LFU")
    gaps = {name: [[] for _ in grid] for name in names}
    for seed in range(runs):
        rng = np.random.RandomState(seed)
        data = logged_stream(universe, s["length"], rng)
        baselines = {
            "LFU": lfu(data, evaluator, K),
            "Epsilon-Greedy": greedy(data, evaluator, K, EXPLORE, rng),
            "Greedy": greedy(data, evaluator, K, 0.0, rng),
            "CUCB-SC": cucb_sc(data, evaluator, K, s["delta"], s["beta_scale"]),
        }
        for i, eps in enumerate(grid):
            cache = cucb_sc_cont(data, evaluator, K, eps, s["delta"], s["ridge"], s["beta_scale"], s["B"], rng)
            gaps["CUCB-SC-Cont"][i].append(evaluator.loss(cache) - optimum)
            for name, cache in baselines.items():  # baselines do not depend on epsilon
                gaps[name][i].append(evaluator.loss(cache) - optimum)
    return {"epsilons": grid.tolist(), "gaps": gaps}


def online_policies(universe, s, seed):
    phi = universe.phi
    return {
        "CLCB-SC-LS-Cont": CLCBContinuous(K, T, phi, s["epsilon"], s["delta"], s["ridge"], s["B"], s["R"]),
        "CLCB-SC-LS": CLCBDiscrete(K, T, phi, s["delta"], s["R_discrete"]),
        "Greedy": FrequencyCostGreedy(K, T, phi, s["every"], seed=seed),
        "Epsilon-Greedy": FrequencyCostGreedy(K, T, phi, s["every"], ONLINE_EXPLORE, seed=seed),
        "LFU": LFU(K, T, phi, s["every"]),
    }


def online(universe, settings, streams):
    """Run every online algorithm on each (stream, seed) pair."""
    results = {"average_regret": {}, "switches": {}, "runtime": {}}
    for (X, cost, ids), seed in streams:
        optimum = optimal_losses(X, cost, K, universe.phi)
        for name, policy in online_policies(universe, settings, seed).items():
            regret, switches, runtime = run_online(policy, X, cost, ids, optimum, settings["noise"], seed)
            results["average_regret"].setdefault(name, []).append((np.cumsum(regret) / np.arange(1, T + 1)).tolist())
            results["switches"].setdefault(name, []).append(switches)
            results["runtime"].setdefault(name, []).append(runtime)
    return results


def online_synthetic(runs):
    """Panels (c) and (d). Run r uses stream seed 123 + r and noise seed r."""
    universe = synthetic_universe()
    streams = [(jittered_stream(universe, T, SYNTHETIC["epsilon"] / 2, seed=123 + r), r) for r in range(runs)]
    return online(universe, SYNTHETIC, streams)


def online_real(runs):
    """Panels (e) and (f). Run r uses seed 123 + r."""
    universe, source = nq_triviaqa_universe()
    streams = [(bursty_stream(universe, source, T, np.random.default_rng(123 + r)), 123 + r) for r in range(runs)]
    return online(universe, REAL, streams)


# name: (function, number of runs in the paper)
EXPERIMENTS = {"stream_length": (stream_length, 10), "epsilon": (epsilon, 20),
               "online_synthetic": (online_synthetic, 5), "online_real": (online_real, 5)}


def summarize(name, results):
    if name in ("stream_length", "epsilon"):
        for method, gaps in results["gaps"].items():
            print(f"  {method:16s} mean gap {np.mean(gaps):.4f}")
    else:
        print(f"  {'':16s} {'final avg regret':>17s} {'switches':>9s} {'runtime (s)':>12s}")
        for method, curves in results["average_regret"].items():
            final = [curve[-1] for curve in curves]
            print(f"  {method:16s} {np.mean(final):9.4f} ± {np.std(final):.4f} "
                  f"{np.mean(results['switches'][method]):9.1f} {np.mean(results['runtime'][method]):12.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="*", metavar="experiment",
                        help=f"any of {', '.join(EXPERIMENTS)} (default: all)")
    parser.add_argument("--runs", type=int, help="number of runs (default: as in the paper)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()
    unknown = set(args.experiments) - set(EXPERIMENTS)
    if unknown:
        parser.error(f"unknown experiment(s): {', '.join(sorted(unknown))}")

    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.experiments or EXPERIMENTS:
        function, runs = EXPERIMENTS[name]
        print(f"Running {name}...", flush=True)
        results = function(args.runs or runs)
        (args.out / f"{name}.json").write_text(json.dumps(results))
        summarize(name, results)

    if all((args.out / f"{name}.json").exists() for name in EXPERIMENTS):
        from plot import make_figure
        print(f"Saved {make_figure(args.out)}")


if __name__ == "__main__":
    main()
