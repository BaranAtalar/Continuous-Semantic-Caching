# Continuous Semantic Caching for Low-Cost LLM Serving

Code for the experiments in [Continuous Semantic Caching for Low-Cost LLM Serving](https://arxiv.org/abs/2604.20021) (ACM MobiHoc 2026).

A semantic cache answers an incoming query with the stored response of a similar cached query, which costs much less than calling the LLM but incurs a mismatch cost. In practice queries live in a continuous embedding space and rarely repeat exactly, and both how often each kind of query arrives and how expensive it is to serve are unknown. The paper covers the query space with an ε-net, estimates LLM costs with kernel ridge regression, and learns which *k* queries to cache offline from logged data (CUCB-SC-Cont) and online while serving queries, with few cache switches (CLCB-SC-LS-Cont).

## Setup

```bash
pip install -r requirements.txt
```

The synthetic experiments download the `paraphrase-MiniLM-L6-v2` sentence embedding model and the GPT-2 tokenizer from Hugging Face on their first run. The Natural Questions + TriviaQA experiment uses the embeddings and token counts in `data/nq_triviaqa.npz`, so it needs no download.

## Reproducing the results

```bash
python run.py
```

This runs all four experiments with the paper's settings and random seeds, prints a summary of each, and saves the results to `results/<experiment>.json` and Figure 2 to `results/figure.pdf`. It takes about five minutes on a laptop CPU. To run only some experiments, name them, for example `python run.py online_real`. `--runs` changes the number of runs, and `python plot.py` redraws the figure from saved results.

| Experiment | Panels | What it measures |
|---|---|---|
| `stream_length` | (a) | Suboptimality gap of caches learned offline from logged streams of 100 to 2,500 queries (10 runs each) |
| `epsilon` | (b) | Suboptimality gap for 20 values of the ε-net radius, with logged streams of 1,000 queries (20 runs) |
| `online_synthetic` | (c), (d) | Average regret over *T* = 300 rounds, and runtime, on the synthetic queries (5 runs) |
| `online_real` | (e), (f) | The same on Natural Questions + TriviaQA (5 runs) |

## Setting

* **Synthetic queries.** The 50 queries in `queries.txt`, from one word to several sentences. Their LLM cost is the GPT-2 token count, min-max normalized to [0, 1], and the mismatch cost of answering with a cached query at distance *d* is min(*d*, 1).
* **Natural Questions + TriviaQA.** 2,500 questions from each dataset. The LLM cost is (token count)<sup>1.5</sup> divided by its mean, and the mismatch cost is *c*<sub>max</sub> · *d*, with *c*<sub>max</sub> the largest LLM cost.
* **Distance.** Euclidean distance between L2-normalized sentence embeddings.
* **Offline data.** Each logged query is a uniformly random synthetic query whose embedding gets Gaussian noise (standard deviation 0.05 per coordinate) with probability 0.6. Half of the logged queries are cache misses, for which the LLM cost is logged with Gaussian noise (standard deviation 0.1).
* **Online streams.** Synthetic: uniformly random queries with Gaussian embedding noise (standard deviation 0.05, clipped to ±ε/2), so no query repeats exactly; a query sent to the LLM reveals its cost with noise (standard deviation 0.1). Natural Questions + TriviaQA: bursts of 5 to 19 queries from one dataset, with log-normal query popularities within each dataset; costs are observed without noise.
* **Loss and regret.** A cache *M* answers each query *x* with the cheaper of its closest cached query and the LLM: loss(*M*) = Σ<sub>*x*</sub> *p*(*x*) · min(φ(*d*(*x*, *M*)), *c*(*x*)). Offline, the suboptimality gap is the loss of the learned cache over the 50 equally likely queries minus that of a reference cache: the best cache (brute force) in panel (b), and the cache built greedily with the true costs in panel (a). Online, the regret of a round is the loss of the current cache over the queries seen so far minus that of the cache Reverse Greedy picks from those queries with their true costs and frequencies; the figure shows its running average.
* **Parameters.** Cache size *k* = 5. All other settings are at the top of `run.py`:

| | (a) | (b) | (c), (d) | (e), (f) |
|---|---|---|---|---|
| ε-net radius | 0.4 | 20 values | 0.3 | 0.5 |
| Confidence parameter δ | 0.8 | 0.95 | 0.01 | 0.01 |
| Kernel ridge regularization | 0.1 | 0.1 | 0.01 | 0.1 |

## Algorithms

The cache is chosen with **Reverse Greedy**: starting from all candidates, repeatedly remove the one whose removal increases the estimated loss the least.

| Algorithm | Setting | How it picks the cache |
|---|---|---|
| CUCB-SC-Cont | offline | Covers the logged queries with an ε-net (at most 50 cells), estimates each center's cost by kernel ridge regression with a pessimistic confidence bonus, and runs Reverse Greedy on the centers weighted by their cell sizes. Each kept center is then represented by its closest query. |
| CUCB-SC | offline | Maps each logged query to its closest query, and runs Reverse Greedy with upper confidence bounds on the costs. |
| CLCB-SC-LS-Cont | online | Builds the ε-net while queries arrive and runs Reverse Greedy on the centers with lower confidence bounds on their kernel ridge regression cost estimates. A query is sent to the LLM when the cost bound of its center is below the mismatch cost. The cache is only recomputed when a center's cost stage or the arrival stage ends, which keeps cache switches low. |
| CLCB-SC-LS | online | The same with every distinct query as its own arm, without generalizing costs across similar queries. |
| Greedy | both | Caches the *k* queries with the highest 0.75 × (frequency / largest frequency) + 0.25 × mean cost. Online, the cache is updated every 25 rounds (synthetic) or 10 rounds (Natural Questions + TriviaQA). |
| Epsilon-Greedy | both | Like Greedy, but with probability 0.4 (offline) or 0.3 (online) the cache is *k* random queries. |
| LFU | both | Caches the *k* most frequent queries. |

## Code layout

| File | Contents |
|---|---|
| `caching.py` | query universes, distances, loss, Reverse Greedy, and kernel ridge regression |
| `offline.py` | logged data and the offline algorithms (panels (a) and (b)) |
| `online.py` | query streams and the online algorithms (panels (c) to (f)) |
| `run.py` | the experiments, with their settings at the top of the file |
| `plot.py` | Figure 2 |
| `queries.txt` | the 50 synthetic queries |
| `data/nq_triviaqa.npz` | embeddings, questions, dataset labels and GPT-2 token counts of the 5,000 Natural Questions and TriviaQA questions |

## Notes

* This code makes the same decisions as the original experiment code, which was checked run by run. On Natural Questions + TriviaQA it reproduces the paper's per-run results for every algorithm except LFU. Its runs can still differ slightly between machines, because a learning algorithm's decisions can depend on the last bits of floating-point results, which vary with the CPU (for example, with or without AVX-512).
* Ties between equally frequent queries used to be broken by NumPy's default sort, whose order depends on the CPU. LFU, Greedy and Epsilon-Greedy now give ties to the query seen first. This only noticeably changes LFU: its final average regret on Natural Questions + TriviaQA is about 0.083 instead of the paper's 0.075.
* Panel (a) averages 10 runs per stream length, as the original experiment code did.
* For Natural Questions, the token counts used in the paper's experiments (stored in `tokens`) were computed on the dataset records, which hold the question text followed by its list of words. They are about five times the token count of the question text alone (53.5 versus 10.7 on average), which makes Natural Questions queries more expensive than TriviaQA queries.
* The runtimes in panels (d) and (f) depend on the machine. Reverse Greedy is implemented much faster here than in the paper's code (it picks the same caches), so runtimes are lower than in the paper, but the order of the algorithms is the same.

## Citation

```bibtex
@misc{atalar2026continuoussemanticcachinglowcost,
      title={Continuous Semantic Caching for Low-Cost LLM Serving}, 
      author={Baran Atalar and Xutong Liu and Jinhang Zuo and Siwei Wang and Wei Chen and Carlee Joe-Wong},
      year={2026},
      eprint={2604.20021},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      note={arXiv preprint arXiv:2604.20021 (2026)}, 
}
```
