# Benchmarks

This folder compares cppbone with the original Python NetBone package. For every backbone method,
`compare_python.py` does three things:

1. It runs the method in both implementations on the same network.
2. It matches the results edge by edge (or node by node for node methods) and compares the backbone property
   (`p_value`, `salience`, …) and the `in_backbone` flag.
3. It times both implementations.

## Contents

| Path | What it is |
|---|---|
| `compare_python.py` | The cross-check and benchmark script |
| `requirements.txt` | The Python dependencies the published numbers were measured with |

The reference is the [`netbone`](https://pypi.org/project/netbone/) package from PyPI, version 0.2.3 (the latest
release). The script prints a warning when a different version is installed.

## Running

From the repository root:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r benchmarks/requirements.txt
pip install --no-deps netbone==0.2.3

cmake -S . -B build-py -DCPPBONE_PYTHON=ON -DPython_EXECUTABLE=$(which python)
cmake --build build-py -j

PYTHONPATH=build-py python benchmarks/compare_python.py               # all methods on data/data.csv
PYTHONPATH=build-py python benchmarks/compare_python.py pmfg glanb    # only some methods
PYTHONPATH=build-py python benchmarks/compare_python.py --csv my.csv  # another edge list (source,target,weight)
PYTHONPATH=build-py python benchmarks/compare_python.py --check       # exit 1 on disagreement (CI)
```

`--reps N` sets how many times each C++ method runs; the best time is reported. Each Python method runs once,
because the slowest ones (pmfg) take more than 30 s.

netbone is installed with `--no-deps` because its `setup.py` pins exact old versions (numpy 1.26, jax 0.4.23,
Cython 0.29, …) that do not install on current Python. `requirements.txt` replaces them with newer versions; `pip
check` will report the mismatches, which is expected. The one dependency kept at the old version is networkx 2.8.8:
`modularity_backbone` calls `nx.to_scipy_sparse_matrix`, and networkx 3 removed it.

## Results

Network: `data/data.csv`, the NetBone sample network (380 nodes, 9,678 edges).
Setup: Apple Silicon Mac, Python 3.15, netbone 0.2.3 with the versions in `requirements.txt`, C++ Release build
(Apple clang).
C++ times are the best of 3 calls through the Python module, including its copy of the input graph. Python times
are a single call.

| Method | Python (s) | C++ (s) | Speedup | Agreement with Python |
|---|---:|---:|---:|---|
| disparity | 0.083 | 0.0017 | 50× | identical |
| lans | 0.216 | 0.0147 | 15× | identical |
| noise_corrected | 0.273 | 0.0080 | 34× | p-values within 2.3e-9 (relative) |
| marginal_likelihood | 0.204 | 0.0031 | 66× | p-values within 2.7e-9 (relative) |
| global_threshold | 0.007 | 0.0005 | 14× | identical |
| degree | 0.008 | 0.0010 | 8× | identical |
| gspar | 0.234 | 0.0240 | 10× | identical |
| plam | 0.012 | 0.0013 | 9× | identical |
| mlam | 0.056 | 0.0022 | 26× | identical |
| maximum_spanning_tree | 0.020 | 0.0030 | 7× | identical |
| doubly_stochastic | 3.213 | 0.0410 | 78× | scores within 5.9e-15; same backbone |
| high_salience_skeleton | 1.119 | 0.0312 | 36× | identical |
| metric_distance_backbone | 0.375 | 0.0159 | 24× | identical |
| ultrametric_distance_backbone | 0.327 | 0.0153 | 21× | identical |
| betweenness | 1.570 | 0.0469 | 33× | within 4.2e-16 |
| h_backbone | 1.684 | 0.0471 | 36× | identical |
| glanb (c = 1) | 1.468 | 0.0342 | 43× | identical |
| modularity_backbone | 0.072 | 0.0034 | 22× | vitality within 3.5e-3 (different random partition) |
| pmfg | 32.541 | 0.3365 | 97× | 26 edges differ; identical when Python sorts stably |
| ecm | 5.348 | 2.2025 | 2× | p-values within 7.3e-6 (relative) |

Timings vary by machine and by run, especially below a millisecond. The speedups are what to compare.

## What `--check` accepts

A method passes when its property matches Python to a relative 1e-8 and its `in_backbone` flags are identical.
The exceptions are the documented differences (see "Differences from Python" in the top-level README):

| Method | Allowed | Why |
|---|---|---|
| `ecm` | relative 1e-4 | Different solver (Levenberg–Marquardt instead of trust-krylov), same optimum |
| `modularity_backbone` | any value | Louvain uses a different random generator, so the partition can differ |
| `pmfg` | up to 26 `in_backbone` flags | pandas' unstable sort orders tied weights arbitrarily; with a stable sort Python matches exactly |

The row sets (edges or nodes) must always be identical.

`ecm` runs only when jax is installed; without jax, NetBone's import skips it.
