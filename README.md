# cppbone

cppbone is a C++17 port of [NetBone](https://gitlab.liris.cnrs.fr/coregraphie/netbone), the Python library for
extracting backbones from simple weighted networks. It has every NetBone backbone method, the three filters and
the `Compare` tools. The core library needs only the standard library. An optional pybind11 module exposes the same
API to Python.

On the sample network the port gives the same results as the Python package. It is 7–97× faster on all but one
method (see [Benchmark](#benchmark)).

## Features

- 20 backbone methods: 5 statistical, 14 structural and 1 hybrid (listed below)
- The NetBone filters: `boolean_filter`, `threshold_filter`, `fraction_filter`
- `Compare` for tables of backbone properties, progressions and KS statistics, plus the `measures.py` properties
- The `distanceclosure` toolkit: metric/ultrametric backbones, distance closure, s/b-values, pairwise proximity
- Optional Python module `netbone_cpp` that accepts networkx graphs or pandas edge lists
- No runtime dependencies

## Methods

| Family | Methods |
|---|---|
| Statistical | `disparity`, `lans`, `noise_corrected`, `marginal_likelihood`, `ecm` |
| Structural | `betweenness`, `degree`, `doubly_stochastic`, `global_threshold`, `gspar`, `h_backbone`, `high_salience_skeleton`, `maximum_spanning_tree`, `metric_distance_backbone`, `ultrametric_distance_backbone`, `mlam`, `plam`, `modularity_backbone`, `pmfg` |
| Hybrid | `glanb` |

## Build

Requires CMake ≥ 3.16 and a C++17 compiler.

```sh
cmake -S . -B build
cmake --build build -j
ctest --test-dir build                         # self-checks + smoke test of every method on data/data.csv
./build/netbone_example data/data.csv          # the usage example below
```

To use it from another CMake project, add it with `add_subdirectory(cppbone)` and link against the `netbone` target.

## Usage (C++)

```cpp
#include "netbone/netbone.hpp"
namespace nb = netbone;

nb::Graph g = nb::Graph::read_csv("edges.csv");          // columns: source,target,weight[,...]
nb::Backbone b = nb::high_salience_skeleton(g);

nb::Graph backbone1 = nb::boolean_filter(b);             // default threshold
nb::Graph backbone2 = nb::threshold_filter(b, 0.7);      // threshold
nb::Graph backbone3 = nb::fraction_filter(b, 0.15);      // keep 15% of the edges
b.to_csv(std::cout);                                     // Backbone.to_dataframe()
```

`nb::Graph` stands in for both `networkx.Graph` and the pandas edge list. Node labels are strings and attributes
are numeric: booleans are stored as 1/0 and a missing attribute reads as NaN. Nodes and edges iterate in networkx
order, so sorting and tie-breaking match the Python package.

`Graph::read_csv` splits on plain commas and does not handle quoted fields, so labels must not contain commas.

## Python module

The distribution is called `cppbone` and the module it installs is `netbone_cpp`. It needs Python ≥ 3.10.

```sh
pip install .                                  # builds the module with scikit-build-core + CMake
pip install ".[interop]"                       # also networkx and pandas, for conversions
```

Once a release is published, `pip install cppbone` installs prebuilt wheels (see [CI](#continuous-integration)).

For development, the CMake option `NETBONE_PYTHON=ON` builds the module in place. CMake uses an installed pybind11
3 if it finds one; otherwise it downloads pybind11 v3.0.1 (pinned by hash). The option is off by default, so the
plain C++ build downloads nothing.

```sh
cmake -S . -B build-py -DNETBONE_PYTHON=ON -DPython_EXECUTABLE=$(which python3)
cmake --build build-py -j
ctest --test-dir build-py                      # also runs tests/test_python.py
PYTHONPATH=build-py python3 -c "import netbone_cpp"
```

The API mirrors the Python package. Methods take a `netbone_cpp.Graph`, a networkx graph or a pandas edge-list
DataFrame:

```python
import networkx as nx
import netbone_cpp as nb

g = nx.les_miserables_graph()
b = nb.high_salience_skeleton(g)
backbone = nb.fraction_filter(b, 0.15)         # netbone_cpp.Graph
backbone.to_networkx(); b.to_dataframe()       # back to networkx / pandas

cmp = nb.Compare()
cmp.set_network(g); cmp.add_backbone(b)
cmp.set_filter(nb.fraction_filter, [0.15])
cmp.add_property("Density", nb.density)        # or any Python callable (original, graph) -> float
cmp.properties()                               # pandas DataFrame
```

When converting, node labels become strings, only numeric attributes are kept, and booleans such as `in_backbone`
come back as 1.0/0.0. networkx and pandas are imported only when you convert to or from them.

## Benchmark

The [`benchmarks/`](benchmarks/README.md) folder runs every method in cppbone and in the original Python NetBone
package (`netbone` 0.2.3 from PyPI). It compares the results edge by edge and times both. Its README has the full
table, setup and instructions.

On the NetBone sample network (380 nodes, 9,678 edges), with C++ called through the Python module:

| Method | Python (s) | C++ (s) | Speedup |
|---|---:|---:|---:|
| pmfg | 32.541 | 0.3365 | 97× |
| doubly_stochastic | 3.213 | 0.0410 | 78× |
| marginal_likelihood | 0.204 | 0.0031 | 66× |
| disparity | 0.083 | 0.0017 | 50× |
| glanb | 1.468 | 0.0342 | 43× |
| high_salience_skeleton | 1.119 | 0.0312 | 36× |
| betweenness | 1.570 | 0.0469 | 33× |
| ecm | 5.348 | 2.2025 | 2× |

Every method agrees with Python, apart from the differences listed in the next section. `ecm` gains little
because its solver does a dense Cholesky factorisation of the 2n × 2n Hessian at every step.

## Continuous integration

Two GitHub Actions workflows are in `.github/workflows/`:

- `ci.yml` builds the library and runs the C++ tests on Linux, macOS and Windows. A second job builds the Python
  module, runs its tests, and runs the benchmark with `--check`, which fails when any method stops agreeing with
  the original Python package.
- `wheels.yml` uses [cibuildwheel](https://cibuildwheel.pypa.io) to build and test wheels for every supported
  CPython version (≥ 3.10) on Linux x86_64/i686/aarch64 (manylinux and musllinux), macOS arm64/x86_64 and
  Windows AMD64/x86/ARM64, plus an sdist. Publishing a GitHub release uploads them to PyPI through
  [trusted publishing](https://docs.pypi.org/trusted-publishers/). This needs a `pypi` environment in the GitHub
  repository and a matching trusted publisher on PyPI.

## Differences from Python

- `marginal_likelihood`, `noise_corrected`: binomial tails are summed directly instead of going through scipy and
  statsmodels. They agree to ~1e-9, and against mpmath the C++ values are the more accurate ones.
- `ecm`: Levenberg–Marquardt with an analytic Hessian replaces trust-krylov with JAX. It reaches the same optimum
  (p-values within 1e-5 relative; Python itself stops at 9% max relative error). `seed` replaces the unseeded
  numpy random start.
- `modularity_backbone`: Louvain uses `std::mt19937`, not numpy's `RandomState(123)`, so the partition and
  therefore the vitalities can differ. Given the same partition, the vitalities match exactly.
- `pmfg`: edges are sorted with a stable sort. pandas' default quicksort orders tied weights arbitrarily, which
  changes 26 edges on the sample data. With a stable sort in Python too, the outputs are identical.
- Python code paths that print a message and return `None` print the same message and return an empty `Graph`.
- `glanb(data, c)` requires `c`. Python's `c=-1` default only prints a message.
- `Compare` keeps one-argument distribution functions apart from two-argument properties: use `add_distribution`
  for `distribution_ks_statistic`. When only some backbones are consensual, it includes the consensual row
  (Python's version raises an error in that case).
- Not ported: `visualize.py` (matplotlib plotting), the directed `MLF` variant (it cannot run in Python either),
  the `maxent_graph` JAX helpers, the sparse-matrix input of `pairwise_proximity`, the dict/matrix conversion
  helpers in `distanceclosure/utils.py`, and the Cython Dijkstra (the C++ Dijkstra replaces it).

## Layout

```
netbone/            library sources; mirrors the Python package file by file
                    (statistical/disparity.py -> netbone/statistical/disparity.cpp, ...)
python/             pybind11 module netbone_cpp
tests/              C++ self-checks and Python module checks
examples/           usage example
benchmarks/         cross-check and benchmark against the Python NetBone package
data/data.csv       NetBone sample network
pyproject.toml      Python packaging (scikit-build-core) and cibuildwheel settings
.github/workflows/  CI and wheel builds
```

## Credits

The original NetBone package is by Ali Yassin and its contributors. The method implementations follow its
Python sources, including the code it adapted from
[Doubly Stochastic / Noise Corrected](https://www.michelecoscia.com/?page_id=287),
[Marginal Likelihood](https://github.com/naviddianati/GraphPruning) and
[distanceclosure](https://github.com/rionbr/distanceclosure). Several graph algorithms are translated from
NetworkX and python-louvain (see `LICENSE`).

## License

MIT, like NetBone. See [LICENSE](LICENSE), which also reproduces the BSD-3-Clause notices of NetworkX and
python-louvain.
