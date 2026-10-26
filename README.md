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
- Optional Python module `cppbone` that accepts networkx graphs or pandas edge lists
- OpenMP parallelism (optional) across nodes, edges and shortest-path sources
- No runtime dependencies besides the OpenMP runtime when it is enabled

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
./build/cppbone_cli data/data.csv -m high_salience_skeleton   # command-line tool, see below
```

The CLI fetches two header-only libraries at configure time:
[Taywee/args](https://github.com/Taywee/args) for argument parsing and
[p-ranav/indicators](https://github.com/p-ranav/indicators) for the progress bar.

To use it from another CMake project, add it with `add_subdirectory(cppbone)` and link against the `cppbone` target.

API documentation: every function carries a doxygen comment. `make -C docs` builds it into `docs/build/` (HTML in
`html/`, LaTeX in `latex/`); `make -C docs pdf` also compiles `docs/build/latex/refman.pdf`. See `docs/Makefile`.

### Parallelism

When the compiler supports OpenMP, the library is built with it (`-DCPPBONE_OPENMP=OFF` turns it off; Apple clang has
no OpenMP, so on macOS it runs single-threaded unless libomp is installed). The thread count is `OMP_NUM_THREADS`,
`cb::set_num_threads(n)`, or `-j N` on the CLI. What runs in parallel:

- the per-source shortest-path searches of `betweenness`, `h_backbone`, `high_salience_skeleton`, `glanb`,
  `metric_distance_backbone`, `ultrametric_distance_backbone`, `all_pairs_dijkstra_path_length` and `diameter`
- the per-node and per-edge loops of the other methods, the measures, the Sinkhorn iterations of
  `doubly_stochastic` and the Hessian of `ecm`
- CSV parsing, graph copies and destruction

Results do not depend on the thread count: each edge or node is computed on its own, and values that come from
several places are combined in the sequential order. The exception is `betweenness` (and so `h_backbone`), whose
per-source contributions are summed in whatever order the threads finish: values can differ in the last bits
(about 1e-15 relative) between runs. `maximum_spanning_tree` and `pmfg` are sequential algorithms, and Louvain in
`modularity_backbone` stays sequential so the partition does not depend on the schedule.

## Usage (C++)

```cpp
#include "cppbone/cppbone.hpp"
namespace cb = cppbone;

cb::Graph g = cb::Graph::read_csv("edges.csv");          // columns: source,target,weight[,...]
// header-less file: pick columns by 0-based index (source, target, weight); weight is stored as "weight"
// cb::Graph g = cb::Graph::read_csv("edges.csv", 0, 1, 2);
cb::Backbone b = cb::high_salience_skeleton(g);

cb::Graph backbone1 = cb::boolean_filter(b);             // default threshold
cb::Graph backbone2 = cb::threshold_filter(b, 0.7);      // threshold
cb::Graph backbone3 = cb::fraction_filter(b, 0.15);      // keep 15% of the edges
b.to_csv(std::cout);                                     // Backbone.to_dataframe()
```

`cb::Graph` stands in for both `networkx.Graph` and the pandas edge list. Node labels are strings and attributes
are numeric: booleans are stored as 1/0 and a missing attribute reads as NaN. Nodes and edges iterate in networkx
order, so sorting and tie-breaking match the Python package.

`Graph::read_csv` splits on plain commas and does not handle quoted fields, so labels must not contain commas.

## Command-line tool

`cppbone_cli` runs any set of methods on an edge list, filters them and compares the results.
`--list` prints the methods, filters, measures and distributions; `--help` prints every option.

```sh
# header-less file: source, target, weight in columns 0, 1, 3
cppbone_cli edges.csv --no-header --weight-col 3 -m disparity -f threshold -v 0.05 -o out/

# every method, keep 10% of the edges, compare them all, write the tables to out/
cppbone_cli data/data.csv -m all -f fraction -v 0.1 -M all --ks all --consensus -o out/

# how the measures change with the fraction of edges kept
cppbone_cli data/data.csv -m disparity,noise_corrected -f fraction --progression 0.05,0.1,0.2 -M density,lcc_size
```

With `-o DIR` it writes `<method>.scores.csv` (every edge with its score), `<method>.backbone.csv` (the filtered
backbone), plus `measures.csv`, `ks.csv`, `consensus.csv` and `progression_<measure>.csv` when asked for. Without it
the summary and the comparison tables go to stdout. A method whose backbone does not support the chosen filter still
writes its scores, and the summary names the filters it does support.

## Python module

The distribution is called `cppbone` and the module it installs is `cppbone`. It needs Python ≥ 3.10.

```sh
pip install .                                  # builds the module with scikit-build-core + CMake
pip install ".[interop]"                       # also networkx and pandas, for conversions
```

Once a release is published, `pip install cppbone` installs prebuilt wheels (see [CI](#continuous-integration)).

For development, the CMake option `CPPBONE_PYTHON=ON` builds the module in place. CMake uses an installed pybind11
3 if it finds one; otherwise it downloads pybind11 v3.0.1 (pinned by hash). The option is off by default, so the
plain C++ build downloads nothing.

```sh
cmake -S . -B build-py -DCPPBONE_PYTHON=ON -DPython_EXECUTABLE=$(which python3)
cmake --build build-py -j
ctest --test-dir build-py                      # also runs tests/test_python.py
PYTHONPATH=build-py python3 -c "import cppbone"
```

The API mirrors the Python package. Methods take a `cppbone.Graph`, a networkx graph or a pandas edge-list
DataFrame:

```python
import networkx as nx
import cppbone as cb

g = nx.les_miserables_graph()
b = cb.high_salience_skeleton(g)
backbone = cb.fraction_filter(b, 0.15)         # cppbone.Graph
backbone.to_networkx(); b.to_dataframe()       # back to networkx / pandas

cmp = cb.Compare()
cmp.set_network(g); cmp.add_backbone(b)
cmp.set_filter(cb.fraction_filter, [0.15])
cmp.add_property("Density", cb.density)        # or any Python callable (original, graph) -> float
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
include/cppbone/    public headers (cppbone.hpp includes the rest)
src/                library sources; mirrors the Python package file by file
                    (statistical/disparity.py -> src/statistical/disparity.cpp, ...), plus the private detail.hpp
apps/               cppbone_cli command-line tool
python/             pybind11 module cppbone
tests/              C++ self-checks and Python module checks
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
