# Regression suite

Runs every public function and every method/filter combination on `data/data.csv` and compares each result with
two stored baselines:

- `baseline/cpp/`: a snapshot of this library. Any change here is a regression unless it was intended.
- `baseline/python/`: the same cases computed with the original Python NetBone 0.2.3, to check the port.

## Running it

The simplest way builds the runner (into `build-regression/`) and runs it in one step, from any directory:

```sh
tests/run_regression.sh                       # every case, one line each
tests/run_regression.sh --only method.lans    # cases whose name contains this
tests/run_regression.sh --quiet               # only the failing cases and the summary
tests/run_regression.sh --update              # rewrite the C++ baseline after an intended change
```

Or through CMake and ctest:

```sh
cmake -S . -B build && cmake --build build -j
ctest --test-dir build -R regression --output-on-failure      # as part of ctest
./build/cppbone_regression data/data.csv tests/regression      # directly
./build/cppbone_regression data/data.csv tests/regression --verbose          # list every case
./build/cppbone_regression data/data.csv tests/regression --only method.lans  # cases whose name contains this
```

Every case gets one line with its verdict against each baseline: `identical`, `ok` (within the tolerance) or
`FAIL`. When a result is not identical, an indented line per baseline says how it differs. A summary follows:

```
[ ok ] method.lans.scores                    cpp: identical   python: identical
[ ok ] method.noise_corrected.scores         cpp: identical   python: ok
         python: max rel diff 2.3e-09 at ESC|TVC p_value: 0.5857139906230069 vs 0.5857139892542863
[FAIL] method.disparity.scores               cpp: FAIL        python: identical
         cpp: max rel diff 1e-09 at ABE|AEX p_value: 0.9985884770724991 vs 0.9985884780710876

109 cases, 1 failed
  vs cpp: 1 FAIL 105 identical 3 ok
  vs python: 91 identical 18 ok
```

`cpp: FAIL` with `python: identical` means a change broke something that used to match; `cpp: FAIL` with
`python: ok` after a fix means the fix moved the result closer to Python and the C++ snapshot needs updating.

## Cases

One file per case in each baseline, named after the case:

| Case | Content |
|---|---|
| `method.<name>.scores` | the method's score (and `in_backbone`) for every edge, or every node for node methods |
| `method.<name>.boolean` / `.threshold_<v>` / `.fraction_<v>` | the edges and nodes kept by each compatible filter (fractions 0.05 and 0.2, thresholds from `thresholds.csv`) |
| `measures` | every measure on the network and on three filtered backbones |
| `compare.*` | `Compare` properties, progression, KS statistics, consensual backbone and `consent()` |
| `distanceclosure.*` | metric/ultrametric closure, s- and b-values, all-pairs Dijkstra (on the 60 smallest labels), metric/ultrametric backbones and distortion (whole network), pairwise proximity for every metric |
| `utils.*` | degrees, cumulative distributions, largest component |
| `misc.scalars` | `jaccard`, `is_planar`, `binom_cdf`/`binom_sf`, `compute_pvalue`, `prox2dist`/`dist2prox` |

Methods with parameters are run in several variants (`betweenness_unweighted`, `betweenness_raw`,
`degree_weighted`, `noise_corrected_exact`, `glanb_c1`, `glanb_c0.5`).

Every file is a canonical CSV: a `key` column, then numeric columns, rows sorted by key, NaN as an empty field.
Keys are `a|b` (labels sorted) for edges, the label for nodes, `#label` for the nodes of a filtered graph, and the
row name for tables. Rows are matched by key and columns by name, so row order and orientation do not matter.

## Tolerances

`tolerances.csv` lists what may differ, per baseline and case pattern (the last matching line wins):
the largest relative difference of values, how many keys (edges, nodes, rows) may be present on one side only,
and how many `in_backbone` / `is_*` flags may differ. The C++ snapshot must match to 1e-12 (parallel
betweenness differs in the last bits); Python must match to 1e-8 except for the documented differences of `ecm`,
`modularity_backbone` and `pmfg`. Relative differences use a floor of 1e-300, so p-values that underflow to
denormals on both sides compare equal.

## Updating the baselines

After an intended change to the C++ results, check the report, then rewrite the snapshot:

```sh
tests/run_regression.sh --update      # or: cmake --build build --target regression_baseline
```

The Python baseline only changes when the cases change. Regenerate it with the original package
(`pip install -r benchmarks/requirements.txt && pip install --no-deps netbone==0.2.3`):

```sh
python tests/regression/python_baseline.py data/data.csv tests/regression
```

It takes about a minute (Python's pmfg alone takes ~35 s). netbone 0.2.3's `distanceclosure/closure.py` cannot be
imported as shipped (it imports a name the package does not export); the script provides that name first.

## Continuous integration

The `regression` job in `.github/workflows/ci.yml` runs `tests/run_regression.sh` on Linux, macOS and Windows.
Only Linux must pass: the C++ snapshot was generated there, and on the other systems `std::shuffle` and
`std::uniform_real_distribution` are implemented differently (they drive `modularity_backbone` and `ecm`) and macOS
builds without OpenMP, so those runs report without failing the workflow. The other CI jobs skip this test in their
ctest runs (`-E regression`).

## Adding a case

Add it to `run_all` in `regression.cpp` and to `python_baseline.py` under the same name and layout, then
regenerate both baselines.
