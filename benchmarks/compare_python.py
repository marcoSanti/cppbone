"""Cross-check the original netbone Python package (pip install netbone) against cppbone and time both.

usage: python benchmarks/compare_python.py [--csv FILE] [--check] [method ...]
--check exits non-zero when a method disagrees with Python beyond its tolerance (used by CI).
"""
import argparse
import os
import sys
import time
import warnings
from importlib.metadata import version

import networkx as nx
import numpy as np
import pandas as pd

import netbone as py
import cppbone as cpp

if version("netbone") != "0.2.3":
    print(f"warning: netbone {version('netbone')} installed; the published results use 0.2.3", file=sys.stderr)

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument("methods", nargs="*", help="methods to run (default: all)")
ap.add_argument("--csv", default=os.path.join(HERE, "..", "data", "data.csv"))
ap.add_argument("--check", action="store_true", help="fail when results disagree beyond tolerance")
ap.add_argument("--reps", type=int, default=3, help="C++ repetitions; the best time is reported")
args = ap.parse_args()

df = pd.read_csv(args.csv)
G = nx.from_pandas_edgelist(df, edge_attr=True)
C = cpp.Graph.read_csv(args.csv)

# name -> (python call, cpp call)
METHODS = {
    "disparity": (lambda: py.disparity(G), lambda: cpp.disparity(C)),
    "lans": (lambda: py.lans(G), lambda: cpp.lans(C)),
    "noise_corrected": (lambda: py.noise_corrected(G), lambda: cpp.noise_corrected(C)),
    "marginal_likelihood": (lambda: py.marginal_likelihood(df), lambda: cpp.marginal_likelihood(C)),
    "global_threshold": (lambda: py.global_threshold(G), lambda: cpp.global_threshold(C)),
    "degree": (lambda: py.degree(G), lambda: cpp.degree(C)),
    "gspar": (lambda: py.gspar(G), lambda: cpp.gspar(C)),
    "plam": (lambda: py.plam(G), lambda: cpp.plam(C)),
    "mlam": (lambda: py.mlam(G), lambda: cpp.mlam(C)),
    "maximum_spanning_tree": (lambda: py.maximum_spanning_tree(G), lambda: cpp.maximum_spanning_tree(C)),
    "doubly_stochastic": (lambda: py.doubly_stochastic(G), lambda: cpp.doubly_stochastic(C)),
    "high_salience_skeleton": (lambda: py.high_salience_skeleton(G), lambda: cpp.high_salience_skeleton(C)),
    "metric_distance_backbone": (lambda: py.metric_distance_backbone(G), lambda: cpp.metric_distance_backbone(C)),
    "ultrametric_distance_backbone": (lambda: py.ultrametric_distance_backbone(G),
                                      lambda: cpp.ultrametric_distance_backbone(C)),
    "betweenness": (lambda: py.betweenness(G), lambda: cpp.betweenness(C)),
    "h_backbone": (lambda: py.h_backbone(G), lambda: cpp.h_backbone(C)),
    "modularity_backbone": (lambda: py.modularity_backbone(G), lambda: cpp.modularity_backbone(C)),
    "glanb": (lambda: py.glanb(G, c=1), lambda: cpp.glanb(C, c=1)),
    "pmfg": (lambda: py.pmfg(G), lambda: cpp.pmfg(C)),
}
if hasattr(py, "ecm"):  # netbone skips ecm when jax is missing
    METHODS["ecm"] = (lambda: py.ecm(G), lambda: cpp.ecm(C))

# Known, documented differences (see the top-level README); everything else must match to 1e-8.
REL_TOL = {"ecm": 1e-4, "modularity_backbone": float("inf")}  # different solver / different random partition
IN_BACKBONE_DIFF = {"pmfg": 26}  # pandas' unstable sort on tied weights


def timed(f, reps):
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t)
    return out, best


def keyed(b):
    """Backbone -> DataFrame indexed by sorted (u, v) for edges, or node label for nodes."""
    d = b.to_dataframe()
    if b.filter_on == "Edges":
        d.index = [tuple(sorted((str(u), str(v)))) for u, v in zip(d["source"], d["target"])]
        d = d.drop(columns=["source", "target"])
    else:
        d.index = d.index.map(str)
    return d.apply(pd.to_numeric, errors="coerce").astype(float)


def compare(name, pb, cb):
    """Returns (description, ok)."""
    p, c = keyed(pb), keyed(cb)
    if set(p.index) != set(c.index):
        return f"row sets differ ({len(p)} vs {len(c)})", False
    c = c.loc[p.index]
    notes, ok = [], True
    for col in [pb.property_name, "in_backbone"]:
        if col not in p.columns:
            continue
        a, b = p[col].to_numpy(), c[col].to_numpy()
        if col == "in_backbone":
            diff = int(np.sum((a != 0) != (b != 0)))
            ok &= diff <= IN_BACKBONE_DIFF.get(name, 0)
            notes.append(f"in_backbone: {'same' if diff == 0 else f'{diff} differ'}")
        else:
            both_nan = np.isnan(a) & np.isnan(b)
            rel = np.nanmax(np.where(both_nan, 0, np.abs(a - b) / (1e-300 + np.maximum(np.abs(a), np.abs(b)))))
            ok &= rel <= REL_TOL.get(name, 1e-8)
            notes.append(f"{col}: max rel diff {rel:.1e}" if rel > 0 else f"{col}: identical")
    return "; ".join(notes), ok


print(f"{G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")
print(f"{'method':32}{'python s':>10}{'c++ s':>10}{'speedup':>9}  agreement")
failed = []
for name, (pf, cf) in METHODS.items():
    if args.methods and name not in args.methods:
        continue
    cb, ct = timed(cf, args.reps)
    pb, pt = timed(pf, 1)
    desc, ok = compare(name, pb, cb)
    if not ok:
        failed.append(name)
    print(f"{name:32}{pt:10.3f}{ct:10.4f}{pt / ct:8.0f}x  {desc}{'' if ok else '  <-- FAIL'}", flush=True)

if args.check and failed:
    sys.exit(f"\nresults disagree with Python: {', '.join(failed)}")
