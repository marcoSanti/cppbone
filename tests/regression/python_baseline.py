"""Writes the Python baseline of the regression suite with the original NetBone package.

usage: python tests/regression/python_baseline.py data/data.csv tests/regression

Needs netbone 0.2.3 and its dependencies (see benchmarks/requirements.txt). Produces, in <dir>/baseline/python,
one canonical CSV per case with the same names and layout as regression.cpp: a "key" column (edge "a|b" with
a < b, node label, "#label" for the nodes of a graph, or a row label), then numeric columns, rows sorted by key,
NaN as an empty field. Run it once; the C++ runner compares against these files.
"""
import csv
import math
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import binom

import netbone as nb
from netbone import filters, measures
from netbone.compare import Compare
from netbone.statistical.lans import compute_pvalue
from netbone.structural.distanceclosure.backbone import (metric_backbone, metric_backbone_distortion,
                                                          ultrametric_backbone)
from netbone.structural.distanceclosure.dijkstra import all_pairs_dijkstra_path_length

# netbone 0.2.3's closure.py does `from netbone import all_pairs_dijkstra_path_length`, which the package does not
# export, so the module cannot be imported as shipped; provide the name first.
nb.all_pairs_dijkstra_path_length = all_pairs_dijkstra_path_length
from netbone.structural.distanceclosure.closure import b_values, distance_closure, s_values  # noqa: E402
from netbone.structural.distanceclosure.distance import pairwise_proximity
from netbone.structural.distanceclosure.utils import dist2prox, prox2dist
from netbone.structural.gspar import jaccard
from netbone.utils.utils import cumulative_dist, lcc


def fmt(x):
    """Number as text: shortest round-trip repr, empty for NaN/None."""
    if x is None:
        return ""
    x = float(x)
    return "" if math.isnan(x) else repr(x)


def pair_key(a, b):
    """Edge key: the two labels sorted, joined by '|'."""
    a, b = str(a), str(b)
    return f"{a}|{b}" if a <= b else f"{b}|{a}"


def write(out_dir, name, cols, rows):
    """Writes one case. rows: dict key -> list of values (one per column)."""
    assert not any("," in k for k in rows), f"comma in a key of {name}"
    with open(os.path.join(out_dir, name + ".csv"), "w", newline="") as f:
        f.write(",".join(["key"] + cols) + "\n")
        for key in sorted(rows):
            f.write(",".join([key] + [fmt(v) for v in rows[key]]) + "\n")


def edge_rows(g, cols):
    """Edge attributes of a networkx graph, NaN where missing."""
    return {pair_key(u, v): [d.get(c, math.nan) for c in cols] for u, v, d in g.edges(data=True)}


def node_rows(g, cols):
    """Node attributes of a networkx graph, NaN where missing."""
    return {str(n): [d.get(c, math.nan) for c in cols] for n, d in g.nodes(data=True)}


def graph_rows(g):
    """Structure of a graph: every edge and every node ('#label')."""
    rows = {pair_key(u, v): [] for u, v in g.edges()}
    rows.update({"#" + str(n): [] for n in g.nodes()})
    return rows


def compare_rows(df, prefix=""):
    """Rows of a Compare DataFrame, keyed by index label."""
    return {prefix + str(i): list(df.loc[i]) for i in df.index}


def index_key(i, j):
    """Zero-padded matrix index pair, as in regression.cpp."""
    return f"{i:02d}|{j:02d}"


def main():
    csv_path, reg_dir = sys.argv[1], sys.argv[2]
    out = os.path.join(reg_dir, "baseline", "python")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(reg_dir, "thresholds.csv")) as f:
        thresholds = {r[0]: float(r[1]) for r in csv.reader(l for l in f if not l.startswith("#")) if r[0] != "method"}

    df = pd.read_csv(csv_path)
    G = nx.from_pandas_edgelist(df, edge_attr=True)

    methods = {
        "disparity": lambda: nb.disparity(G),
        "lans": lambda: nb.lans(G),
        "noise_corrected": lambda: nb.noise_corrected(G),
        "noise_corrected_exact": lambda: nb.noise_corrected(G, approximation=False),
        "marginal_likelihood": lambda: nb.marginal_likelihood(df),
        "ecm": lambda: nb.ecm(G),
        "betweenness": lambda: nb.betweenness(G),
        "betweenness_unweighted": lambda: nb.betweenness(G, weighted=False, normalized=True),
        "betweenness_raw": lambda: nb.betweenness(G, weighted=True, normalized=False),
        "degree": lambda: nb.degree(G),
        "degree_weighted": lambda: nb.degree(G, weighted=True),
        "doubly_stochastic": lambda: nb.doubly_stochastic(G),
        "global_threshold": lambda: nb.global_threshold(G),
        "gspar": lambda: nb.gspar(G),
        "h_backbone": lambda: nb.h_backbone(G),
        "high_salience_skeleton": lambda: nb.high_salience_skeleton(G),
        "maximum_spanning_tree": lambda: nb.maximum_spanning_tree(G),
        "metric_distance_backbone": lambda: nb.metric_distance_backbone(G),
        "ultrametric_distance_backbone": lambda: nb.ultrametric_distance_backbone(G),
        "mlam": lambda: nb.mlam(G),
        "plam": lambda: nb.plam(G),
        "modularity_backbone": lambda: nb.modularity_backbone(G),
        "pmfg": lambda: nb.pmfg(G),
        "glanb_c1": lambda: nb.glanb(G, c=1),
        "glanb_c0.5": lambda: nb.glanb(G, c=0.5),
    }
    if not hasattr(nb, "ecm"):
        del methods["ecm"]
        print("ecm unavailable (jax missing): skipped", file=sys.stderr)

    kept = {}
    for name, run in methods.items():
        print(name, file=sys.stderr, flush=True)
        b = run()
        prefix = "method." + name
        if name == "noise_corrected_exact":
            write(out, prefix + ".scores", ["score", "nc_sdev"], edge_rows(b.graph, ["score", "nc_sdev"]))
            continue
        cols = [b.property_name]
        if b.filter_on == "Nodes":
            write(out, prefix + ".scores", cols, node_rows(b.graph, cols))
        else:
            if any("in_backbone" in d for _, _, d in b.graph.edges(data=True)) and b.property_name != "in_backbone":
                cols.append("in_backbone")
            write(out, prefix + ".scores", cols, edge_rows(b.graph, cols))
        for f in b.compatible_filters():
            if f.__name__ == "boolean_filter":
                write(out, prefix + ".boolean", [], graph_rows(filters.boolean_filter(b, narrate=False)))
            elif f.__name__ == "threshold_filter":
                v = thresholds[name]
                write(out, f"{prefix}.threshold_{fmt_value(v)}", [],
                      graph_rows(filters.threshold_filter(b, v, narrate=False)))
            elif f.__name__ == "fraction_filter":
                for v in (0.05, 0.2):
                    write(out, f"{prefix}.fraction_{fmt_value(v)}", [],
                          graph_rows(filters.fraction_filter(b, v, narrate=False)))
        kept[name] = b

    # measures
    graphs = {
        "original": G,
        "disparity_threshold_0.05": filters.threshold_filter(kept["disparity"], 0.05, narrate=False),
        "high_salience_skeleton_boolean": filters.boolean_filter(kept["high_salience_skeleton"], narrate=False),
        "maximum_spanning_tree_boolean": filters.boolean_filter(kept["maximum_spanning_tree"], narrate=False),
    }
    props = {
        "density": measures.density, "weight_entropy": measures.weight_entropy,
        "average_degree": measures.average_degree, "lcc_size": measures.lcc_size,
        "node_fraction": measures.node_fraction, "edge_fraction": measures.edge_fraction,
        "weight_fraction": measures.weight_fraction, "reachability": measures.reachability,
        "number_connected_components": measures.number_connected_components, "diameter": measures.diameter,
        "lcc_node_fraction": lambda o, x: measures.lcc_node_fraction(x),
        "lcc_edge_fraction": measures.lcc_edge_fraction, "lcc_weight_fraction": measures.lcc_weight_fraction,
        "average_clustering_coefficient": measures.average_clustering_coefficient,
        "transitivity": measures.transitivity,
    }
    write(out, "measures", list(graphs), {p: [f(G, x) for x in graphs.values()] for p, f in props.items()})

    # degrees and cumulative distributions
    d, wd = dict(G.degree()), dict(G.degree(weight="weight"))
    write(out, "utils.degrees", ["degree", "weighted_degree"], {str(n): [d[n], wd[n]] for n in G.nodes()})
    deg = measures.degrees(G)
    for increasing in (True, False):
        cd = cumulative_dist("degree", "y", deg, increasing)
        rows = {index_key(i // 100, i % 100): [x, y] for i, (x, y) in enumerate(zip(cd.index, cd["y"]))}
        write(out, "utils.cumulative_dist_" + ("increasing" if increasing else "decreasing"), ["x", "y"], rows)

    # Compare
    cprops = {"density": measures.density, "edge_fraction": measures.edge_fraction,
              "node_fraction": measures.node_fraction, "weight_fraction": measures.weight_fraction,
              "lcc_size": measures.lcc_size, "number_connected_components": measures.number_connected_components}
    cmp = Compare()
    cmp.set_network(G)
    for m in ("high_salience_skeleton", "maximum_spanning_tree", "plam"):
        cmp.add_backbone(kept[m])
    for n, p in cprops.items():
        cmp.add_property(n, p)
    cmp.set_filter(filters.boolean_filter)
    t = cmp.properties()
    write(out, "compare.properties", list(t.columns), compare_rows(t))
    write(out, "compare.consent", [], graph_rows(cmp.consent()))

    cmp = Compare()
    cmp.set_network(G)
    for m in ("disparity", "lans", "high_salience_skeleton"):
        cmp.add_backbone(kept[m])
    for n, p in cprops.items():
        cmp.add_property(n, p)
    cmp.set_filter(filters.fraction_filter, [0.05, 0.1, 0.2])
    rows, cols = {}, None
    for pname, table in cmp.properties_progression().items():
        cols = list(table.columns)
        for v in table.index:
            rows[f"{pname}|{fmt_value(v)}"] = list(table.loc[v])
    write(out, "compare.properties_progression", cols, rows)

    cmp = Compare()
    cmp.set_network(G)
    for m in ("disparity", "noise_corrected", "lans"):
        cmp.add_backbone(kept[m])
    cmp.add_property("degree", lambda x: measures.degrees(x))
    cmp.add_property("weighted_degree", lambda x: measures.degrees(x, weight="weight"))
    cmp.add_property("weight", lambda x: measures.weights(x))
    cmp.set_filter(filters.fraction_filter, [0.1, 0.1, 0.1])
    ks, _, consensual = cmp.distribution_ks_statistic(increasing=True, consent=[True, True, True])
    write(out, "compare.ks_statistic", list(ks.columns), compare_rows(ks))
    write(out, "compare.ks_consensual_backbone", [], graph_rows(consensual))

    # distance closure toolkit, on the induced subgraph of the 60 smallest labels with distance = 1 / weight
    small = G.subgraph(sorted(G.nodes(), key=str)[:60]).copy()
    for u, v, dd in small.edges(data=True):
        dd["distance"] = 1 / dd["weight"]
    cm = distance_closure(small, kind="metric", weight="distance")
    write(out, "distanceclosure.closure_metric", ["distance", "metric_distance", "is_metric"],
          edge_rows(cm, ["distance", "metric_distance", "is_metric"]))
    cu = distance_closure(small, kind="ultrametric", weight="distance")
    write(out, "distanceclosure.closure_ultrametric", ["distance", "ultrametric_distance", "is_ultrametric"],
          edge_rows(cu, ["distance", "ultrametric_distance", "is_ultrametric"]))
    write(out, "distanceclosure.s_values", ["s-value"], edge_rows(s_values(cm, "distance", "metric_distance"), ["s-value"]))
    bv = b_values(cm, "distance", "metric_distance")
    rows = {}
    for u, v, dd in bv.edges(data=True):
        ij, ji = dd.get("b_ij-value"), dd.get("b_ji-value")
        if ij is None and ji is None:
            continue
        rows[pair_key(u, v)] = [ij, ji] if str(u) < str(v) else [ji, ij]
    write(out, "distanceclosure.b_values", ["b_low", "b_high"], rows)
    for name, disj in (("sum", sum), ("max", max)):
        rows = {}
        for u, lengths in all_pairs_dijkstra_path_length(small, weight="distance", disjunction=disj):
            for v, length in lengths.items():
                if length < math.inf:
                    rows[f"{u}>{v}"] = [length]
        write(out, "distanceclosure.all_pairs_dijkstra_" + name, ["length"], rows)
    full = G.copy()
    for u, v, dd in full.edges(data=True):
        dd["distance"] = 1 / dd["weight"]
    write(out, "distanceclosure.metric_backbone", [], graph_rows(metric_backbone(full, weight="distance")))
    write(out, "distanceclosure.ultrametric_backbone", [], graph_rows(ultrametric_backbone(full, weight="distance")))
    _, sv = metric_backbone_distortion(full, weight="distance")
    items = sv.items() if isinstance(sv, dict) else ((tuple(e), s) for e, s in sv)
    write(out, "distanceclosure.metric_backbone_distortion", ["s"], {pair_key(*e): [s] for e, s in items})
    m = np.array([[(i * 7 + j * 3) % 5 for j in range(15)] for i in range(20)], dtype=float)
    for metric in ("jaccard", "jaccard_binary", "jaccard_set", "jaccard_weighted"):
        p = np.asarray(pairwise_proximity(m, metric=metric))
        write(out, "distanceclosure.pairwise_proximity_" + metric, ["proximity"],
              {index_key(i, j): [p[i, j]] for i in range(p.shape[0]) for j in range(p.shape[1])})
    write(out, "utils.lcc", [], graph_rows(lcc(small)))

    # scalar functions
    k33 = nx.complete_bipartite_graph(3, 3)
    rows = {
        "prox2dist(0)": [prox2dist(0)], "prox2dist(0.25)": [prox2dist(0.25)], "prox2dist(1)": [prox2dist(1)],
        "dist2prox(0)": [dist2prox(0)], "dist2prox(3)": [dist2prox(3)], "dist2prox(inf)": [dist2prox(math.inf)],
        "jaccard([1;2;3];[2;3;4])": [jaccard([1, 2, 3], [2, 3, 4])], "jaccard([1;1;2];[2])": [jaccard([1, 1, 2], [2])],
        "is_planar(K4)": [nx.is_planar(nx.complete_graph(4))], "is_planar(K5)": [nx.is_planar(nx.complete_graph(5))],
        "is_planar(K3_3)": [nx.is_planar(k33)], "is_planar(network)": [nx.is_planar(G)],
        "is_planar(mst)": [nx.is_planar(filters.boolean_filter(kept["maximum_spanning_tree"], narrate=False))],
    }
    for k, n, p in ((2, 10, 0.3), (0, 10, 0.5), (10, 10, 0.5), (5, 100, 0.01), (100, 620000000, 3e-7),
                    (1, 620000000, 1e-9), (3000, 10000, 0.25)):
        a = f"({fmt_value(k)};{fmt_value(n)};{fmt_value(p)})"
        rows["binom_cdf" + a] = [binom.cdf(k, n, p)]
        rows["binom_sf" + a] = [binom.sf(k, n, p)]
    for key, u, v in sorted((pair_key(u, v), u, v) for u, v in G.edges())[:5]:
        node = u if str(u) < str(v) else v
        rows[f"compute_pvalue({key})"] = [compute_pvalue(G, node, G[u][v]["weight"])]
    write(out, "misc.scalars", ["value"], rows)
    print("wrote", len(os.listdir(out)), "files to", out, file=sys.stderr)


def fmt_value(v):
    """Formats a parameter in a case name like C++ format_double: integers without a decimal point."""
    v = float(v)
    return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)


if __name__ == "__main__":
    main()
