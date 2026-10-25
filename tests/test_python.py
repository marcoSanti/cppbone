"""Self-checks for the netbone_cpp binding. argv[1]: an edge-list CSV. networkx/pandas checks run when installed."""
import importlib.util
import math
import sys

import netbone_cpp as nb

g = nb.Graph()
g.add_edge("a", "b", {"weight": 10})
g.add_edge("b", "c", {"weight": 10})
g.add_edge("a", "c", {"weight": 1})
assert len(g) == 3 and g.number_of_edges() == 3

b = nb.disparity(g)
assert b.method_name == "Disparity Filter" and b.filter_on == "Edges"
assert b.compatible_filters() == ["threshold_filter", "fraction_filter"]
p = {tuple(sorted((u, v))): d["p_value"] for u, v, d in b.graph.edges(data=True)}
assert math.isclose(p[("a", "c")], 10 / 11)
assert nb.threshold_filter(b, 0.5, narrate=False).number_of_edges() == 2
assert nb.boolean_filter(nb.maximum_spanning_tree(g), narrate=False).number_of_edges() == 2
assert nb.degree(g, weighted=True).graph.node_attrs("b")["weighted-degree"] == 20
air = nb.Graph.read_csv(sys.argv[1])
assert nb.boolean_filter(nb.maximum_spanning_tree(air), narrate=False).number_of_edges() == len(air) - 1

cmp = nb.Compare()
cmp.set_network(g)
cmp.add_backbone(nb.maximum_spanning_tree(g))
cmp.set_filter(nb.boolean_filter)
cmp.add_property("Density", nb.density)
cmp.add_property("Custom", lambda original, graph: graph.number_of_edges() / original.number_of_edges())

if importlib.util.find_spec("pandas") and importlib.util.find_spec("networkx"):
    import networkx as nx
    import pandas as pd

    props = cmp.properties()
    assert list(props.index) == ["Original", "Maximum Spanning Tree"]
    assert math.isclose(props.loc["Maximum Spanning Tree", "Custom"], 2 / 3)

    nxg = nx.Graph()
    nxg.add_edge(1, 2, weight=3.0)
    nxg.add_edge(2, 3, weight=1.0)
    df = nb.high_salience_skeleton(nxg).to_dataframe()
    assert set(df.columns) == {"source", "target", "weight", "distance", "salience", "in_backbone"}
    assert len(nb.lans(pd.DataFrame({"source": ["x", "y"], "target": ["y", "z"], "weight": [1, 2]})).graph) == 3
    assert nb.Graph.from_networkx(nxg).to_networkx().number_of_edges() == 2
    assert list(nb.degree(nxg).to_dataframe().index) == ["1", "2", "3"]

print("python binding checks passed")
