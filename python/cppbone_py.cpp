/// @file
/// @brief Python bindings: module `cppbone`, mirroring the original Python API.
///
/// Methods accept a cppbone.Graph, a networkx graph or a pandas edge-list DataFrame. Node labels become strings;
/// booleans come back as 1.0 / 0.0.
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <ranges>
#include <set>
#include <sstream>
#include <string_view>

#include "cppbone/cppbone.hpp"

namespace py = pybind11;
using namespace cppbone;

namespace {

/// @brief Converts a Python attribute dict to Attrs.
/// @param data Python dict.
/// @return The numeric attributes; values that do not convert to float (strings, None) are skipped.
Attrs to_attrs(const py::handle& data) {
    Attrs a;
    for (auto kv : py::reinterpret_borrow<py::dict>(data)) {
        try {
            a[py::str(kv.first)] = py::cast<double>(kv.second);
        } catch (const py::cast_error&) {
        }
    }
    return a;
}

/// @brief Converts an undirected networkx graph, keeping node order, edge order and numeric edge attributes.
/// @param g networkx.Graph.
/// @return The graph, with labels converted to strings.
/// @throws py::type_error for a directed graph.
Graph from_networkx(const py::object& g) {
    if (py::cast<bool>(g.attr("is_directed")())) throw py::type_error("only undirected graphs are supported");
    Graph out;
    for (auto n : g.attr("nodes")()) out.add_node(std::string(py::str(n)));
    for (auto e : g.attr("edges")(py::arg("data") = true)) {
        auto t = py::reinterpret_borrow<py::tuple>(e);
        out.add_edge(std::string(py::str(t[0])), std::string(py::str(t[1])), to_attrs(t[2]));
    }
    return out;
}

/// @brief Converts a pandas edge list with "source" and "target" columns; other numeric columns become attributes.
/// @param df pandas.DataFrame.
/// @return The graph, with labels converted to strings.
/// @throws py::value_error if a column is missing.
Graph from_pandas(const py::object& df) {
    std::vector<std::string> cols;
    for (auto c : df.attr("columns")) cols.push_back(py::str(c));
    const auto src = static_cast<size_t>(std::ranges::find(cols, "source") - cols.begin());
    const auto dst = static_cast<size_t>(std::ranges::find(cols, "target") - cols.begin());
    if (src == cols.size() || dst == cols.size()) throw py::value_error("DataFrame needs 'source' and 'target' columns");
    std::vector<std::vector<py::object>> values;
    for (const auto& c : cols) values.push_back(py::cast<std::vector<py::object>>(df[py::str(c)].attr("tolist")()));
    Graph out;
    for (size_t r = 0; r < values[0].size(); ++r) {
        Attrs a;
        for (size_t c = 0; c < cols.size(); ++c) {
            if (c == src || c == dst) continue;
            try {
                a[cols[c]] = py::cast<double>(values[c][r]);
            } catch (const py::cast_error&) {
            }
        }
        out.add_edge(std::string(py::str(values[src][r])), std::string(py::str(values[dst][r])), a);
    }
    return out;
}

/// @brief Accepts any supported input as a Graph.
/// @param data cppbone.Graph, pandas.DataFrame or networkx graph.
/// @return The graph (copied or converted).
/// @throws py::type_error for any other type.
Graph to_graph(const py::object& data) {
    if (py::isinstance<Graph>(data)) return py::cast<Graph>(data);
    if (py::hasattr(data, "columns")) return from_pandas(data);
    if (py::hasattr(data, "edges") && py::hasattr(data, "is_directed")) return from_networkx(data);
    throw py::type_error("data should be a cppbone.Graph, a pandas dataframe or a networkx graph");
}

/// @brief Builds a pandas DataFrame from columns.
/// @param order Column order.
/// @param cols Column name to list of values.
/// @param index Row index, or None for the default.
/// @return The DataFrame.
py::object columns_to_pandas(const std::vector<std::string>& order, py::dict cols, py::object index = py::none()) {
    py::list names;
    for (const auto& c : order) names.append(c);
    return py::module_::import("pandas").attr("DataFrame")(cols, py::arg("columns") = names, py::arg("index") = index);
}

/// @brief Converts a graph to a pandas edge list (networkx to_pandas_edgelist).
/// @param g Graph.
/// @return DataFrame with source, target and every edge attribute in alphabetical order.
py::object edges_to_pandas(const Graph& g) {
    const auto edges = g.edges();
    std::set<std::string> keys;
    for (const auto& e : edges) keys.insert_range(g.edge_attrs(e.id) | std::views::keys);
    std::vector<std::string> order{"source", "target"};
    order.append_range(keys);
    py::dict cols;
    py::list src, dst;
    for (const auto& e : edges) src.append(g.label(e.u)), dst.append(g.label(e.v));
    cols["source"] = src;
    cols["target"] = dst;
    for (const auto& k : keys) {
        py::list v;
        for (const auto& e : edges) v.append(g.get(e.id, k));
        cols[py::str(k)] = v;
    }
    return columns_to_pandas(order, cols);
}

/// @brief Converts the node attributes of a graph to a pandas DataFrame indexed by label.
/// @param g Graph.
/// @return DataFrame with one column per node attribute, in alphabetical order.
py::object nodes_to_pandas(const Graph& g) {
    std::set<std::string> keys;
    for (const int u : g.nodes()) keys.insert_range(g.node_attrs(u) | std::views::keys);
    py::dict cols;
    py::list index;
    for (const int u : g.nodes()) index.append(g.label(u));
    for (const auto& k : keys) {
        py::list v;
        for (const int u : g.nodes()) v.append(g.node_get(u, k));
        cols[py::str(k)] = v;
    }
    return columns_to_pandas({keys.begin(), keys.end()}, cols, index);
}

/// @brief Converts a graph to networkx, keeping node and edge attributes.
/// @param g Graph.
/// @return networkx.Graph.
py::object to_networkx(const Graph& g) {
    py::object out = py::module_::import("networkx").attr("Graph")();
    for (const int u : g.nodes()) {
        py::dict attrs;
        for (const auto& [name, value] : g.node_attrs(u)) attrs[py::str(name)] = value;
        out.attr("add_node")(g.label(u), **attrs);
    }
    for (const auto& e : g.edges()) {
        py::dict attrs;
        for (const auto& [name, value] : g.edge_attrs(e.id)) attrs[py::str(name)] = value;
        out.attr("add_edge")(g.label(e.u), g.label(e.v), **attrs);
    }
    return out;
}

/// @brief Converts a Compare table to a pandas DataFrame.
/// @param t Table.
/// @return DataFrame with t's columns and a named index.
py::object table_to_pandas(const Table& t) {
    py::dict cols;
    for (size_t c = 0; c < t.columns.size(); ++c) {
        py::list v;
        for (const auto& row : t.data) v.append(row[c]);
        cols[py::str(t.columns[c])] = v;
    }
    py::object idx = py::module_::import("pandas").attr("Index")(t.index, py::arg("name") = t.index_name);
    return columns_to_pandas(t.columns, cols, idx);
}

/// @brief Maps a Python filter (a cppbone function or its name) to the C++ filter.
/// @param f Filter function or name.
/// @return The matching filter.
/// @throws py::value_error for an unknown name.
Filter filter_from(const py::object& f) {
    std::string name = py::hasattr(f, "__name__") ? std::string(py::str(f.attr("__name__"))) : std::string(py::str(f));
    if (name == "boolean_filter") return boolean_filter;
    if (name == "threshold_filter") return threshold_filter;
    if (name == "fraction_filter") return fraction_filter;
    throw py::value_error("unknown filter: " + name);
}

}  // namespace

/// @brief Defines the `cppbone` Python module: the Graph class, the methods, filters, measures and Compare.
PYBIND11_MODULE(cppbone, m) {
    m.doc() = "Backbone extraction from simple weighted networks";

    py::class_<Graph>(m, "Graph")
        .def(py::init<>())
        .def("add_node", &Graph::add_node)
        .def("add_edge", [](Graph& g, const std::string& u, const std::string& v, const Attrs& a) { g.add_edge(u, v, a); },
             py::arg("u"), py::arg("v"), py::arg("attrs") = Attrs{})
        .def("number_of_nodes", &Graph::number_of_nodes)
        .def("number_of_edges", &Graph::number_of_edges)
        .def("__len__", &Graph::number_of_nodes)
        .def("nodes", [](const Graph& g) {
            return g.nodes() | std::views::transform([&](const int u) { return g.label(u); }) |
                   std::ranges::to<std::vector>();
        })
        .def("edges", [](const Graph& g, bool data) {
            py::list out;
            for (const auto& e : g.edges()) {
                if (data)
                    out.append(py::make_tuple(g.label(e.u), g.label(e.v), g.edge_attrs(e.id)));
                else
                    out.append(py::make_tuple(g.label(e.u), g.label(e.v)));
            }
            return out;
        }, py::arg("data") = false)
        .def("node_attrs", [](const Graph& g, const std::string& n) {
            int u = g.node(n);
            if (u < 0) throw py::key_error(n);
            return g.node_attrs(u);
        })
        .def_static("read_csv", py::overload_cast<const std::string&>(&Graph::read_csv), py::arg("path"))
        .def_static("read_csv", py::overload_cast<const std::string&, int, int, int>(&Graph::read_csv),
                    py::arg("path"), py::arg("source_col"), py::arg("target_col"), py::arg("weight_col") = 2)
        .def("to_csv", [](const Graph& g) { std::ostringstream s; g.to_csv(s); return s.str(); })
        .def_static("from_networkx", &from_networkx)
        .def_static("from_pandas", &from_pandas)
        .def("to_networkx", &to_networkx)
        .def("to_pandas", &edges_to_pandas)
        .def("__repr__", [](const Graph& g) {
            return "<cppbone.Graph with " + std::to_string(g.number_of_nodes()) + " nodes and " +
                   std::to_string(g.number_of_edges()) + " edges>";
        });

    py::class_<Backbone>(m, "Backbone")
        .def_readonly("graph", &Backbone::graph)
        .def_readonly("method_name", &Backbone::method_name)
        .def_readonly("property_name", &Backbone::property_name)
        .def_readonly("ascending", &Backbone::ascending)
        .def_property_readonly("filter_on",
                               [](const Backbone& b) { return b.filter_on == FilterOn::Edges ? "Edges" : "Nodes"; })
        .def("compatible_filters", [](const Backbone& b) {
            return b.filters | std::views::transform(filter_name) | std::ranges::to<std::vector>();
        })
        .def("narrate", &Backbone::narrate)
        .def("to_dataframe", [](const Backbone& b) {
            return b.filter_on == FilterOn::Edges ? edges_to_pandas(b.graph) : nodes_to_pandas(b.graph);
        });

    // filters.py
    m.def("boolean_filter", [](const Backbone& b, bool narrate) { return boolean_filter(b, 0, narrate); },
          py::arg("backbone"), py::arg("narrate") = true);
    m.def("threshold_filter", &threshold_filter, py::arg("backbone"), py::arg("value"), py::arg("narrate") = true,
          py::arg("secondary_property") = "weight", py::arg("secondary_property_ascending") = false);
    m.def("fraction_filter", &fraction_filter, py::arg("backbone"), py::arg("value"), py::arg("narrate") = true,
          py::arg("secondary_property") = "weight", py::arg("secondary_property_ascending") = false);

    // backbone methods
    auto method = [&m](const char* name, Backbone (*f)(const Graph&)) {
        m.def(name, [f](const py::object& data) { return f(to_graph(data)); }, py::arg("data"));
    };
    method("lans", lans);
    method("marginal_likelihood", marginal_likelihood);
    method("doubly_stochastic", doubly_stochastic);
    method("global_threshold", global_threshold);
    method("gspar", gspar);
    method("h_backbone", h_backbone);
    method("high_salience_skeleton", high_salience_skeleton);
    method("maximum_spanning_tree", maximum_spanning_tree);
    method("metric_distance_backbone", metric_distance_backbone);
    method("ultrametric_distance_backbone", ultrametric_distance_backbone);
    method("mlam", mlam);
    method("plam", plam);
    method("pmfg", pmfg);
    m.def("disparity", [](const py::object& d, const std::string& w) { return disparity(to_graph(d), w); },
          py::arg("data"), py::arg("weight") = "weight");
    m.def("noise_corrected", [](const py::object& d, bool a) { return noise_corrected(to_graph(d), a); },
          py::arg("data"), py::arg("approximation") = true);
    m.def("ecm", [](const py::object& d, std::uint32_t seed) { return ecm(to_graph(d), seed); }, py::arg("data"),
          py::arg("seed") = 0);
    m.def("betweenness", [](const py::object& d, bool w, bool n) { return betweenness(to_graph(d), w, n); },
          py::arg("data"), py::arg("weighted") = true, py::arg("normalized") = true);
    m.def("degree", [](const py::object& d, bool w) { return degree(to_graph(d), w); }, py::arg("data"),
          py::arg("weighted") = false);
    m.def("modularity_backbone", [](const py::object& d, std::uint32_t s) { return modularity_backbone(to_graph(d), s); },
          py::arg("data"), py::arg("seed") = 123);
    m.def("set_num_threads", &set_num_threads, py::arg("n"),
          "Threads used by the methods (OpenMP builds); n <= 0 restores the default.");
    m.def("num_threads", &num_threads);
    m.def("glanb", [](const py::object& d, double c) { return glanb(to_graph(d), c); }, py::arg("data"), py::arg("c"));
    m.def("is_planar", [](const py::object& d) { return is_planar(to_graph(d)); }, py::arg("data"));

    // measures.py
    using Measure = double (*)(const Graph&, const Graph&);
    for (auto [name, f] : std::vector<std::pair<const char*, Measure>>{
             {"density", density},
             {"weight_entropy", weight_entropy},
             {"average_degree", average_degree},
             {"lcc_size", lcc_size},
             {"node_fraction", node_fraction},
             {"edge_fraction", edge_fraction},
             {"weight_fraction", weight_fraction},
             {"reachability", reachability},
             {"number_connected_components", number_connected_components},
             {"diameter", diameter},
             {"lcc_edge_fraction", lcc_edge_fraction},
             {"lcc_weight_fraction", lcc_weight_fraction},
             {"average_clustering_coefficient", average_clustering_coefficient},
             {"transitivity", transitivity}})
        m.def(name, f, py::arg("original"), py::arg("graph"));
    m.def("lcc_node_fraction", &lcc_node_fraction);
    m.def("weights", &weights);
    m.def("degrees", &degrees, py::arg("graph"), py::arg("weight") = "");

    // utils
    m.def("lcc", &lcc);
    m.def("cumulative_dist", &cumulative_dist, py::arg("values"), py::arg("increasing") = true);
    m.def("ks_statistic", &ks_statistic);

    // structural/distanceclosure
    m.def("metric_backbone", [](const py::object& d, const std::string& w) { return metric_backbone(to_graph(d), w); },
          py::arg("D"), py::arg("weight") = "weight");
    m.def("ultrametric_backbone",
          [](const py::object& d, const std::string& w) { return ultrametric_backbone(to_graph(d), w); }, py::arg("D"),
          py::arg("weight") = "weight");
    m.def("metric_backbone_distortion",
          [](const py::object& d, const std::string& w) { return metric_backbone_distortion(to_graph(d), w); },
          py::arg("D"), py::arg("weight") = "weight");
    m.def("distance_closure",
          [](const py::object& d, const std::string& kind, const std::string& algorithm, const std::string& weight,
             bool only_backbone) { return distance_closure(to_graph(d), kind, algorithm, weight, only_backbone); },
          py::arg("D"), py::arg("kind") = "metric", py::arg("algorithm") = "dijkstra", py::arg("weight") = "weight",
          py::arg("only_backbone") = false);
    m.def("s_values", &s_values, py::arg("Cm"), py::arg("weight_distance") = "distance",
          py::arg("weight_metric_distance") = "metric_distance");
    m.def("b_values", &b_values, py::arg("Cm"), py::arg("weight_distance") = "distance",
          py::arg("weight_metric_distance") = "metric_distance");
    m.def("pairwise_proximity", &pairwise_proximity, py::arg("M"), py::arg("metric") = "jaccard",
          py::arg("min_support") = -1);
    m.def("prox2dist", &prox2dist);
    m.def("dist2prox", &dist2prox);

    // compare.py
    py::class_<Compare>(m, "Compare")
        .def(py::init<>())
        .def("set_network", [](Compare& c, const py::object& d) { c.set_network(to_graph(d)); })
        .def("add_backbone", &Compare::add_backbone)
        .def("add_property", &Compare::add_property, py::arg("name"), py::arg("property"))
        .def("add_distribution", &Compare::add_distribution, py::arg("name"), py::arg("distribution"))
        .def("set_filter",
             [](Compare& c, const py::object& f, std::vector<double> values) { c.set_filter(filter_from(f), values); },
             py::arg("filter"), py::arg("values") = std::vector<double>{})
        .def("properties", [](Compare& c) { return table_to_pandas(c.properties()); })
        .def("properties_progression", [](Compare& c) {
            py::dict out;
            for (const auto& [name, t] : c.properties_progression()) out[py::str(name)] = table_to_pandas(t);
            return out;
        })
        .def("distribution_ks_statistic", [](Compare& c, bool increasing, std::vector<bool> consent) -> py::tuple {
            auto r = c.distribution_ks_statistic(increasing, consent);
            py::object ks = table_to_pandas(r.ks_statistics);
            if (r.consensual_backbone) return py::make_tuple(ks, r.dist, *r.consensual_backbone);
            return py::make_tuple(ks, r.dist);
        }, py::arg("increasing") = true, py::arg("consent") = std::vector<bool>{})
        .def("consent", &Compare::consent);
}
