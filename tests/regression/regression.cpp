/// @file
/// @brief Regression runner: computes every case on a network and checks it against stored baselines.
///
/// Every public function and every method/filter combination is turned into a canonical table (see Table) and
/// compared with two baselines: `baseline/cpp` (a snapshot of this library, to catch regressions) and
/// `baseline/python` (written by python_baseline.py with the original NetBone, to check the port). Allowed
/// differences come from `tolerances.csv`; threshold_filter values from `thresholds.csv`.
///
/// Usage: `cppbone_regression EDGES_CSV REGRESSION_DIR [--write-cpp-baseline] [--only SUBSTRING] [--verbose]`

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "cppbone/cppbone.hpp"

namespace fs = std::filesystem;
using namespace cppbone;

namespace {

/// @brief Canonical form of one case: named numeric columns, rows keyed by a string and kept sorted by key.
///
/// Edge keys are "a|b" with a < b; node keys are the label ("#label" when listed next to edges); table rows use
/// their index label. A table without columns is a plain set of keys (a filtered graph).
struct Table {
    std::vector<std::string> cols;                 ///< Column names, after the key column.
    std::map<std::string, std::vector<double>> rows;  ///< Values by key, in column order.
};

/// @brief Key of an edge: its endpoint labels in sorted order.
/// @param a One label.
/// @param b The other label.
/// @return "a|b" with the smaller label first.
std::string pair_key(std::string a, std::string b) {
    if (b < a) std::swap(a, b);
    return a + "|" + b;
}

/// @brief Edge attributes as a table.
/// @param g Graph.
/// @param cols Attributes to include (missing values become NaN).
/// @return One row per edge.
Table edge_table(const Graph& g, const std::vector<std::string>& cols) {
    Table t{cols, {}};
    for (const auto& e : g.edges()) {
        std::vector<double> row;
        for (const auto& c : cols) row.push_back(g.get(e.id, c));
        t.rows[pair_key(g.label(e.u), g.label(e.v))] = row;
    }
    return t;
}

/// @brief Node attributes as a table.
/// @param g Graph.
/// @param cols Attributes to include (missing values become NaN).
/// @return One row per node.
Table node_table(const Graph& g, const std::vector<std::string>& cols) {
    Table t{cols, {}};
    for (int u = 0; u < g.number_of_nodes(); ++u) {
        std::vector<double> row;
        for (const auto& c : cols) row.push_back(g.node_get(u, c));
        t.rows[g.label(u)] = row;
    }
    return t;
}

/// @brief Structure of a graph as a set of keys: every edge ("a|b") and every node ("#label").
/// @param g Graph.
/// @return Table without columns.
Table graph_table(const Graph& g) {
    Table t;
    for (const auto& e : g.edges()) t.rows[pair_key(g.label(e.u), g.label(e.v))] = {};
    for (int u = 0; u < g.number_of_nodes(); ++u) t.rows["#" + g.label(u)] = {};
    return t;
}

/// @brief Converts a Compare table: one row per index label, the same columns.
/// @param in Table returned by Compare.
/// @param prefix Prepended to every key (used to merge several tables into one).
/// @param out Table to fill; its columns are set from `in`.
void add_compare_table(const cppbone::Table& in, const std::string& prefix, Table& out) {
    out.cols = in.columns;
    for (size_t r = 0; r < in.index.size(); ++r) out.rows[prefix + in.index[r]] = in.data[r];
}

/// @brief Writes a table as canonical CSV: "key,<columns>", rows in key order, NaN as an empty field.
/// @param path Destination file.
/// @param t Table.
/// @throws std::runtime_error if the file cannot be written or a key contains a comma.
void write_table(const fs::path& path, const Table& t) {
    for (const auto& [key, row] : t.rows)
        if (key.find(',') != std::string::npos) throw std::runtime_error("comma in key '" + key + "'");
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path.string());
    out << "key";
    for (const auto& c : t.cols) out << ',' << c;
    out << '\n';
    for (const auto& [key, row] : t.rows) {
        out << key;
        for (double v : row) out << ',' << format_double(v);
        out << '\n';
    }
}

/// @brief Splits a CSV line on commas (keys and labels never contain commas).
/// @param line Line.
/// @return The fields, including a trailing empty one.
std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) out.push_back(cell);
    if (!line.empty() && line.back() == ',') out.emplace_back();
    return out;
}

/// @brief Reads a canonical CSV written by write_table or python_baseline.py.
/// @param path File.
/// @return The table, or nothing if the file does not exist.
/// @throws std::runtime_error for a malformed row.
std::optional<Table> read_table(const fs::path& path) {
    std::ifstream in(path);
    if (!in) return std::nullopt;
    Table t;
    std::string line;
    std::getline(in, line);
    auto header = split(line);
    t.cols.assign(header.begin() + 1, header.end());
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto cells = split(line);
        if (cells.size() != t.cols.size() + 1) throw std::runtime_error("bad row in " + path.string() + ": " + line);
        std::vector<double> row;
        for (size_t i = 1; i < cells.size(); ++i) row.push_back(cells[i].empty() ? NaN : std::strtod(cells[i].c_str(), nullptr));
        t.rows[cells[0]] = row;
    }
    return t;
}

/// @brief Reads a two-column CSV (with header) into a map.
/// @param path File.
/// @return First column to second column; empty if the file is missing.
std::map<std::string, double> read_map(const fs::path& path) {
    std::map<std::string, double> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto cells = split(line);
        if (cells.size() >= 2 && cells[0] != "method") out[cells[0]] = std::strtod(cells[1].c_str(), nullptr);
    }
    return out;
}

/// @brief Allowed differences for the cases matching a pattern.
struct Tolerance {
    std::string baseline;    ///< "cpp" or "python".
    std::string pattern;     ///< Case name pattern; '*' matches any run of characters.
    double rel = 0;          ///< Largest relative difference of numeric values.
    double keys = 0;         ///< Largest number of keys present in only one of the two tables.
    double flags = 0;        ///< Largest number of differing in_backbone / is_* values.
};

/// @brief Glob match with '*' as the only wildcard.
/// @param pattern Pattern.
/// @param s Text.
/// @return true if s matches the whole pattern.
bool glob(const std::string& pattern, const std::string& s) {
    size_t p = 0, t = 0, star = std::string::npos, mark = 0;
    while (t < s.size()) {
        if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            mark = t;
        } else if (p < pattern.size() && pattern[p] == s[t]) {
            ++p, ++t;
        } else if (star != std::string::npos) {
            p = star + 1;
            t = ++mark;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') ++p;
    return p == pattern.size();
}

/// @brief Reads tolerances.csv: "baseline,pattern,rel_tol,max_key_diff,max_flag_diff", '#' for comments.
/// @param path File.
/// @return The rules, in file order (the last matching rule wins).
std::vector<Tolerance> read_tolerances(const fs::path& path) {
    std::vector<Tolerance> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto c = split(line);
        if (c.size() < 5 || c[0] == "baseline") continue;
        out.push_back({c[0], c[1], std::strtod(c[2].c_str(), nullptr), std::strtod(c[3].c_str(), nullptr),
                       std::strtod(c[4].c_str(), nullptr)});
    }
    return out;
}

/// @brief Finds the rule for a case.
/// @param rules All rules.
/// @param baseline "cpp" or "python".
/// @param name Case name.
/// @return The last matching rule, or a zero tolerance.
Tolerance tolerance_for(const std::vector<Tolerance>& rules, const std::string& baseline, const std::string& name) {
    Tolerance t{baseline, "", 0, 0, 0};
    for (const auto& r : rules)
        if (r.baseline == baseline && glob(r.pattern, name)) t = r;
    return t;
}

/// @brief Differences between a result and its baseline.
struct Diff {
    int keys = 0;          ///< Keys present in only one table.
    int flags = 0;         ///< Differing in_backbone / is_* values.
    double rel = 0;        ///< Largest relative difference of the other columns (inf for NaN against a number).
    std::string worst;     ///< Where `rel` was found.
    std::string columns;   ///< Columns present in only one table.
};

/// @brief Relative difference of two values; equal NaNs and equal infinities count as identical.
///
/// The denominator is floored at 1e-300 so that values in the denormal range (p-values underflowing to ~1e-320 on
/// both sides) do not count as completely different.
/// @param a First value.
/// @param b Second value.
/// @return |a - b| / max(|a|, |b|, 1e-300), 0 when equal, inf when only one is NaN or infinite.
double rel_diff(double a, double b) {
    if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b) ? 0 : INF;
    if (a == b) return 0;
    if (std::isinf(a) || std::isinf(b)) return INF;
    return std::abs(a - b) / std::max({std::abs(a), std::abs(b), 1e-300});
}

/// @brief Compares a result with its baseline, matching rows by key and columns by name.
/// @param got Result.
/// @param want Baseline.
/// @return The differences.
Diff compare(const Table& got, const Table& want) {
    Diff d;
    for (const auto& c : got.cols)
        if (std::find(want.cols.begin(), want.cols.end(), c) == want.cols.end()) d.columns += " +" + c;
    for (const auto& c : want.cols)
        if (std::find(got.cols.begin(), got.cols.end(), c) == got.cols.end()) d.columns += " -" + c;
    for (const auto& [key, row] : got.rows)
        if (!want.rows.count(key)) ++d.keys;
    for (const auto& [key, row] : want.rows) {
        auto it = got.rows.find(key);
        if (it == got.rows.end()) {
            ++d.keys;
            continue;
        }
        for (size_t j = 0; j < want.cols.size(); ++j) {
            auto c = std::find(got.cols.begin(), got.cols.end(), want.cols[j]);
            if (c == got.cols.end()) continue;
            double a = it->second[c - got.cols.begin()], b = row[j];
            const std::string& name = want.cols[j];
            if (name == "in_backbone" || name.rfind("is_", 0) == 0) {
                bool fa = !std::isnan(a) && a != 0, fb = !std::isnan(b) && b != 0;
                d.flags += fa != fb;
            } else {
                double r = rel_diff(a, b);
                if (r > d.rel) {
                    d.rel = r;
                    d.worst = key + " " + name + ": " + format_double(a) + " vs " + format_double(b);
                }
            }
        }
    }
    return d;
}

/// @brief Collects the cases in run order.
struct Cases {
    std::vector<std::pair<std::string, Table>> list;  ///< (case name, table).
    std::string only;                                 ///< Run only cases whose name contains this.

    /// @brief Tells whether a case (or group of cases) is selected.
    /// @param name Case name or prefix.
    /// @return true if no filter is set or the name contains it.
    bool wanted(const std::string& name) const { return only.empty() || name.find(only) != std::string::npos; }

    /// @brief Adds a case if it is selected.
    /// @param name Case name (also the baseline file name, without ".csv").
    /// @param t Its table.
    void add(const std::string& name, Table t) {
        if (wanted(name)) list.emplace_back(name, std::move(t));
    }
};

/// @brief Builds the complete graph on n nodes.
/// @param n Number of nodes.
/// @return The graph, labels "0".."n-1", weight 1.
Graph complete(int n) {
    Graph g;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) g.add_edge(std::to_string(i), std::to_string(j), {{"weight", 1}});
    return g;
}

/// @brief The induced subgraph on the n smallest labels, with distance = 1 / weight (the distance-closure input).
/// @param g Network.
/// @param n Number of nodes kept.
/// @return The small distance graph.
Graph small_distance_graph(const Graph& g, int n) {
    std::vector<std::string> labels;
    for (int u = 0; u < g.number_of_nodes(); ++u) labels.push_back(g.label(u));
    std::sort(labels.begin(), labels.end());
    std::set<std::string> chosen(labels.begin(), labels.begin() + std::min<size_t>(n, labels.size()));
    std::vector<bool> keep(g.number_of_nodes());
    for (int u = 0; u < g.number_of_nodes(); ++u) keep[u] = chosen.count(g.label(u)) > 0;
    Graph s = g.subgraph(keep);
    for (const auto& e : s.edges()) s.set(e.id, "distance", 1 / s.get(e.id, "weight"));
    return s;
}

/// @brief The deterministic matrix used for the pairwise_proximity cases.
/// @return 20 x 15 matrix with entries (7 i + 3 j) mod 5.
Matrix proximity_input() {
    Matrix m;
    for (int i = 0; i < 20; ++i) {
        m.emplace_back();
        for (int j = 0; j < 15; ++j) m.back().push_back((i * 7 + j * 3) % 5);
    }
    return m;
}

/// @brief Formats a matrix index pair as a sortable key.
/// @param i Row.
/// @param j Column.
/// @return "ii|jj", zero-padded.
std::string index_key(int i, int j) {
    std::ostringstream s;
    s << std::setw(2) << std::setfill('0') << i << '|' << std::setw(2) << std::setfill('0') << j;
    return s.str();
}

/// @brief Computes every case.
/// @param g Network.
/// @param thresholds threshold_filter value per method name.
/// @param cases Receives the tables.
void run_all(const Graph& g, const std::map<std::string, double>& thresholds, Cases& cases) {
    // ---- methods and their filters ----
    std::vector<std::pair<std::string, std::function<Backbone()>>> methods = {
        {"disparity", [&] { return disparity(g); }},
        {"lans", [&] { return lans(g); }},
        {"noise_corrected", [&] { return noise_corrected(g); }},
        {"noise_corrected_exact", [&] { return noise_corrected(g, false); }},
        {"marginal_likelihood", [&] { return marginal_likelihood(g); }},
        {"ecm", [&] { return ecm(g); }},
        {"betweenness", [&] { return betweenness(g); }},
        {"betweenness_unweighted", [&] { return betweenness(g, false, true); }},
        {"betweenness_raw", [&] { return betweenness(g, true, false); }},
        {"degree", [&] { return degree(g); }},
        {"degree_weighted", [&] { return degree(g, true); }},
        {"doubly_stochastic", [&] { return doubly_stochastic(g); }},
        {"global_threshold", [&] { return global_threshold(g); }},
        {"gspar", [&] { return gspar(g); }},
        {"h_backbone", [&] { return h_backbone(g); }},
        {"high_salience_skeleton", [&] { return high_salience_skeleton(g); }},
        {"maximum_spanning_tree", [&] { return maximum_spanning_tree(g); }},
        {"metric_distance_backbone", [&] { return metric_distance_backbone(g); }},
        {"ultrametric_distance_backbone", [&] { return ultrametric_distance_backbone(g); }},
        {"mlam", [&] { return mlam(g); }},
        {"plam", [&] { return plam(g); }},
        {"modularity_backbone", [&] { return modularity_backbone(g); }},
        {"pmfg", [&] { return pmfg(g); }},
        {"glanb_c1", [&] { return glanb(g, 1); }},
        {"glanb_c0.5", [&] { return glanb(g, 0.5); }},
    };
    std::map<std::string, Backbone> kept;  // backbones reused by the comparison cases
    for (auto& [name, run] : methods) {
        if (!cases.wanted("method." + name) && !cases.wanted("compare") && !cases.wanted("measures")) continue;
        Backbone b = run();
        std::string prefix = "method." + name;
        if (name == "noise_corrected_exact") {
            cases.add(prefix + ".scores", edge_table(b.graph, {"score", "nc_sdev"}));
            continue;  // no p_value to filter on
        }
        std::vector<std::string> cols{b.property_name};
        if (b.filter_on == FilterOn::Nodes) {
            cases.add(prefix + ".scores", node_table(b.graph, cols));
        } else {
            bool flagged = false;
            for (const auto& e : b.graph.edges()) flagged |= !std::isnan(b.graph.get(e.id, "in_backbone"));
            if (flagged && b.property_name != "in_backbone") cols.push_back("in_backbone");
            cases.add(prefix + ".scores", edge_table(b.graph, cols));
        }
        for (Filter f : b.compatible_filters()) {
            if (f == boolean_filter) {
                cases.add(prefix + ".boolean", graph_table(boolean_filter(b, 0, false)));
            } else if (f == threshold_filter) {
                auto it = thresholds.find(name);
                if (it == thresholds.end()) throw std::runtime_error("no threshold for " + name + " in thresholds.csv");
                cases.add(prefix + ".threshold_" + format_double(it->second),
                          graph_table(threshold_filter(b, it->second, false)));
            } else if (f == fraction_filter) {
                for (double v : {0.05, 0.2})
                    cases.add(prefix + ".fraction_" + format_double(v), graph_table(fraction_filter(b, v, false)));
            }
        }
        kept.emplace(name, std::move(b));
    }

    // ---- measures on the original network and three backbones ----
    if (cases.wanted("measures")) {
        std::vector<std::pair<std::string, Graph>> graphs = {
            {"original", g},
            {"disparity_threshold_0.05", threshold_filter(kept.at("disparity"), 0.05, false)},
            {"high_salience_skeleton_boolean", boolean_filter(kept.at("high_salience_skeleton"), 0, false)},
            {"maximum_spanning_tree_boolean", boolean_filter(kept.at("maximum_spanning_tree"), 0, false)},
        };
        std::vector<std::pair<std::string, Property>> props = {
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
            {"lcc_node_fraction", [](const Graph&, const Graph& x) { return lcc_node_fraction(x); }},
            {"lcc_edge_fraction", lcc_edge_fraction},
            {"lcc_weight_fraction", lcc_weight_fraction},
            {"average_clustering_coefficient", average_clustering_coefficient},
            {"transitivity", transitivity},
        };
        Table t;
        for (const auto& gr : graphs) t.cols.push_back(gr.first);
        for (const auto& [pname, prop] : props) {
            std::vector<double> row;
            for (const auto& gr : graphs) row.push_back(prop(g, gr.second));
            t.rows[pname] = row;
        }
        cases.add("measures", t);
    }

    // ---- degrees and their cumulative distributions ----
    {
        Table t{{"degree", "weighted_degree"}, {}};
        auto d = degrees(g), w = degrees(g, "weight");
        for (int u = 0; u < g.number_of_nodes(); ++u) t.rows[g.label(u)] = {d[u], w[u]};
        cases.add("utils.degrees", t);
        for (bool increasing : {true, false}) {
            Table c{{"x", "y"}, {}};
            auto pts = cumulative_dist(d, increasing);
            for (size_t i = 0; i < pts.size(); ++i) c.rows[index_key((int)i / 100, (int)i % 100)] = {pts[i].first, pts[i].second};
            cases.add(std::string("utils.cumulative_dist_") + (increasing ? "increasing" : "decreasing"), c);
        }
    }

    // ---- Compare ----
    if (cases.wanted("compare")) {
        std::vector<std::pair<std::string, Property>> props = {
            {"density", density},           {"edge_fraction", edge_fraction},
            {"node_fraction", node_fraction}, {"weight_fraction", weight_fraction},
            {"lcc_size", lcc_size},         {"number_connected_components", number_connected_components},
        };
        {  // properties with the boolean filter
            Compare cmp;
            cmp.set_network(g);
            for (const char* m : {"high_salience_skeleton", "maximum_spanning_tree", "plam"}) cmp.add_backbone(kept.at(m));
            for (const auto& [n, p] : props) cmp.add_property(n, p);
            cmp.set_filter(boolean_filter);
            Table t;
            add_compare_table(cmp.properties(), "", t);
            cases.add("compare.properties", t);
            cases.add("compare.consent", graph_table(cmp.consent()));
        }
        {  // progression with the fraction filter
            Compare cmp;
            cmp.set_network(g);
            for (const char* m : {"disparity", "lans", "high_salience_skeleton"}) cmp.add_backbone(kept.at(m));
            for (const auto& [n, p] : props) cmp.add_property(n, p);
            cmp.set_filter(fraction_filter, {0.05, 0.1, 0.2});
            Table t;
            for (const auto& [pname, table] : cmp.properties_progression()) add_compare_table(table, pname + "|", t);
            cases.add("compare.properties_progression", t);
        }
        {  // KS statistics with a consensual backbone
            Compare cmp;
            cmp.set_network(g);
            for (const char* m : {"disparity", "noise_corrected", "lans"}) cmp.add_backbone(kept.at(m));
            cmp.add_distribution("degree", [](const Graph& x) { return degrees(x); });
            cmp.add_distribution("weighted_degree", [](const Graph& x) { return degrees(x, "weight"); });
            cmp.add_distribution("weight", [](const Graph& x) { return weights(x); });
            cmp.set_filter(fraction_filter, {0.1, 0.1, 0.1});
            auto ks = cmp.distribution_ks_statistic(true, {true, true, true});
            Table t;
            add_compare_table(ks.ks_statistics, "", t);
            cases.add("compare.ks_statistic", t);
            if (ks.consensual_backbone) cases.add("compare.ks_consensual_backbone", graph_table(*ks.consensual_backbone));
        }
    }

    // ---- distance closure toolkit ----
    if (cases.wanted("distanceclosure")) {
        Graph small = small_distance_graph(g, 60);
        Graph cm = distance_closure(small, "metric", "dijkstra", "distance");
        cases.add("distanceclosure.closure_metric", edge_table(cm, {"distance", "metric_distance", "is_metric"}));
        Graph cu = distance_closure(small, "ultrametric", "dijkstra", "distance");
        cases.add("distanceclosure.closure_ultrametric",
                  edge_table(cu, {"distance", "ultrametric_distance", "is_ultrametric"}));
        cases.add("distanceclosure.s_values", edge_table(s_values(cm, "distance", "metric_distance"), {"s-value"}));
        {  // b-values, oriented by label so the table does not depend on edge orientation
            Graph bv = b_values(cm, "distance", "metric_distance");
            Table t{{"b_low", "b_high"}, {}};
            for (const auto& e : bv.edges()) {
                double ij = bv.get(e.id, "b_ij-value"), ji = bv.get(e.id, "b_ji-value");
                if (std::isnan(ij) && std::isnan(ji)) continue;
                bool ordered = bv.label(e.u) < bv.label(e.v);
                t.rows[pair_key(bv.label(e.u), bv.label(e.v))] = ordered ? std::vector<double>{ij, ji}
                                                                         : std::vector<double>{ji, ij};
            }
            cases.add("distanceclosure.b_values", t);
        }
        for (auto [name, disj] : {std::pair{"sum", Disjunction::Sum}, std::pair{"max", Disjunction::Max}}) {
            auto d = all_pairs_dijkstra_path_length(small, "distance", disj);
            Table t{{"length"}, {}};
            for (int u = 0; u < small.number_of_nodes(); ++u)
                for (int v = 0; v < small.number_of_nodes(); ++v)
                    if (d[u][v] < INF) t.rows[small.label(u) + ">" + small.label(v)] = {d[u][v]};
            cases.add(std::string("distanceclosure.all_pairs_dijkstra_") + name, t);
        }
        Graph full = g;
        for (const auto& e : full.edges()) full.set(e.id, "distance", 1 / full.get(e.id, "weight"));
        cases.add("distanceclosure.metric_backbone", graph_table(metric_backbone(full, "distance")));
        cases.add("distanceclosure.ultrametric_backbone", graph_table(ultrametric_backbone(full, "distance")));
        {
            auto [b, sv] = metric_backbone_distortion(full, "distance");
            Table t{{"s"}, {}};
            for (const auto& [e, s] : sv) t.rows[pair_key(e.first, e.second)] = {s};
            cases.add("distanceclosure.metric_backbone_distortion", t);
        }
        Matrix m = proximity_input();
        for (const char* metric : {"jaccard", "jaccard_binary", "jaccard_set", "jaccard_weighted"}) {
            Matrix p = pairwise_proximity(m, metric);
            Table t{{"proximity"}, {}};
            for (size_t i = 0; i < p.size(); ++i)
                for (size_t j = 0; j < p.size(); ++j) t.rows[index_key((int)i, (int)j)] = {p[i][j]};
            cases.add(std::string("distanceclosure.pairwise_proximity_") + metric, t);
        }
        cases.add("utils.lcc", graph_table(lcc(small)));
    }

    // ---- scalar functions ----
    {
        Table t{{"value"}, {}};
        t.rows["prox2dist(0)"] = {prox2dist(0)};
        t.rows["prox2dist(0.25)"] = {prox2dist(0.25)};
        t.rows["prox2dist(1)"] = {prox2dist(1)};
        t.rows["dist2prox(0)"] = {dist2prox(0)};
        t.rows["dist2prox(3)"] = {dist2prox(3)};
        t.rows["dist2prox(inf)"] = {dist2prox(INF)};
        t.rows["jaccard([1;2;3];[2;3;4])"] = {jaccard({1, 2, 3}, {2, 3, 4})};
        t.rows["jaccard([1;1;2];[2])"] = {jaccard({1, 1, 2}, {2})};
        Graph k33;
        for (const char* a : {"a1", "a2", "a3"})
            for (const char* b : {"b1", "b2", "b3"}) k33.add_edge(a, b);
        t.rows["is_planar(K4)"] = {double(is_planar(complete(4)))};
        t.rows["is_planar(K5)"] = {double(is_planar(complete(5)))};
        t.rows["is_planar(K3_3)"] = {double(is_planar(k33))};
        t.rows["is_planar(network)"] = {double(is_planar(g))};
        t.rows["is_planar(mst)"] = {double(is_planar(boolean_filter(maximum_spanning_tree(g), 0, false)))};
        for (auto [k, n, p] : {std::tuple{2., 10., 0.3}, {0., 10., 0.5}, {10., 10., 0.5}, {5., 100., 0.01},
                               {100., 620000000., 3e-7}, {1., 620000000., 1e-9}, {3000., 10000., 0.25}}) {
            std::string args = "(" + format_double(k) + ";" + format_double(n) + ";" + format_double(p) + ")";
            t.rows["binom_cdf" + args] = {binom_cdf(k, n, p)};
            t.rows["binom_sf" + args] = {binom_sf(k, n, p)};
        }
        // compute_pvalue on the first edges in key order, from the endpoint with the smaller label
        std::vector<std::pair<std::string, int>> edge_keys;
        for (const auto& e : g.edges()) edge_keys.emplace_back(pair_key(g.label(e.u), g.label(e.v)), e.id);
        std::sort(edge_keys.begin(), edge_keys.end());
        for (size_t i = 0; i < std::min<size_t>(5, edge_keys.size()); ++i) {
            auto [u, v] = g.ends(edge_keys[i].second);
            int node = g.label(u) < g.label(v) ? u : v;
            t.rows["compute_pvalue(" + edge_keys[i].first + ")"] = {compute_pvalue(g, node, g.get(edge_keys[i].second, "weight"))};
        }
        cases.add("misc.scalars", t);
    }
}

}  // namespace

/// @brief Computes every case, then writes the C++ baseline or checks both baselines.
/// @param argc Argument count.
/// @param argv See the file description.
/// @return 0 when every case is within tolerance (or the baseline was written), 1 otherwise, 2 on bad usage.
int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    bool write = false, verbose = false;
    std::string only;
    std::vector<std::string> pos;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--write-cpp-baseline") write = true;
        else if (args[i] == "--verbose") verbose = true;
        else if (args[i] == "--only" && i + 1 < args.size()) only = args[++i];
        else pos.push_back(args[i]);
    }
    if (pos.size() != 2) {
        std::cerr << "usage: " << argv[0]
                  << " <edges.csv> <regression dir> [--write-cpp-baseline] [--only SUBSTRING] [--verbose]\n";
        return 2;
    }
    fs::path dir = pos[1];
    try {
        Graph g = Graph::read_csv(pos[0]);
        Cases cases;
        cases.only = only;
        run_all(g, read_map(dir / "thresholds.csv"), cases);

        if (write) {
            fs::create_directories(dir / "baseline" / "cpp");
            for (const auto& [name, t] : cases.list) write_table(dir / "baseline" / "cpp" / (name + ".csv"), t);
            std::cout << "wrote " << cases.list.size() << " cases to " << (dir / "baseline" / "cpp").string() << '\n';
            return 0;
        }

        // One line per case: "[ ok ] name   cpp: <verdict>   python: <verdict>", then one indented line per
        // baseline that is not identical, saying how it differs.
        auto rules = read_tolerances(dir / "tolerances.csv");
        size_t width = 0;
        for (const auto& c : cases.list) width = std::max(width, c.first.size());
        int failures = 0;
        std::map<std::string, std::map<std::string, int>> tally;  // baseline -> verdict -> cases
        for (const auto& [name, t] : cases.list) {
            bool failed = false;
            std::string verdicts, details;
            for (const std::string base : {"cpp", "python"}) {
                auto want = read_table(dir / "baseline" / base / (name + ".csv"));
                std::string verdict;
                if (!want) {
                    verdict = base == "cpp" ? "MISSING" : "none";
                    failed |= base == "cpp";
                    if (base == "cpp") details += "\n         cpp: no baseline file (run with --write-cpp-baseline)";
                } else {
                    Diff d = compare(t, *want);
                    Tolerance tol = tolerance_for(rules, base, name);
                    bool ok = d.columns.empty() && d.keys <= tol.keys && d.flags <= tol.flags && d.rel <= tol.rel;
                    bool identical = d.columns.empty() && d.keys == 0 && d.flags == 0 && d.rel == 0;
                    verdict = ok ? (identical ? "identical" : "ok") : "FAIL";
                    failed |= !ok;
                    if (!identical) {
                        std::ostringstream s;
                        s << "\n         " << base << ": ";
                        if (!d.columns.empty()) s << "columns" << d.columns << "; ";
                        if (d.keys) s << d.keys << " keys on one side only; ";
                        if (d.flags) s << d.flags << " flags differ; ";
                        if (d.rel > 0) s << "max rel diff " << std::setprecision(2) << d.rel << " at " << d.worst;
                        details += s.str();
                    }
                }
                ++tally[base][verdict];
                std::ostringstream v;
                v << "   " << base << ": " << std::left << std::setw(base == "cpp" ? 9 : 0) << verdict;
                verdicts += v.str();
            }
            failures += failed;
            if (failed || verbose)
                std::cout << (failed ? "[FAIL] " : "[ ok ] ") << std::left << std::setw((int)width) << name << verdicts
                          << details << '\n';
        }
        std::cout << '\n' << cases.list.size() << " cases, " << failures << " failed\n";
        for (const auto& [base, counts] : tally) {
            std::cout << "  vs " << base << ":";
            for (const auto& [verdict, n] : counts) std::cout << ' ' << n << ' ' << verdict;
            std::cout << '\n';
        }
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 2;
    }
}
