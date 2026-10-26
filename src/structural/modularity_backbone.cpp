/// @file
/// @brief Modularity backbone (structural/modulairy_backbone.py).
///
/// Louvain communities (python-louvain best_partition), then each node's vitality = |Q - Q without
/// the node|.
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <ranges>

#include "../../include/cppbone/detail.hpp"

namespace cppbone {

namespace {

/// @brief Weighted graph on integer nodes with self-loops, the shape Louvain's induced graphs need.
struct WGraph {
    std::vector<std::vector<std::pair<int, double>>>
        adj;                   ///< (neighbour, weight) per node, self-loops excluded.
    std::vector<double> loops; ///< Self-loop weight per node.

    /// @brief Weighted degree of a node; a self-loop counts twice, like networkx.
    /// @param u Node.
    /// @return The weighted degree.
    [[nodiscard]] double degree(const int u) const {
        double d = 2 * loops[u];
        for (const double w : adj[u] | std::views::values) {
            d += w;
        }
        return d;
    }
    /// @brief Total edge weight (networkx size(weight)).
    /// @return Half the sum of the weighted degrees.
    [[nodiscard]] double size() const {
        double s = 0;
        for (const int u : std::views::iota(0, static_cast<int>(adj.size()))) {
            s += degree(u);
        }
        return s / 2;
    }
};

/// @brief Smallest modularity gain that keeps Louvain iterating (python-louvain __MIN).
constexpr double MIN_GAIN = 0.0000001;

/// @brief Louvain state of one level (python-louvain Status): node communities and per-community
/// totals.
struct Status {
    std::vector<int> node2com;     ///< Community of every node (-1 while a node is being moved).
    std::vector<double> degrees;   ///< Total weighted degree of every community.
    std::vector<double> gdegrees;  ///< Weighted degree of every node.
    std::vector<double> internals; ///< Weight inside every community.
    std::vector<double> loops;     ///< Self-loop weight of every node.
    double total_weight = 0;       ///< Total edge weight of the graph.

    /// @brief Puts every node in its own community.
    /// @param g Graph of this level.
    void init(const WGraph &g) {
        const int n = static_cast<int>(g.adj.size());
        node2com.resize(n);
        degrees.resize(n);
        gdegrees.resize(n);
        internals.resize(n);
        loops        = g.loops;
        total_weight = g.size();
        for (int u = 0; u < n; ++u) {
            node2com[u] = u;
            degrees[u] = gdegrees[u] = g.degree(u);
            internals[u]             = loops[u];
        }
    }

    /// @brief Modularity of the current partition.
    /// @return Sum over communities of internal / total - (degree / (2 total))^2.
    [[nodiscard]] double modularity() const {
        double result = 0;
        std::vector seen(degrees.size(), false);
        for (const int c : node2com) {
            if (seen[c]) {
                continue;
            }
            seen[c] = true;
            if (total_weight > 0) {
                result +=
                    internals[c] / total_weight - std::pow(degrees[c] / (2. * total_weight), 2);
            }
        }
        return result;
    }
};

/// @brief One Louvain level: moves nodes between neighbouring communities while the modularity
/// improves.
/// @param g Graph of this level.
/// @param st State, updated in place.
/// @param rng Generator for the random node and community orders.
void one_level(const WGraph &g, Status &st, std::mt19937 &rng) {
    const int n    = static_cast<int>(g.adj.size());
    auto order     = std::views::iota(0, n) | std::ranges::to<std::vector>();
    double new_mod = st.modularity();
    bool modified  = true;
    while (modified) {
        const double cur_mod = new_mod;
        modified             = false;
        std::shuffle(order.begin(), order.end(), rng);
        for (const int node : order) {
            const int com_node     = st.node2com[node];
            const double degc_totw = st.gdegrees[node] / (st.total_weight * 2.);
            std::map<int, double> neigh;
            for (const auto [v, w] : g.adj[node]) {
                neigh[st.node2com[v]] += w;
            }
            const double remove_cost =
                -neigh[com_node] + (st.degrees[com_node] - st.gdegrees[node]) * degc_totw;
            // __remove
            st.degrees[com_node] -= st.gdegrees[node];
            st.internals[com_node] -= neigh[com_node] + st.loops[node];
            st.node2com[node] = -1;

            std::vector<std::pair<int, double>> items(neigh.begin(), neigh.end());
            std::shuffle(items.begin(), items.end(), rng);
            int best_com         = com_node;
            double best_increase = 0;
            for (const auto [com, dnc] : items) {
                const double incr = remove_cost + dnc - st.degrees[com] * degc_totw;
                if (incr > best_increase) {
                    best_increase = incr, best_com = com;
                }
            }
            // __insert
            st.node2com[node] = best_com;
            st.degrees[best_com] += st.gdegrees[node];
            st.internals[best_com] += neigh[best_com] + st.loops[node];
            if (best_com != com_node) {
                modified = true;
            }
        }
        new_mod = st.modularity();
        if (new_mod - cur_mod < MIN_GAIN) {
            break;
        }
    }
}

/// @brief Renumbers communities 0..k-1 in order of first appearance.
/// @param node2com Community of every node.
/// @return The renumbered communities.
std::vector<int> renumber(const std::vector<int> &node2com) {
    std::map<int, int> ids;
    std::vector<int> out;
    out.reserve(node2com.size());
    for (const int c : node2com) {
        out.push_back(ids.try_emplace(c, static_cast<int>(ids.size())).first->second);
    }
    return out;
}

/// @brief Collapses every community into one node (python-louvain induced_graph).
/// @param part Community of every node, numbered 0..k-1.
/// @param g Graph to collapse.
/// @return Graph with one node per community; internal weight becomes self-loops.
WGraph induced_graph(const std::vector<int> &part, const WGraph &g) {
    const int k = part.empty() ? 0 : std::ranges::max(part) + 1;
    std::vector<std::map<int, double>> w(k);
    WGraph out;
    out.loops.assign(k, 0);
    for (size_t u = 0; u < g.adj.size(); ++u) {
        out.loops[part[u]] += g.loops[u];
        for (const auto [v, wt] : g.adj[u]) {
            if (static_cast<int>(u) < v) {
                const int a = part[u];
                if (const int b = part[v]; a == b) {
                    out.loops[a] += wt;
                } else {
                    w[a][b] += wt, w[b][a] += wt;
                }
            }
        }
    }
    out.adj.resize(k);
    for (int a = 0; a < k; ++a) {
        out.adj[a] = w[a] | std::ranges::to<std::vector<std::pair<int, double>>>();
    }
    return out;
}

/// @brief Louvain communities (python-louvain best_partition): the last level of the dendrogram.
/// @param graph Graph.
/// @param rng Generator for the random orders.
/// @return Community of every node, numbered 0..k-1.
std::vector<int> best_partition(const WGraph &graph, std::mt19937 &rng) {
    const auto n   = static_cast<int>(graph.adj.size());
    auto partition = std::views::iota(0, n) | std::ranges::to<std::vector>();
    if (graph.size() == 0) {
        return partition;
    }

    WGraph current = graph;
    Status st;
    st.init(current);
    one_level(current, st, rng);
    double mod = st.modularity();
    auto level = renumber(st.node2com);
    partition  = level;
    current    = induced_graph(level, current);
    st.init(current);
    while (true) {
        one_level(current, st, rng);
        const double new_mod = st.modularity();
        if (new_mod - mod < MIN_GAIN) {
            break;
        }
        level = renumber(st.node2com);
        for (int &c : partition) {
            c = level[c];
        }
        mod     = new_mod;
        current = induced_graph(level, current);
        st.init(current);
    }
    return partition;
}

} // namespace

Backbone modularity_backbone(const Graph &data, std::uint32_t seed) {
    // shortcut: std::mt19937 instead of numpy's RandomState(123), so the partition can differ from
    // Python's.
    Graph g     = data;
    const int n = g.number_of_nodes();
    WGraph wg;
    wg.adj.resize(n);
    wg.loops.assign(n, 0);
    const auto weight = g.edge_values("weight");
    for (const auto &[u, v, id] : g.edges()) {
        const double w = weight[id];
        if (u == v) {
            wg.loops[u] += w;
        } else {
            wg.adj[u].emplace_back(v, w);
            wg.adj[v].emplace_back(u, w);
        }
    }
    std::mt19937 rng(seed);
    const auto membership = best_partition(wg, rng);
    const int k           = membership.empty() ? 0 : std::ranges::max(membership) + 1;

    // community.modularity(partition, g) with weights
    const double links = wg.size();
    std::vector<double> inc(k, 0), deg(k, 0);
    for (const int u : g.nodes()) {
        deg[membership[u]] += wg.degree(u);
        inc[membership[u]] += wg.loops[u];
        for (const auto [v, w] : wg.adj[u]) {
            if (membership[v] == membership[u]) {
                inc[membership[u]] += w / 2;
            }
        }
    }
    double modularity_value = 0;
    for (int c = 0; c < k; ++c) {
        modularity_value += inc[c] / links - std::pow(deg[c] / (2. * links), 2);
    }

    // Sparse-matrix algebra of the Python version, written per node. As there, m counts edges
    // (weight_key=None) while the adjacency matrix A is weighted.
    const double m = g.number_of_edges();
    std::vector<std::map<int, double>> node_deg_by_group(n); // sparse rows of A * group indicator
    std::vector<double> diag(n, 0);
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) {
        diag[u] = wg.loops[u];
        node_deg_by_group[u][membership[u]] += wg.loops[u];
        for (const auto [v, w] : wg.adj[u]) {
            node_deg_by_group[u][membership[v]] += w;
        }
    }
    double internal_edges = 0;
    std::vector<double> degrees(n, 0);
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) {
        for (const double x : node_deg_by_group[u] | std::views::values) {
            degrees[u] += x;
        }
    }
    for (const int u : g.nodes()) {
        internal_edges += node_deg_by_group[u][membership[u]] / 2;
    }
    std::vector<double> group_degs(k, 0);
    for (const int u : g.nodes()) {
        node_deg_by_group[u][membership[u]] += degrees[u];
        group_degs[membership[u]] += degrees[u] + diag[u];
    }
    double sum_group_degs_sq = 0;
    for (const double x : group_degs) {
        sum_group_degs_sq += x * x;
    }

#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < n; ++u) {
        const double internal_deg = node_deg_by_group[u][membership[u]] - degrees[u];
        const double q1_links     = (internal_edges - internal_deg) / (m - degrees[u]);
        double dot = 0, sq = 0;
        for (const auto [c, x] : node_deg_by_group[u]) {
            dot += x * group_degs[c];
            sq += x * x;
        }
        const double expected_impact = sum_group_degs_sq - 2 * dot + sq;
        const double q1_degrees      = expected_impact / (4 * std::pow(m - degrees[u], 2));
        g.node_set(u, "vitality", std::abs(modularity_value - (q1_links - q1_degrees)));
    }
    return {.graph         = std::move(g),
            .method_name   = "Modularity Filter",
            .property_name = "vitality",
            .filters       = {threshold_filter, fraction_filter},
            .filter_on     = FilterOn::Nodes};
}

} // namespace cppbone
