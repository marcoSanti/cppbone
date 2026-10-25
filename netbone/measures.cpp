#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

#include "netbone/netbone.hpp"

namespace netbone {

std::vector<double> weights(const Graph& g) {
    std::vector<double> out;
    for (const auto& e : g.edges())
        if (!std::isnan(g.get(e.id, "weight"))) out.push_back(g.get(e.id, "weight"));
    return out;
}

std::vector<double> degrees(const Graph& g, const std::string& weight) {
    std::vector<double> out;
    for (int u = 0; u < g.number_of_nodes(); ++u) out.push_back(weight.empty() ? g.degree(u) : g.degree(u, weight));
    return out;
}

static double sum(const std::vector<double>& v) { return std::accumulate(v.begin(), v.end(), 0.0); }

double density(const Graph&, const Graph& graph) {
    double n = graph.number_of_nodes();
    double d = n <= 1 ? 0 : 2.0 * graph.number_of_edges() / (n * (n - 1));
    return std::round(d * 10000) / 10000;
}

double weight_entropy(const Graph&, const Graph& graph) {
    auto w = weights(graph);
    double total = sum(w), h = 0;
    for (double x : w)
        if (x > 0) h -= x / total * std::log2(x / total);
    return h;
}

double average_degree(const Graph&, const Graph& graph) {
    auto d = degrees(graph);
    return d.empty() ? NaN : sum(d) / d.size();
}

double lcc_size(const Graph&, const Graph& graph) { return lcc(graph).number_of_nodes(); }

double node_fraction(const Graph& network, const Graph& b) {
    return double(b.number_of_nodes()) / network.number_of_nodes();
}

double edge_fraction(const Graph& network, const Graph& b) {
    return double(b.number_of_edges()) / network.number_of_edges();
}

double weight_fraction(const Graph& network, const Graph& b) { return sum(weights(b)) / sum(weights(network)); }

double reachability(const Graph&, const Graph& g) {
    double r = 0, n = g.number_of_nodes();
    for (const auto& c : connected_components(g)) r += double(c.size()) * (c.size() - 1);
    return r / (n * (n - 1));
}

double number_connected_components(const Graph&, const Graph& g) { return connected_components(g).size(); }

double diameter(const Graph&, const Graph& g) {
    Graph l = lcc(g);  // unweighted, as igraph's diameter(directed=False, unconn=True)
    int n = l.number_of_nodes(), best = 0;
    std::vector<int> dist(n);
    for (int s = 0; s < n; ++s) {
        std::fill(dist.begin(), dist.end(), -1);
        std::vector<int> q{s};
        dist[s] = 0;
        for (size_t i = 0; i < q.size(); ++i)
            for (auto [v, id] : l.adj(q[i]))
                if (dist[v] < 0) dist[v] = dist[q[i]] + 1, best = std::max(best, dist[v]), q.push_back(v);
    }
    return best;
}

double lcc_node_fraction(const Graph& g) { return node_fraction(g, lcc(g)); }

double lcc_edge_fraction(const Graph&, const Graph& g) { return edge_fraction(g, lcc(g)); }

double lcc_weight_fraction(const Graph&, const Graph& g) { return weight_fraction(g, lcc(g)); }

// Neighbour pairs of u that are connected, counted once per unordered pair.
static double triangles(const Graph& g, int u) {
    std::set<int> nb;
    for (auto [v, id] : g.adj(u))
        if (v != u) nb.insert(v);
    double t = 0;
    for (int v : nb)
        for (auto [w, id] : g.adj(v))
            if (w != v && w != u && nb.count(w)) t += 1;
    return t / 2;
}

double average_clustering_coefficient(const Graph&, const Graph& g) {
    double s = 0;
    int count = 0;
    for (int u = 0; u < g.number_of_nodes(); ++u) {
        double k = g.degree(u);
        if (k < 2) continue;  // igraph mode="nan"; NaNs are dropped from the mean
        s += triangles(g, u) / (k * (k - 1) / 2);
        ++count;
    }
    return count ? s / count : NaN;
}

double transitivity(const Graph&, const Graph& graph) {
    double tri = 0, triads = 0;
    for (int u = 0; u < graph.number_of_nodes(); ++u) {
        double d = graph.degree(u);
        tri += 2 * triangles(graph, u);
        triads += d * (d - 1);
    }
    return triads == 0 ? 0 : tri / triads;
}

}  // namespace netbone
