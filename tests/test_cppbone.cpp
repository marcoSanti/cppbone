/// @file
/// @brief Assert-based self-checks of the library. Optional argv[1]: an edge-list CSV to smoke-test every method on.
#undef NDEBUG
#include <cassert>
#include <cmath>
#include <format>
#include <print>

#include "cppbone/cppbone.hpp"

using namespace cppbone;

/// @brief Compares two doubles with a relative tolerance.
/// @param a Value.
/// @param b Reference.
/// @param tol Tolerance, relative to 1 + |b|.
/// @return true if |a - b| <= tol (1 + |b|).
static bool near(double a, double b, double tol = 1e-9) { return std::abs(a - b) <= tol * (1 + std::abs(b)); }

/// @brief Builds the complete graph on n nodes labelled "0".."n-1", every edge of weight 1.
/// @param n Number of nodes.
/// @return The graph.
static Graph complete(int n) {
    Graph g;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) g.add_edge(std::to_string(i), std::to_string(j), {{"weight", 1}});
    return g;
}

/// @brief Runs the self-checks, then the smoke test when a CSV is given.
/// @param argc Argument count.
/// @param argv argv[1], if present, is the CSV to smoke-test.
/// @return 0 when every check passes (a failed check aborts).
int main(int argc, char** argv) {
    // planarity
    assert(is_planar(complete(4)));
    assert(!is_planar(complete(5)));
    Graph k33;
    for (const char* a : {"a", "b", "c"})
        for (const char* b : {"x", "y", "z"}) k33.add_edge(a, b);
    assert(!is_planar(k33));
    Graph grid;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            auto id = [](int a, int b) { return std::format("{}_{}", a, b); };
            if (i + 1 < 6) grid.add_edge(id(i, j), id(i + 1, j));
            if (j + 1 < 6) grid.add_edge(id(i, j), id(i, j + 1));
        }
    assert(is_planar(grid));
    assert(pmfg(complete(6)).graph.number_of_edges() == 15);
    assert(boolean_filter(pmfg(complete(6)), 0, false).number_of_edges() == 12);  // 3(n-2)

    // binomial tails (scipy reference values)
    assert(near(binom_sf(2, 10, 0.3), 0.6172172136));
    assert(near(binom_cdf(2, 10, 0.3), 0.3827827864));
    assert(std::isnan(binom_cdf(2, 10.5, 0.3)));
    assert(near(binom_cdf(100, 620000000, 3e-7), 3.4037820208616282e-12, 1e-12));  // mpmath; small tail, huge n
    assert(near(binom_cdf(1, 620000000, 1e-9), 0.87146998896674252, 1e-12));

    // triangle with a heavy pair: a-b (10), b-c (10), a-c (1)
    Graph tri;
    tri.add_edge("a", "b", {{"weight", 10}});
    tri.add_edge("b", "c", {{"weight", 10}});
    tri.add_edge("a", "c", {{"weight", 1}});
    int ac = tri.find_edge(tri.node("a"), tri.node("c"));

    auto mst = maximum_spanning_tree(tri);
    assert(mst.graph.get(ac, "in_backbone") == 0 && boolean_filter(mst, 0, false).number_of_edges() == 2);

    auto metric = metric_distance_backbone(tri);  // a-c distance 1 > 0.1 + 0.1
    assert(metric.graph.get(ac, "in_backbone") == 0);
    assert(ultrametric_distance_backbone(tri).graph.get(ac, "in_backbone") == 0);

    auto disp = disparity(tri);  // node a: k=2, p=1/11, alpha=(1-1/11)^1; node c: same; min wins
    assert(near(disp.graph.get(ac, "p_value"), 10.0 / 11));
    assert(threshold_filter(disp, 0.5, false).number_of_edges() == 2);
    assert(fraction_filter(disp, 0.34, false).number_of_edges() == 2);  // ceil(0.34 * 3)

    auto hss = high_salience_skeleton(tri);
    assert(near(hss.graph.get(ac, "salience"), 0) && near(hss.graph.get(0, "salience"), 1));

    // edge betweenness on a path a-b-c: each edge carries 2 of the 6 ordered pairs
    Graph path;
    path.add_edge("a", "b", {{"weight", 1}});
    path.add_edge("b", "c", {{"weight", 1}});
    auto bc = edge_betweenness_centrality(path, true, "weight");
    assert(near(bc[0], 2.0 / 3) && near(bc[1], 2.0 / 3));
    assert(near(edge_betweenness_centrality(path, false, "")[0], 2));

    auto deg = degree(tri, true);
    assert(deg.graph.node_get(tri.node("b"), "weighted-degree") == 20);
    assert(fraction_filter(deg, 0.3, false).number_of_nodes() == 1);

    // measures
    assert(density(tri, tri) == 1 && transitivity(tri, tri) == 1 && average_clustering_coefficient(tri, tri) == 1);
    assert(diameter(path, path) == 2 && reachability(path, path) == 1);
    assert(near(ks_statistic({1, 2, 3}, {1, 2, 3}), 0) && near(ks_statistic({1, 2}, {3, 4}), 1));

    // proximity (docstring example of distance.py)
    Matrix B = {{1, 1, 1, 1}, {1, 1, 1, 0}, {1, 1, 0, 0}, {1, 0, 0, 0}};
    auto pb = pairwise_proximity(B, "jaccard_binary");
    assert(near(pb[0][1], 0.75) && near(pb[1][2], 2.0 / 3) && pb[0][0] == 1);

    // compare + consent
    Compare cmp;
    cmp.set_network(tri);
    cmp.add_backbone(mst);
    cmp.add_backbone(metric);
    cmp.set_filter(boolean_filter);
    cmp.add_property("Density", density);
    auto props = cmp.properties();
    assert(props.index.size() == 3 && near(props.data[1][0], 2.0 / 3, 1e-4));
    assert(cmp.consent().number_of_edges() == 2);

    if (argc > 1) {  // smoke test on a real network
        Graph g = Graph::read_csv(argv[1]);
        std::vector<std::pair<const char*, Backbone>> all = {
            {"disparity", disparity(g)},
            {"lans", lans(g)},
            {"noise_corrected", noise_corrected(g)},
            {"marginal_likelihood", marginal_likelihood(g)},
            {"ecm", ecm(g)},
            {"betweenness", betweenness(g)},
            {"degree", degree(g)},
            {"doubly_stochastic", doubly_stochastic(g)},
            {"global_threshold", global_threshold(g)},
            {"gspar", gspar(g)},
            {"h_backbone", h_backbone(g)},
            {"high_salience_skeleton", high_salience_skeleton(g)},
            {"maximum_spanning_tree", maximum_spanning_tree(g)},
            {"metric_distance_backbone", metric_distance_backbone(g)},
            {"ultrametric_distance_backbone", ultrametric_distance_backbone(g)},
            {"mlam", mlam(g)},
            {"plam", plam(g)},
            {"modularity_backbone", modularity_backbone(g)},
            {"glanb", glanb(g, 1)},
        };
        for (auto& [name, b] : all) {
            Graph x = b.filters.front()(b, 0.1, false, "weight", false);
            std::println("{}: {} nodes, {} edges", name, x.number_of_nodes(), x.number_of_edges());
        }
        auto mst_g = boolean_filter(maximum_spanning_tree(g), 0, false);
        assert(mst_g.number_of_edges() == g.number_of_nodes() - (int)connected_components(g).size());
    }
    std::println("all checks passed");
}
