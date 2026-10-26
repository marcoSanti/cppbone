#pragma once

/// @file
/// @brief Umbrella header: every backbone method, the distance-closure toolkit, the measures and the utilities.
///
/// Every function mirrors the Python function of the same name; each description names the module it comes from.
/// Inputs are cppbone::Graph (use Graph::read_csv for a pandas-style edge list).

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "cppbone/backbone.hpp"
#include "cppbone/compare.hpp"
#include "cppbone/graph.hpp"

/// @brief Backbone extraction from simple weighted networks.
namespace cppbone {

/// @name Parallelism
/// Methods use OpenMP when the library is built with it (CMake option CPPBONE_OPENMP, on by default).
/// @{

/// @brief Sets the number of threads used by the parallel regions of every method.
/// @param n Thread count; n <= 0 restores the OpenMP default (OMP_NUM_THREADS or the core count). Ignored without
///          OpenMP.
void set_num_threads(int n);

/// @brief Returns how many threads a parallel region will use.
/// @return The OpenMP maximum thread count, or 1 when the library is built without OpenMP.
int num_threads();
/// @}

/// @name Statistical methods (statistical/)
/// @{

/// @brief Disparity filter (disparity.py).
///
/// For every endpoint of degree k > 1 and strength s, an edge of weight w gets alpha = (1 - w / s)^(k - 1), the
/// probability of a share at least this large under uniform random splitting. The edge keeps the smaller alpha of
/// its two endpoints as "p_value".
/// @param data Input graph with a "weight" edge attribute.
/// @param weight Attribute used for the node strength.
/// @return Backbone on "p_value" (ascending), filtered with threshold_filter or fraction_filter.
Backbone disparity(const Graph& data, const std::string& weight = "weight");

/// @brief Locally Adaptive Network Sparsification (lans.py).
///
/// Each edge's "p_value" is the smaller, over its endpoints, of the fraction of the endpoint's edges whose share
/// of its strength is larger than this edge's share.
/// @param data Input graph with a "weight" edge attribute.
/// @return Backbone on "p_value" (ascending), filtered with threshold_filter or fraction_filter.
Backbone lans(const Graph& data);

/// @brief LANS p-value of one edge seen from one endpoint (lans.py).
/// @param g Graph with a "weight" edge attribute.
/// @param node Endpoint whose neighbourhood is used.
/// @param w Weight of the edge.
/// @return 1 minus the fraction of the node's edges whose normalised weight is at most w / strength(node).
double compute_pvalue(const Graph& g, int node, double w);

/// @brief Noise Corrected filter (noise_corrected.py).
///
/// Models each edge weight as a binomial draw whose probability is set by the strengths of its endpoints.
/// @param data Input graph with a "weight" edge attribute.
/// @param approximation true: store "p_value" = 1 - BinomCDF(w; N, s_u s_v / N^2), N the total weight.
///        false: store the Bayesian "score" and its standard deviation "nc_sdev" instead (no "p_value").
/// @return Backbone on "p_value" (ascending), filtered with threshold_filter or fraction_filter.
Backbone noise_corrected(const Graph& data, bool approximation = true);

/// @brief Marginal Likelihood filter (__init__.py and marginal_likelihood.py), undirected variant.
///
/// "p_value" is P(X >= w) for X ~ Binomial(T, k_u k_v / (2 T^2)), with k the strengths and T half their sum.
/// @param data Input graph with non-negative integer "weight" values.
/// @return Backbone whose graph keeps only "weight" and "p_value", on "p_value" (ascending).
/// @throws std::invalid_argument if a weight is negative.
Backbone marginal_likelihood(const Graph& data);

/// @brief Enhanced Configuration Model filter (maxent_graph/ecm_main.py).
///
/// Fits the maximum-entropy model that preserves every node's degree and strength (Levenberg-Marquardt on the
/// log-likelihood), then sets "p_value" = p_ij y_i y_j^(w - 1), the probability of a weight at least w. Self-loops
/// and zero-weight edges get no p-value.
/// @param data Input graph with non-negative integer "weight" values.
/// @param seed Seed of the random starting point.
/// @return Backbone on "p_value" (ascending), filtered with threshold_filter or fraction_filter.
/// @throws std::runtime_error if the fit fails with a large residual.
Backbone ecm(const Graph& data, std::uint32_t seed = 0);
/// @}

/// @name Structural methods (structural/)
/// @{

/// @brief Edge betweenness backbone (betweenness.py).
/// @param data Input graph.
/// @param weighted true: use "weight" as the path length and store "weighted-betweenness"; false: hop count and
///        "betweenness".
/// @param normalized Divide by n (n - 1) instead of 2.
/// @return Backbone on the betweenness (descending), filtered with threshold_filter or fraction_filter.
Backbone betweenness(const Graph& data, bool weighted = true, bool normalized = true);

/// @brief Degree backbone (degree.py): keeps the nodes with the largest degree.
/// @param data Input graph.
/// @param weighted true: node property "weighted-degree" (strength); false: "degree".
/// @return Node backbone on the degree (descending), filtered with threshold_filter or fraction_filter.
Backbone degree(const Graph& data, bool weighted = false);

/// @brief Doubly Stochastic filter (doubly_stochastic.py).
///
/// Rescales the source-by-target weight table with Sinkhorn-Knopp until rows and columns sum to one; the result is
/// the edge "score". "in_backbone" marks the edges added by decreasing score until the graph is connected and spans
/// every node.
/// @param data Input graph with a "weight" edge attribute.
/// @return Backbone on "score" (descending); an empty graph if Sinkhorn-Knopp does not converge in 1000 rounds.
Backbone doubly_stochastic(const Graph& data);

/// @brief Global threshold (global_threshold.py): ranks the edges by their own weight.
/// @param data Input graph with a "weight" edge attribute.
/// @return Backbone on "weight" (descending), filtered with fraction_filter or threshold_filter.
Backbone global_threshold(const Graph& data);

/// @brief Jaccard similarity of two sets given as lists (gspar.py).
/// @param a First set; duplicates are ignored.
/// @param b Second set; duplicates are ignored.
/// @return |a and b| / |a or b|.
double jaccard(std::vector<int> a, std::vector<int> b);

/// @brief Global Sparsification (gspar.py).
///
/// "jaccard-sim" is the Jaccard similarity of the closed neighbourhoods (node plus neighbours) of the endpoints.
/// @param data Input graph.
/// @return Backbone on "jaccard-sim" (descending), filtered with threshold_filter or fraction_filter.
Backbone gspar(const Graph& data);

/// @brief H-Backbone (h_backbone.py).
///
/// "h_bridge" is the unnormalised weighted edge betweenness divided by n, rounded to 3 decimals. An edge is
/// "in_backbone" when its bridge value reaches the h-index of all bridge values or its weight reaches the h-index
/// of all weights.
/// @param data Input graph with a "weight" edge attribute.
/// @return Backbone on "h_bridge", filtered with boolean_filter.
Backbone h_backbone(const Graph& data);

/// @brief High Salience Skeleton (high_salience_skeleton.py).
///
/// With distance = 1 / weight, "salience" is the fraction of nodes whose shortest-path tree contains the edge;
/// "in_backbone" marks salience above 0.8.
/// @param data Input graph with a "weight" edge attribute.
/// @return Backbone on "salience" (descending), filtered with any of the three filters.
Backbone high_salience_skeleton(const Graph& data);

/// @brief Maximum spanning tree (maximum_spanning_tree.py).
///
/// Kruskal on decreasing weight (stable on ties); "in_backbone" marks the tree edges.
/// @param data Input graph with a "weight" edge attribute.
/// @return Backbone on "weight", filtered with boolean_filter.
Backbone maximum_spanning_tree(const Graph& data);

/// @brief Metric distance backbone (metric_distance_backbone.py).
///
/// With distance = 1 / weight, "in_backbone" marks the edges that are a shortest path between their endpoints.
/// @param data Input graph with a "weight" edge attribute.
/// @return Backbone on "distance", filtered with boolean_filter.
Backbone metric_distance_backbone(const Graph& data);

/// @brief Ultrametric distance backbone (ultrametric_distance_backbone.py).
///
/// Like metric_distance_backbone, with the length of a path being its largest distance instead of the sum.
/// @param data Input graph with a "weight" edge attribute.
/// @return Backbone on "distance", filtered with boolean_filter.
Backbone ultrametric_distance_backbone(const Graph& data);

/// @brief Multiple Linkage Analysis (mlam.py).
///
/// Each node keeps its k strongest edges, k chosen where the coefficient of determination between its sorted
/// weight shares and the ideal "k equal links" distribution stops increasing. "in_backbone" marks edges kept by
/// either endpoint.
/// @param data Input graph with a "weight" edge attribute.
/// @return Backbone on "weight", filtered with boolean_filter.
Backbone mlam(const Graph& data);

/// @brief Primary Linkage Analysis (plam.py): every node keeps its strongest edge ("in_backbone").
/// @param data Input graph with a "weight" edge attribute.
/// @return Backbone on "weight", filtered with boolean_filter.
Backbone plam(const Graph& data);

/// @brief Modularity backbone (modulairy_backbone.py).
///
/// Finds Louvain communities, then gives every node its "vitality": how much the modularity changes when the
/// node is removed.
/// @param data Input graph with a "weight" edge attribute.
/// @param seed Seed of the Louvain node order.
/// @return Node backbone on "vitality" (descending), filtered with threshold_filter or fraction_filter.
Backbone modularity_backbone(const Graph& data, std::uint32_t seed = 123);

/// @brief Planar Maximally Filtered Graph (pmfg.py).
///
/// Adds edges by decreasing weight, skipping those that would make the graph non-planar, until it has
/// 3 (n - 2) edges. "in_backbone" marks the added edges.
/// @param data Input graph with a "weight" edge attribute.
/// @return Backbone on "weight", filtered with boolean_filter.
Backbone pmfg(const Graph& data);

/// @brief Planarity test (pmfg.py, networkx is_planar), using the Left-Right algorithm.
/// @param g Graph to test; self-loops are ignored.
/// @return true if g can be drawn in the plane without crossings.
bool is_planar(const Graph& g);

/// @brief Edge betweenness centrality (networkx edge_betweenness_centrality), including its tie handling.
/// @param g Graph.
/// @param normalized Divide by n (n - 1) instead of 2.
/// @param weight Edge attribute used as the path length; "" counts hops.
/// @return Betweenness indexed by edge id (size edge_capacity()).
std::vector<double> edge_betweenness_centrality(const Graph& g, bool normalized, const std::string& weight);
/// @}

/// @name Hybrid methods (hybrid/)
/// @{

/// @brief Globally and Locally Adaptive Backbone (glanb.py).
///
/// For an edge (i, j), I_ij is the share of i's shortest paths (distance = 1 / weight) whose first hop is j, and
/// S_ij = (1 - I_ij)^((k_i - 1)^c). The edge keeps the smaller S of its endpoints of degree > 1 as "SI".
/// @param data Input graph with a "weight" edge attribute.
/// @param c Exponent balancing the local (degree) and global (paths) terms.
/// @return Backbone on "SI" (ascending), filtered with threshold_filter or fraction_filter.
Backbone glanb(const Graph& data, double c);
/// @}

/// @name Distance closure (structural/distanceclosure/)
/// @{

/// @brief How path lengths combine: Sum gives metric distances, Max ultrametric (minimax) distances.
enum class Disjunction {
    Sum,  ///< Length of a path is the sum of its edge distances.
    Max   ///< Length of a path is its largest edge distance.
};

/// @brief Shortest-path lengths from one node (dijkstra.py).
/// @param g Graph.
/// @param source Start node.
/// @param weight Edge distances indexed by edge id; must be non-negative.
/// @param disjunction How distances combine along a path.
/// @param parent If given, receives the shortest-path tree: parent[v] is v's predecessor, -1 for the source and
///        unreachable nodes.
/// @return Distance of every node from source, INF when unreachable.
/// @throws std::invalid_argument when a negative distance makes the paths contradictory.
std::vector<double> single_source_dijkstra_path_length(const Graph& g, int source, const std::vector<double>& weight,
                                                       Disjunction disjunction = Disjunction::Sum,
                                                       std::vector<int>* parent = nullptr);

/// @brief Shortest-path lengths between every pair of nodes (dijkstra.py).
/// @param g Graph.
/// @param weight Edge attribute holding the distances.
/// @param disjunction How distances combine along a path.
/// @return Dense matrix: result[u][v] is the distance from u to v, INF when unreachable.
std::vector<std::vector<double>> all_pairs_dijkstra_path_length(const Graph& g, const std::string& weight = "weight",
                                                                Disjunction disjunction = Disjunction::Sum);

/// @brief Metric backbone (backbone.py): removes every edge that is not a shortest path between its endpoints.
/// @param d Graph whose `weight` attribute holds distances.
/// @param weight Distance attribute.
/// @return Copy of d without the redundant edges.
Graph metric_backbone(const Graph& d, const std::string& weight = "weight");

/// @brief Ultrametric backbone (backbone.py): like metric_backbone with minimax path lengths.
/// @param d Graph whose `weight` attribute holds distances.
/// @param weight Distance attribute.
/// @return Copy of d without the redundant edges.
Graph ultrametric_backbone(const Graph& d, const std::string& weight = "weight");

/// @brief Metric backbone and the distortion of the removed edges (backbone.py).
/// @param d Graph whose `weight` attribute holds distances.
/// @param weight Distance attribute.
/// @return The backbone, and for every removed edge its endpoint labels and s-value (distance / shortest path).
std::pair<Graph, std::vector<std::pair<std::pair<std::string, std::string>, double>>> metric_backbone_distortion(
    const Graph& d, const std::string& weight = "weight");

/// @brief Transitive closure of a distance graph (closure.py).
///
/// Every connected pair gets its shortest-path length as "<kind>_distance". Existing edges also get
/// "is_<kind>" (their distance is the shortest path); missing pairs are added with an infinite `weight`.
/// @param d Graph whose `weight` attribute holds distances.
/// @param kind "metric" or "ultrametric".
/// @param algorithm Only "dijkstra" is implemented.
/// @param weight Distance attribute.
/// @param only_backbone Do not add the missing pairs.
/// @return The closure graph.
/// @throws std::invalid_argument for an unknown kind or algorithm; std::logic_error for "dense".
Graph distance_closure(const Graph& d, const std::string& kind = "metric", const std::string& algorithm = "dijkstra",
                       const std::string& weight = "weight", bool only_backbone = false);

/// @brief s-values of a closure graph (closure.py): "s-value" = distance / closure distance.
/// @param cm Closure graph from distance_closure.
/// @param weight_distance Original distance attribute.
/// @param weight_metric_distance Closure distance attribute.
/// @return Copy of cm with "s-value" on the edges of finite distance and positive closure distance.
Graph s_values(const Graph& cm, const std::string& weight_distance = "distance",
               const std::string& weight_metric_distance = "metric_distance");

/// @brief b-values of a closure graph (closure.py).
///
/// For the pairs added by the closure, "b_ij-value" and "b_ji-value" are the mean original distance around each
/// endpoint divided by the closure distance.
/// @param cm Closure graph from distance_closure.
/// @param weight_distance Original distance attribute (infinite on the added pairs).
/// @param weight_metric_distance Closure distance attribute.
/// @return Copy of cm with the b-values set.
Graph b_values(const Graph& cm, const std::string& weight_distance = "distance",
               const std::string& weight_metric_distance = "metric_distance");

/// @brief Dense matrix, row-major: m[row][column].
using Matrix = std::vector<std::vector<double>>;

/// @brief Jaccard proximity between every pair of rows of a dense matrix (distance.py).
/// @param m Non-negative matrix.
/// @param metric "jaccard" (alias "scipy", "j"), "jaccard_binary" ("jb"), "jaccard_set" ("js") or
///        "jaccard_weighted" ("weighted_jaccard", "wj").
/// @param min_support Pairs with less support get 0; negative selects the metric's default (10 for weighted,
///        1 otherwise).
/// @return Symmetric matrix of proximities with ones on the diagonal.
/// @throws std::invalid_argument for negative entries or an unknown metric.
Matrix pairwise_proximity(const Matrix& m, const std::string& metric = "jaccard", double min_support = -1);

/// @brief Converts a [0, 1] proximity to a [0, inf] distance, d = 1 / p - 1 (utils.py).
/// @param p Proximity.
/// @return Distance; INF for p = 0.
double prox2dist(double p);

/// @brief Converts a [0, inf] distance to a [0, 1] proximity, p = 1 / (d + 1) (utils.py).
/// @param d Distance.
/// @return Proximity; 0 for d = INF.
double dist2prox(double d);
/// @}

/// @name Measures (measures.py)
/// Properties comparing a backbone with the original network, usable with Compare::add_property.
/// @{

/// @brief Edge density of a graph, rounded to 4 decimals.
/// @param original Unused (kept for the Property signature).
/// @param graph Graph measured.
/// @return 2 m / (n (n - 1)), 0 for fewer than two nodes.
double density(const Graph& original, const Graph& graph);

/// @brief Shannon entropy, in bits, of the edge weights normalised to sum to one.
/// @param original Unused (kept for the Property signature).
/// @param graph Graph measured.
/// @return The entropy; zero weights do not contribute.
double weight_entropy(const Graph& original, const Graph& graph);

/// @brief Average node degree.
/// @param original Unused (kept for the Property signature).
/// @param graph Graph measured.
/// @return Mean degree, NaN for an empty graph.
double average_degree(const Graph& original, const Graph& graph);

/// @brief Number of nodes in the largest connected component.
/// @param original Unused (kept for the Property signature).
/// @param graph Graph measured.
/// @return Size of the largest component.
double lcc_size(const Graph& original, const Graph& graph);

/// @brief Fraction of the network's nodes kept by the backbone.
/// @param network Original network.
/// @param b Backbone.
/// @return nodes(b) / nodes(network).
double node_fraction(const Graph& network, const Graph& b);

/// @brief Fraction of the network's edges kept by the backbone.
/// @param network Original network.
/// @param b Backbone.
/// @return edges(b) / edges(network).
double edge_fraction(const Graph& network, const Graph& b);

/// @brief Fraction of the network's total weight kept by the backbone.
/// @param network Original network.
/// @param b Backbone.
/// @return Sum of b's weights / sum of the network's weights.
double weight_fraction(const Graph& network, const Graph& b);

/// @brief Fraction of ordered node pairs joined by a path.
/// @param original Unused (kept for the Property signature).
/// @param g Graph measured.
/// @return Sum over components of s (s - 1), divided by n (n - 1).
double reachability(const Graph& original, const Graph& g);

/// @brief Number of connected components.
/// @param original Unused (kept for the Property signature).
/// @param g Graph measured.
/// @return The component count.
double number_connected_components(const Graph& original, const Graph& g);

/// @brief Unweighted diameter of the largest connected component (igraph diameter, unconn=True).
/// @param original Unused (kept for the Property signature).
/// @param g Graph measured.
/// @return Longest shortest path, in hops.
double diameter(const Graph& original, const Graph& g);

/// @brief Fraction of a graph's nodes in its largest connected component.
/// @param g Graph measured.
/// @return nodes(lcc(g)) / nodes(g).
double lcc_node_fraction(const Graph& g);

/// @brief Fraction of a graph's edges in its largest connected component.
/// @param original Unused (kept for the Property signature).
/// @param g Graph measured.
/// @return edges(lcc(g)) / edges(g).
double lcc_edge_fraction(const Graph& original, const Graph& g);

/// @brief Fraction of a graph's total weight in its largest connected component.
/// @param original Unused (kept for the Property signature).
/// @param g Graph measured.
/// @return Weight of lcc(g) / weight of g.
double lcc_weight_fraction(const Graph& original, const Graph& g);

/// @brief Edge weights of a graph.
/// @param g Graph.
/// @return The "weight" of every edge that has one, in edges() order.
std::vector<double> weights(const Graph& g);

/// @brief Node degrees of a graph.
/// @param g Graph.
/// @param weight "" for the degree, or the attribute whose sum gives the strength.
/// @return One value per node, in node order.
std::vector<double> degrees(const Graph& g, const std::string& weight = "");

/// @brief Mean local clustering coefficient over the nodes of degree >= 2 (igraph mode="nan").
/// @param original Unused (kept for the Property signature).
/// @param g Graph measured.
/// @return The average, NaN when no node has degree >= 2.
double average_clustering_coefficient(const Graph& original, const Graph& g);

/// @brief Global clustering coefficient: closed triads over all triads.
/// @param original Unused (kept for the Property signature).
/// @param graph Graph measured.
/// @return The transitivity, 0 when there are no triads.
double transitivity(const Graph& original, const Graph& graph);
/// @}

/// @name Utilities (utils/utils.py)
/// @{

/// @brief Connected components (Boost.Graph).
/// @param g Graph.
/// @return The components, ordered by their lowest node, each listing its nodes in increasing order.
std::vector<std::vector<int>> connected_components(const Graph& g);

/// @brief Largest connected component (the first one on ties).
/// @param g Graph.
/// @return Induced subgraph on the largest component; g itself when it has no nodes.
Graph lcc(const Graph& g);

/// @brief Points of the cumulative distribution of a sample.
/// @param values Sample.
/// @param increasing true: values sorted in decreasing order, so y is the fraction of values >= x;
///        false: increasing order, the fraction <= x.
/// @return (value, rank / size) for every value, in sorted order.
std::vector<std::pair<double, double>> cumulative_dist(std::vector<double> values, bool increasing = true);

/// @brief Binomial cumulative distribution P(X <= k), X ~ Binomial(n, p) (scipy binom.cdf).
/// @param k Number of successes (floored).
/// @param n Number of trials.
/// @param p Success probability.
/// @return The probability; NaN when n is not a non-negative integer or p is outside [0, 1].
double binom_cdf(double k, double n, double p);

/// @brief Binomial survival function P(X > k), X ~ Binomial(n, p) (scipy binom.sf).
/// @param k Number of successes (floored).
/// @param n Number of trials.
/// @param p Success probability.
/// @return The probability; NaN when n is not a non-negative integer or p is outside [0, 1].
double binom_sf(double k, double n, double p);
/// @}

}  // namespace cppbone
