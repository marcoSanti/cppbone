#pragma once

// cppbone - backbone extraction from simple weighted networks.
// Every function mirrors the Python function of the same name; the comment names the module it comes from.
// Inputs are cppbone::Graph (use Graph::read_csv for a pandas-style edge list).

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "cppbone/backbone.hpp"
#include "cppbone/compare.hpp"
#include "cppbone/graph.hpp"

namespace cppbone {

// ---- parallelism ------------------------------------------------------------------------------------------
// Methods use OpenMP when the library is built with it (CMake option CPPBONE_OPENMP, on by default).
void set_num_threads(int n);  // n <= 0 restores the OpenMP default (OMP_NUM_THREADS or the core count)
int num_threads();            // threads a parallel region will use; 1 without OpenMP

// ---- statistical/ -------------------------------------------------------------------------------------------
Backbone disparity(const Graph& data, const std::string& weight = "weight");               // disparity.py
Backbone lans(const Graph& data);                                                          // lans.py
double compute_pvalue(const Graph& g, int node, double w);                                 // lans.py
Backbone noise_corrected(const Graph& data, bool approximation = true);                    // noise_corrected.py
Backbone marginal_likelihood(const Graph& data);                                           // __init__.py + marginal_likelihood.py
Backbone ecm(const Graph& data, std::uint32_t seed = 0);                                   // maxent_graph/ecm_main.py

// ---- structural/ --------------------------------------------------------------------------------------------
Backbone betweenness(const Graph& data, bool weighted = true, bool normalized = true);     // betweenness.py
Backbone degree(const Graph& data, bool weighted = false);                                 // degree.py
Backbone doubly_stochastic(const Graph& data);                                             // doubly_stochastic.py
Backbone global_threshold(const Graph& data);                                              // global_threshold.py
double jaccard(std::vector<int> a, std::vector<int> b);                                    // gspar.py
Backbone gspar(const Graph& data);                                                         // gspar.py
Backbone h_backbone(const Graph& data);                                                    // h_backbone.py
Backbone high_salience_skeleton(const Graph& data);                                        // high_salience_skeleton.py
Backbone maximum_spanning_tree(const Graph& data);                                         // maximum_spanning_tree.py
Backbone metric_distance_backbone(const Graph& data);                                      // metric_distance_backbone.py
Backbone ultrametric_distance_backbone(const Graph& data);                                 // ultrametric_distance_backbone.py
Backbone mlam(const Graph& data);                                                          // mlam.py
Backbone plam(const Graph& data);                                                          // plam.py
Backbone modularity_backbone(const Graph& data, std::uint32_t seed = 123);                 // modulairy_backbone.py
Backbone pmfg(const Graph& data);                                                          // pmfg.py
bool is_planar(const Graph& g);                                                            // pmfg.py (nx.is_planar)

// Edge betweenness by edge id (nx.edge_betweenness_centrality); weight "" means unweighted.
std::vector<double> edge_betweenness_centrality(const Graph& g, bool normalized, const std::string& weight);

// ---- hybrid/ ------------------------------------------------------------------------------------------------
Backbone glanb(const Graph& data, double c);                                               // glanb.py

// ---- structural/distanceclosure/ ----------------------------------------------------------------------------
enum class Disjunction { Sum, Max };  // sum: metric, max: ultrametric

// dijkstra.py. `weight` is indexed by edge id. Unreachable nodes get INF. `parent`, if given, receives the
// shortest-path tree (parent of the source and unreachable nodes is -1).
std::vector<double> single_source_dijkstra_path_length(const Graph& g, int source, const std::vector<double>& weight,
                                                       Disjunction disjunction = Disjunction::Sum,
                                                       std::vector<int>* parent = nullptr);
std::vector<std::vector<double>> all_pairs_dijkstra_path_length(const Graph& g, const std::string& weight = "weight",
                                                                Disjunction disjunction = Disjunction::Sum);

// backbone.py
Graph metric_backbone(const Graph& d, const std::string& weight = "weight");
Graph ultrametric_backbone(const Graph& d, const std::string& weight = "weight");
// Returns the backbone and the s-values ((u, v) labels, value) of removed edges.
std::pair<Graph, std::vector<std::pair<std::pair<std::string, std::string>, double>>> metric_backbone_distortion(
    const Graph& d, const std::string& weight = "weight");

// closure.py
Graph distance_closure(const Graph& d, const std::string& kind = "metric", const std::string& algorithm = "dijkstra",
                       const std::string& weight = "weight", bool only_backbone = false);
Graph s_values(const Graph& cm, const std::string& weight_distance = "distance",
               const std::string& weight_metric_distance = "metric_distance");
Graph b_values(const Graph& cm, const std::string& weight_distance = "distance",
               const std::string& weight_metric_distance = "metric_distance");

// distance.py (dense matrices only)
using Matrix = std::vector<std::vector<double>>;
Matrix pairwise_proximity(const Matrix& m, const std::string& metric = "jaccard", double min_support = -1);

// utils.py
double prox2dist(double p);
double dist2prox(double d);

// ---- measures.py --------------------------------------------------------------------------------------------
double density(const Graph& original, const Graph& graph);
double weight_entropy(const Graph& original, const Graph& graph);
double average_degree(const Graph& original, const Graph& graph);
double lcc_size(const Graph& original, const Graph& graph);
double node_fraction(const Graph& network, const Graph& b);
double edge_fraction(const Graph& network, const Graph& b);
double weight_fraction(const Graph& network, const Graph& b);
double reachability(const Graph& original, const Graph& g);
double number_connected_components(const Graph& original, const Graph& g);
double diameter(const Graph& original, const Graph& g);
double lcc_node_fraction(const Graph& g);
double lcc_edge_fraction(const Graph& original, const Graph& g);
double lcc_weight_fraction(const Graph& original, const Graph& g);
std::vector<double> weights(const Graph& g);
std::vector<double> degrees(const Graph& g, const std::string& weight = "");
double average_clustering_coefficient(const Graph& original, const Graph& g);
double transitivity(const Graph& original, const Graph& graph);

// ---- utils/utils.py -----------------------------------------------------------------------------------------
std::vector<std::vector<int>> connected_components(const Graph& g);
Graph lcc(const Graph& g);
// (x, y) points of the cumulative distribution.
std::vector<std::pair<double, double>> cumulative_dist(std::vector<double> values, bool increasing = true);
// scipy binom.cdf / binom.sf; NaN when n is not integral, like scipy.
double binom_cdf(double k, double n, double p);
double binom_sf(double k, double n, double p);

}  // namespace cppbone
