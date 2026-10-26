#pragma once

#include <cstdint>
#include <iosfwd>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cppbone {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();

// Attribute values are doubles; booleans are stored as 1.0 / 0.0 and a missing value reads as NaN (like pandas).
using Attrs = std::map<std::string, double>;

struct EdgeRef {
    int u, v, id;
};

struct CsvRow;  // graph.cpp

// Undirected simple graph with string node labels, standing in for networkx.Graph and the pandas edge list.
// Node order, adjacency order and edges() order follow networkx iteration order.
class Graph {
public:
    // Copying and destroying a large graph touch one attribute map per edge, so both run in parallel.
    Graph() = default;
    Graph(const Graph& other);
    Graph(Graph&&) noexcept = default;
    Graph& operator=(const Graph& other);
    Graph& operator=(Graph&&) noexcept = default;
    ~Graph();

    int add_node(const std::string& label);
    // Adding an existing edge updates its attributes, like networkx.
    int add_edge(const std::string& u, const std::string& v, const Attrs& attrs = {});
    int add_edge(int u, int v, const Attrs& attrs = {});
    void remove_edge(int id);
    // remove_edge for every id with drop[id] set, in one pass (drop is indexed by edge id).
    void remove_edges(const std::vector<char>& drop);

    int number_of_nodes() const { return (int)labels_.size(); }
    int number_of_edges() const { return n_edges_; }
    int node(const std::string& label) const;  // -1 if absent
    const std::string& label(int u) const { return labels_[u]; }
    int find_edge(int u, int v) const;  // -1 if absent
    bool has_edge(int u, int v) const { return find_edge(u, v) >= 0; }
    std::vector<EdgeRef> edges() const;
    const std::vector<std::pair<int, int>>& adj(int u) const { return adj_[u]; }  // (neighbour, edge id)
    int degree(int u) const { return (int)adj_[u].size(); }
    double degree(int u, const std::string& weight) const;  // a missing weight counts as 1, like networkx
    std::pair<int, int> ends(int id) const { return ends_[id]; }

    double get(int id, const std::string& key) const;  // NaN when missing
    void set(int id, const std::string& key, double value) { attrs_[id][key] = value; }
    const Attrs& edge_attrs(int id) const { return attrs_[id]; }
    // Values indexed by edge id, `missing` where the attribute is absent.
    std::vector<double> edge_values(const std::string& key, double missing = 1.0) const;
    int edge_capacity() const { return (int)ends_.size(); }

    double node_get(int u, const std::string& key) const;
    void node_set(int u, const std::string& key, double value) { node_attrs_[u][key] = value; }
    const Attrs& node_attrs(int u) const { return node_attrs_[u]; }

    // Induced subgraph on the kept nodes, preserving order and attributes.
    Graph subgraph(const std::vector<bool>& keep) const;

    // shortcut: plain comma split, no quoted fields; upgrade to a real CSV parser if labels contain commas.
    static Graph read_csv(const std::string& path);
    // Header-less variant: columns are picked by 0-based index and the weight column (if >= 0) is stored as "weight";
    // other columns are ignored.
    static Graph read_csv(const std::string& path, int source_col, int target_col, int weight_col = 2);
    void to_csv(std::ostream& out) const;        // nx.to_pandas_edgelist
    void nodes_to_csv(std::ostream& out) const;  // node attribute table

private:
    static std::uint64_t key(int u, int v);
    void add_rows(std::vector<CsvRow>& rows);  // read_csv: insert parsed rows in file order

    std::vector<std::string> labels_;
    std::unordered_map<std::string, int> index_;
    std::vector<std::vector<std::pair<int, int>>> adj_;
    std::vector<Attrs> node_attrs_;
    std::vector<std::pair<int, int>> ends_;
    std::vector<Attrs> attrs_;
    // (u, v) key -> edge id, sharded so large graphs can copy the shards in parallel.
    static constexpr int EDGE_SHARDS = 256;
    std::vector<std::unordered_map<std::uint64_t, int>> edge_index_ =
        std::vector<std::unordered_map<std::uint64_t, int>>(EDGE_SHARDS);
    static int shard(std::uint64_t k) { return int((k * 0x9E3779B97F4A7C15ull) >> 56); }
    int n_edges_ = 0;
};

// Order of node labels: numeric when both parse as numbers (pandas int columns), lexicographic otherwise.
bool label_less(const std::string& a, const std::string& b);

std::string format_double(double x);

}  // namespace cppbone
